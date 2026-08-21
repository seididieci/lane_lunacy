// SPDX-License-Identifier: MIT

//! Ray-traced renderer backend.
//!
//! [`RayTraceResources`] owns everything the ray-tracing path needs: the
//! bottom-level acceleration structures (one per mesh slot: the player, the
//! four traffic car models, and each chunk in the sliding window), the top-level
//! structure rebuilt per image-in-flight, the ray-tracing pipeline + shader
//! binding table, and the compact per-frame vertex pool the closest-hit shader
//! shades from (vec4-aligned, 3 vec4 per vertex: position / normal /
//! uv+material).
//!
//! [`RayTraceResources::record`] is recorded into the renderer's ordinary
//! command buffer *before* the bloom + post composite. It fires the rays into
//! an internal storage image and then copies it into the offscreen color
//! target, so the bloom chain, post composite and HUD keep reading the same
//! images as the raster path and never know ray tracing exists. The raster
//! scene, puddle mask and planar-reflection passes are simply skipped by the
//! caller when the RT path runs, and the composite samples the offscreen as the
//! reflection source via [`ReflectionMethod::RayTraced`].
//!
//! Every mutating resource is indexed by image-in-flight so a rebuild can never
//! race a command buffer that is still executing: the renderer waits on the
//! fence for `image_i` before recording, so replacing `image_i`'s buffers and
//! acceleration structures is always safe.

use std::ffi::CString;
use std::mem::MaybeUninit;
use std::num::NonZeroU64;
use std::ptr;
use std::sync::Arc;

use glam::{Mat4, Vec3};
use smallvec::smallvec;
use vulkano::acceleration_structure::{
    AccelerationStructure, AccelerationStructureBuildGeometryInfo,
    AccelerationStructureBuildRangeInfo, AccelerationStructureBuildType,
    AccelerationStructureCreateInfo, AccelerationStructureGeometries,
    AccelerationStructureGeometryInstancesData, AccelerationStructureGeometryInstancesDataType,
    AccelerationStructureGeometryTrianglesData, AccelerationStructureInstance, AccelerationStructureType,
    BuildAccelerationStructureMode, TransformMatrix,
};
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, IndexBuffer, Subbuffer};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, BlitImageInfo, PrimaryAutoCommandBuffer,
};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::{
    AllocationCreateInfo, DeviceLayout, MemoryTypeFilter, StandardMemoryAllocator,
};
use vulkano::memory::DeviceAlignment;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::ray_tracing::{
    RayTracingPipeline, RayTracingPipelineCreateInfo, RayTracingShaderGroupCreateInfo,
    ShaderBindingTable,
};
use vulkano::pipeline::{PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::shader::{ShaderModule, ShaderModuleCreateInfo};
use vulkano::shader::ShaderStage;
use vulkano::{Packed24_8, VulkanObject};

use crate::game::Game;
use crate::render::frame::{traffic_rotation, Frame};
use crate::render::frame_builder::WorldChunk;
use crate::render::scene::SceneResources;
use crate::road::road_curve;
use crate::shaders::{self, RtUniforms};
use crate::vertex::Vertex3d;

/// Vertex pool capacities. Must match the `#define`s in
/// `shaders/raytrace.rchit.glsl`. One vertex occupies 3 `vec4` (position /
/// normal / uv+material), so `VERT_CAP_VEC4` covers the measured worst-case
/// window (eight HIGH chunks plus the player and traffic cars: 3.24 M vertices
/// / 4.86 M indices) with ~40% headroom for denser roadside scenery.
const VERT_CAP_VEC4: usize = 14_000_000;
const INDEX_CAP: usize = 7_000_000;
const SLOT_CAP: usize = 256;
/// Player + traffic + the 8-chunk window + slack.
const INSTANCE_CAP: usize = 1 + 8 + 8 + 8;
/// Acceleration-structure storage and slices must be 256-byte aligned.
const AS_ALIGN_BYTES: u64 = 256;
/// The raygen's shadow probes trace with cull mask `0x01` (see
/// `raytrace.rgen.glsl`); chunk (world) instances carry bit 0 set (`0x01`) so
/// only terrain/walls/trees occlude, and the car statics carry bit 0 clear
/// (`0xfe`) so the player + traffic cars never cast a shadow. The primary ray
/// uses mask `0xFF` and still hits everything.
const STATIC_INSTANCE_MASK: u32 = 0xfe;
const CHUNK_INSTANCE_MASK: u32 = 0x01;

/// Chunk slots in the window: 1 behind + current + 6 ahead. Each chunk index
/// maps to a FIXED slot (`static_slots + (index mod CHUNK_SLOT_COUNT)`), so a
/// window crossing only rewrites the entering chunk's slot instead of the whole
/// pool.
const CHUNK_SLOT_COUNT: usize = 8;

/// Ray-traced output is rendered at this × this resolution and downscaled into
/// the offscreen, averaging per-pixel normal/texture noise the way the raster
/// path's MSAA does (the RT stage can't select texture mips on this driver).
const SUPERSAMPLE: u32 = 2;

/// Per-chunk vertex/index capacity. Sizes every chunk slot for the worst case
/// (measured: HIGH ~404K verts / ~605K indices) so the slot's pool region and
/// BLAS storage never move when neighbours enter/leave.
const CHUNK_VERT_CAP: u32 = 430_000;
const CHUNK_INDEX_CAP: u32 = 630_000;

/// Fixed pool region for one mesh slot. `vertex_base` is in `vec4` units and
/// never changes once the layout is built, so the BLAS over a slot stays valid
/// unless that slot's own geometry is rewritten.
#[derive(Clone, Copy, Debug)]
struct SlotLayout {
    vertex_base: u32,
    vertex_cap: u32,
    index_base: u32,
    index_cap: u32,
}

/// One mesh in the RT vertex pool. Mirrors the `RtSlot` struct in
/// `raytrace.rchit.glsl`. `vertex_base` is in `vec4` units (vertex `i` runs at
/// `vertex_base + i * 3`).
#[derive(BufferContents, Clone, Copy, Debug)]
#[repr(C)]
pub struct RtSlot {
    pub vertex_base: u32,
    pub vertex_count: u32,
    pub index_base: u32,
    pub index_count: u32,
}

/// Allocates a 256-byte-aligned, host-writable buffer.
fn aligned_buffer(
    allocator: Arc<StandardMemoryAllocator>,
    size: u64,
    usage: BufferUsage,
) -> Subbuffer<[u8]> {
    let layout = DeviceLayout::new(
        NonZeroU64::new(size).expect("non-zero buffer size"),
        DeviceAlignment::new(AS_ALIGN_BYTES).expect("valid alignment"),
    )
    .expect("valid buffer layout");
    let buffer = Buffer::new(
        allocator,
        BufferCreateInfo {
            usage,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        layout,
    )
    .expect("ray tracing buffer");
    Subbuffer::new(buffer)
}

/// Converts a `glam::Mat4` (column-major) into the row-major 3x4 instance
/// transform the acceleration structure expects.
fn to_transform(m: &Mat4) -> TransformMatrix {
    let cols = m.to_cols_array_2d();
    let mut t = [[0.0f32; 4]; 3];
    for r in 0..3 {
        for c in 0..3 {
            t[r][c] = cols[c][r];
        }
        t[r][3] = cols[3][r];
    }
    t
}

pub struct RayTraceResources {
    device: Arc<Device>,
    pipeline: Arc<RayTracingPipeline>,
    layout: Arc<PipelineLayout>,
    sbt: ShaderBindingTable,

    verts_pools: Vec<Subbuffer<[f32]>>,
    index_pools: Vec<Subbuffer<[u32]>>,
    slot_pools: Vec<Subbuffer<[RtSlot]>>,
    instance_buffers: Vec<Subbuffer<[AccelerationStructureInstance]>>,
    blas_scratch: Vec<Subbuffer<[u8]>>,
    tlas_scratch: Vec<Subbuffer<[u8]>>,
    blas: Vec<Vec<Arc<AccelerationStructure>>>,
    tlas: Vec<Arc<AccelerationStructure>>,

    output_image: Option<Arc<Image>>,
    output_view: Option<Arc<ImageView>>,
    output_size: [u32; 2],

    /// R32 linear eye-distance depth written by the raygen, sampled by the RT
    /// particle overlay pass for per-pixel occlusion of rain/mist/dust.
    depth_image: Option<Arc<Image>>,
    depth_view: Option<Arc<ImageView>>,
    /// 1x depth resolved from the supersampled chain for the particle overlay.
    depth_resolved_image: Option<Arc<Image>>,
    depth_resolved_view: Option<Arc<ImageView>>,

    /// Fixed per-slot pool regions (statics + chunk slots), built once.
    slot_layouts: Vec<SlotLayout>,
    /// Number of static slots (player + traffic models); chunk slots follow.
    static_slots: usize,
    /// Which chunk index currently occupies each chunk slot (for instance
    /// writes / TLAS). Only the entering chunk changes per window crossing.
    chunk_owner: Vec<Option<i32>>,
    /// CPU mirror of the packed pool layout, one [`RtSlot`] per slot.
    slots: Vec<RtSlot>,
    /// Per-slot CPU copy of the packed vertex/index arrays. Statics are staged
    /// once; a chunk slot is rewritten only when its owning chunk changes.
    slot_data: Vec<Option<(Vec<f32>, Vec<u32>)>>,
    /// Per-image, per-slot flag: that slot's pool region + BLAS need (re)writing
    /// on that image's own frame (after its fence wait).
    blas_dirty: Vec<Vec<bool>>,
    last_chunk_indices: Vec<i32>,
}

impl RayTraceResources {
    /// Builds every GPU resource for the ray-traced path. Panics if ray
    /// tracing is unavailable; the caller gates on `gpu::ray_tracing_supported`.
    pub fn new(device: Arc<Device>, scene: &SceneResources, images: usize) -> Arc<Self> {
        let rgen = unsafe {
            ShaderModule::new(
                device.clone(),
                ShaderModuleCreateInfo::new(&shaders::spv_words(shaders::RAYTRACE_RGEN_SPV)),
            )
        }
        .expect("raygen shader module");
        let rmiss = unsafe {
            ShaderModule::new(
                device.clone(),
                ShaderModuleCreateInfo::new(&shaders::spv_words(shaders::RAYTRACE_RMISS_SPV)),
            )
        }
        .expect("miss shader module");
        let rchit = unsafe {
            ShaderModule::new(
                device.clone(),
                ShaderModuleCreateInfo::new(&shaders::spv_words(shaders::RAYTRACE_RCHIT_SPV)),
            )
        }
        .expect("closest-hit shader module");
        let rshadowmiss = unsafe {
            ShaderModule::new(
                device.clone(),
                ShaderModuleCreateInfo::new(&shaders::spv_words(shaders::RAYTRACE_RSMISS_SPV)),
            )
        }
        .expect("shadow miss shader module");
        let rshadowhit = unsafe {
            ShaderModule::new(
                device.clone(),
                ShaderModuleCreateInfo::new(&shaders::spv_words(shaders::RAYTRACE_RSHAD_SPV)),
            )
        }
        .expect("shadow any-hit shader module");

        let stages = vec![
            PipelineShaderStageCreateInfo::new(rgen.entry_point("main").unwrap()),
            PipelineShaderStageCreateInfo::new(rmiss.entry_point("main").unwrap()),
            PipelineShaderStageCreateInfo::new(rchit.entry_point("main").unwrap()),
            PipelineShaderStageCreateInfo::new(rshadowmiss.entry_point("main").unwrap()),
            PipelineShaderStageCreateInfo::new(rshadowhit.entry_point("main").unwrap()),
        ];
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .expect("ray tracing pipeline layout");

        // vulkano 0.35's `RayTracingPipelineCreateInfo` never sets
        // `max_pipeline_ray_payload_size` / `max_pipeline_ray_hit_attribute_size`
        // (both default to 0), and the spec (VUID-03447) requires the payload
        // size to be at least the largest payload declared by any shader in the
        // pipeline. With 0 the driver silently drops all payload writes, so the
        // raygen reads back its own pre-trace value and the image is black. Build
        // the pipeline with the raw `vkCreateRayTracingPipelinesKHR` instead.
        let rgen_stage = &stages[0];
        let rmiss_stage = &stages[1];
        let rchit_stage = &stages[2];
        let rshadowmiss_stage = &stages[3];
        let rshadowhit_stage = &stages[4];
        let name_rgen = CString::new(rgen_stage.entry_point.info().name.as_str()).unwrap();
        let name_rmiss = CString::new(rmiss_stage.entry_point.info().name.as_str()).unwrap();
        let name_rchit = CString::new(rchit_stage.entry_point.info().name.as_str()).unwrap();
        let name_rshadowmiss =
            CString::new(rshadowmiss_stage.entry_point.info().name.as_str()).unwrap();
        let name_rshadowhit = CString::new(rshadowhit_stage.entry_point.info().name.as_str()).unwrap();

        // 6x vec4 = 96 bytes (the single shared `RTShade` payload; the shadow
        // probe reuses it, so no separate shadow payload channel is needed),
        // and the default triangle hit attributes are 2 floats (barycentric
        // coords) = 8 bytes.
        let max_pipeline_ray_payload_size: u32 = 96;
        let max_pipeline_ray_hit_attribute_size: u32 = 8;

        let raw_stages = [
            ash::vk::PipelineShaderStageCreateInfo::default()
                .stage(ash::vk::ShaderStageFlags::from(ShaderStage::from(
                    rgen_stage.entry_point.info().execution_model,
                )))
                .module(rgen_stage.entry_point.module().handle())
                .name(&name_rgen),
            ash::vk::PipelineShaderStageCreateInfo::default()
                .stage(ash::vk::ShaderStageFlags::from(ShaderStage::from(
                    rmiss_stage.entry_point.info().execution_model,
                )))
                .module(rmiss_stage.entry_point.module().handle())
                .name(&name_rmiss),
            ash::vk::PipelineShaderStageCreateInfo::default()
                .stage(ash::vk::ShaderStageFlags::from(ShaderStage::from(
                    rchit_stage.entry_point.info().execution_model,
                )))
                .module(rchit_stage.entry_point.module().handle())
                .name(&name_rchit),
            ash::vk::PipelineShaderStageCreateInfo::default()
                .stage(ash::vk::ShaderStageFlags::from(ShaderStage::from(
                    rshadowmiss_stage.entry_point.info().execution_model,
                )))
                .module(rshadowmiss_stage.entry_point.module().handle())
                .name(&name_rshadowmiss),
            ash::vk::PipelineShaderStageCreateInfo::default()
                .stage(ash::vk::ShaderStageFlags::from(ShaderStage::from(
                    rshadowhit_stage.entry_point.info().execution_model,
                )))
                .module(rshadowhit_stage.entry_point.module().handle())
                .name(&name_rshadowhit),
        ];
        let raw_groups = [
            ash::vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(ash::vk::RayTracingShaderGroupTypeKHR::GENERAL)
                .general_shader(0)
                .closest_hit_shader(ash::vk::SHADER_UNUSED_KHR)
                .any_hit_shader(ash::vk::SHADER_UNUSED_KHR)
                .intersection_shader(ash::vk::SHADER_UNUSED_KHR),
            ash::vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(ash::vk::RayTracingShaderGroupTypeKHR::GENERAL)
                .general_shader(1)
                .closest_hit_shader(ash::vk::SHADER_UNUSED_KHR)
                .any_hit_shader(ash::vk::SHADER_UNUSED_KHR)
                .intersection_shader(ash::vk::SHADER_UNUSED_KHR),
            ash::vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(ash::vk::RayTracingShaderGroupTypeKHR::GENERAL)
                .general_shader(3)
                .closest_hit_shader(ash::vk::SHADER_UNUSED_KHR)
                .any_hit_shader(ash::vk::SHADER_UNUSED_KHR)
                .intersection_shader(ash::vk::SHADER_UNUSED_KHR),
            ash::vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(ash::vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                .general_shader(ash::vk::SHADER_UNUSED_KHR)
                .closest_hit_shader(2)
                .any_hit_shader(ash::vk::SHADER_UNUSED_KHR)
                .intersection_shader(ash::vk::SHADER_UNUSED_KHR),
            ash::vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(ash::vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                .general_shader(ash::vk::SHADER_UNUSED_KHR)
                .closest_hit_shader(ash::vk::SHADER_UNUSED_KHR)
                .any_hit_shader(4)
                .intersection_shader(ash::vk::SHADER_UNUSED_KHR),
        ];
        let library_interface =
            &ash::vk::RayTracingPipelineInterfaceCreateInfoKHR::default()
                .max_pipeline_ray_payload_size(max_pipeline_ray_payload_size)
                .max_pipeline_ray_hit_attribute_size(max_pipeline_ray_hit_attribute_size);
        let create_info_vk = ash::vk::RayTracingPipelineCreateInfoKHR::default()
            .stages(&raw_stages)
            .groups(&raw_groups)
            .layout(layout.handle())
            .max_pipeline_ray_recursion_depth(1)
            .library_interface(library_interface)
            .base_pipeline_index(-1);

        let fns = device.fns();
        let mut handle = MaybeUninit::uninit();
        unsafe {
            (fns.khr_ray_tracing_pipeline
                .create_ray_tracing_pipelines_khr)(
                device.handle(),
                ash::vk::DeferredOperationKHR::null(),
                ash::vk::PipelineCache::null(),
                1,
                &create_info_vk,
                ptr::null(),
                handle.as_mut_ptr(),
            )
        }
        .result()
        .expect("create ray tracing pipeline");
        let handle = unsafe { handle.assume_init() };

        let pipeline = unsafe {
            RayTracingPipeline::from_handle(
                device.clone(),
                handle,
                RayTracingPipelineCreateInfo {
                    stages: smallvec![
                        stages[0].clone(),
                        stages[1].clone(),
                        stages[2].clone(),
                        stages[3].clone(),
                        stages[4].clone(),
                    ],
                    groups: smallvec![
                        RayTracingShaderGroupCreateInfo::General { general_shader: 0 },
                        RayTracingShaderGroupCreateInfo::General { general_shader: 1 },
                        // Shadow-miss record (the raygen's shadow probes trace at
                        // miss-record offset 2).
                        RayTracingShaderGroupCreateInfo::General { general_shader: 3 },
                        RayTracingShaderGroupCreateInfo::TrianglesHit {
                            closest_hit_shader: Some(2),
                            any_hit_shader: None,
                        },
                        // Shadow-any-hit record (shadow probes trace at
                        // hit-record offset 1 so GLSL resolves this group).
                        RayTracingShaderGroupCreateInfo::TrianglesHit {
                            closest_hit_shader: None,
                            any_hit_shader: Some(4),
                        },
                    ],
                    max_pipeline_ray_recursion_depth: 1,
                    layout: layout.clone(),
                    ..RayTracingPipelineCreateInfo::layout(layout.clone())
                },
            )
        };
        let sbt = ShaderBindingTable::new(scene.memory_allocator.clone(), &pipeline)
            .expect("shader binding table");

        let mem = scene.memory_allocator.clone();
        let mut verts_pools = Vec::with_capacity(images);
        let mut index_pools = Vec::with_capacity(images);
        let mut slot_pools = Vec::with_capacity(images);
        let mut instance_buffers = Vec::with_capacity(images);
        let mut blas_scratch = Vec::with_capacity(images);
        let mut tlas_scratch = Vec::with_capacity(images);

        for _ in 0..images {
            verts_pools.push(
                aligned_buffer(
                    mem.clone(),
                    VERT_CAP_VEC4 as u64 * 4 * 4,
                    BufferUsage::STORAGE_BUFFER
                        | BufferUsage::SHADER_DEVICE_ADDRESS
                        | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY,
                )
                .reinterpret(),
            );
            index_pools.push(
                aligned_buffer(
                    mem.clone(),
                    INDEX_CAP as u64 * 4,
                    BufferUsage::STORAGE_BUFFER
                        | BufferUsage::SHADER_DEVICE_ADDRESS
                        | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY,
                )
                .reinterpret(),
            );
            slot_pools.push(
                aligned_buffer(
                    mem.clone(),
                    SLOT_CAP as u64 * size_of::<RtSlot>() as u64,
                    BufferUsage::STORAGE_BUFFER,
                )
                .reinterpret(),
            );
            instance_buffers.push(
                aligned_buffer(
                    mem.clone(),
                    INSTANCE_CAP as u64 * size_of::<AccelerationStructureInstance>() as u64,
                    BufferUsage::STORAGE_BUFFER
                        | BufferUsage::SHADER_DEVICE_ADDRESS
                        | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY,
                )
                .reinterpret(),
            );
        }

        // Worst-case TLAS size + scratch for the instance cap.
        let tlas_geom = AccelerationStructureGeometries::Instances(
            AccelerationStructureGeometryInstancesData::new(
                AccelerationStructureGeometryInstancesDataType::Values(None),
            ),
        );
        let tlas_info = AccelerationStructureBuildGeometryInfo::new(tlas_geom);
        let tlas_sizes = device
            .acceleration_structure_build_sizes(
                AccelerationStructureBuildType::Device,
                &tlas_info,
                &[INSTANCE_CAP as u32],
            )
            .expect("tlas build sizes");
        let tlas_size = tlas_sizes.acceleration_structure_size;

        // Worst-case BLAS scratch for the index cap. The geometry mirrors the
        // real build (`vertex_stride`/`max_vertex`) because the build-size
        // query accounts for them.
        let mut tri = AccelerationStructureGeometryTrianglesData::new(Format::R32G32B32_SFLOAT);
        tri.vertex_stride = 48;
        tri.max_vertex = (VERT_CAP_VEC4 / 3).saturating_sub(1) as u32;
        let blas_info = AccelerationStructureBuildGeometryInfo::new(
            AccelerationStructureGeometries::Triangles(vec![tri]),
        );
        let blas_sizes = device
            .acceleration_structure_build_sizes(
                AccelerationStructureBuildType::Device,
                &blas_info,
                &[INDEX_CAP as u32 / 3],
            )
            .expect("blas build sizes");

        let mut tlas_storage = Vec::with_capacity(images);
        for _ in 0..images {
            tlas_storage.push(aligned_buffer(
                mem.clone(),
                tlas_size,
                BufferUsage::ACCELERATION_STRUCTURE_STORAGE,
            ));
            blas_scratch.push(aligned_buffer(
                mem.clone(),
                blas_sizes.build_scratch_size,
                BufferUsage::STORAGE_BUFFER | BufferUsage::SHADER_DEVICE_ADDRESS,
            ));
            tlas_scratch.push(aligned_buffer(
                mem.clone(),
                tlas_sizes.build_scratch_size,
                BufferUsage::STORAGE_BUFFER | BufferUsage::SHADER_DEVICE_ADDRESS,
            ));
        }
        // The top-level structures keep their storage buffers alive via
        // `AccelerationStructureCreateInfo::buffer`, so `tlas_storage` is only
        // needed for the construction loop above.

        // Stage the static meshes once on the CPU so a repack never re-reads
        // the GPU buffers.
        let mut static_geometry = Vec::with_capacity(1 + scene.traffic_meshes.len());
        let car_v: Vec<Vertex3d> = scene
            .car_vertices
            .read()
            .expect("read player car vertices")
            .to_vec();
        let car_i: Vec<u32> = scene
            .car_indices
            .read()
            .expect("read player car indices")
            .to_vec();
        static_geometry.push((car_v, car_i));
        for (v, i, _anchors) in &scene.traffic_meshes {
            let tv: Vec<Vertex3d> = v.read().expect("read traffic vertices").to_vec();
            let ti: Vec<u32> = i.read().expect("read traffic indices").to_vec();
            static_geometry.push((tv, ti));
        }

        // ---- Fixed per-slot pool layout ----
        // Statics (player + traffic) get their own fixed regions first; the 8
        // chunk slots follow, each sized for the worst-case chunk. Because the
        // region of a slot never moves, the BLAS over it stays valid unless the
        // slot's own geometry is rewritten (only the entering chunk per crossing).
        let static_slots = static_geometry.len();
        let total_slots = static_slots + CHUNK_SLOT_COUNT;
        let mut slot_layouts = Vec::with_capacity(total_slots);
        let mut slots = Vec::with_capacity(total_slots);
        let mut slot_data: Vec<Option<(Vec<f32>, Vec<u32>)>> =
            (0..total_slots).map(|_| None).collect();

        let mut float_base = 0u32; // in vec4 units
        let mut index_base = 0u32;
        for (v, i) in &static_geometry {
            let layout = SlotLayout {
                vertex_base: float_base,
                vertex_cap: v.len() as u32,
                index_base,
                index_cap: i.len() as u32,
            };
            float_base += v.len() as u32 * 3; // 3 vec4 per vertex
            index_base += i.len() as u32;
            let packed = pack_mesh(v, i);
            slot_data[slot_layouts.len()] = Some(packed);
            slot_layouts.push(layout);
            slots.push(RtSlot {
                vertex_base: layout.vertex_base,
                vertex_count: layout.vertex_cap,
                index_base: layout.index_base,
                index_count: layout.index_cap,
            });
        }
        for _ in 0..CHUNK_SLOT_COUNT {
            let layout = SlotLayout {
                vertex_base: float_base,
                vertex_cap: CHUNK_VERT_CAP,
                index_base,
                index_cap: CHUNK_INDEX_CAP,
            };
            float_base += CHUNK_VERT_CAP * 3;
            index_base += CHUNK_INDEX_CAP;
            slot_layouts.push(layout);
            slots.push(RtSlot {
                vertex_base: layout.vertex_base,
                vertex_count: 0,
                index_base: layout.index_base,
                index_count: 0,
            });
        }
        let total_floats = float_base as usize;
        let total_indices = index_base as usize;
        assert!(
            total_floats <= VERT_CAP_VEC4 * 4,
            "RT vertex pool layout {total_floats} > cap {}",
            VERT_CAP_VEC4 * 4
        );
        assert!(
            total_indices <= INDEX_CAP,
            "RT index pool layout {total_indices} > cap {INDEX_CAP}"
        );
        assert!(
            total_slots <= SLOT_CAP,
            "RT slot pool layout {total_slots} > cap {SLOT_CAP}"
        );

        // Write the static geometry into every image's pool once.
        for i in 0..images {
            for (slot, data) in slot_data.iter().enumerate() {
                let Some((floats, indices)) = data else { continue };
                let layout = slot_layouts[slot];
                let vstart: usize = layout.vertex_base as usize * 4;
                let vend: usize = vstart + floats.len();
                let mut vg: vulkano::buffer::BufferWriteGuard<'_, [f32]> =
                    verts_pools[i].write().expect("write vert pool");
                vg[vstart..vend].copy_from_slice(floats);
                let istart: usize = layout.index_base as usize;
                let iend: usize = istart + indices.len();
                let mut ig: vulkano::buffer::BufferWriteGuard<'_, [u32]> =
                    index_pools[i].write().expect("write index pool");
                ig[istart..iend].copy_from_slice(indices);
            }
            let mut sg: vulkano::buffer::BufferWriteGuard<'_, [RtSlot]> =
                slot_pools[i].write().expect("write slot pool");
            sg[..slots.len()].copy_from_slice(&slots);
        }

        // Per-slot BLAS storage, one fixed 256-aligned slice per slot. Built
        // once; rebuilds reuse the same slice (only dirty slots are rebuilt).
        let mut blas = Vec::with_capacity(images);
        for _ in 0..images {
            let mut sizes = Vec::with_capacity(total_slots);
            let mut offsets = Vec::with_capacity(total_slots);
            let mut total = 0u64;
            for layout in &slot_layouts {
                offsets.push(total);
                let size = align_up(
                    blas_size_for_cap(&device, layout.vertex_cap, layout.index_cap),
                    AS_ALIGN_BYTES,
                );
                sizes.push(size);
                total += size;
            }
            let storage = aligned_buffer(
                mem.clone(),
                total,
                BufferUsage::ACCELERATION_STRUCTURE_STORAGE,
            );
            let mut img_blas = Vec::with_capacity(total_slots);
            for (idx, _) in slot_layouts.iter().enumerate() {
                let slice = storage.clone().slice(offsets[idx]..offsets[idx] + sizes[idx]);
                // Safety: the create info supplies a valid 256-aligned backing
                // buffer slice and type.
                img_blas.push(
                    unsafe {
                        AccelerationStructure::new(
                            device.clone(),
                            AccelerationStructureCreateInfo {
                                ty: AccelerationStructureType::BottomLevel,
                                buffer: slice.clone(),
                                ..AccelerationStructureCreateInfo::new(slice)
                            },
                        )
                    }
                    .expect("bottom-level acceleration structure"),
                );
            }
            // Each bottom-level structure keeps its storage slice alive via
            // `AccelerationStructureCreateInfo::buffer`, so the local pool can
            // be dropped once the structures are created.
            blas.push(img_blas);
        }

        let mut tlas = Vec::with_capacity(images);
        for storage in tlas_storage {
            // Safety: the create info supplies a valid backing buffer and type.
            tlas.push(
                unsafe {
                    AccelerationStructure::new(
                        device.clone(),
                        AccelerationStructureCreateInfo {
                            ty: AccelerationStructureType::TopLevel,
                            buffer: storage.clone(),
                            ..AccelerationStructureCreateInfo::new(storage)
                        },
                    )
                }
                .expect("top-level acceleration structure"),
            );
        }

        Arc::new(RayTraceResources {
            device,
            pipeline,
            layout,
            sbt,
            verts_pools,
            index_pools,
            slot_pools,
            instance_buffers,
            blas_scratch,
            tlas_scratch,
            blas,
            tlas,
            output_image: None,
            output_view: None,
            output_size: [0, 0],
            depth_image: None,
            depth_view: None,
            depth_resolved_image: None,
            depth_resolved_view: None,
            slot_layouts,
            static_slots,
            chunk_owner: vec![None; CHUNK_SLOT_COUNT],
            slots,
            slot_data,
            blas_dirty: vec![vec![true; total_slots]; images],
            last_chunk_indices: Vec::new(),
        })
    }

    /// Records the RT render into the caller's command buffer: rebuilds the
    /// TLAS with this frame's instances, fires the rays into the internal
    /// storage image, then copies it into the offscreen color target for
    /// bloom/post/HUD. Geometry slots are fixed: only the entering chunk (if
    /// the window moved) has its pool region + BLAS rewritten, lazily per image
    /// on that image's own frame.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        scene: &SceneResources,
        game: &Game,
        frame: &Frame,
        world_chunks: &[WorldChunk],
        chunk_indices: &[i32],
        image_i: usize,
        offscreen_view: Arc<ImageView>,
        extent: [u32; 2],
    ) {
        self.ensure_output(scene, extent);
        self.sync_geometry(world_chunks, chunk_indices);
        let dirty: Vec<usize> = self.dirty_slots(image_i);
        if !dirty.is_empty() {
            self.apply_dirty_slots(image_i, &dirty);
            self.record_blas_builds(builder, image_i, &dirty);
        }
        self.write_instances(game, scene, image_i, chunk_indices);
        self.record_tlas_build(builder, image_i, game, world_chunks.len());
        // Supersample the ray-traced output (SS × SS rays per output pixel) and
        // downscale it into the offscreen, so per-pixel normal/texture noise is
        // averaged like the raster path's MSAA and the terrain reads as smooth
        // as the raster render.
        let ss = extent.map(|e| e * SUPERSAMPLE);
        self.record_trace(builder, scene, frame, image_i, ss);
        self.copy_output(builder, image_i, offscreen_view, extent);
    }

    /// (Re)creates the internal storage images when the window size changed.
    fn ensure_output(&mut self, scene: &SceneResources, extent: [u32; 2]) {
        if self.output_size == extent {
            return;
        }
        let ss = [extent[0] * SUPERSAMPLE, extent[1] * SUPERSAMPLE];
        // Safety: no further work is submitted to this device afterwards; it is
        // dropped immediately after this call.
        unsafe { self.device.wait_idle() }.expect("wait idle on RT resize");        let image = Image::new(
            scene.memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R16G16B16A16_SFLOAT,
                extent: [ss[0], ss[1], 1],
                usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC | ImageUsage::SAMPLED,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .expect("ray tracing output image");
        let view = ImageView::new_default(image.clone()).expect("ray tracing output view");
        self.output_image = Some(image);
        self.output_view = Some(view);
        self.output_size = extent;

        let depth_image = Image::new(
            scene.memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R32_SFLOAT,
                extent: [ss[0], ss[1], 1],
                usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC | ImageUsage::SAMPLED,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .expect("ray tracing depth image");
        self.depth_image = Some(depth_image.clone());
        self.depth_view = Some(
            ImageView::new_default(depth_image).expect("ray tracing depth view"),
        );

        // 1x depth resolved from the supersampled chain, sampled by the RT
        // particle overlay pass for rain/mist occlusion.
        let resolved = Image::new(
            scene.memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R32_SFLOAT,
                extent: [extent[0], extent[1], 1],
                usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .expect("ray tracing resolved depth image");
        self.depth_resolved_image = Some(resolved.clone());
        self.depth_resolved_view = Some(
            ImageView::new_default(resolved).expect("ray tracing resolved depth view"),
        );
    }

    /// Compares the chunk window against the last repack; when it changed,
    /// re-packs only the entering chunk(s) into their fixed slot regions and
    /// marks those slots dirty for every image. Chunks that stay in the window
    /// keep their pool regions + BLAS untouched.
    fn sync_geometry(&mut self, world_chunks: &[WorldChunk], chunk_indices: &[i32]) {
        if self.last_chunk_indices.as_slice() == chunk_indices {
            return;
        }

        // Map each window chunk to its fixed slot (`static + index mod 8`).
        // Chunks are consecutive, so on a +1 slide the entering chunk reuses
        // the leaving chunk's slot and every other slot is untouched.
        let mut repacked = 0usize;
        for (j, &idx) in chunk_indices.iter().enumerate() {
            let slot = self.chunk_slot(idx);
            let owner_i = slot - self.static_slots;
            if self.chunk_owner[owner_i] == Some(idx) {
                continue;
            }
            let (world_vertices, world_indices) = &world_chunks[j];
            let v: Vec<Vertex3d> = world_vertices
                .read()
                .expect("read chunk vertices for RT repack")
                .to_vec();
            let i: Vec<u32> = world_indices
                .read()
                .expect("read chunk indices for RT repack")
                .to_vec();
            let layout = self.slot_layouts[slot];
            assert!(
                v.len() <= layout.vertex_cap as usize,
                "chunk {idx} {} verts > slot cap {}",
                v.len(),
                layout.vertex_cap
            );
            assert!(
                i.len() <= layout.index_cap as usize,
                "chunk {idx} {} indices > slot cap {}",
                i.len(),
                layout.index_cap
            );
            self.slot_data[slot] = Some(pack_mesh(&v, &i));
            self.slots[slot].vertex_count = v.len() as u32;
            self.slots[slot].index_count = i.len() as u32;
            self.chunk_owner[owner_i] = Some(idx);
            for flags in &mut self.blas_dirty {
                flags[slot] = true;
            }
            repacked += 1;
        }
        self.last_chunk_indices = chunk_indices.to_vec();
        eprintln!("RT pack: {} chunk(s) repacked, {} slots total", repacked, self.static_slots + CHUNK_SLOT_COUNT);
    }

    /// The fixed pool slot for a world chunk index.
    fn chunk_slot(&self, index: i32) -> usize {
        self.static_slots + index.rem_euclid(CHUNK_SLOT_COUNT as i32) as usize
    }

    /// Slots that still need their pool region + BLAS rewritten on `image_i`'s
    /// next frame.
    fn dirty_slots(&self, image_i: usize) -> Vec<usize> {
        self.blas_dirty[image_i]
            .iter()
            .enumerate()
            .filter(|(_, &d)| d)
            .map(|(s, _)| s)
            .collect()
    }

    /// Writes the packed arrays of the given slots into `image_i`'s pools and
    /// clears their dirty flags. Safe because the fence for `image_i` was
    /// already waited before recording, so no in-flight command buffer can
    /// still read those buffers.
    fn apply_dirty_slots(&mut self, image_i: usize, dirty: &[usize]) {
        for &slot in dirty {
            let Some((floats, indices)) = self.slot_data[slot].as_ref() else { continue };
            let layout = self.slot_layouts[slot];
            {
                let mut guard = self.verts_pools[image_i].write().expect("write vert pool");
                let vstart = layout.vertex_base as usize * 4;
                let vend = vstart + floats.len();
                guard[vstart..vend].copy_from_slice(floats);
            }
            {
                let mut guard = self.index_pools[image_i].write().expect("write index pool");
                let istart = layout.index_base as usize;
                let iend = istart + indices.len();
                guard[istart..iend].copy_from_slice(indices);
            }
            {
                let mut guard = self.slot_pools[image_i].write().expect("write slot pool");
                guard[slot] = self.slots[slot];
            }
            self.blas_dirty[image_i][slot] = false;
        }
    }

    /// Rebuilds the BLAS for the given slots. Each slot's acceleration
    /// structure was allocated once with a fixed storage slice, so a rebuild
    /// reuses the same object + slice.
    fn record_blas_builds(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        image_i: usize,
        dirty: &[usize],
    ) {
        for &idx in dirty {
            let slot = &self.slots[idx];
            let mut info = self.blas_info(image_i, slot);
            info.dst_acceleration_structure = Some(self.blas[image_i][idx].clone());
            info.scratch_data = Some(self.blas_scratch[image_i].clone());
            unsafe {
                builder
                    .build_acceleration_structure(
                        info,
                        smallvec![AccelerationStructureBuildRangeInfo {
                            primitive_count: slot.index_count / 3,
                            ..Default::default()
                        }],
                    )
                    .expect("build bottom-level acceleration structure");
            }
        }
    }

    fn blas_info(&self, image_i: usize, slot: &RtSlot) -> AccelerationStructureBuildGeometryInfo {
        let vstart = slot.vertex_base as u64 * 4 * 4;
        let vend = vstart + slot.vertex_count as u64 * 12 * 4;
        let mut tri = AccelerationStructureGeometryTrianglesData::new(Format::R32G32B32_SFLOAT);
        tri.vertex_stride = 48;
        tri.max_vertex = slot.vertex_count.saturating_sub(1);
        tri.vertex_data =
            Some(self.verts_pools[image_i].clone().reinterpret::<[u8]>().slice(vstart..vend));
        tri.index_data = Some(IndexBuffer::U32(
            self.index_pools[image_i]
                .clone()
                .slice(slot.index_base as u64..(slot.index_base + slot.index_count) as u64),
        ));
        AccelerationStructureBuildGeometryInfo::new(AccelerationStructureGeometries::Triangles(vec![tri]))
    }

    fn write_instances(
        &mut self,
        game: &Game,
        scene: &SceneResources,
        image_i: usize,
        chunk_indices: &[i32],
    ) {
        let mut instances = Vec::with_capacity(INSTANCE_CAP);
        // `is_shadow_caster` toggles the instance mask bit the raygen's shadow
        // probes cull on (`SHADOW_RAY_MASK`): chunk (static) geometry is caster,
        // car statics are not, so the shadow rays never even test the car hulls.
        let push = |instances: &mut Vec<AccelerationStructureInstance>,
                    slot: u32,
                    transform: TransformMatrix,
                    is_shadow_caster: bool| {
            let mask: u8 = if is_shadow_caster {
                CHUNK_INSTANCE_MASK as u8
            } else {
                STATIC_INSTANCE_MASK as u8
            };
            instances.push(AccelerationStructureInstance {
                transform,
                instance_custom_index_and_mask: Packed24_8::new(slot, mask),
                acceleration_structure_reference: self.blas[image_i][slot as usize]
                    .device_address()
                    .get(),
                ..Default::default()
            });
        };

        push(
            &mut instances,
            0,
            to_transform(&Mat4::from_scale_rotation_translation(
                Vec3::ONE,
                glam::Quat::from_rotation_y(-game.vehicle.heading),
                Vec3::new(game.player_world_x(), 0.03, game.player_world_z()),
            )),
            false,
        );
        for (idx, t) in game.traffic.iter().enumerate() {
            let tvx = road_curve(t.distance) + t.lane;
            let traffic_rot = traffic_rotation(t.lane, t.distance);
            push(
                &mut instances,
                1 + (idx % scene.traffic_meshes.len()) as u32,
                to_transform(&Mat4::from_scale_rotation_translation(
                    Vec3::ONE,
                    traffic_rot,
                    Vec3::new(tvx, 0.35, -t.distance),
                )),
                false,
            );
        }
        for &idx in chunk_indices {
            let slot = self.chunk_slot(idx);
            push(&mut instances, slot as u32, to_transform(&Mat4::IDENTITY), true);
        }

        let mut guard = self.instance_buffers[image_i]
            .write()
            .expect("write instance buffer");
        guard.fill(AccelerationStructureInstance::default());
        guard[..instances.len()].copy_from_slice(&instances);
    }

    fn record_tlas_build(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        image_i: usize,
        game: &Game,
        chunk_count: usize,
    ) {
        let instance_count = 1 + game.traffic.len() + chunk_count;
        let instances = AccelerationStructureGeometryInstancesData::new(
            AccelerationStructureGeometryInstancesDataType::Values(Some(
                self.instance_buffers[image_i].clone(),
            )),
        );
        let mut info = AccelerationStructureBuildGeometryInfo::new(
            AccelerationStructureGeometries::Instances(instances),
        );
        info.mode = BuildAccelerationStructureMode::Build;
        info.dst_acceleration_structure = Some(self.tlas[image_i].clone());
        info.scratch_data = Some(self.tlas_scratch[image_i].clone());
        unsafe {
            builder
                .build_acceleration_structure(
                    info,
                    smallvec![AccelerationStructureBuildRangeInfo {
                        primitive_count: instance_count as u32,
                        ..Default::default()
                    }],
                )
                .expect("build top-level acceleration structure");
        }
    }

    fn record_trace(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        scene: &SceneResources,
        frame: &Frame,
        image_i: usize,
        extent: [u32; 2],
    ) {
        let view_proj = frame.uniforms.proj * frame.uniforms.view;
        let rt = RtUniforms {
            inv_view_proj: view_proj.inverse().to_cols_array_2d(),
            eye: [frame.uniforms.eye.x, frame.uniforms.eye.y, frame.uniforms.eye.z, 1.0],
            fog_color: frame.uniforms.fog_color,
            zenith: frame.sky_uniform.zenith,
            horizon: frame.sky_uniform.horizon,
            cloud_tint: frame.sky_uniform.cloud_tint,
            light_dir: frame.sky_uniform.light_dir,
            cloud_amount: frame.sky_uniform.cloud_amount,
            _pad: [0.0; 3],
            sun_state: frame.sky_uniform.sun_state,
            time: frame.sky_uniform.time,
            _pad2: [0.0; 3],
        };
        let rt_buf = Buffer::from_data(
            scene.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            rt,
        )
        .expect("rt uniforms buffer");
        let mvp = scene.mvp_buffer(Mat4::IDENTITY, &frame.uniforms, &frame.headlights, [0.0, 0.0, 0.0, -1.0], false);

        let set_layout_0 = self.layout.set_layouts()[0].clone();
        let set_0 = DescriptorSet::new(
            scene.descriptor_set_allocator.clone(),
            set_layout_0,
            [
                WriteDescriptorSet::buffer(0, rt_buf),
                WriteDescriptorSet::buffer(1, mvp),
                WriteDescriptorSet::image_view(2, self.output_view.clone().expect("rt output view")),
                WriteDescriptorSet::image_view(
                    10,
                    self.depth_view.clone().expect("rt depth view"),
                ),
                WriteDescriptorSet::buffer(3, self.verts_pools[image_i].clone()),
                WriteDescriptorSet::buffer(4, self.index_pools[image_i].clone()),
                WriteDescriptorSet::buffer(5, self.slot_pools[image_i].clone()),
                WriteDescriptorSet::image_view_sampler(
                    6,
                    scene.world_texture_view.clone(),
                    scene.mesh_sampler.clone(),
                ),
                WriteDescriptorSet::image_view_sampler(
                    7,
                    scene.car_texture_view.clone(),
                    scene.mesh_sampler.clone(),
                ),
                WriteDescriptorSet::image_view_sampler(
                    8,
                    scene.cloud_a_view.clone(),
                    scene.mesh_sampler.clone(),
                ),
                WriteDescriptorSet::image_view_sampler(
                    9,
                    scene.cloud_b_view.clone(),
                    scene.mesh_sampler.clone(),
                ),
                WriteDescriptorSet::image_view_sampler(
                    11,
                    scene.world_mid_view.clone(),
                    scene.mesh_sampler.clone(),
                ),
                WriteDescriptorSet::image_view_sampler(
                    12,
                    scene.world_far_view.clone(),
                    scene.mesh_sampler.clone(),
                ),
            ],
            [],
        )
        .expect("rt descriptor set");
        let set_layout_1 = self.layout.set_layouts()[1].clone();
        let set_1 = DescriptorSet::new(
            scene.descriptor_set_allocator.clone(),
            set_layout_1,
            [WriteDescriptorSet::acceleration_structure(0, self.tlas[image_i].clone())],
            [],
        )
        .expect("tlas descriptor set");

        builder
            .bind_pipeline_ray_tracing(self.pipeline.clone())
            .expect("bind rt pipeline")
            .bind_descriptor_sets(
                PipelineBindPoint::RayTracing,
                self.layout.clone(),
                0,
                set_0,
            )
            .expect("bind rt set 0")
            .bind_descriptor_sets(
                PipelineBindPoint::RayTracing,
                self.layout.clone(),
                1,
                set_1,
            )
            .expect("bind rt set 1");
        unsafe {
            builder
                .trace_rays(self.sbt.addresses().clone(), [extent[0], extent[1], 1])
                .expect("trace rays");
        }
    }

    fn copy_output(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        image_i: usize,
        offscreen_view: Arc<ImageView>,
        extent: [u32; 2],
    ) {
        let _ = image_i;
        // Downscale the supersampled color into the offscreen with a linear
        // (box-ish) filter, which averages the SS² rays per output pixel.
        let src = self.output_image.clone().expect("rt output image");
        let dst = offscreen_view.image().clone();
        let src_layers = src.subresource_layers();
        let dst_layers = dst.subresource_layers();
        let mut info = BlitImageInfo::images(src, dst);
        info.filter = vulkano::image::sampler::Filter::Linear;
        let region = &mut info.regions[0];
        region.src_subresource = src_layers;
        region.src_offsets = [[0, 0, 0], region_extent(self.output_size, SUPERSAMPLE)];
        region.dst_subresource = dst_layers;
        region.dst_offsets = [[0, 0, 0], [extent[0], extent[1], 1]];
        builder
            .blit_image(info)
            .expect("blit supersampled rt output into offscreen");

        // Resolve the supersampled depth down to 1x for the particle overlay.
        let dsrc = self.depth_image.clone().expect("rt depth image");
        let ddst = self
            .depth_resolved_image
            .clone()
            .expect("rt resolved depth image");
        let dsrc_layers = dsrc.subresource_layers();
        let ddst_layers = ddst.subresource_layers();
        let mut dinfo = BlitImageInfo::images(dsrc, ddst);
        dinfo.filter = vulkano::image::sampler::Filter::Linear;
        let dregion = &mut dinfo.regions[0];
        dregion.src_subresource = dsrc_layers;
        dregion.src_offsets = [[0, 0, 0], region_extent(self.output_size, SUPERSAMPLE)];
        dregion.dst_subresource = ddst_layers;
        dregion.dst_offsets = [[0, 0, 0], [extent[0], extent[1], 1]];
        builder
            .blit_image(dinfo)
            .expect("blit supersampled rt depth into resolved");
    }

    /// The resolved 1x primary-ray depth image (linear eye distance, R32),
    /// sampled by the RT particle overlay pass to occlude rain/mist/dust.
    pub fn depth_view(&self) -> Arc<ImageView> {
        self.depth_resolved_view.clone().expect("rt resolved depth view")
    }
}

fn align_up(x: u64, align: u64) -> u64 {
    x.div_ceil(align) * align
}

/// Blit region offset for a supersampled image: full size × `scale`.
fn region_extent(extent: [u32; 2], scale: u32) -> [u32; 3] {
    [extent[0] * scale, extent[1] * scale, 1]
}

/// Serializes one mesh into the RT pool layout, matching `raytrace.rchit.glsl`:
///   verts[base + i*3 + 0] = position.xyz, uv.x
///   verts[base + i*3 + 1] = normal.xyz,  material
///   verts[base + i*3 + 2] = color.xyz,   uv.y
fn pack_mesh(verts: &[Vertex3d], idx: &[u32]) -> (Vec<f32>, Vec<u32>) {
    let mut floats = Vec::with_capacity(verts.len() * 12);
    for v in verts {
        floats.extend_from_slice(&v.position);
        floats.push(v.tex_coord[0]);
        floats.extend_from_slice(&v.normal);
        floats.push(v.material);
        floats.extend_from_slice(&v.color);
        floats.push(v.tex_coord[1]);
    }
    (floats, idx.to_vec())
}

/// Acceleration-structure storage size for a slot with the given caps. Mirrors
/// `blas_info`: the size query accounts for `max_vertex` and `vertex_stride`,
/// so a bare geometry would under-size the structure.
fn blas_size_for_cap(device: &Arc<Device>, vertex_cap: u32, index_cap: u32) -> u64 {
    let mut tri = AccelerationStructureGeometryTrianglesData::new(Format::R32G32B32_SFLOAT);
    tri.vertex_stride = 48;
    tri.max_vertex = vertex_cap.saturating_sub(1);
    let info = AccelerationStructureBuildGeometryInfo::new(
        AccelerationStructureGeometries::Triangles(vec![tri]),
    );
    device
        .acceleration_structure_build_sizes(
            AccelerationStructureBuildType::Device,
            &info,
            &[index_cap / 3],
        )
        .expect("blas build size")
        .acceleration_structure_size
}
