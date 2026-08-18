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

use std::num::NonZeroU64;
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
    AutoCommandBufferBuilder, CopyImageInfo, PrimaryAutoCommandBuffer,
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
use vulkano::Packed24_8;

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
/// Meshes a repack stages: player + 4 traffic models + up to 8 window chunks.
const MAX_SLOTS: usize = 5 + 8;
/// Player + traffic + the 8-chunk window + slack.
const INSTANCE_CAP: usize = 1 + 8 + 8 + 8;
/// Acceleration-structure storage and slices must be 256-byte aligned.
const AS_ALIGN_BYTES: u64 = 256;

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
    memory_allocator: Arc<StandardMemoryAllocator>,
    pipeline: Arc<RayTracingPipeline>,
    layout: Arc<PipelineLayout>,
    sbt: ShaderBindingTable,

    images: usize,
    verts_pools: Vec<Subbuffer<[f32]>>,
    index_pools: Vec<Subbuffer<[u32]>>,
    slot_pools: Vec<Subbuffer<[RtSlot]>>,
    instance_buffers: Vec<Subbuffer<[AccelerationStructureInstance]>>,
    blas_scratch: Vec<Subbuffer<[u8]>>,
    tlas_scratch: Vec<Subbuffer<[u8]>>,
    blas: Vec<Vec<Arc<AccelerationStructure>>>,
    tlas: Vec<Arc<AccelerationStructure>>,

    /// CPU mirror of the statics (player + 4 traffic models), staged once.
    static_geometry: Vec<(Vec<Vertex3d>, Vec<u32>)>,
    /// Current pool layout: one [`RtSlot`] per mesh, in pool order.
    slots: Vec<RtSlot>,

    output_image: Option<Arc<Image>>,
    output_view: Option<Arc<ImageView>>,
    output_size: [u32; 2],

    geometry_dirty: bool,
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

        let stages = vec![
            PipelineShaderStageCreateInfo::new(rgen.entry_point("main").unwrap()),
            PipelineShaderStageCreateInfo::new(rmiss.entry_point("main").unwrap()),
            PipelineShaderStageCreateInfo::new(rchit.entry_point("main").unwrap()),
        ];
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .expect("ray tracing pipeline layout");
        let pipeline = RayTracingPipeline::new(
            device.clone(),
            None,
            RayTracingPipelineCreateInfo {
                stages: smallvec![stages[0].clone(), stages[1].clone(), stages[2].clone()],
                groups: smallvec![
                    RayTracingShaderGroupCreateInfo::General { general_shader: 0 },
                    RayTracingShaderGroupCreateInfo::General { general_shader: 1 },
                    RayTracingShaderGroupCreateInfo::TrianglesHit {
                        closest_hit_shader: Some(2),
                        any_hit_shader: None,
                    },
                ],
                max_pipeline_ray_recursion_depth: 1,
                layout: layout.clone(),
                ..RayTracingPipelineCreateInfo::layout(layout.clone())
            },
        )
        .expect("ray tracing pipeline");
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
            memory_allocator: scene.memory_allocator.clone(),
            pipeline,
            layout,
            sbt,
            images,
            verts_pools,
            index_pools,
            slot_pools,
            instance_buffers,
            blas_scratch,
            tlas_scratch,
            blas: vec![Vec::new(); images],
            tlas,
            static_geometry,
            slots: Vec::new(),
            output_image: None,
            output_view: None,
            output_size: [0, 0],
            geometry_dirty: true,
            last_chunk_indices: Vec::new(),
        })
    }

    /// Records the RT render into the caller's command buffer: rebuilds the
    /// BLAS when the world window changed, rebuilds the TLAS with this frame's
    /// instances, fires the rays into the internal storage image, then copies
    /// it into the offscreen color target for bloom/post/HUD.
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
        let rebuilt = self.sync_geometry(scene, world_chunks, chunk_indices, image_i);
        if rebuilt {
            self.build_blas_objects(image_i);
        }
        self.write_instances(game, scene, image_i);
        if rebuilt {
            self.record_blas_builds(builder, image_i);
        }
        self.record_tlas_build(builder, image_i, game, world_chunks.len());
        self.record_trace(builder, scene, frame, image_i, extent);
        self.copy_output(builder, image_i, offscreen_view);
    }

    /// (Re)creates the internal storage image when the window size changed.
    fn ensure_output(&mut self, scene: &SceneResources, extent: [u32; 2]) {
        if self.output_size == extent {
            return;
        }
        // Safety: no further work is submitted to this device afterwards; it is
        // dropped immediately after this call.
        unsafe { self.device.wait_idle() }.expect("wait idle on RT resize");        let image = Image::new(
            scene.memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R16G16B16A16_SFLOAT,
                extent: [extent[0], extent[1], 1],
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
    }

    /// Compares the chunk window against the last repack; when it changed,
    /// re-packs the vertex/index/slot pools (writing every image's copy) and
    /// flags the BLAS for a rebuild. Returns true when the BLAS must be
    /// rebuilt this frame.
    fn sync_geometry(
        &mut self,
        scene: &SceneResources,
        world_chunks: &[WorldChunk],
        chunk_indices: &[i32],
        _image_i: usize,
    ) -> bool {
        if !self.geometry_dirty && self.last_chunk_indices.as_slice() == chunk_indices {
            return false;
        }

        let (floats, indices, slots) = self.pack(scene, world_chunks);
        eprintln!(
            "RT pack: {} floats ({} verts), {} indices, {} slots",
            floats.len(),
            floats.len() / 12,
            indices.len(),
            slots.len()
        );
        assert!(
            floats.len() <= VERT_CAP_VEC4 * 4,
            "RT vertex pool overflow: {} floats > cap {}",
            floats.len(),
            VERT_CAP_VEC4 * 4
        );
        assert!(
            indices.len() <= INDEX_CAP,
            "RT index pool overflow: {} indices > cap {}",
            indices.len(),
            INDEX_CAP
        );
        assert!(
            slots.len() <= SLOT_CAP,
            "RT slot pool overflow: {} slots > cap {}",
            slots.len(),
            SLOT_CAP
        );

        for i in 0..self.images {
            {
                let mut guard = self.verts_pools[i].write().expect("write vert pool");
                guard.fill(0.0);
                guard[..floats.len()].copy_from_slice(&floats);
            }
            {
                let mut guard = self.index_pools[i].write().expect("write index pool");
                guard.fill(0);
                guard[..indices.len()].copy_from_slice(&indices);
            }
            {
                let mut guard = self.slot_pools[i].write().expect("write slot pool");
                guard[..slots.len()].copy_from_slice(&slots);
            }
        }

        self.slots = slots;
        self.last_chunk_indices = chunk_indices.to_vec();
        self.geometry_dirty = false;
        true
    }

    /// Stages every mesh (statics + chunks) into contiguous CPU arrays and
    /// returns the packed pools plus the per-mesh [`RtSlot`] table.
    fn pack(&self, _scene: &SceneResources, world_chunks: &[WorldChunk]) -> (Vec<f32>, Vec<u32>, Vec<RtSlot>) {
        let mut floats = Vec::with_capacity(VERT_CAP_VEC4 * 4);
        let mut indices = Vec::with_capacity(INDEX_CAP);
        let mut slots = Vec::with_capacity(MAX_SLOTS);

        let mut push_mesh = |verts: &[Vertex3d], idx: &[u32], slots: &mut Vec<RtSlot>| {
            let vertex_base = (floats.len() / 4) as u32;
            let index_base = indices.len() as u32;
            // 3 vec4 per vertex, matching `raytrace.rchit.glsl`:
            //   verts[base + i*3 + 0] = position.xyz, uv.x
            //   verts[base + i*3 + 1] = normal.xyz,  material
            //   verts[base + i*3 + 2] = color.xyz,   uv.y
            for v in verts {
                floats.extend_from_slice(&v.position);
                floats.push(v.tex_coord[0]);
                floats.extend_from_slice(&v.normal);
                floats.push(v.material);
                floats.extend_from_slice(&v.color);
                floats.push(v.tex_coord[1]);
            }
            indices.extend_from_slice(idx);
            slots.push(RtSlot {
                vertex_base,
                vertex_count: verts.len() as u32,
                index_base,
                index_count: idx.len() as u32,
            });
        };

        for (v, i) in &self.static_geometry {
            push_mesh(v, i, &mut slots);
        }
        for (world_vertices, world_indices) in world_chunks {
            let v: Vec<Vertex3d> = world_vertices
                .read()
                .expect("read chunk vertices for RT repack")
                .to_vec();
            let i: Vec<u32> = world_indices
                .read()
                .expect("read chunk indices for RT repack")
                .to_vec();
            push_mesh(&v, &i, &mut slots);
        }

        (floats, indices, slots)
    }

    /// Creates (or replaces) the per-image bottom-level structures, slicing the
    /// blas storage buffer at 256-aligned offsets. Safe because the fence for
    /// `image_i` was already waited before recording. Rebuilds *every* image's
    /// BLAS because `sync_geometry` repacks all the pools.
    fn build_blas_objects(&mut self, _image_i: usize) {
        let mut sizes = Vec::with_capacity(self.slots.len());
        let mut total = 0u64;
        let mut offsets = Vec::with_capacity(self.slots.len());
        for slot in &self.slots {
            offsets.push(total);
            let size = align_up(self.blas_size_for_slot(slot), AS_ALIGN_BYTES);
            sizes.push(size);
            total += size;
        }
        assert!(total > 0, "RT BLAS storage must have at least one slot");

        // Allocate fresh per-image blas storage big enough for the current
        // slot set (sized at repack time, so no cap waste).
        let mem = self.memory_allocator.clone();
        let mut blas_storage = Vec::with_capacity(self.images);
        for _ in 0..self.images {
            blas_storage.push(aligned_buffer(
                mem.clone(),
                total,
                BufferUsage::ACCELERATION_STRUCTURE_STORAGE,
            ));
        }

        // Slice a per-slot bottom-level structure out of each image's buffer.
        let mut blas = Vec::with_capacity(self.images);
        for storage in blas_storage {
            let mut img_blas = Vec::with_capacity(self.slots.len());
            for (i, _slot) in self.slots.iter().enumerate() {
                let slice = storage.clone().slice(offsets[i]..offsets[i] + sizes[i]);
                // Safety: the create info supplies a valid 256-aligned backing
                // buffer slice and type.
                img_blas.push(
                    unsafe {
                        AccelerationStructure::new(
                            self.device.clone(),
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
            blas.push(img_blas);
        }

        // Each bottom-level structure keeps its storage slice alive via
        // `AccelerationStructureCreateInfo::buffer`, so the local pool can be
        // dropped once the structures are created.
        self.blas = blas;
    }

    fn blas_size_for_slot(&self, slot: &RtSlot) -> u64 {
        // Mirrors `blas_info`: the size query accounts for `max_vertex` and
        // `vertex_stride`, so a bare geometry would under-size the structure.
        let mut tri = AccelerationStructureGeometryTrianglesData::new(Format::R32G32B32_SFLOAT);
        tri.vertex_stride = 48;
        tri.max_vertex = slot.vertex_count.saturating_sub(1);
        let info = AccelerationStructureBuildGeometryInfo::new(
            AccelerationStructureGeometries::Triangles(vec![tri]),
        );
        self.device
            .acceleration_structure_build_sizes(
                AccelerationStructureBuildType::Device,
                &info,
                &[slot.index_count / 3],
            )
            .expect("blas build size")
            .acceleration_structure_size
    }

    fn record_blas_builds(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        image_i: usize,
    ) {
        for (idx, slot) in self.slots.iter().enumerate() {
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

    fn write_instances(&mut self, game: &Game, scene: &SceneResources, image_i: usize) {
        let mut instances = Vec::with_capacity(INSTANCE_CAP);
        let push = |instances: &mut Vec<AccelerationStructureInstance>, slot: u32, transform: TransformMatrix| {
            instances.push(AccelerationStructureInstance {
                transform,
                instance_custom_index_and_mask: Packed24_8::new(slot, 0xff),
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
            );
        }
        let chunk_slot_base = 1 + scene.traffic_meshes.len();
        for j in 0..self.slots.len().saturating_sub(chunk_slot_base) {
            push(&mut instances, (chunk_slot_base + j) as u32, to_transform(&Mat4::IDENTITY));
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
        let mvp = scene.mvp_buffer(Mat4::IDENTITY, &frame.uniforms, &frame.headlights, [0.0, 0.0, 0.0, -1.0]);

        let set_layout_0 = self.layout.set_layouts()[0].clone();
        let set_0 = DescriptorSet::new(
            scene.descriptor_set_allocator.clone(),
            set_layout_0,
            [
                WriteDescriptorSet::buffer(0, rt_buf),
                WriteDescriptorSet::buffer(1, mvp),
                WriteDescriptorSet::image_view(2, self.output_view.clone().expect("rt output view")),
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
    ) {
        let src = self.output_image.clone().expect("rt output image");
        builder
            .copy_image(CopyImageInfo::images(src, offscreen_view.image().clone()))
            .expect("copy rt output into offscreen");
        let _ = image_i;
    }
}

fn align_up(x: u64, align: u64) -> u64 {
    x.div_ceil(align) * align
}
