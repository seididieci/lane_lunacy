// SPDX-License-Identifier: MIT

//! GPU resources shared by every presenter (windowed `Renderer` and the
//! headless snapshot path): pipelines, textures, samplers, models and mesh
//! buffers. Built once against a render pass whose attachment layout matches
//! the target; the same `SceneResources` then serves both windowed frames and
//! offscreen captures, so the two can never drift apart.

use std::sync::Arc;

use glam::Mat4;

use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferToImageInfo, PrimaryAutoCommandBuffer,
};
use vulkano::descriptor_set::allocator::{
    StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo,
};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::{Device, Queue};
use vulkano::format::Format;
use vulkano::image::sampler::{Sampler, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode};
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage, SampleCount};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::rasterization::CullMode;
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint};
use vulkano::render_pass::{RenderPass, Subpass};
use vulkano::sync::{self, GpuFuture};

use crate::font::FontAtlas;
use crate::mesh::build_sky_dome;
use crate::model::{load_gltf_mesh_from_bytes, CarLightAnchors};
use crate::render::cloud::{generate_cloud_tile, generate_foliage_tile, generate_rock_tile};
use crate::render::flare;
use crate::render::frame::{FrameUniforms, Headlights};
use crate::render::particles::{generate_cloud_sprite, generate_soft_sprite};
use crate::render::pipeline::{graphics_pipeline, load_shaders, Blend, Depth, PipelineSpec};
use crate::render::texture::{
    make_mesh_buffers, upload_rgba8_texture, upload_rgba8_texture_mipmapped,
};
use crate::shaders::{self, MVP};
use crate::vertex::{FlareVertex, HudVertex, ParticleVertex, Vertex3d};

const CLOUD_TILE: u32 = 256;
/// Foliage tile must match the other world-atlas slot dimensions (512×512).
const FOLIAGE_TILE: u32 = 512;

const PLAYER_MODEL_GLB: &[u8] = include_bytes!("../../assets/models/player_race_future.glb");
const TRAFFIC_SEDAN_GLB: &[u8] = include_bytes!("../../assets/models/traffic_sedan.glb");
const TRAFFIC_SUV_GLB: &[u8] = include_bytes!("../../assets/models/traffic_suv.glb");
const TRAFFIC_TAXI_GLB: &[u8] = include_bytes!("../../assets/models/traffic_taxi.glb");
const TRAFFIC_VAN_GLB: &[u8] = include_bytes!("../../assets/models/traffic_van.glb");
const CAR_COLORMAP_PNG: &[u8] = include_bytes!("../../assets/models/colormap.png");
const ASPHALT_BASE_PNG: &[u8] = include_bytes!("../../assets/textures/asphalt_base.png");
const ASPHALT_WORN_PNG: &[u8] = include_bytes!("../../assets/textures/asphalt_worn.png");
const ASPHALT_CRACKED_PNG: &[u8] = include_bytes!("../../assets/textures/asphalt_cracked.png");
const GRASS_PNG: &[u8] = include_bytes!("../../assets/textures/grass.png");

/// Decodes an embedded PNG to RGBA. Runs on a startup worker thread.
fn decode_png(bytes: &[u8], label: &str) -> image::RgbaImage {
    image::load_from_memory(bytes)
        .unwrap_or_else(|e| panic!("failed to decode embedded {label} texture: {e}"))
        .to_rgba8()
}

/// One traffic car's mesh: vertex buffer, index buffer, and its lamp anchors.
pub type TrafficMesh = (Subbuffer<[Vertex3d]>, Subbuffer<[u32]>, CarLightAnchors);

/// All GPU state a frame needs except the swapchain/framebuffer and the
/// per-frame mutable state (particles, camera smoothing, world chunks).
pub struct SceneResources {
    pub memory_allocator: Arc<StandardMemoryAllocator>,
    pub command_allocator: Arc<StandardCommandBufferAllocator>,
    pub descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    /// Queue family the frame command buffers are recorded against.
    pub queue_family_index: u32,

    pub mesh_pipeline: Arc<GraphicsPipeline>,
    pub hud_pipeline: Arc<GraphicsPipeline>,
    pub sky_pipeline: Arc<GraphicsPipeline>,
    pub particle_pipeline: Arc<GraphicsPipeline>,
    pub dust_pipeline: Arc<GraphicsPipeline>,
    pub flare_pipeline: Arc<GraphicsPipeline>,

    /// Color-only overlay pass the ray-traced backend uses to composite the
    /// CPU particle quads (rain/mist/dust/glows) over the RT image. The color
    /// attachment loads the existing RT output; the shaders occlude per-pixel
    /// against the raygen's depth image, so no depth attachment is needed.
    pub rt_particle_render_pass: Arc<RenderPass>,
    pub rt_particle_pipeline: Arc<GraphicsPipeline>,
    pub rt_dust_pipeline: Arc<GraphicsPipeline>,

    pub flare_core_view: Arc<ImageView>,
    pub flare_streak_view: Arc<ImageView>,
    pub flare_ring_view: Arc<ImageView>,
    pub flare_sampler: Arc<Sampler>,
    pub particle_sprite_view: Arc<ImageView>,
    pub particle_sampler: Arc<Sampler>,

    pub hud_descriptor_set: Arc<DescriptorSet>,
    pub mesh_sampler: Arc<Sampler>,
    pub world_texture_view: Arc<ImageView>,
    /// Downsampled atlas copies for the ray-traced hit shader's software mip
    /// chain: `world_mid_view` = 1/4 res, `world_far_view` = 1/16 res. Explicit
    /// LOD is ignored in RT stages on this driver, so the shader blends these
    /// by distance instead of `textureLod`.
    pub world_mid_view: Arc<ImageView>,
    pub world_far_view: Arc<ImageView>,
    pub car_texture_view: Arc<ImageView>,
    pub cloud_a_view: Arc<ImageView>,
    pub cloud_b_view: Arc<ImageView>,

    pub sky_dome_vertices: Subbuffer<[Vertex3d]>,
    pub sky_dome_indices: Subbuffer<[u32]>,

    pub car_vertices: Subbuffer<[Vertex3d]>,
    pub car_indices: Subbuffer<[u32]>,
    pub player_anchors: CarLightAnchors,
    pub traffic_meshes: Vec<TrafficMesh>,
    pub traffic_anchors: Vec<CarLightAnchors>,
}

impl SceneResources {
    /// Builds every pipeline, texture, sampler and model. The `render_pass`
    /// must have one color + one depth attachment (swapchain or offscreen);
    /// the pipelines are recorded against its subpass. `samples` is the
    /// multisampling of the render pass color/depth attachments.
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        render_pass: Arc<RenderPass>,
        font_atlas: &FontAtlas,
        seed: u64,
        samples: SampleCount,
    ) -> Self {
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let command_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        ));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            StandardDescriptorSetAllocatorCreateInfo::default(),
        ));

        // Parallel CPU phase: everything independent and pure-CPU (embedded PNG
        // decodes, procedural tile/sprite/flare generation, sky dome geometry,
        // glTF model parsing) runs concurrently on every core before any GPU
        // work starts. This dominates the debug startup time; the uploads,
        // pipelines and buffers that follow are serialized on this thread.
        let (
            asphalt_base,
            asphalt_worn,
            asphalt_cracked,
            grass,
            colormap,
            foliage_tile,
            rock_tile,
            cloud_a,
            cloud_b,
            sprite_atlas,
            (flare_core, flare_streak, flare_ring),
            (dome_vertices, dome_indices),
            (car_vertices, car_indices, player_anchors),
            (traffic_0, traffic_1, traffic_2, traffic_3),
        ) = std::thread::scope(|s| {
            let asphalt_base = s.spawn(|| decode_png(ASPHALT_BASE_PNG, "asphalt_base"));
            let asphalt_worn = s.spawn(|| decode_png(ASPHALT_WORN_PNG, "asphalt_worn"));
            let asphalt_cracked = s.spawn(|| decode_png(ASPHALT_CRACKED_PNG, "asphalt_cracked"));
            // Grass is mildly blurred at load so its fine grain reads as soft
            // turf instead of a grainy speckle (most visible on the verges).
            let grass = s.spawn(|| {
                let img = decode_png(GRASS_PNG, "grass");
                image::imageops::blur(&img, 1.2)
            });
            let colormap = s.spawn(|| decode_png(CAR_COLORMAP_PNG, "car colormap"));
            let foliage = s.spawn(|| generate_foliage_tile(FOLIAGE_TILE, seed));
            let rock = s.spawn(|| generate_rock_tile(FOLIAGE_TILE, seed ^ 0x0BAD_F00D));
            let cloud_a = s.spawn(|| generate_cloud_tile(CLOUD_TILE, seed));
            let cloud_b = s.spawn(|| {
                generate_cloud_tile(
                    CLOUD_TILE,
                    seed.wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407),
                )
            });
            let sprite_atlas = s.spawn(|| {
                let mut atlas = Vec::with_capacity((512 * 128 * 4) as usize);
                atlas.extend_from_slice(&generate_soft_sprite(128));
                for seed in [
                    0x9E3779B97F4A7C15u64,
                    0xBF58476D1CE4E5B9u64,
                    0x94D049BB133111EBu64,
                ] {
                    atlas.extend_from_slice(&generate_cloud_sprite(128, seed));
                }
                atlas
            });
            let flare = s.spawn(|| {
                (
                    flare::generate_sun_core(64),
                    flare::generate_flare_streak(256, 32),
                    flare::generate_flare_ring(256),
                )
            });
            let dome = s.spawn(|| build_sky_dome(32, 128));
            let player = s.spawn(|| {
                load_gltf_mesh_from_bytes(PLAYER_MODEL_GLB, "player_race_future.glb")
                    .expect("failed to load embedded player model")
            });
            let sedan = s.spawn(|| {
                load_gltf_mesh_from_bytes(TRAFFIC_SEDAN_GLB, "traffic_sedan.glb")
                    .unwrap_or_else(|e| panic!("failed to load embedded traffic model {e}"))
            });
            let suv = s.spawn(|| {
                load_gltf_mesh_from_bytes(TRAFFIC_SUV_GLB, "traffic_suv.glb")
                    .unwrap_or_else(|e| panic!("failed to load embedded traffic model {e}"))
            });
            let taxi = s.spawn(|| {
                load_gltf_mesh_from_bytes(TRAFFIC_TAXI_GLB, "traffic_taxi.glb")
                    .unwrap_or_else(|e| panic!("failed to load embedded traffic model {e}"))
            });
            let van = s.spawn(|| {
                load_gltf_mesh_from_bytes(TRAFFIC_VAN_GLB, "traffic_van.glb")
                    .unwrap_or_else(|e| panic!("failed to load embedded traffic model {e}"))
            });

            (
                asphalt_base.join().expect("startup worker panicked"),
                asphalt_worn.join().expect("startup worker panicked"),
                asphalt_cracked.join().expect("startup worker panicked"),
                grass.join().expect("startup worker panicked"),
                colormap.join().expect("startup worker panicked"),
                foliage.join().expect("startup worker panicked"),
                rock.join().expect("startup worker panicked"),
                cloud_a.join().expect("startup worker panicked"),
                cloud_b.join().expect("startup worker panicked"),
                sprite_atlas.join().expect("startup worker panicked"),
                flare.join().expect("startup worker panicked"),
                dome.join().expect("startup worker panicked"),
                player.join().expect("startup worker panicked"),
                (
                    sedan.join().expect("startup worker panicked"),
                    suv.join().expect("startup worker panicked"),
                    taxi.join().expect("startup worker panicked"),
                    van.join().expect("startup worker panicked"),
                ),
            )
        });

        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();
        let mesh =
            load_shaders::<Vertex3d>(&device, shaders::MESH_VERT_SPV, shaders::MESH_FRAG_SPV);
        let mesh_pipeline = graphics_pipeline(
            &device,
            &subpass,
            PipelineSpec {
                label: "mesh pipeline",
                cull_mode: CullMode::Back,
                depth: Depth::Test { write: true },
                blend: Blend::Opaque,
            },
            mesh.stages,
            mesh.vertex_input,
            mesh.layout,
            samples,
        );

        let hud = load_shaders::<HudVertex>(&device, shaders::HUD_VERT_SPV, shaders::HUD_FRAG_SPV);
        let hud_pipeline = graphics_pipeline(
            &device,
            &subpass,
            PipelineSpec {
                label: "hud pipeline",
                cull_mode: CullMode::None,
                depth: Depth::None,
                blend: Blend::Alpha,
            },
            hud.stages,
            hud.vertex_input,
            hud.layout,
            samples,
        );

        let sky = load_shaders::<Vertex3d>(&device, shaders::SKY_VERT_SPV, shaders::SKY_FRAG_SPV);
        let sky_pipeline = graphics_pipeline(
            &device,
            &subpass,
            PipelineSpec {
                label: "sky pipeline",
                cull_mode: CullMode::None,
                depth: Depth::None,
                blend: Blend::Opaque,
            },
            sky.stages,
            sky.vertex_input,
            sky.layout,
            samples,
        );

        // ---- Rain particles ----
        let particle = load_shaders::<ParticleVertex>(
            &device,
            shaders::PARTICLE_VERT_SPV,
            shaders::PARTICLE_FRAG_SPV,
        );
        let particle_pipeline = graphics_pipeline(
            &device,
            &subpass,
            PipelineSpec {
                label: "particle pipeline",
                cull_mode: CullMode::None,
                depth: Depth::Test { write: false },
                blend: Blend::Additive,
            },
            particle.stages.clone(),
            particle.vertex_input.clone(),
            particle.layout.clone(),
            samples,
        );

        // Dust uses normal alpha blending instead of additive: the cloud must
        // accumulate opacity smoothly where puffs overlap, rather than adding
        // RGB and producing bright bokeh/interference rings at the seams.
        let dust_pipeline = graphics_pipeline(
            &device,
            &subpass,
            PipelineSpec {
                label: "dust pipeline",
                cull_mode: CullMode::None,
                depth: Depth::Test { write: false },
                blend: Blend::Alpha,
            },
            particle.stages,
            particle.vertex_input,
            particle.layout,
            samples,
        );

        // ---- RT particle overlay pass ----
        // The ray-traced backend writes the scene color directly into the
        // offscreen and has no depth buffer, so the CPU particle quads are
        // composited in a dedicated color-only pass that *loads* the RT output.
        // Occlusion happens in `rt_particle.frag.glsl` against the raygen's
        // linear-distance depth image (sampled at binding 2), so no depth
        // attachment is required. Always single-sampled: the offscreen is 1x
        // even under 2x/4x AA.
        let rt_particle_render_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    format: Format::R16G16B16A16_SFLOAT,
                    samples: 1,
                    load_op: Load,
                    store_op: Store,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {},
            }
        )
        .expect("rt particle overlay render pass");
        let rt_particle_subpass = Subpass::from(rt_particle_render_pass.clone(), 0).unwrap();
        let rt_particle =
            load_shaders::<ParticleVertex>(&device, shaders::PARTICLE_VERT_SPV, shaders::RT_PARTICLE_FRAG_SPV);
        let rt_particle_pipeline = graphics_pipeline(
            &device,
            &rt_particle_subpass,
            PipelineSpec {
                label: "rt particle pipeline",
                cull_mode: CullMode::None,
                depth: Depth::None,
                blend: Blend::Additive,
            },
            rt_particle.stages.clone(),
            rt_particle.vertex_input.clone(),
            rt_particle.layout.clone(),
            vulkano::image::SampleCount::Sample1,
        );
        let rt_dust_pipeline = graphics_pipeline(
            &device,
            &rt_particle_subpass,
            PipelineSpec {
                label: "rt dust pipeline",
                cull_mode: CullMode::None,
                depth: Depth::None,
                blend: Blend::Alpha,
            },
            rt_particle.stages,
            rt_particle.vertex_input,
            rt_particle.layout,
            vulkano::image::SampleCount::Sample1,
        );

        // Sprite atlas for the particle pipeline: a horizontal strip of four
        // 128x128 cells. Cell 0 = rain gaussian (also used by taillights and
        // headlights, variant 0); cells 1..=3 = organic cloud shapes for dust.
        let sprite_atlas_w = 512u32;
        let sprite_atlas_h = 128u32;
        let particle_sprite_view = upload_rgba8_texture(
            memory_allocator.clone(),
            command_allocator.clone(),
            queue.clone(),
            sprite_atlas_w,
            sprite_atlas_h,
            sprite_atlas,
        );
        let particle_sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: vulkano::image::sampler::Filter::Linear,
                min_filter: vulkano::image::sampler::Filter::Linear,
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                lod: 0.0..=0.0,
                ..Default::default()
            },
        )
        .expect("particle sampler");

        // ---- Sun lens flare ----
        let flare =
            load_shaders::<FlareVertex>(&device, shaders::FLARE_VERT_SPV, shaders::FLARE_FRAG_SPV);
        let flare_pipeline = graphics_pipeline(
            &device,
            &subpass,
            PipelineSpec {
                label: "flare pipeline",
                // Screen-space quads use the same winding as the HUD, so no
                // back-face culling (all triangles would be culled).
                cull_mode: CullMode::None,
                depth: Depth::None,
                blend: Blend::Additive,
            },
            flare.stages,
            flare.vertex_input,
            flare.layout,
            samples,
        );

        let (flare_core_view, flare_core_mips) = upload_rgba8_texture_mipmapped(
            memory_allocator.clone(),
            command_allocator.clone(),
            queue.clone(),
            64,
            64,
            flare_core,
        );
        let (flare_streak_view, flare_streak_mips) = upload_rgba8_texture_mipmapped(
            memory_allocator.clone(),
            command_allocator.clone(),
            queue.clone(),
            256,
            32,
            flare_streak,
        );
        let (flare_ring_view, flare_ring_mips) = upload_rgba8_texture_mipmapped(
            memory_allocator.clone(),
            command_allocator.clone(),
            queue.clone(),
            256,
            256,
            flare_ring,
        );
        let flare_sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: vulkano::image::sampler::Filter::Linear,
                min_filter: vulkano::image::sampler::Filter::Linear,
                mipmap_mode: SamplerMipmapMode::Linear,
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                lod: 0.0..=flare_core_mips
                    .max(flare_streak_mips)
                    .max(flare_ring_mips)
                    .saturating_sub(1) as f32,
                ..Default::default()
            },
        )
        .expect("flare sampler");

        // ---- Font atlas texture ----
        let atlas_staging = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            font_atlas.pixels.iter().copied(),
        )
        .expect("atlas staging buffer");

        let atlas_image = Image::new(
            memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R8_UNORM,
                extent: [font_atlas.width, font_atlas.height, 1],
                usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .expect("atlas image");

        let mut upload_builder = AutoCommandBufferBuilder::primary(
            command_allocator.clone(),
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .expect("upload command builder");
        upload_builder
            .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
                atlas_staging,
                atlas_image.clone(),
            ))
            .expect("copy atlas to image");
        let upload_cb = upload_builder.build().expect("build upload command buffer");
        sync::now(device.clone())
            .then_execute(queue.clone(), upload_cb)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        let atlas_view = ImageView::new_default(atlas_image).expect("atlas view");
        let atlas_sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: vulkano::image::sampler::Filter::Linear,
                min_filter: vulkano::image::sampler::Filter::Linear,
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                lod: 0.0..=1.0,
                ..Default::default()
            },
        )
        .expect("atlas sampler");

        let hud_set_layout = hud_pipeline.layout().set_layouts()[0].clone();
        let hud_descriptor_set = DescriptorSet::new(
            descriptor_set_allocator.clone(),
            hud_set_layout,
            [WriteDescriptorSet::image_view_sampler(
                0,
                atlas_view,
                atlas_sampler,
            )],
            [],
        )
        .expect("hud descriptor set");

        // World texture atlas, one row of slots left-to-right:
        //   slot 0 = asphalt base, slot 1 = asphalt worn, slot 2 = asphalt cracked,
        //   slot 3 = grass, slot 4 = foliage, slot 5 = rock.
        // See mesh.frag.glsl for the material-based atlas offset.
        let slot_textures = [
            asphalt_base,
            asphalt_worn,
            asphalt_cracked,
            grass,
            image::RgbaImage::from_raw(FOLIAGE_TILE, FOLIAGE_TILE, foliage_tile)
                .expect("foliage tile has the right size"),
            image::RgbaImage::from_raw(FOLIAGE_TILE, FOLIAGE_TILE, rock_tile)
                .expect("rock tile has the right size"),
        ];
        // Each slot is padded with a horizontal gutter of cloned edge columns so
        // the mip chain (generated over the whole atlas) blurs each slot into
        // itself instead of bleeding the neighbouring slot's color across the
        // boundary at low mips (e.g. green foliage tinting the rock slot). The
        // shaders inset their sampled UV by this gutter; see `mesh.frag.glsl`
        // and `raytrace.rchit.glsl` (SLOT_STRIDE / ATLAS_W must match here).
        let gutter = 8u32;
        let slot_w = slot_textures[0].dimensions().0;
        let slot_stride = slot_w + 2 * gutter;
        let atlas_h = slot_textures
            .iter()
            .map(|t| t.dimensions().1)
            .max()
            .unwrap();
        let atlas_w = slot_stride * slot_textures.len() as u32;
        let mut atlas = vec![0u8; (atlas_w * atlas_h * 4) as usize];
        for (slot, tex) in slot_textures.iter().enumerate() {
            let (sw, sh) = tex.dimensions();
            let cell_base = (slot as u32 * slot_stride) as usize;
            for y in 0..sh {
                // Clamp the sampled x to the slot content, so the gutter columns
                // clone the first/last content column of the slot.
                for x in 0..slot_stride as usize {
                    let src_x = (x as i64 - gutter as i64).clamp(0, sw as i64 - 1) as usize;
                    let dst = (y * atlas_w * 4) as usize + (cell_base + x) * 4;
                    let src = (y * sw * 4) as usize + src_x * 4;
                    atlas[dst..dst + 4].copy_from_slice(&tex.as_raw()[src..src + 4]);
                }
            }
        }
        // Mipmapped so distant ground/road samples a pre-filtered level instead
        // of aliasing into shimmering texels. The raster path selects the level
        // from screen-space derivatives; the ray-traced hit shader picks it
        // explicitly from the hit distance (implicit LOD is undefined in RT).
        // CPU-downsampled copies of the atlas for the ray-traced hit shader.
        // Explicit-LOD sampling (`textureLod`) is not honoured in the ray-tracing
        // stage on this driver (always mip 0), so the RT path blends between the
        // sharp atlas and two pre-filtered, downsampled atlases selected by
        // distance instead. The downsamples keep the atlas (and its slot gutters)
        // proportionally, so the same atlas-space UV math applies. See
        // `raytrace.rchit.glsl`.
        let atlas_img =
            image::RgbaImage::from_raw(atlas_w, atlas_h, atlas).expect("world atlas image");
        let (world_texture_view, world_mip_levels) = upload_rgba8_texture_mipmapped(
            memory_allocator.clone(),
            command_allocator.clone(),
            queue.clone(),
            atlas_w,
            atlas_h,
            atlas_img.clone().into_raw(),
        );
        let world_mid_view = upload_rgba8_texture(
            memory_allocator.clone(),
            command_allocator.clone(),
            queue.clone(),
            (atlas_w / 4).max(1),
            (atlas_h / 4).max(1),
            image::imageops::resize(
                &atlas_img,
                (atlas_w / 4).max(1),
                (atlas_h / 4).max(1),
                image::imageops::FilterType::Triangle,
            )
            .into_raw(),
        );
        let world_far_view = upload_rgba8_texture(
            memory_allocator.clone(),
            command_allocator.clone(),
            queue.clone(),
            (atlas_w / 16).max(1),
            (atlas_h / 16).max(1),
            image::imageops::resize(
                &atlas_img,
                (atlas_w / 16).max(1),
                (atlas_h / 16).max(1),
                image::imageops::FilterType::Triangle,
            )
            .into_raw(),
        );

        // Sampled by the mesh/RT particle paths (world atlas, car texture,
        // clouds) and the ray-traced hit/miss shaders. LOD range covers the
        // full world-atlas mip chain; textures with fewer levels clamp.
        let mesh_sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: vulkano::image::sampler::Filter::Linear,
                min_filter: vulkano::image::sampler::Filter::Linear,
                mipmap_mode: vulkano::image::sampler::SamplerMipmapMode::Linear,
                address_mode: [SamplerAddressMode::Repeat; 3],
                lod: 0.0..=world_mip_levels as f32 - 1.0,
                ..Default::default()
            },
        )
        .expect("mesh sampler");

        let (cmap_w, cmap_h) = colormap.dimensions();
        let car_texture_view = upload_rgba8_texture(
            memory_allocator.clone(),
            command_allocator.clone(),
            queue.clone(),
            cmap_w,
            cmap_h,
            colormap.into_raw(),
        );

        // ---- Sky cloud layer ----
        // Two decorrelated seamless tiles cross-faded in the sky shader so the
        // clouds drift and evolve. A fixed scene seed -> the same sky every
        // run; an unpredictable seed (default) -> a different sky each launch.
        let cloud_a_view = upload_rgba8_texture(
            memory_allocator.clone(),
            command_allocator.clone(),
            queue.clone(),
            CLOUD_TILE,
            CLOUD_TILE,
            cloud_a,
        );
        let cloud_b_view = upload_rgba8_texture(
            memory_allocator.clone(),
            command_allocator.clone(),
            queue.clone(),
            CLOUD_TILE,
            CLOUD_TILE,
            cloud_b,
        );

        let (sky_dome_vertices, sky_dome_indices) =
            make_mesh_buffers(memory_allocator.clone(), dome_vertices, dome_indices);

        let (car_vertices, car_indices) =
            make_mesh_buffers(memory_allocator.clone(), car_vertices, car_indices);

        let traffic_cpu = [traffic_0, traffic_1, traffic_2, traffic_3];
        let mut traffic_meshes = Vec::new();
        for (vertices, indices, anchors) in traffic_cpu {
            let (vertices, indices) =
                make_mesh_buffers(memory_allocator.clone(), vertices, indices);
            traffic_meshes.push((vertices, indices, anchors));
        }
        let traffic_anchors: Vec<_> = traffic_meshes.iter().map(|m| m.2).collect();

        Self {
            memory_allocator,
            command_allocator,
            descriptor_set_allocator,
            queue_family_index: queue.queue_family_index(),
            mesh_pipeline,
            hud_pipeline,
            sky_pipeline,
            particle_pipeline,
            dust_pipeline,
            flare_pipeline,
            rt_particle_render_pass,
            rt_particle_pipeline,
            rt_dust_pipeline,
            flare_core_view,
            flare_streak_view,
            flare_ring_view,
            flare_sampler,
            particle_sprite_view,
            particle_sampler,
            hud_descriptor_set,
            mesh_sampler,
            world_texture_view,
            world_mid_view,
            world_far_view,
            car_texture_view,
            cloud_a_view,
            cloud_b_view,
            sky_dome_vertices,
            sky_dome_indices,
            car_vertices,
            car_indices,
            player_anchors,
            traffic_meshes,
            traffic_anchors,
        }
    }

    /// Uploads the per-draw uniform block for the mesh shader (also used by
    /// the particle pipeline via the same UBO layout). `clip_plane` is the
    /// world-space `(n, d)` clip plane; the ordinary scene passes pass the
    /// disabled sentinel `(0,0,0,-1)`. `shadows` gates the mesh shader's
    /// shadow-map sampling (`shadow_state.x`); particle/RT consumers ignore it.
    pub fn mvp_buffer(
        &self,
        model: Mat4,
        uniforms: &FrameUniforms,
        headlights: &Headlights,
        clip_plane: [f32; 4],
        shadows: bool,
    ) -> Subbuffer<MVP> {
        let FrameUniforms {
            view,
            proj,
            lights,
            wet_fac,
            fog_color,
            eye,
            light_view_proj,
        } = *uniforms;
        let Headlights {
            pos: headlight_pos,
            dir: headlight_dir,
            traffic_pos: traffic_head_pos,
            traffic_dir: traffic_head_dir,
            traffic_state: traffic_head_state,
            lamp_pos,
            lamp_dir,
            lamp_state,
        } = *headlights;
        let mvp = MVP {
            model: model.to_cols_array_2d(),
            view: view.to_cols_array_2d(),
            projection: proj.to_cols_array_2d(),
            light_dir: lights.light_dir,
            fog_color,
            camera_pos: [eye.x, eye.y, eye.z, 1.0],
            light_state: [
                lights.ambient,
                lights.sun_intensity,
                wet_fac,
                lights.night_fac,
            ],
            headlight_pos,
            headlight_dir,
            traffic_head_pos,
            traffic_head_dir,
            traffic_head_state,
            lamp_pos,
            lamp_dir,
            lamp_state,
            terrain_state: [
                lights.terrain_tint[0],
                lights.terrain_tint[1],
                lights.terrain_tint[2],
                0.0,
            ],
            clip_plane,
            shadow_view_proj: light_view_proj.to_cols_array_2d(),
            shadow_state: [if shadows { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0],
        };
        Buffer::from_data(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            mvp,
        )
        .expect("mvp buffer")
    }

    /// Binds the particle pipeline and draws camera-facing billboard quads.
    /// Shared by rain and the night taillights. `shadows` feeds the shared MVP
    /// block (the particle shaders never read it).
    pub fn draw_particles(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        pipeline: &Arc<GraphicsPipeline>,
        verts: &[ParticleVertex],
        uniforms: &FrameUniforms,
        headlights: &Headlights,
        shadows: bool,
    ) {
        if verts.is_empty() {
            return;
        }
        let particle_count = verts.len() as u32;
        let mvp = self.mvp_buffer(Mat4::IDENTITY, uniforms, headlights, [0.0, 0.0, 0.0, -1.0], shadows);
        let set_layout = pipeline.layout().set_layouts()[0].clone();
        let set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            set_layout,
            [
                WriteDescriptorSet::buffer(0, mvp),
                WriteDescriptorSet::image_view_sampler(
                    1,
                    self.particle_sprite_view.clone(),
                    self.particle_sampler.clone(),
                ),
            ],
            [],
        )
        .expect("particle descriptor set");
        let particle_buf = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            verts.iter().copied(),
        )
        .expect("particle buffer");
        builder
            .bind_pipeline_graphics(pipeline.clone())
            .expect("bind particle pipeline")
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                pipeline.layout().clone(),
                0,
                set,
            )
            .expect("bind particle descriptor sets")
            .bind_vertex_buffers(0, particle_buf)
            .expect("bind particle vertex buffers");
        unsafe {
            builder
                .draw(particle_count, 1, 0, 0)
                .expect("draw particles");
        }
    }

    /// Draws particles into the ray-traced overlay pass. Identical to
    /// [`Self::draw_particles`] but binds a third descriptor (binding 2) with
    /// the raygen's linear-depth image so `rt_particle.frag.glsl` can discard
    /// fragments hidden behind geometry. `depth_sampler` must be a NEAREST
    /// sampler (exact texel depth reads, like the post composite's). `shadows`
    /// feeds the shared MVP block (the RT overlay shader never reads it).
    #[allow(clippy::too_many_arguments)]
    pub fn draw_rt_particles(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        pipeline: &Arc<GraphicsPipeline>,
        verts: &[ParticleVertex],
        uniforms: &FrameUniforms,
        headlights: &Headlights,
        depth_view: Arc<ImageView>,
        depth_sampler: Arc<Sampler>,
        shadows: bool,
    ) {
        if verts.is_empty() {
            return;
        }
        let particle_count = verts.len() as u32;
        let mvp = self.mvp_buffer(Mat4::IDENTITY, uniforms, headlights, [0.0, 0.0, 0.0, -1.0], shadows);
        let set_layout = pipeline.layout().set_layouts()[0].clone();
        let set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            set_layout,
            [
                WriteDescriptorSet::buffer(0, mvp),
                WriteDescriptorSet::image_view_sampler(
                    1,
                    self.particle_sprite_view.clone(),
                    self.particle_sampler.clone(),
                ),
                WriteDescriptorSet::image_view_sampler(2, depth_view, depth_sampler),
            ],
            [],
        )
        .expect("rt particle descriptor set");
        let particle_buf = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            verts.iter().copied(),
        )
        .expect("rt particle buffer");
        builder
            .bind_pipeline_graphics(pipeline.clone())
            .expect("bind rt particle pipeline")
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                pipeline.layout().clone(),
                0,
                set,
            )
            .expect("bind rt particle descriptor sets")
            .bind_vertex_buffers(0, particle_buf)
            .expect("bind rt particle vertex buffers");
        unsafe {
            builder
                .draw(particle_count, 1, 0, 0)
                .expect("draw rt particles");
        }
    }
}
