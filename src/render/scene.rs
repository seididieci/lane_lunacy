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
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::rasterization::CullMode;
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint};
use vulkano::render_pass::{RenderPass, Subpass};
use vulkano::sync::{self, GpuFuture};

use crate::font::FontAtlas;
use crate::mesh::build_sky_dome;
use crate::model::{load_gltf_mesh_from_bytes, CarLightAnchors};
use crate::render::cloud::generate_cloud_tile;
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

    pub flare_core_view: Arc<ImageView>,
    pub flare_streak_view: Arc<ImageView>,
    pub flare_ring_view: Arc<ImageView>,
    pub flare_sampler: Arc<Sampler>,
    pub particle_sprite_view: Arc<ImageView>,
    pub particle_sampler: Arc<Sampler>,

    pub hud_descriptor_set: Arc<DescriptorSet>,
    pub mesh_sampler: Arc<Sampler>,
    pub world_texture_view: Arc<ImageView>,
    pub car_texture_view: Arc<ImageView>,
    pub cloud_a_view: Arc<ImageView>,
    pub cloud_b_view: Arc<ImageView>,

    pub sky_dome_vertices: Subbuffer<[Vertex3d]>,
    pub sky_dome_indices: Subbuffer<[u32]>,

    pub car_vertices: Subbuffer<[Vertex3d]>,
    pub car_indices: Subbuffer<[u32]>,
    pub player_anchors: CarLightAnchors,
    pub traffic_meshes: Vec<(Subbuffer<[Vertex3d]>, Subbuffer<[u32]>, CarLightAnchors)>,
    pub traffic_anchors: Vec<CarLightAnchors>,
}

impl SceneResources {
    /// Builds every pipeline, texture, sampler and model. The `render_pass`
    /// must have one color + one depth attachment (swapchain or offscreen);
    /// the pipelines are recorded against its subpass.
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        render_pass: Arc<RenderPass>,
        font_atlas: &FontAtlas,
        seed: u64,
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
        );

        // Sprite atlas for the particle pipeline: a horizontal strip of four
        // 128x128 cells. Cell 0 = rain gaussian (also used by taillights and
        // headlights, variant 0); cells 1..=3 = organic cloud shapes for dust.
        let sprite_atlas_w = 512u32;
        let sprite_atlas_h = 128u32;
        let mut sprite_atlas = Vec::with_capacity((sprite_atlas_w * sprite_atlas_h * 4) as usize);
        let gaussian = generate_soft_sprite(128);
        sprite_atlas.extend_from_slice(&gaussian);
        for seed in [
            0x9E3779B97F4A7C15u64,
            0xBF58476D1CE4E5B9u64,
            0x94D049BB133111EBu64,
        ] {
            sprite_atlas.extend_from_slice(&generate_cloud_sprite(128, seed));
        }
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
        );

        let (flare_core_view, flare_core_mips) = upload_rgba8_texture_mipmapped(
            memory_allocator.clone(),
            command_allocator.clone(),
            queue.clone(),
            64,
            64,
            flare::generate_sun_core(64),
        );
        let (flare_streak_view, flare_streak_mips) = upload_rgba8_texture_mipmapped(
            memory_allocator.clone(),
            command_allocator.clone(),
            queue.clone(),
            256,
            32,
            flare::generate_flare_streak(256, 32),
        );
        let (flare_ring_view, flare_ring_mips) = upload_rgba8_texture_mipmapped(
            memory_allocator.clone(),
            command_allocator.clone(),
            queue.clone(),
            256,
            256,
            flare::generate_flare_ring(256),
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

        let mesh_sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: vulkano::image::sampler::Filter::Linear,
                min_filter: vulkano::image::sampler::Filter::Linear,
                address_mode: [SamplerAddressMode::Repeat; 3],
                lod: 0.0..=1.0,
                ..Default::default()
            },
        )
        .expect("mesh sampler");

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
        //   slot 0 = asphalt base, slot 1 = asphalt worn, slot 2 = asphalt cracked, slot 3 = grass.
        // See mesh.frag.glsl for the material-based atlas offset.
        let slot_textures = [
            image::load_from_memory(ASPHALT_BASE_PNG)
                .expect("failed to decode embedded asphalt_base texture")
                .to_rgba8(),
            image::load_from_memory(ASPHALT_WORN_PNG)
                .expect("failed to decode embedded asphalt_worn texture")
                .to_rgba8(),
            image::load_from_memory(ASPHALT_CRACKED_PNG)
                .expect("failed to decode embedded asphalt_cracked texture")
                .to_rgba8(),
            image::load_from_memory(GRASS_PNG)
                .expect("failed to decode embedded grass texture")
                .to_rgba8(),
        ];
        let slot_w = slot_textures[0].dimensions().0;
        let atlas_h = slot_textures
            .iter()
            .map(|t| t.dimensions().1)
            .max()
            .unwrap();
        let atlas_w = slot_w * slot_textures.len() as u32;
        let mut atlas = vec![0u8; (atlas_w * atlas_h * 4) as usize];
        for (slot, tex) in slot_textures.iter().enumerate() {
            let (sw, sh) = tex.dimensions();
            let x0 = (slot as u32 * slot_w) as usize;
            for y in 0..sh {
                let dst = (y * atlas_w * 4) as usize + x0 * 4;
                let src = (y * sw * 4) as usize;
                atlas[dst..dst + (sw as usize) * 4]
                    .copy_from_slice(&tex.as_raw()[src..src + (sw as usize) * 4]);
            }
        }
        let world_texture_view = upload_rgba8_texture(
            memory_allocator.clone(),
            command_allocator.clone(),
            queue.clone(),
            atlas_w,
            atlas_h,
            atlas,
        );

        let colormap = image::load_from_memory(CAR_COLORMAP_PNG)
            .expect("failed to decode embedded car colormap texture")
            .to_rgba8();
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
        let cloud_a = generate_cloud_tile(CLOUD_TILE, seed);
        let cloud_b = generate_cloud_tile(
            CLOUD_TILE,
            seed.wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407),
        );
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

        let (dome_vertices, dome_indices) = build_sky_dome(32, 128);
        let (sky_dome_vertices, sky_dome_indices) =
            make_mesh_buffers(memory_allocator.clone(), dome_vertices, dome_indices);

        let (car_vertices, car_indices, player_anchors) =
            load_gltf_mesh_from_bytes(PLAYER_MODEL_GLB, "player_race_future.glb")
                .expect("failed to load embedded player model");
        let (car_vertices, car_indices) =
            make_mesh_buffers(memory_allocator.clone(), car_vertices, car_indices);

        let traffic_models = [
            ("traffic_sedan.glb", TRAFFIC_SEDAN_GLB),
            ("traffic_suv.glb", TRAFFIC_SUV_GLB),
            ("traffic_taxi.glb", TRAFFIC_TAXI_GLB),
            ("traffic_van.glb", TRAFFIC_VAN_GLB),
        ];
        let mut traffic_meshes = Vec::new();
        for (label, bytes) in traffic_models {
            let (vertices, indices, anchors) = load_gltf_mesh_from_bytes(bytes, label)
                .unwrap_or_else(|e| panic!("failed to load embedded traffic model {label}: {e}"));
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
            flare_core_view,
            flare_streak_view,
            flare_ring_view,
            flare_sampler,
            particle_sprite_view,
            particle_sampler,
            hud_descriptor_set,
            mesh_sampler,
            world_texture_view,
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
    /// the particle pipeline via the same UBO layout).
    pub fn mvp_buffer(
        &self,
        model: Mat4,
        uniforms: &FrameUniforms,
        headlights: &Headlights,
    ) -> Subbuffer<MVP> {
        let FrameUniforms {
            view,
            proj,
            lights,
            wet_fac,
            fog_color,
        } = *uniforms;
        let Headlights {
            pos: headlight_pos,
            dir: headlight_dir,
            traffic_pos: traffic_head_pos,
            traffic_dir: traffic_head_dir,
            traffic_state: traffic_head_state,
        } = *headlights;
        let mvp = MVP {
            model: model.to_cols_array_2d(),
            view: view.to_cols_array_2d(),
            projection: proj.to_cols_array_2d(),
            light_dir: lights.light_dir,
            fog_color,
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
    /// Shared by rain and the night taillights.
    pub fn draw_particles(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        pipeline: &Arc<GraphicsPipeline>,
        verts: &[ParticleVertex],
        uniforms: &FrameUniforms,
        headlights: &Headlights,
    ) {
        if verts.is_empty() {
            return;
        }
        let particle_count = verts.len() as u32;
        let mvp = self.mvp_buffer(Mat4::IDENTITY, uniforms, headlights);
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
}
