// SPDX-License-Identifier: MIT

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use glam::{Mat4, Vec3};
use image::load_from_memory;
use winit::window::Window;

use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferToImageInfo, PrimaryAutoCommandBuffer,
    RenderPassBeginInfo, SubpassBeginInfo, SubpassContents, SubpassEndInfo,
};
use vulkano::descriptor_set::allocator::{
    StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo,
};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::{Device, Queue};
use vulkano::format::Format;
use vulkano::image::sampler::{
    Sampler, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode,
};
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::color_blend::{
    AttachmentBlend, BlendFactor, BlendOp, ColorBlendAttachmentState, ColorBlendState,
};
use vulkano::pipeline::graphics::depth_stencil::{CompareOp, DepthState, DepthStencilState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{CullMode, RasterizationState};
use vulkano::pipeline::graphics::vertex_input::{Vertex as VertexTrait, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
    PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::{Framebuffer, RenderPass, Subpass};
use vulkano::shader::{ShaderModule, ShaderModuleCreateInfo};
use vulkano::swapchain::{
    acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
};
use vulkano::sync::future::FenceSignalFuture;
use vulkano::sync::{self, GpuFuture};
use vulkano::{Validated, VulkanError};

use crate::font::FontAtlas;
use crate::game::Game;
use crate::mesh::{build_sky_dome, build_world_chunk};
use crate::model::load_gltf_mesh_from_bytes;
use crate::render::camera::{perspective_vulkan, Camera};
use crate::render::cloud::generate_cloud_tile;
use crate::render::particles::{generate_soft_sprite, RainSystem};
use crate::render::texture::{
    make_mesh_buffers, upload_rgba8_texture, upload_rgba8_texture_mipmapped,
};
use crate::road::road_tangent;
use crate::shaders::{self, MVP, SkyUniform};
use crate::vertex::{HudVertex, ParticleVertex, Vertex3d};

pub mod camera;
pub mod cloud;
pub mod particles;
pub mod texture;

const LIGHT_DIR: [f32; 4] = [0.25, 1.0, 0.4, 0.0];
const WORLD_CHUNK_LEN: f32 = 260.0;
const WORLD_CHUNKS_BEHIND: i32 = 1;
const WORLD_CHUNKS_AHEAD: i32 = 6;

const SKY_RADIUS: f32 = 550.0;
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

pub struct Renderer {
    device: Arc<Device>,
    queue: Arc<Queue>,
    window: Arc<Window>,
    swapchain: Arc<Swapchain>,
    render_pass: Arc<RenderPass>,
    framebuffers: Vec<Arc<Framebuffer>>,
    depth_view: Arc<ImageView>,
    mesh_pipeline: Arc<GraphicsPipeline>,
    hud_pipeline: Arc<GraphicsPipeline>,
    sky_pipeline: Arc<GraphicsPipeline>,
    particle_pipeline: Arc<GraphicsPipeline>,
    particle_sprite_view: Arc<ImageView>,
    particle_sampler: Arc<Sampler>,
    rain: RainSystem,
    hud_descriptor_set: Arc<DescriptorSet>,
    mesh_sampler: Arc<Sampler>,
    world_texture_view: Arc<ImageView>,
    car_texture_view: Arc<ImageView>,
    cloud_a_view: Arc<ImageView>,
    cloud_b_view: Arc<ImageView>,
    sky_dome_vertices: Subbuffer<[Vertex3d]>,
    sky_dome_indices: Subbuffer<[u32]>,
    sky_time: f32,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    world_chunks: Vec<(Subbuffer<[Vertex3d]>, Subbuffer<[u32]>)>,
    world_anchor_chunk: i32,
    car_vertices: Subbuffer<[Vertex3d]>,
    car_indices: Subbuffer<[u32]>,
    traffic_meshes: Vec<(Subbuffer<[Vertex3d]>, Subbuffer<[u32]>)>,
    viewport: Viewport,
    pub recreate: bool,
    camera_heading: f32,
    fences: Vec<Option<Arc<FenceSignalFuture<Box<dyn GpuFuture>>>>>,
    previous_fence_i: u32,
}

impl Renderer {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        surface: Arc<Surface>,
        window: Arc<Window>,
        physical: &Arc<vulkano::device::physical::PhysicalDevice>,
        font_atlas: &FontAtlas,
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

        let caps = physical
            .surface_capabilities(&surface, Default::default())
            .expect("surface capabilities");
        let window_size = window.inner_size();
        let composite_alpha = caps.supported_composite_alpha.into_iter().next().unwrap();
        let (image_format, _) = physical
            .surface_formats(&surface, Default::default())
            .expect("surface formats")[0];

        let (swapchain, images) = Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: caps.min_image_count,
                image_format,
                image_extent: [window_size.width, window_size.height],
                image_usage: vulkano::image::ImageUsage::COLOR_ATTACHMENT,
                composite_alpha,
                ..Default::default()
            },
        )
        .expect("swapchain");

        let render_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: { format: swapchain.image_format(), samples: 1, load_op: Clear, store_op: Store },
                depth: { format: Format::D32_SFLOAT, samples: 1, load_op: Clear, store_op: DontCare },
            },
            pass: {
                color: [color],
                depth_stencil: { depth },
            }
        )
        .expect("render pass");

        let vs = unsafe {
            ShaderModule::new(
                device.clone(),
                ShaderModuleCreateInfo::new(&shaders::spv_words(shaders::MESH_VERT_SPV)),
            )
        }
        .expect("mesh vertex shader");
        let fs = unsafe {
            ShaderModule::new(
                device.clone(),
                ShaderModuleCreateInfo::new(&shaders::spv_words(shaders::MESH_FRAG_SPV)),
            )
        }
        .expect("mesh fragment shader");
        let vs_ep = vs.entry_point("main").unwrap();
        let fs_ep = fs.entry_point("main").unwrap();
        let mesh_vertex_input = Vertex3d::per_vertex().definition(&vs_ep).unwrap();
        let mesh_stages = [
            PipelineShaderStageCreateInfo::new(vs_ep),
            PipelineShaderStageCreateInfo::new(fs_ep),
        ];
        let mesh_layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&mesh_stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();
        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();
        let mesh_pipeline = GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: mesh_stages.into_iter().collect(),
                vertex_input_state: Some(mesh_vertex_input),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState::default()),
                rasterization_state: Some(RasterizationState {
                    cull_mode: CullMode::Back,
                    ..Default::default()
                }),
                multisample_state: Some(MultisampleState::default()),
                depth_stencil_state: Some(DepthStencilState {
                    depth: Some(DepthState {
                        write_enable: true,
                        compare_op: CompareOp::Less,
                    }),
                    ..Default::default()
                }),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.num_color_attachments(),
                    ColorBlendAttachmentState::default(),
                )),
                dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(mesh_layout)
            },
        )
        .expect("mesh pipeline");

        let hvs = unsafe {
            ShaderModule::new(
                device.clone(),
                ShaderModuleCreateInfo::new(&shaders::spv_words(shaders::HUD_VERT_SPV)),
            )
        }
        .expect("hud vertex shader");
        let hfs = unsafe {
            ShaderModule::new(
                device.clone(),
                ShaderModuleCreateInfo::new(&shaders::spv_words(shaders::HUD_FRAG_SPV)),
            )
        }
        .expect("hud fragment shader");
        let hvs_ep = hvs.entry_point("main").unwrap();
        let hfs_ep = hfs.entry_point("main").unwrap();
        let hud_vertex_input = HudVertex::per_vertex().definition(&hvs_ep).unwrap();
        let hud_stages = [
            PipelineShaderStageCreateInfo::new(hvs_ep),
            PipelineShaderStageCreateInfo::new(hfs_ep),
        ];
        let hud_layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&hud_stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();
        let hud_subpass = Subpass::from(render_pass.clone(), 0).unwrap();
        let hud_pipeline = GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: hud_stages.into_iter().collect(),
                vertex_input_state: Some(hud_vertex_input),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState::default()),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState::default()),
                depth_stencil_state: Some(DepthStencilState {
                    depth: None,
                    ..Default::default()
                }),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    hud_subpass.num_color_attachments(),
                    ColorBlendAttachmentState {
                        blend: Some(AttachmentBlend::alpha()),
                        ..Default::default()
                    },
                )),
                dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                subpass: Some(hud_subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(hud_layout)
            },
        )
        .expect("hud pipeline");

        let svs = unsafe {
            ShaderModule::new(
                device.clone(),
                ShaderModuleCreateInfo::new(&shaders::spv_words(shaders::SKY_VERT_SPV)),
            )
        }
        .expect("sky vertex shader");
        let sfs = unsafe {
            ShaderModule::new(
                device.clone(),
                ShaderModuleCreateInfo::new(&shaders::spv_words(shaders::SKY_FRAG_SPV)),
            )
        }
        .expect("sky fragment shader");
        let svs_ep = svs.entry_point("main").unwrap();
        let sfs_ep = sfs.entry_point("main").unwrap();
        let sky_vertex_input = Vertex3d::per_vertex().definition(&svs_ep).unwrap();
        let sky_stages = [
            PipelineShaderStageCreateInfo::new(svs_ep),
            PipelineShaderStageCreateInfo::new(sfs_ep),
        ];
        let sky_layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&sky_stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();
        let sky_subpass = Subpass::from(render_pass.clone(), 0).unwrap();
        let sky_pipeline = GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: sky_stages.into_iter().collect(),
                vertex_input_state: Some(sky_vertex_input),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState::default()),
                rasterization_state: Some(RasterizationState {
                    cull_mode: CullMode::None,
                    ..Default::default()
                }),
                multisample_state: Some(MultisampleState::default()),
                depth_stencil_state: Some(DepthStencilState {
                    depth: None,
                    ..Default::default()
                }),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    sky_subpass.num_color_attachments(),
                    ColorBlendAttachmentState::default(),
                )),
                dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                subpass: Some(sky_subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(sky_layout)
            },
        )
        .expect("sky pipeline");

        // ---- Rain particles ----
        let pvs = unsafe {
            ShaderModule::new(
                device.clone(),
                ShaderModuleCreateInfo::new(&shaders::spv_words(shaders::PARTICLE_VERT_SPV)),
            )
        }
        .expect("particle vertex shader");
        let pfs = unsafe {
            ShaderModule::new(
                device.clone(),
                ShaderModuleCreateInfo::new(&shaders::spv_words(shaders::PARTICLE_FRAG_SPV)),
            )
        }
        .expect("particle fragment shader");
        let pvs_ep = pvs.entry_point("main").unwrap();
        let pfs_ep = pfs.entry_point("main").unwrap();
        let particle_vertex_input = ParticleVertex::per_vertex().definition(&pvs_ep).unwrap();
        let particle_stages = [
            PipelineShaderStageCreateInfo::new(pvs_ep),
            PipelineShaderStageCreateInfo::new(pfs_ep),
        ];
        let particle_layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&particle_stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();
        let particle_subpass = Subpass::from(render_pass.clone(), 0).unwrap();
        let particle_pipeline = GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: particle_stages.into_iter().collect(),
                vertex_input_state: Some(particle_vertex_input),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState::default()),
                rasterization_state: Some(RasterizationState {
                    cull_mode: CullMode::None,
                    ..Default::default()
                }),
                multisample_state: Some(MultisampleState::default()),
                depth_stencil_state: Some(DepthStencilState {
                    depth: Some(DepthState {
                        write_enable: false,
                        compare_op: CompareOp::Less,
                    }),
                    ..Default::default()
                }),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    particle_subpass.num_color_attachments(),
                    ColorBlendAttachmentState {
                        blend: Some(AttachmentBlend {
                            src_color_blend_factor: BlendFactor::SrcAlpha,
                            dst_color_blend_factor: BlendFactor::One,
                            color_blend_op: BlendOp::Add,
                            src_alpha_blend_factor: BlendFactor::SrcAlpha,
                            dst_alpha_blend_factor: BlendFactor::One,
                            alpha_blend_op: BlendOp::Add,
                        }),
                        ..Default::default()
                    },
                )),
                dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                subpass: Some(particle_subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(particle_layout)
            },
        )
        .expect("particle pipeline");

        let (particle_sprite_view, particle_sprite_mips) = upload_rgba8_texture_mipmapped(
            memory_allocator.clone(),
            command_allocator.clone(),
            queue.clone(),
            128,
            128,
            generate_soft_sprite(128),
        );
        let particle_sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: vulkano::image::sampler::Filter::Linear,
                min_filter: vulkano::image::sampler::Filter::Linear,
                mipmap_mode: SamplerMipmapMode::Linear,
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                lod: 0.0..=particle_sprite_mips.saturating_sub(1) as f32,
                ..Default::default()
            },
        )
        .expect("particle sampler");

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
            load_from_memory(ASPHALT_BASE_PNG)
                .expect("failed to decode embedded asphalt_base texture")
                .to_rgba8(),
            load_from_memory(ASPHALT_WORN_PNG)
                .expect("failed to decode embedded asphalt_worn texture")
                .to_rgba8(),
            load_from_memory(ASPHALT_CRACKED_PNG)
                .expect("failed to decode embedded asphalt_cracked texture")
                .to_rgba8(),
            load_from_memory(GRASS_PNG)
                .expect("failed to decode embedded grass texture")
                .to_rgba8(),
        ];
        let slot_w = slot_textures[0].dimensions().0;
        let atlas_h = slot_textures.iter().map(|t| t.dimensions().1).max().unwrap();
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

        let colormap = load_from_memory(CAR_COLORMAP_PNG)
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
        // clouds drift and evolve. Per-run seed -> a different sky each launch.
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before epoch")
            .as_nanos() as u64;
        let cloud_a = generate_cloud_tile(CLOUD_TILE, seed);
        let cloud_b = generate_cloud_tile(
            CLOUD_TILE,
            seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407),
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

        let (dome_vertices, dome_indices) = build_sky_dome(10, 32);
        let (sky_dome_vertices, sky_dome_indices) =
            make_mesh_buffers(memory_allocator.clone(), dome_vertices, dome_indices);

        let player_mesh = load_gltf_mesh_from_bytes(PLAYER_MODEL_GLB, "player_race_future.glb")
            .expect("failed to load embedded player model");
        let (car_vertices, car_indices) =
            make_mesh_buffers(memory_allocator.clone(), player_mesh.0, player_mesh.1);

        let traffic_models = [
            ("traffic_sedan.glb", TRAFFIC_SEDAN_GLB),
            ("traffic_suv.glb", TRAFFIC_SUV_GLB),
            ("traffic_taxi.glb", TRAFFIC_TAXI_GLB),
            ("traffic_van.glb", TRAFFIC_VAN_GLB),
        ];
        let mut traffic_meshes = Vec::new();
        for (label, bytes) in traffic_models {
            let mesh = load_gltf_mesh_from_bytes(bytes, label)
                .unwrap_or_else(|e| panic!("failed to load embedded traffic model {label}: {e}"));
            traffic_meshes.push(make_mesh_buffers(memory_allocator.clone(), mesh.0, mesh.1));
        }

        let extent = swapchain.image_extent();
        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: [extent[0] as f32, extent[1] as f32],
            depth_range: 0.0..=1.0,
        };

        let mut renderer = Renderer {
            device: device.clone(),
            queue: queue.clone(),
            window: window.clone(),
            swapchain,
            render_pass,
            framebuffers: Vec::new(),
            depth_view: ImageView::new_default(
                Image::new(
                    memory_allocator.clone(),
                    ImageCreateInfo {
                        image_type: ImageType::Dim2d,
                        format: Format::D32_SFLOAT,
                        extent: [extent[0], extent[1], 1],
                        usage: vulkano::image::ImageUsage::DEPTH_STENCIL_ATTACHMENT,
                        ..Default::default()
                    },
                    AllocationCreateInfo::default(),
                )
                .expect("depth image"),
            )
            .expect("depth view"),
            mesh_pipeline,
            hud_pipeline,
            sky_pipeline,
            particle_pipeline,
            particle_sprite_view,
            particle_sampler,
            rain: RainSystem::new(),
            hud_descriptor_set,
            mesh_sampler,
            world_texture_view,
            car_texture_view,
            cloud_a_view,
            cloud_b_view,
            sky_dome_vertices,
            sky_dome_indices,
            sky_time: 0.0,
            memory_allocator,
            command_allocator,
            descriptor_set_allocator,
            world_chunks: Vec::new(),
            world_anchor_chunk: i32::MIN,
            car_vertices,
            car_indices,
            traffic_meshes,
            viewport,
            recreate: false,
            camera_heading: 0.0,
            fences: Vec::new(),
            previous_fence_i: 0,
        };

        renderer.framebuffers = renderer.create_framebuffers(&images);
        renderer.fences = vec![None; renderer.framebuffers.len()];
        renderer.rebuild_world_chunks(0);
        renderer
    }

    fn rebuild_world_chunks(&mut self, anchor_chunk: i32) {
        self.world_chunks.clear();
        for rel in -WORLD_CHUNKS_BEHIND..=WORLD_CHUNKS_AHEAD {
            let chunk_idx = anchor_chunk + rel;
            let start_s = chunk_idx as f32 * WORLD_CHUNK_LEN;
            let (wv, wi) = build_world_chunk(start_s, WORLD_CHUNK_LEN);
            let world_vertices = Buffer::from_iter(
                self.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::VERTEX_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                wv,
            )
            .expect("world chunk vertices");
            let world_indices = Buffer::from_iter(
                self.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::INDEX_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                wi,
            )
            .expect("world chunk indices");
            self.world_chunks.push((world_vertices, world_indices));
        }
        self.world_anchor_chunk = anchor_chunk;
    }

    fn ensure_world_chunks_for_player(&mut self, player_distance: f32) {
        let current_chunk = (player_distance / WORLD_CHUNK_LEN).floor() as i32;
        if current_chunk != self.world_anchor_chunk {
            self.rebuild_world_chunks(current_chunk);
        }
    }

    fn create_framebuffers(&self, images: &[Arc<Image>]) -> Vec<Arc<Framebuffer>> {
        images
            .iter()
            .map(|img| {
                let color_view = ImageView::new_default(img.clone()).unwrap();
                Framebuffer::new(
                    self.render_pass.clone(),
                    vulkano::render_pass::FramebufferCreateInfo {
                        attachments: vec![color_view, self.depth_view.clone()],
                        ..Default::default()
                    },
                )
                .unwrap()
            })
            .collect()
    }

    fn recreate_swapchain(&mut self) {
        let dims = self.window.inner_size();
        let (new_swapchain, new_images) = self
            .swapchain
            .recreate(SwapchainCreateInfo {
                image_extent: [dims.width, dims.height],
                ..self.swapchain.create_info().clone()
            })
            .expect("recreate swapchain");
        self.swapchain = new_swapchain;
        let extent = self.swapchain.image_extent();
        let depth = Image::new(
            self.memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::D32_SFLOAT,
                extent: [extent[0], extent[1], 1],
                usage: vulkano::image::ImageUsage::DEPTH_STENCIL_ATTACHMENT,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .expect("depth image");
        self.depth_view = ImageView::new_default(depth).expect("depth view");
        self.framebuffers = self.create_framebuffers(&new_images);
        self.viewport.extent = [dims.width as f32, dims.height as f32];
        self.recreate = false;
    }

    fn mvp_buffer(&self, model: Mat4, view: Mat4, proj: Mat4, fog_color: [f32; 4]) -> Subbuffer<MVP> {
        let mvp = MVP {
            model: model.to_cols_array_2d(),
            view: view.to_cols_array_2d(),
            projection: proj.to_cols_array_2d(),
            light_dir: LIGHT_DIR,
            fog_color,
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

    /// Waits for all pending work on the backing device, so its swapchain and
    /// resources can be torn down safely. Called before a GPU switch.
    pub(crate) fn wait_idle(&self) {
        // Safety: no further work is submitted to this device afterwards; it is
        // dropped immediately after this call.
        unsafe { self.device.wait_idle() }.expect("failed to wait for device idle");
    }

    pub fn render(&mut self, game: &Game, dt: std::time::Duration, hud_verts: &[HudVertex]) {
        if self.recreate {
            self.recreate_swapchain();
        }

        let (image_i, suboptimal, acquire_future) =
            match acquire_next_image(self.swapchain.clone(), None) {
                Ok(r) => r,
                Err(Validated::Error(VulkanError::OutOfDate)) => {
                    self.recreate = true;
                    return;
                }
                Err(e) => panic!("failed to acquire next image: {:?}", e),
            };
        if suboptimal {
            self.recreate = true;
        }

        if let Some(fence) = &self.fences[image_i as usize] {
            fence.wait(None).unwrap();
        }

        let aspect = self.viewport.extent[0] / self.viewport.extent[1];
        let proj = perspective_vulkan(60.0f32.to_radians(), aspect, 0.1, 600.0);

        // Fog color mirrors the sky shader's horizon at t=0 so the mesh fades
        // into exactly the same color the sky dome shows at the horizon line,
        // including the weather dim/overcast shift.
        let cover = {
            let t = ((game.cloud_amount() - 0.10) / 0.90).clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        };
        let dim = 1.0 - 0.22 * cover;
        let fog_color = [
            (0.55 + (0.60 - 0.55) * cover) * dim,
            (0.70 + (0.60 - 0.70) * cover) * dim,
            (0.92 + (0.63 - 0.92) * cover) * dim,
            1.0,
        ];

        self.ensure_world_chunks_for_player(game.vehicle.distance);
        let car_pos = Vec3::new(game.player_world_x(), 0.0, game.player_world_z());
        let dt_secs = dt.as_secs_f32().min(0.05);
        self.sky_time += dt_secs;
        let diff = game.vehicle.heading - self.camera_heading;
        self.camera_heading += diff * (dt_secs * 3.0).min(1.0);
        let cam_forward = Vec3::new(self.camera_heading.sin(), 0.0, -self.camera_heading.cos());
        let eye = car_pos - cam_forward * 8.0 + Vec3::new(0.0, 4.0, 0.0);
        let look_at = car_pos + cam_forward * 4.0 + Vec3::new(0.0, 0.5, 0.0);
        let cam = Camera {
            eye,
            forward: (look_at - eye).normalize(),
        };
        let view = cam.view();

        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
        )
        .expect("command buffer builder");

        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([0.9, 0.7, 0.5, 1.0].into()), Some(1.0f32.into())],
                    ..RenderPassBeginInfo::framebuffer(self.framebuffers[image_i as usize].clone())
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )
            .expect("begin render pass")
            .set_viewport(0, [self.viewport.clone()].into_iter().collect())
            .expect("set viewport");

        // ---- Sky dome (background) ----
        // Drawn first with depth disabled so the 3D scene overdraws it.
        builder
            .bind_pipeline_graphics(self.sky_pipeline.clone())
            .expect("bind sky pipeline");

        let sky_uniform = SkyUniform {
            model: Mat4::from_scale_rotation_translation(
                Vec3::splat(SKY_RADIUS),
                glam::Quat::IDENTITY,
                eye,
            )
            .to_cols_array_2d(),
            view: view.to_cols_array_2d(),
            projection: proj.to_cols_array_2d(),
            time: self.sky_time,
            _pad: [0.0; 3],
            zenith: [0.18, 0.42, 0.83, 1.0],
            horizon: [0.55, 0.70, 0.92, 1.0],
            cloud_tint: [1.0, 0.97, 0.92, 1.0],
            light_dir: LIGHT_DIR,
            cloud_amount: game.cloud_amount(),
            _pad2: [0.0; 3],
        };
        let sky_buf = Buffer::from_data(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            sky_uniform,
        )
        .expect("sky uniform buffer");
        let sky_set_layout = self.sky_pipeline.layout().set_layouts()[0].clone();
        let sky_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            sky_set_layout,
            [
                WriteDescriptorSet::buffer(0, sky_buf),
                WriteDescriptorSet::image_view_sampler(
                    1,
                    self.cloud_a_view.clone(),
                    self.mesh_sampler.clone(),
                ),
                WriteDescriptorSet::image_view_sampler(
                    2,
                    self.cloud_b_view.clone(),
                    self.mesh_sampler.clone(),
                ),
            ],
            [],
        )
        .expect("sky descriptor set");
        let sky_index_count = self.sky_dome_indices.len() as u32;
        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.sky_pipeline.layout().clone(),
                0,
                sky_set,
            )
            .expect("bind sky descriptor sets")
            .bind_vertex_buffers(0, self.sky_dome_vertices.clone())
            .expect("bind sky vertex buffers")
            .bind_index_buffer(self.sky_dome_indices.clone())
            .expect("bind sky index buffer");
        unsafe {
            builder
                .draw_indexed(sky_index_count, 1, 0, 0, 0)
                .expect("draw sky");
        }

        // ---- 3D scene ----
        builder
            .bind_pipeline_graphics(self.mesh_pipeline.clone())
            .expect("bind mesh pipeline");

        let draw = |builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
                    vertices: Subbuffer<[Vertex3d]>,
                    indices: Subbuffer<[u32]>,
                    texture: Arc<ImageView>,
                    model: Mat4| {
            let index_count = indices.len() as u32;
            let mvp = self.mvp_buffer(model, view, proj, fog_color);
            let set_layout = self.mesh_pipeline.layout().set_layouts()[0].clone();
            let set = DescriptorSet::new(
                self.descriptor_set_allocator.clone(),
                set_layout,
                [
                    WriteDescriptorSet::buffer(0, mvp.clone()),
                    WriteDescriptorSet::image_view_sampler(1, texture, self.mesh_sampler.clone()),
                ],
                [],
            )
            .expect("descriptor set");
            builder
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.mesh_pipeline.layout().clone(),
                    0,
                    set,
                )
                .expect("bind descriptor sets")
                .bind_vertex_buffers(0, vertices)
                .expect("bind vertex buffers")
                .bind_index_buffer(indices)
                .expect("bind index buffer");
            unsafe {
                builder
                    .draw_indexed(index_count, 1, 0, 0, 0)
                    .expect("draw indexed");
            }
        };

        for (world_vertices, world_indices) in &self.world_chunks {
            draw(
                &mut builder,
                world_vertices.clone(),
                world_indices.clone(),
                self.world_texture_view.clone(),
                Mat4::IDENTITY,
            );
        }
        // player car
        draw(
            &mut builder,
            self.car_vertices.clone(),
            self.car_indices.clone(),
            self.car_texture_view.clone(),
            Mat4::from_scale_rotation_translation(
                Vec3::ONE,
                glam::Quat::from_rotation_y(-game.vehicle.heading),
                Vec3::new(game.player_world_x(), 0.03, game.player_world_z()),
            ),
        );
        // traffic
        for (idx, t) in game.traffic.iter().enumerate() {
            let tvx = crate::road::road_curve(t.distance) + t.lane;
            let traffic_rot = if t.lane > 0.0 {
                glam::Quat::from_rotation_y(f32::atan2(-road_tangent(t.distance), 1.0))
            } else {
                glam::Quat::from_rotation_y(f32::atan2(road_tangent(t.distance), -1.0))
            };
            let (traffic_vertices, traffic_indices) =
                &self.traffic_meshes[idx % self.traffic_meshes.len()];
            draw(
                &mut builder,
                traffic_vertices.clone(),
                traffic_indices.clone(),
                self.car_texture_view.clone(),
                Mat4::from_scale_rotation_translation(
                    Vec3::ONE,
                    traffic_rot,
                    Vec3::new(tvx, 0.35, -t.distance),
                ),
            );
        }

        // ---- Rain particles ----
        // Additive, depth-tested (no depth write) so rain falls over the road
        // but behind cars, and fades into the sky fog like everything else.
        let rain_intensity = game.rain_intensity();
        if rain_intensity > 0.0 {
            let cam_right = cam_forward.cross(Vec3::Y);
            self.rain.update(dt_secs, eye, game.vehicle.speed);
            let particle_verts = self.rain.build_vertices(eye, cam_right, rain_intensity);
            if !particle_verts.is_empty() {
                let particle_count = particle_verts.len() as u32;
                let mvp = self.mvp_buffer(Mat4::IDENTITY, view, proj, fog_color);
                let particle_set_layout = self.particle_pipeline.layout().set_layouts()[0].clone();
                let particle_set = DescriptorSet::new(
                    self.descriptor_set_allocator.clone(),
                    particle_set_layout,
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
                    particle_verts.iter().copied(),
                )
                .expect("particle buffer");
                builder
                    .bind_pipeline_graphics(self.particle_pipeline.clone())
                    .expect("bind particle pipeline")
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        self.particle_pipeline.layout().clone(),
                        0,
                        particle_set,
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

        // ---- HUD ----
        builder
            .bind_pipeline_graphics(self.hud_pipeline.clone())
            .expect("bind hud pipeline")
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.hud_pipeline.layout().clone(),
                0,
                self.hud_descriptor_set.clone(),
            )
            .expect("bind hud descriptor set");
        let hud_vertex_count = hud_verts.len() as u32;
        let hud_buf = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            hud_verts.iter().copied(),
        )
        .expect("hud buffer");
        builder
            .bind_vertex_buffers(0, hud_buf)
            .expect("bind hud vertex buffers");
        unsafe {
            builder.draw(hud_vertex_count, 1, 0, 0).expect("draw hud");
        }

        builder
            .end_render_pass(SubpassEndInfo::default())
            .expect("end render pass");
        let command_buffer = builder.build().expect("build command buffer");

        let previous_future: Box<dyn GpuFuture> =
            match self.fences[self.previous_fence_i as usize].clone() {
                Some(f) => f.boxed(),
                None => {
                    let mut now = sync::now(self.device.clone());
                    now.cleanup_finished();
                    now.boxed()
                }
            };

        let future = previous_future
            .join(acquire_future)
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(
                self.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(self.swapchain.clone(), image_i),
            )
            .boxed()
            .then_signal_fence_and_flush();

        self.fences[image_i as usize] = match future.map_err(Validated::unwrap) {
            Ok(v) => Some(Arc::new(v)),
            Err(VulkanError::OutOfDate) => {
                self.recreate = true;
                None
            }
            Err(e) => {
                println!("failed to flush future: {:?}", e);
                None
            }
        };
        self.previous_fence_i = image_i;
    }
}
