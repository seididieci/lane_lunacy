// SPDX-License-Identifier: MIT
use std::sync::Arc;

use glam::{Mat4, Vec3};
use winit::window::Window;

use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, RenderPassBeginInfo,
    SubpassBeginInfo, SubpassContents, SubpassEndInfo,
};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::{Device, Queue};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
use vulkano::render_pass::{Framebuffer, RenderPass};
use vulkano::swapchain::{
    acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
};
use vulkano::sync::future::FenceSignalFuture;
use vulkano::sync::{self, GpuFuture};
use vulkano::{Validated, VulkanError};

use crate::font::FontAtlas;
use crate::game::Game;
use crate::mesh::build_world_chunk;
use crate::render::frame::{build_frame, traffic_rotation};
use crate::render::particles::{DustSystem, RainSystem};
use crate::render::scene::SceneResources;
use crate::vertex::{HudVertex, Vertex3d};

pub mod camera;
pub mod cloud;
pub mod daynight;
pub mod flare;
pub mod frame;
pub mod particles;
pub mod scene;
pub mod texture;

const WORLD_CHUNK_LEN: f32 = 260.0;
const WORLD_CHUNKS_BEHIND: i32 = 1;
const WORLD_CHUNKS_AHEAD: i32 = 6;

pub struct Renderer {
    device: Arc<Device>,
    queue: Arc<Queue>,
    window: Arc<Window>,
    swapchain: Arc<Swapchain>,
    render_pass: Arc<RenderPass>,
    framebuffers: Vec<Arc<Framebuffer>>,
    depth_view: Arc<ImageView>,
    /// Everything shared with the headless snapshot path: pipelines, textures,
    /// samplers, models, buffers, allocators.
    scene: SceneResources,
    rain: RainSystem,
    dust: DustSystem,
    sky_time: f32,
    world_chunks: Vec<(Subbuffer<[Vertex3d]>, Subbuffer<[u32]>)>,
    world_anchor_chunk: i32,
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
        seed: u64,
    ) -> Self {
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

        // Every GPU resource that is independent of the swapchain/framebuffer
        // lives in `SceneResources`, shared with the headless snapshot path.
        let scene = SceneResources::new(device.clone(), queue.clone(), render_pass.clone(), font_atlas, seed);

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
                    scene.memory_allocator.clone(),
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
            scene,
            rain: RainSystem::new(),
            dust: DustSystem::new(),
            sky_time: 0.0,
            world_chunks: Vec::new(),
            world_anchor_chunk: i32::MIN,
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
                self.scene.memory_allocator.clone(),
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
                self.scene.memory_allocator.clone(),
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
            self.scene.memory_allocator.clone(),
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

        // Everything per-frame is computed on the CPU into a pure `Frame`:
        // camera, day/night lights, sky uniform, headlight projectors,
        // particles, and flare. The same builder drives the headless snapshot
        // path, so windowed and offline renders are pixel-identical.
        self.ensure_world_chunks_for_player(game.vehicle.distance);
        let frame = build_frame(
            game,
            dt,
            aspect,
            &mut self.sky_time,
            &mut self.camera_heading,
            &mut self.rain,
            &mut self.dust,
            &self.scene.player_anchors,
            &self.scene.traffic_anchors,
            hud_verts.to_vec(),
        );

        let view = frame.view;
        let proj = frame.proj;
        let lights = frame.lights;
        let fog_color = frame.fog_color;
        let wet_fac = frame.wet_fac;
        let headlight_pos = frame.headlight_pos;
        let headlight_dir = frame.headlight_dir;
        let traffic_head_pos = frame.traffic_head_pos;
        let traffic_head_dir = frame.traffic_head_dir;
        let traffic_head_state = frame.traffic_head_state;
        let sky_uniform = frame.sky_uniform;
        let particle_verts = frame.particle_verts;
        let dust_verts = frame.dust_verts;
        let flare_verts = frame.flare_verts;

        let mut builder = AutoCommandBufferBuilder::primary(
            self.scene.command_allocator.clone(),
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
            .bind_pipeline_graphics(self.scene.sky_pipeline.clone())
            .expect("bind sky pipeline");

        let sky_buf = Buffer::from_data(
            self.scene.memory_allocator.clone(),
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
        let sky_set_layout = self.scene.sky_pipeline.layout().set_layouts()[0].clone();
        let sky_set = DescriptorSet::new(
            self.scene.descriptor_set_allocator.clone(),
            sky_set_layout,
            [
                WriteDescriptorSet::buffer(0, sky_buf),
                WriteDescriptorSet::image_view_sampler(
                    1,
                    self.scene.cloud_a_view.clone(),
                    self.scene.mesh_sampler.clone(),
                ),
                WriteDescriptorSet::image_view_sampler(
                    2,
                    self.scene.cloud_b_view.clone(),
                    self.scene.mesh_sampler.clone(),
                ),
            ],
            [],
        )
        .expect("sky descriptor set");
        let sky_index_count = self.scene.sky_dome_indices.len() as u32;
        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.scene.sky_pipeline.layout().clone(),
                0,
                sky_set,
            )
            .expect("bind sky descriptor sets")
            .bind_vertex_buffers(0, self.scene.sky_dome_vertices.clone())
            .expect("bind sky vertex buffers")
            .bind_index_buffer(self.scene.sky_dome_indices.clone())
            .expect("bind sky index buffer");
        unsafe {
            builder
                .draw_indexed(sky_index_count, 1, 0, 0, 0)
                .expect("draw sky");
        }

        // ---- 3D scene ----
        builder
            .bind_pipeline_graphics(self.scene.mesh_pipeline.clone())
            .expect("bind mesh pipeline");

        let draw = |builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
                    vertices: Subbuffer<[Vertex3d]>,
                    indices: Subbuffer<[u32]>,
                    texture: Arc<ImageView>,
                    model: Mat4| {
            let index_count = indices.len() as u32;
            let mvp = self.scene.mvp_buffer(
                model,
                view,
                proj,
                &lights,
                wet_fac,
                fog_color,
                headlight_pos,
                headlight_dir,
                traffic_head_pos,
                traffic_head_dir,
                traffic_head_state,
            );
            let set_layout = self.scene.mesh_pipeline.layout().set_layouts()[0].clone();
            let set = DescriptorSet::new(
                self.scene.descriptor_set_allocator.clone(),
                set_layout,
                [
                    WriteDescriptorSet::buffer(0, mvp.clone()),
                    WriteDescriptorSet::image_view_sampler(1, texture, self.scene.mesh_sampler.clone()),
                ],
                [],
            )
            .expect("descriptor set");
            builder
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.scene.mesh_pipeline.layout().clone(),
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
                self.scene.world_texture_view.clone(),
                Mat4::IDENTITY,
            );
        }
        // player car
        draw(
            &mut builder,
            self.scene.car_vertices.clone(),
            self.scene.car_indices.clone(),
            self.scene.car_texture_view.clone(),
            Mat4::from_scale_rotation_translation(
                Vec3::ONE,
                glam::Quat::from_rotation_y(-game.vehicle.heading),
                Vec3::new(game.player_world_x(), 0.03, game.player_world_z()),
            ),
        );
        // traffic
        for (idx, t) in game.traffic.iter().enumerate() {
            let tvx = crate::road::road_curve(t.distance) + t.lane;
            let traffic_rot = traffic_rotation(t.lane, t.distance);
            let (traffic_vertices, traffic_indices, _anchors) =
                &self.scene.traffic_meshes[idx % self.scene.traffic_meshes.len()];
            draw(
                &mut builder,
                traffic_vertices.clone(),
                traffic_indices.clone(),
                self.scene.car_texture_view.clone(),
                Mat4::from_scale_rotation_translation(
                    Vec3::ONE,
                    traffic_rot,
                    Vec3::new(tvx, 0.35, -t.distance),
                ),
            );
        }

        // ---- Particles (rain + night taillights + drift dust) ----
        // Additive, depth-tested (no depth write) so particles fall over the
        // road but behind cars, and fade into the sky fog like everything else.
        if !dust_verts.is_empty() {
            self.scene.draw_particles(
                &mut builder,
                &self.scene.dust_pipeline,
                &dust_verts,
                view,
                proj,
                &lights,
                wet_fac,
                fog_color,
                headlight_pos,
                headlight_dir,
                traffic_head_pos,
                traffic_head_dir,
                traffic_head_state,
            );
        }
        if !particle_verts.is_empty() {
            self.scene.draw_particles(
                &mut builder,
                &self.scene.particle_pipeline,
                &particle_verts,
                view,
                proj,
                &lights,
                wet_fac,
                fog_color,
                headlight_pos,
                headlight_dir,
                traffic_head_pos,
                traffic_head_dir,
                traffic_head_state,
            );
        }

        // ---- Sun lens flare ----
        // Quads are baked into the CPU Frame (NDC positions, fan layout,
        // intensity); we only upload and draw them here.
        if !flare_verts.is_empty() {
            let flare_set_layout = self.scene.flare_pipeline.layout().set_layouts()[0].clone();
            let flare_set = DescriptorSet::new(
                self.scene.descriptor_set_allocator.clone(),
                flare_set_layout,
                [
                    WriteDescriptorSet::image_view_sampler(
                        0,
                        self.scene.flare_core_view.clone(),
                        self.scene.flare_sampler.clone(),
                    ),
                    WriteDescriptorSet::image_view_sampler(
                        1,
                        self.scene.flare_streak_view.clone(),
                        self.scene.flare_sampler.clone(),
                    ),
                    WriteDescriptorSet::image_view_sampler(
                        2,
                        self.scene.flare_ring_view.clone(),
                        self.scene.flare_sampler.clone(),
                    ),
                ],
                [],
            )
            .expect("flare descriptor set");
            let flare_buf = Buffer::from_iter(
                self.scene.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::VERTEX_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                flare_verts.iter().copied(),
            )
            .expect("flare buffer");
            let flare_vertex_count = flare_buf.len() as u32;
            builder
                .bind_pipeline_graphics(self.scene.flare_pipeline.clone())
                .expect("bind flare pipeline")
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.scene.flare_pipeline.layout().clone(),
                    0,
                    flare_set,
                )
                .expect("bind flare descriptor sets")
                .bind_vertex_buffers(0, flare_buf)
                .expect("bind flare vertex buffers");
            unsafe {
                builder
                    .draw(flare_vertex_count, 1, 0, 0)
                    .expect("draw flare");
            }
        }

        // ---- HUD ----
        builder
            .bind_pipeline_graphics(self.scene.hud_pipeline.clone())
            .expect("bind hud pipeline")
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.scene.hud_pipeline.layout().clone(),
                0,
                self.scene.hud_descriptor_set.clone(),
            )
            .expect("bind hud descriptor set");
        let hud_vertex_count = hud_verts.len() as u32;
        let hud_buf = Buffer::from_iter(
            self.scene.memory_allocator.clone(),
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
