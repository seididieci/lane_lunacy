// SPDX-License-Identifier: MIT
use std::sync::Arc;

use winit::window::Window;

use vulkano::device::{Device, Queue};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType};
use vulkano::memory::allocator::AllocationCreateInfo;
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::render_pass::{Framebuffer, RenderPass};
use vulkano::swapchain::{
    acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
};
use vulkano::sync::future::FenceSignalFuture;
use vulkano::sync::{self, GpuFuture};
use vulkano::{Validated, VulkanError};

use crate::font::FontAtlas;
use crate::game::Game;
use crate::render::frame_builder::FrameBuilder;
use crate::render::record::record_frame;
use crate::render::scene::SceneResources;
use crate::vertex::HudVertex;

pub mod camera;
pub mod cloud;
pub mod daynight;
pub mod flare;
pub mod frame;
pub mod frame_builder;
pub mod particles;
pub mod pipeline;
pub mod probe;
pub mod record;
pub mod scene;
pub mod snapshot;
pub mod texture;

pub(crate) const WORLD_CHUNK_LEN: f32 = 260.0;
pub(crate) const WORLD_CHUNKS_BEHIND: i32 = 1;
pub(crate) const WORLD_CHUNKS_AHEAD: i32 = 6;

/// An in-flight present future, pinned per swapchain image so we can wait on
/// the oldest one before re-using the image. Stored as `Arc` because vulkano's
/// `FenceSignalFuture` implements `GpuFuture` only behind an `Arc` (see
/// `vulkano::sync::future::fence_signal`).
type FrameFence = Arc<FenceSignalFuture<Box<dyn GpuFuture>>>;

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
    /// Mutable per-frame state (particles, camera smoothing, world chunks).
    /// Math lives in `FrameBuilder`; this struct only feeds it the swapchain.
    frame_builder: FrameBuilder,
    viewport: Viewport,
    pub recreate: bool,
    fences: Vec<Option<FrameFence>>,
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
        let scene = SceneResources::new(
            device.clone(),
            queue.clone(),
            render_pass.clone(),
            font_atlas,
            seed,
        );

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
            frame_builder: FrameBuilder::new(),
            viewport,
            recreate: false,
            fences: Vec::new(),
            previous_fence_i: 0,
        };

        renderer.framebuffers = renderer.create_framebuffers(&images);
        renderer.fences = vec![None; renderer.framebuffers.len()];
        renderer
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

    // Single-threaded by design (this is the window presenter); the `Arc` is
    // mandated by vulkano's `GpuFuture` impl for `FenceSignalFuture`.
    #[allow(clippy::arc_with_non_send_sync)]
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
        let frame = self
            .frame_builder
            .build(&self.scene, game, dt, aspect, hud_verts.to_vec());

        // Everything between begin/end render pass is recorded identically for
        // the swapchain and the headless snapshot target.
        let command_buffer = record_frame(
            &self.scene,
            game,
            &frame,
            self.frame_builder.world_chunks(),
            self.framebuffers[image_i as usize].clone(),
            &self.viewport,
        );

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
