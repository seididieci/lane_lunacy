// SPDX-License-Identifier: MIT
use std::sync::Arc;

use winit::window::Window;

use vulkano::device::{Device, Queue};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage, SampleCount};
use vulkano::memory::allocator::{AllocationCreateInfo, StandardMemoryAllocator};
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass};
use vulkano::swapchain::{
    acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
};
use vulkano::sync::future::FenceSignalFuture;
use vulkano::sync::{self, GpuFuture};
use vulkano::{Validated, VulkanError};

use crate::font::FontAtlas;
use crate::game::Game;
use crate::render::frame_builder::FrameBuilder;
use crate::render::post::PostResources;
use crate::render::record::record_frame_posted;
use crate::render::scene::SceneResources;
use crate::shaders::{
    PostSettings, POST_BLOOM, POST_CHROMA, POST_FXAA, POST_GRAIN, POST_SATURATION, POST_VIGNETTE,
};
use crate::vertex::HudVertex;

pub mod camera;
pub mod cloud;
pub mod daynight;
pub mod flare;
pub mod frame;
pub mod frame_builder;
pub mod particles;
pub mod pipeline;
pub mod post;
pub mod probe;
pub mod record;
pub mod scene;
pub mod snapshot;
pub mod texture;

pub(crate) const WORLD_CHUNK_LEN: f32 = 260.0;
pub(crate) const WORLD_CHUNKS_BEHIND: i32 = 1;
pub(crate) const WORLD_CHUNKS_AHEAD: i32 = 6;

/// Per-frame toggles for the post-processing composite.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FxSettings {
    pub fxaa: bool,
    pub bloom: bool,
    pub vignette: bool,
    pub grain: bool,
    pub saturation: bool,
    pub chroma: bool,
}

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
    /// Scene render pass: one offscreen color attachment (+ optional MSAA
    /// resolve) plus one depth attachment, sample count == `aa_samples`.
    scene_render_pass: Arc<RenderPass>,
    /// Offscreen HDR scene target: the color attachment at 1x, or the resolve
    /// target under MSAA. Also sampled by the bloom chain and the composite.
    offscreen_view: Arc<ImageView>,
    /// MSAA color attachment, present only when `aa_samples > 1`.
    msaa_color_view: Option<Arc<ImageView>>,
    /// Depth attachment, at `aa_samples` samples.
    depth_view: Arc<ImageView>,
    /// Single scene framebuffer (independent of the swapchain images: the
    /// resolve target is the shared offscreen image, not the swapchain).
    scene_framebuffer: Arc<Framebuffer>,
    /// One composite framebuffer per swapchain image.
    post_framebuffers: Vec<Arc<Framebuffer>>,
    /// One HUD/text framebuffer per swapchain image, bound to `post.hud_pass`
    /// (`load_op: Load`) so text composites flat over the post output.
    hud_framebuffers: Vec<Arc<Framebuffer>>,
    /// Everything shared with the headless snapshot path: pipelines, textures,
    /// samplers, models, buffers, allocators.
    scene: SceneResources,
    /// Bloom chain + composite pass/pipeline/sampler.
    post: PostResources,
    /// Mutable per-frame state (particles, camera smoothing, world chunks).
    /// Math lives in `FrameBuilder`; this struct only feeds it the swapchain.
    frame_builder: FrameBuilder,
    /// Rebuilt together with `scene` when the sample count changes.
    font_atlas: FontAtlas,
    seed: u64,
    aa_samples: SampleCount,
    /// Monotonic seconds since renderer start; drives animated post effects
    /// (film grain). Not driven by game time so pausing doesn't freeze it.
    post_clock: f32,
    viewport: Viewport,
    pub recreate: bool,
    fences: Vec<Option<FrameFence>>,
    previous_fence_i: u32,
}

impl Renderer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        surface: Arc<Surface>,
        window: Arc<Window>,
        physical: &Arc<vulkano::device::physical::PhysicalDevice>,
        font_atlas: &FontAtlas,
        seed: u64,
        aa_samples: SampleCount,
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

        let scene_render_pass = build_scene_render_pass(&device, aa_samples);

        // Every GPU resource that is independent of the swapchain/framebuffer
        // lives in `SceneResources`, shared with the headless snapshot path.
        let scene = SceneResources::new(
            device.clone(),
            queue.clone(),
            scene_render_pass.clone(),
            font_atlas,
            seed,
            aa_samples,
        );

        let extent = swapchain.image_extent();
        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: [extent[0] as f32, extent[1] as f32],
            depth_range: 0.0..=1.0,
        };

        let offscreen_view = create_offscreen_view(&scene.memory_allocator, extent);
        let msaa_color_view = create_msaa_color_view(&scene.memory_allocator, extent, aa_samples);
        let depth_view = create_depth_view(&scene.memory_allocator, extent, aa_samples);
        let scene_framebuffer = create_scene_framebuffer(
            &scene_render_pass,
            &offscreen_view,
            &depth_view,
            msaa_color_view.as_ref(),
        );
        let post = PostResources::new(
            &device,
            swapchain.image_format(),
            &scene.memory_allocator,
            extent,
        );
        let post_framebuffers = create_post_framebuffers(&post.pass, &images);
        let hud_framebuffers = create_post_framebuffers(&post.hud_pass, &images);
        let frame_count = post_framebuffers.len();

        Renderer {
            device,
            queue,
            window,
            swapchain,
            scene_render_pass,
            offscreen_view,
            msaa_color_view,
            depth_view,
            scene_framebuffer,
            post_framebuffers,
            hud_framebuffers,
            scene,
            post,
            frame_builder: FrameBuilder::new(),
            font_atlas: font_atlas.clone(),
            seed,
            aa_samples,
            post_clock: 0.0,
            viewport,
            recreate: false,
            fences: vec![None; frame_count],
            previous_fence_i: 0,
        }
    }

    /// Rebuilds everything that depends on the sample count: the scene render
    /// pass, the MSAA color attachment, the depth attachment and every scene
    /// pipeline. Cheap when `aa` already equals the current sample count.
    pub fn set_aa(&mut self, aa: SampleCount) {
        if aa == self.aa_samples {
            return;
        }
        self.wait_idle();
        self.aa_samples = aa;
        let extent = self.swapchain.image_extent();
        self.scene_render_pass = build_scene_render_pass(&self.device, aa);
        self.msaa_color_view = create_msaa_color_view(&self.scene.memory_allocator, extent, aa);
        self.depth_view = create_depth_view(&self.scene.memory_allocator, extent, aa);
        self.scene = SceneResources::new(
            self.device.clone(),
            self.queue.clone(),
            self.scene_render_pass.clone(),
            &self.font_atlas,
            self.seed,
            aa,
        );
        self.scene_framebuffer = create_scene_framebuffer(
            &self.scene_render_pass,
            &self.offscreen_view,
            &self.depth_view,
            self.msaa_color_view.as_ref(),
        );
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
        let aa = self.aa_samples;
        self.offscreen_view = create_offscreen_view(&self.scene.memory_allocator, extent);
        self.msaa_color_view = create_msaa_color_view(&self.scene.memory_allocator, extent, aa);
        self.depth_view = create_depth_view(&self.scene.memory_allocator, extent, aa);
        self.scene_framebuffer = create_scene_framebuffer(
            &self.scene_render_pass,
            &self.offscreen_view,
            &self.depth_view,
            self.msaa_color_view.as_ref(),
        );
        self.post_framebuffers = create_post_framebuffers(&self.post.pass, &new_images);
        self.hud_framebuffers = create_post_framebuffers(&self.post.hud_pass, &new_images);
        self.post
            .create_bloom_images(&self.scene.memory_allocator, extent);
        self.viewport.extent = [dims.width as f32, dims.height as f32];
        self.recreate = false;
    }

    pub(crate) fn wait_idle(&self) {
        // Safety: no further work is submitted to this device afterwards; it is
        // dropped immediately after this call.
        unsafe { self.device.wait_idle() }.expect("failed to wait for device idle");
    }

    /// Cached world-mesh volume and rebuild timing (debug HUD).
    pub fn world_stats(&self) -> frame_builder::WorldStats {
        self.frame_builder.world_stats()
    }

    // Single-threaded by design (this is the window presenter); the `Arc` is
    // mandated by vulkano's `GpuFuture` impl for `FenceSignalFuture`.
    #[allow(clippy::arc_with_non_send_sync)]
    pub fn render(
        &mut self,
        game: &Game,
        dt: std::time::Duration,
        hud_verts: &[HudVertex],
        fx: &FxSettings,
    ) {
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

        let extent = self.viewport.extent;
        let mut flags = 0u32;
        if fx.fxaa {
            flags |= POST_FXAA;
        }
        if fx.bloom {
            flags |= POST_BLOOM;
        }
        if fx.vignette {
            flags |= POST_VIGNETTE;
        }
        if fx.grain {
            flags |= POST_GRAIN;
        }
        if fx.saturation {
            flags |= POST_SATURATION;
        }
        if fx.chroma {
            flags |= POST_CHROMA;
        }
        let post_settings = PostSettings {
            flags,
            time: self.post_clock,
            vignette_strength: 0.35,
            grain_amount: 0.04,
            saturation_boost: 1.15,
            bloom_strength: 0.5,
            chroma_strength: 0.0015,
            texel_x: 1.0 / extent[0],
            texel_y: 1.0 / extent[1],
            _pad: [0.0; 3],
        };
        self.post_clock += dt.as_secs_f32();

        let command_buffer = record_frame_posted(
            &self.scene,
            &self.post,
            game,
            &frame,
            self.frame_builder.world_chunks(),
            self.scene_framebuffer.clone(),
            self.post_framebuffers[image_i as usize].clone(),
            self.hud_framebuffers[image_i as usize].clone(),
            &self.post.bloom_fbs,
            &self.viewport,
            self.offscreen_view.clone(),
            &self.post.bloom_views,
            &post_settings,
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

/// Builds the scene render pass. At 1x the offscreen image is the color
/// attachment; at 2x/4x it becomes the resolve target of an MSAA color
/// attachment, with a matching sample-count depth attachment.
fn build_scene_render_pass(device: &Arc<Device>, aa: SampleCount) -> Arc<RenderPass> {
    if aa == SampleCount::Sample1 {
        vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    format: Format::R16G16B16A16_SFLOAT,
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
                depth: {
                    format: Format::D32_SFLOAT,
                    samples: 1,
                    load_op: Clear,
                    store_op: DontCare,
                },
            },
            pass: {
                color: [color],
                depth_stencil: { depth },
            }
        )
        .expect("scene render pass")
    } else {
        let samples = aa as u32;
        vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    format: Format::R16G16B16A16_SFLOAT,
                    samples: samples,
                    load_op: Clear,
                    store_op: DontCare,
                },
                resolve: {
                    format: Format::R16G16B16A16_SFLOAT,
                    samples: 1,
                    load_op: DontCare,
                    store_op: Store,
                },
                depth: {
                    format: Format::D32_SFLOAT,
                    samples: samples,
                    load_op: Clear,
                    store_op: DontCare,
                },
            },
            pass: {
                color: [color],
                color_resolve: [resolve],
                depth_stencil: { depth },
            }
        )
        .expect("scene render pass")
    }
}

fn create_offscreen_view(
    allocator: &Arc<StandardMemoryAllocator>,
    extent: [u32; 2],
) -> Arc<ImageView> {
    ImageView::new_default(
        Image::new(
            allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R16G16B16A16_SFLOAT,
                extent: [extent[0], extent[1], 1],
                usage: ImageUsage::COLOR_ATTACHMENT
                    | ImageUsage::SAMPLED
                    | ImageUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .expect("offscreen image"),
    )
    .expect("offscreen view")
}

fn create_msaa_color_view(
    allocator: &Arc<StandardMemoryAllocator>,
    extent: [u32; 2],
    aa: SampleCount,
) -> Option<Arc<ImageView>> {
    if aa == SampleCount::Sample1 {
        return None;
    }
    Some(
        ImageView::new_default(
            Image::new(
                allocator.clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: Format::R16G16B16A16_SFLOAT,
                    extent: [extent[0], extent[1], 1],
                    samples: aa,
                    usage: ImageUsage::COLOR_ATTACHMENT,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .expect("msaa color image"),
        )
        .expect("msaa color view"),
    )
}

fn create_depth_view(
    allocator: &Arc<StandardMemoryAllocator>,
    extent: [u32; 2],
    aa: SampleCount,
) -> Arc<ImageView> {
    ImageView::new_default(
        Image::new(
            allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::D32_SFLOAT,
                extent: [extent[0], extent[1], 1],
                samples: aa,
                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .expect("depth image"),
    )
    .expect("depth view")
}

/// Scene framebuffer. Attachment order must match the render pass: at 1x
/// `[color, depth]`, under MSAA `[color, resolve, depth]`.
fn create_scene_framebuffer(
    render_pass: &Arc<RenderPass>,
    offscreen: &Arc<ImageView>,
    depth: &Arc<ImageView>,
    msaa_color: Option<&Arc<ImageView>>,
) -> Arc<Framebuffer> {
    let mut attachments = Vec::new();
    match msaa_color {
        Some(color) => {
            attachments.push(color.clone());
            attachments.push(offscreen.clone());
        }
        None => attachments.push(offscreen.clone()),
    }
    attachments.push(depth.clone());
    Framebuffer::new(
        render_pass.clone(),
        FramebufferCreateInfo {
            attachments,
            ..Default::default()
        },
    )
    .expect("scene framebuffer")
}

fn create_post_framebuffers(
    post_pass: &Arc<RenderPass>,
    images: &[Arc<Image>],
) -> Vec<Arc<Framebuffer>> {
    images
        .iter()
        .map(|img| {
            let color_view = ImageView::new_default(img.clone()).unwrap();
            Framebuffer::new(
                post_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![color_view],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect()
}
