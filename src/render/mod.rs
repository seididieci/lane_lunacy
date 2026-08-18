// SPDX-License-Identifier: MIT
use std::path::PathBuf;
use std::sync::Arc;

use winit::window::Window;

use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyImageToBufferInfo,
};
use vulkano::device::{Device, Queue};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage, SampleCount};
use vulkano::memory::allocator::{
    AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator,
};
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
use crate::render::puddle_mask::PuddleMaskResources;
use crate::render::record::record_frame_posted;
use crate::render::reflection::{reflected_view, ReflectionResources, REFLECTION_PLANE_Y};
use crate::render::scene::SceneResources;
use crate::shaders::{
    PostSettings, POST_BLOOM, POST_CHROMA, POST_DEBUG_MASK, POST_DEBUG_PLANAR, POST_DEBUG_REFLTEX,
    POST_FXAA, POST_GRAIN, POST_RAINDROPS, REFLECT_OFF, REFLECT_PLANAR, POST_REFLECT,
    POST_SATURATION, POST_VIGNETTE,
};
use crate::vertex::HudVertex;

pub mod camera;
pub mod chunk_cache;
pub mod cloud;
pub mod daynight;
pub mod drive;
pub mod flare;
pub mod frame;
pub mod frame_builder;
pub mod particles;
pub mod pipeline;
pub mod post;
pub mod probe;
pub mod puddle_mask;
pub mod record;
pub mod reflection;
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
    /// Wet-lens rain droplets (visible only when the weather is wet).
    pub rain_fx: bool,
    /// Puddle-reflection quality uniform:
    /// 0 = off, 1 = low, 2 = medium, 3 = high.
    pub puddle_quality: f32,
}

#[derive(Clone, Copy, Debug)]
struct PuddleQualityProfile {
    enabled: bool,
    reflection_scale_div: u32,
    reflection_cadence: u32,
}

fn puddle_quality_profile(q: f32) -> PuddleQualityProfile {
    if q < 0.5 {
        PuddleQualityProfile {
            enabled: false,
            reflection_scale_div: 1,
            reflection_cadence: 1,
        }
    } else if q < 1.5 {
        // LOW: quarter-res reflection, updated every 2 frames.
        PuddleQualityProfile {
            enabled: true,
            reflection_scale_div: 4,
            reflection_cadence: 2,
        }
    } else if q < 2.5 {
        // MED: half-res reflection, updated every frame.
        PuddleQualityProfile {
            enabled: true,
            reflection_scale_div: 2,
            reflection_cadence: 1,
        }
    } else {
        // HIGH: full-res reflection, updated every frame.
        PuddleQualityProfile {
            enabled: true,
            reflection_scale_div: 1,
            reflection_cadence: 1,
        }
    }
}

/// Temporary post-composite diagnostics selected by `LANE_DEBUG_POST`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DebugPostMode {
    None,
    Mask,
    Planar,
    ReflTex,
}

/// Reads the `LANE_DEBUG_POST` env var (cached) into a [`DebugPostMode`].
pub fn debug_post_mode() -> DebugPostMode {
    use std::sync::OnceLock;
    static MODE: OnceLock<DebugPostMode> = OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("LANE_DEBUG_POST").as_deref() {
        Ok("mask") => DebugPostMode::Mask,
        Ok("planar") => DebugPostMode::Planar,
        Ok("refltex") => DebugPostMode::ReflTex,
        _ => DebugPostMode::None,
    })
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
    swapchain_images: Vec<Arc<Image>>,
    swapchain_can_transfer_src: bool,
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
    /// Single-sampled depth target: the depth resolve attachment under MSAA, so
    /// the composite pass can sample a readable depth buffer for the puddle
    /// reflections. Under 1x the composite samples `depth_view` directly.
    depth_resolve_view: Arc<ImageView>,
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
    /// Planar road-reflection pass: mirrored-camera target + pipelines.
    reflection: ReflectionResources,
    /// Dedicated puddle-mask pass: renders a stable asphalt mask texture from
    /// the main camera without relying on post depth reconstruction.
    puddle_mask: PuddleMaskResources,
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
    capture_request: Option<PathBuf>,
    reflection_scale_div: u32,
    reflection_cadence_flip: bool,
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
        present_mode: vulkano::swapchain::PresentMode,
    ) -> Self {
        let caps = physical
            .surface_capabilities(&surface, Default::default())
            .expect("surface capabilities");
        let window_size = window.inner_size();
        let composite_alpha = caps.supported_composite_alpha.into_iter().next().unwrap();
        let (image_format, _) = physical
            .surface_formats(&surface, Default::default())
            .expect("surface formats")[0];

        let supported_present_modes = physical
            .surface_present_modes(&surface, Default::default())
            .expect("surface present modes");
        let present_mode = if supported_present_modes.contains(&present_mode) {
            println!(
                "present mode: {} (supported: {})",
                present_mode_name(&present_mode),
                present_modes_names(&supported_present_modes)
            );
            present_mode
        } else {
            println!(
                "present mode {} not supported (supported: {}) — falling back to FIFO",
                present_mode_name(&present_mode),
                present_modes_names(&supported_present_modes)
            );
            vulkano::swapchain::PresentMode::Fifo
        };

        // Mailbox only beats Fifo once the presentation engine never blocks
        // acquisition: with 2 images it behaves exactly like Fifo. Ask for at
        // least 3 images so there is always a free one to acquire and render.
        let min_image_count = if present_mode == vulkano::swapchain::PresentMode::Mailbox {
            caps.min_image_count
                .max(3)
                .min(caps.max_image_count.unwrap_or(u32::MAX))
        } else {
            caps.min_image_count
        };

        let mut swapchain_usage = vulkano::image::ImageUsage::COLOR_ATTACHMENT;
        if caps
            .supported_usage_flags
            .intersects(vulkano::image::ImageUsage::TRANSFER_SRC)
        {
            swapchain_usage |= vulkano::image::ImageUsage::TRANSFER_SRC;
        }

        let (swapchain, images) = Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count,
                image_format,
                image_extent: [window_size.width, window_size.height],
                image_usage: swapchain_usage,
                composite_alpha,
                present_mode,
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
        let depth_resolve_view = create_depth_resolve_view(&scene.memory_allocator, extent);
        let scene_framebuffer = create_scene_framebuffer(
            &scene_render_pass,
            &offscreen_view,
            &depth_view,
            msaa_color_view.as_ref(),
            Some(&depth_resolve_view),
        );
        let post = PostResources::new(
            &device,
            swapchain.image_format(),
            &scene.memory_allocator,
            extent,
        );
        let reflection_scale_div = 1;
        let reflection = ReflectionResources::new(
            &device,
            &scene.memory_allocator,
            extent,
            reflection_scale_div,
        );
        let puddle_mask = PuddleMaskResources::new(&device, &scene.memory_allocator, extent);
        let post_framebuffers = create_post_framebuffers(&post.pass, &images);
        let hud_framebuffers = create_post_framebuffers(&post.hud_pass, &images);
        let frame_count = post_framebuffers.len();

        Renderer {
            device,
            queue,
            window,
            swapchain,
            swapchain_images: images,
            swapchain_can_transfer_src: swapchain_usage
                .intersects(vulkano::image::ImageUsage::TRANSFER_SRC),
            scene_render_pass,
            offscreen_view,
            msaa_color_view,
            depth_view,
            depth_resolve_view,
            scene_framebuffer,
            post_framebuffers,
            hud_framebuffers,
            scene,
            post,
            reflection,
            puddle_mask,
            frame_builder: FrameBuilder::new(),
            font_atlas: font_atlas.clone(),
            seed,
            aa_samples,
            post_clock: 0.0,
            viewport,
            recreate: false,
            fences: vec![None; frame_count],
            previous_fence_i: 0,
            capture_request: None,
            reflection_scale_div,
            reflection_cadence_flip: false,
        }
    }

    /// Request a one-shot PNG capture from the windowed renderer path.
    pub fn request_window_capture(&mut self, path: PathBuf) {
        self.capture_request = Some(path);
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
        self.depth_resolve_view = create_depth_resolve_view(&self.scene.memory_allocator, extent);
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
            Some(&self.depth_resolve_view),
        );
    }

    /// Changes the terrain ribbon density. The cached world chunks are
    /// invalidated and rebuilt at the new detail on the next frame. Cheap when
    /// `detail` already equals the current setting.
    pub fn set_terrain_detail(&mut self, detail: crate::mesh::TerrainDetail) {
        if self.frame_builder.set_terrain_detail(detail) {
            self.wait_idle();
        }
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
        self.swapchain_images = new_images.clone();
        let extent = self.swapchain.image_extent();
        let aa = self.aa_samples;
        self.offscreen_view = create_offscreen_view(&self.scene.memory_allocator, extent);
        self.msaa_color_view = create_msaa_color_view(&self.scene.memory_allocator, extent, aa);
        self.depth_view = create_depth_view(&self.scene.memory_allocator, extent, aa);
        self.depth_resolve_view = create_depth_resolve_view(&self.scene.memory_allocator, extent);
        self.scene_framebuffer = create_scene_framebuffer(
            &self.scene_render_pass,
            &self.offscreen_view,
            &self.depth_view,
            self.msaa_color_view.as_ref(),
            Some(&self.depth_resolve_view),
        );
        self.post_framebuffers = create_post_framebuffers(&self.post.pass, &new_images);
        self.hud_framebuffers = create_post_framebuffers(&self.post.hud_pass, &new_images);
        self.post
            .create_bloom_images(&self.scene.memory_allocator, extent);
        self.reflection
            .resize(&self.scene.memory_allocator, extent, self.reflection_scale_div);
        self.puddle_mask
            .resize(&self.scene.memory_allocator, extent);
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
        timings: &mut crate::profiler::FrameTimings,
    ) -> bool {
        let render_started = std::time::Instant::now();
        if self.recreate {
            self.recreate_swapchain();
        }

        let acquire_started = std::time::Instant::now();
        let (image_i, suboptimal, acquire_future) =
            match acquire_next_image(self.swapchain.clone(), None) {
                Ok(r) => r,
                Err(Validated::Error(VulkanError::OutOfDate)) => {
                    self.recreate = true;
                    return false;
                }
                Err(e) => panic!("failed to acquire next image: {:?}", e),
            };
        if suboptimal {
            self.recreate = true;
        }
        timings.acquire_ms = acquire_started.elapsed().as_secs_f32() * 1000.0;

        let fence_started = std::time::Instant::now();
        if let Some(fence) = &self.fences[image_i as usize] {
            fence.wait(None).unwrap();
        }
        timings.fence_ms = fence_started.elapsed().as_secs_f32() * 1000.0;
        timings.gpu_wait_ms = timings.acquire_ms + timings.fence_ms;

        let aspect = self.viewport.extent[0] / self.viewport.extent[1];

        // Everything per-frame is computed on the CPU into a pure `Frame`:
        // camera, day/night lights, sky uniform, headlight projectors,
        // particles, and flare. The same builder drives the headless snapshot
        // path, so windowed and offline renders are pixel-identical.
        let frame_started = std::time::Instant::now();
        let frame = self
            .frame_builder
            .build(&self.scene, game, dt, aspect, hud_verts.to_vec());
        timings.frame_ms = frame_started.elapsed().as_secs_f32() * 1000.0;
        let ws = self.frame_builder.world_stats();
        timings.rebuild_ms = ws.last_rebuild_ms;
        timings.chunks_rebuilt = ws.chunks_rebuilt;

        let extent = self.viewport.extent;
        let puddle_profile = puddle_quality_profile(fx.puddle_quality);
        if puddle_profile.enabled && self.reflection_scale_div != puddle_profile.reflection_scale_div {
            self.wait_idle();
            self.reflection.resize(
                &self.scene.memory_allocator,
                [extent[0] as u32, extent[1] as u32],
                puddle_profile.reflection_scale_div,
            );
            self.reflection_scale_div = puddle_profile.reflection_scale_div;
        }
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
        if fx.rain_fx {
            flags |= POST_RAINDROPS;
        }
        if puddle_profile.enabled {
            flags |= POST_REFLECT;
        }
        // Temporary diagnostics: LANE_DEBUG_POST=mask|planar visualizes the
        // puddle mask or the planar reflection sample in the composite.
        match crate::render::debug_post_mode() {
            crate::render::DebugPostMode::Mask => flags |= POST_DEBUG_MASK,
            crate::render::DebugPostMode::Planar => flags |= POST_DEBUG_PLANAR,
            crate::render::DebugPostMode::ReflTex => flags |= POST_DEBUG_REFLTEX,
            crate::render::DebugPostMode::None => {}
        }
        // The composite samples a readable 1x depth buffer: `depth_resolve_view`
        // under MSAA (resolved in-subpass), the plain depth attachment at 1x.
        let post_depth_view = if self.aa_samples == SampleCount::Sample1 {
            self.depth_view.clone()
        } else {
            self.depth_resolve_view.clone()
        };
        let view_proj = frame.uniforms.proj * frame.uniforms.view;
        // Planar reflections mirror the camera across the road plane; the
        // composite projects a road point through this to sample the target.
        let reflection_method = if puddle_profile.enabled {
            REFLECT_PLANAR
        } else {
            REFLECT_OFF
        };
        let should_record_reflection = if puddle_profile.enabled && frame.uniforms.wet_fac > 0.001 {
            if puddle_profile.reflection_cadence <= 1 {
                self.reflection_cadence_flip = false;
                true
            } else {
                self.reflection_cadence_flip = !self.reflection_cadence_flip;
                self.reflection_cadence_flip
            }
        } else {
            self.reflection_cadence_flip = false;
            false
        };
        let planar_view_proj = frame.uniforms.proj * reflected_view(frame.uniforms.view, REFLECTION_PLANE_Y);
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
            wet_fac: frame.uniforms.wet_fac,
            puddle_quality: fx.puddle_quality,
            reflection_method,
            planar_plane_y: REFLECTION_PLANE_Y,
            _pad: [0.0; 2],
            inv_view_proj: view_proj.inverse().to_cols_array_2d(),
            view_proj: view_proj.to_cols_array_2d(),
            planar_view_proj: planar_view_proj.to_cols_array_2d(),
            eye: [
                frame.uniforms.eye.x,
                frame.uniforms.eye.y,
                frame.uniforms.eye.z,
                1.0,
            ],
            fog_color: frame.uniforms.fog_color,
        };
        self.post_clock += dt.as_secs_f32();

        let record_started = std::time::Instant::now();
        let command_buffer = record_frame_posted(
            &self.scene,
            &self.post,
            &self.reflection,
            &self.puddle_mask,
            should_record_reflection,
            game,
            &frame,
            self.frame_builder.world_chunks(),
            self.scene_framebuffer.clone(),
            self.post_framebuffers[image_i as usize].clone(),
            self.hud_framebuffers[image_i as usize].clone(),
            &self.post.bloom_fbs,
            &self.viewport,
            self.offscreen_view.clone(),
            post_depth_view,
            &self.post.bloom_views,
            &post_settings,
            timings,
        );
        timings.record_ms = record_started.elapsed().as_secs_f32() * 1000.0;

        let submit_started = std::time::Instant::now();

        let capture_path = self.capture_request.take();
        let mut capture_readback: Option<Subbuffer<[u8]>> = None;
        if capture_path.is_some() && !self.swapchain_can_transfer_src {
            if let Some(path) = capture_path.as_ref() {
                println!(
                    "window capture failed ({}): swapchain image doesn't support TRANSFER_SRC",
                    path.display()
                );
            }
            timings.submit_ms = submit_started.elapsed().as_secs_f32() * 1000.0;
            timings.render_ms = render_started.elapsed().as_secs_f32() * 1000.0;
            return true;
        }
        let capture_bpp = capture_bytes_per_pixel(self.swapchain.image_format());
        if capture_path.is_some() && capture_bpp.is_none() {
            if let Some(path) = capture_path.as_ref() {
                println!(
                    "window capture failed ({}): unsupported swapchain format {:?}",
                    path.display(),
                    self.swapchain.image_format()
                );
            }
            timings.submit_ms = submit_started.elapsed().as_secs_f32() * 1000.0;
            timings.render_ms = render_started.elapsed().as_secs_f32() * 1000.0;
            return true;
        }
        let capture_copy_cb = if capture_path.is_some() {
            let extent = self.swapchain.image_extent();
            let pixel_count = (extent[0] as u64) * (extent[1] as u64);
            let bpp = capture_bpp.expect("capture format checked");
            let readback = Buffer::new_slice::<u8>(
                self.scene.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::HOST_RANDOM_ACCESS,
                    ..Default::default()
                },
                pixel_count * bpp,
            )
            .expect("window capture readback");
            let mut copy_builder = AutoCommandBufferBuilder::primary(
                self.scene.command_allocator.clone(),
                self.scene.queue_family_index,
                CommandBufferUsage::OneTimeSubmit,
            )
            .expect("window capture builder");
            copy_builder
                .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
                    self.swapchain_images[image_i as usize].clone(),
                    readback.clone(),
                ))
                .expect("copy swapchain image for capture");
            capture_readback = Some(readback);
            Some(copy_builder.build().expect("build window capture command buffer"))
        } else {
            None
        };
        let previous_future: Box<dyn GpuFuture> =
            match self.fences[self.previous_fence_i as usize].clone() {
                Some(f) => f.boxed(),
                None => {
                    let mut now = sync::now(self.device.clone());
                    now.cleanup_finished();
                    now.boxed()
                }
            };

        let render_future = previous_future
            .join(acquire_future)
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_semaphore_and_flush()
            .expect("flush render before capture/present")
            .boxed();

        let after_copy: Box<dyn GpuFuture> = if let Some(copy_cb) = capture_copy_cb {
            render_future
                .then_execute(self.queue.clone(), copy_cb)
                .unwrap()
                .boxed()
        } else {
            render_future
        };

        let future = after_copy
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

        let capture_done = if let (Some(path), Some(readback)) = (capture_path, capture_readback) {
            if let Some(fence) = &self.fences[image_i as usize] {
                let saved = fence
                    .wait(None)
                    .map_err(|e| format!("capture wait failed: {e}"))
                    .and_then(|_| {
                        write_capture_png(
                            &readback,
                            self.swapchain.image_format(),
                            self.swapchain.image_extent(),
                            &path,
                        )
                    });
                match saved {
                    Ok(()) => println!("wrote window capture: {}", path.display()),
                    Err(e) => println!("window capture failed ({}): {e}", path.display()),
                }
            }
            true
        } else {
            false
        };

        timings.submit_ms = submit_started.elapsed().as_secs_f32() * 1000.0;
        timings.render_ms = render_started.elapsed().as_secs_f32() * 1000.0;
        capture_done
    }
}

fn write_capture_png(
    readback: &Subbuffer<[u8]>,
    format: Format,
    extent: [u32; 2],
    path: &std::path::Path,
) -> Result<(), String> {
    let guard = readback
        .read()
        .map_err(|e| format!("readback mapping failed: {e}"))?;
    let mut out = Vec::with_capacity((extent[0] * extent[1] * 4) as usize);
    match format {
        Format::B8G8R8A8_UNORM | Format::B8G8R8A8_SRGB => {
            for px in guard.chunks_exact(4) {
                out.extend_from_slice(&[px[2], px[1], px[0], px[3]]);
            }
        }
        Format::R8G8B8A8_UNORM | Format::R8G8B8A8_SRGB => out.extend_from_slice(&guard),
        Format::R16G16B16A16_SFLOAT => {
            for px in guard.chunks_exact(8) {
                let r = half::f16::from_bits(u16::from_le_bytes([px[0], px[1]])).to_f32();
                let g = half::f16::from_bits(u16::from_le_bytes([px[2], px[3]])).to_f32();
                let b = half::f16::from_bits(u16::from_le_bytes([px[4], px[5]])).to_f32();
                out.push(linear_to_srgb_u8(r));
                out.push(linear_to_srgb_u8(g));
                out.push(linear_to_srgb_u8(b));
                out.push(255);
            }
        }
        other => {
            return Err(format!(
                "unsupported swapchain format for capture: {other:?}"
            ));
        }
    }
    let image = image::RgbaImage::from_raw(extent[0], extent[1], out)
        .ok_or_else(|| "invalid capture image size".to_string())?;
    image
        .save_with_format(path, image::ImageFormat::Png)
        .map_err(|e| format!("png write failed: {e}"))
}

fn capture_bytes_per_pixel(format: Format) -> Option<u64> {
    match format {
        Format::B8G8R8A8_UNORM
        | Format::B8G8R8A8_SRGB
        | Format::R8G8B8A8_UNORM
        | Format::R8G8B8A8_SRGB => Some(4),
        Format::R16G16B16A16_SFLOAT => Some(8),
        _ => None,
    }
}

fn linear_to_srgb_u8(linear: f32) -> u8 {
    let c = linear.clamp(0.0, 1.0);
    let s = if c <= 0.0031308 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0).round() as u8
}

/// Lowercase human name for a swapchain present mode, for diagnostics.
fn present_mode_name(mode: &vulkano::swapchain::PresentMode) -> &'static str {
    match mode {
        vulkano::swapchain::PresentMode::Fifo => "fifo",
        vulkano::swapchain::PresentMode::Mailbox => "mailbox",
        vulkano::swapchain::PresentMode::Immediate => "immediate",
        vulkano::swapchain::PresentMode::FifoRelaxed => "relaxed",
        _ => "unknown",
    }
}

/// Comma-joined lowercased present-mode list, e.g. `fifo,mailbox`.
fn present_modes_names(modes: &[vulkano::swapchain::PresentMode]) -> String {
    modes
        .iter()
        .map(present_mode_name)
        .collect::<Vec<_>>()
        .join(",")
}

/// Builds the scene render pass. At 1x the offscreen image is the color
/// attachment and the depth attachment is sampled directly by the composite;
/// at 2x/4x the offscreen image becomes the resolve target of an MSAA color
/// attachment, and the MSAA depth is resolved in-subpass into a single-sampled
/// depth target (the only way to get a readable depth buffer: Vulkan's
/// `vkCmdResolveImage` cannot resolve depth, only the subpass
/// `depth_stencil_resolve` can).
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
                    store_op: Store,
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
                depth_resolve: {
                    format: Format::D32_SFLOAT,
                    samples: 1,
                    load_op: DontCare,
                    store_op: Store,
                },
            },
            pass: {
                color: [color],
                color_resolve: [resolve],
                depth_stencil: { depth },
                depth_stencil_resolve: { depth_resolve },
                depth_resolve_mode: SampleZero,
                stencil_resolve_mode: SampleZero,
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
                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::SAMPLED,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .expect("depth image"),
    )
    .expect("depth view")
}

/// Single-sampled depth image the composite pass samples for the puddle
/// reflections. Under MSAA it is the subpass depth resolve target; the MSAA
/// depth attachment is resolved into it in-pass, so no extra copy pass is
/// needed.
fn create_depth_resolve_view(
    allocator: &Arc<StandardMemoryAllocator>,
    extent: [u32; 2],
) -> Arc<ImageView> {
    ImageView::new_default(
        Image::new(
            allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::D32_SFLOAT,
                extent: [extent[0], extent[1], 1],
                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT
                    | ImageUsage::SAMPLED
                    | ImageUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .expect("depth resolve image"),
    )
    .expect("depth resolve view")
}

/// Scene framebuffer. Attachment order must match the render pass: at 1x
/// `[color, depth]`, under MSAA `[color, resolve, depth, depth_resolve]`.
fn create_scene_framebuffer(
    render_pass: &Arc<RenderPass>,
    offscreen: &Arc<ImageView>,
    depth: &Arc<ImageView>,
    msaa_color: Option<&Arc<ImageView>>,
    depth_resolve: Option<&Arc<ImageView>>,
) -> Arc<Framebuffer> {
    let mut attachments = Vec::new();
    match msaa_color {
        Some(color) => {
            attachments.push(color.clone());
            attachments.push(offscreen.clone());
            attachments.push(depth.clone());
            if let Some(resolve) = depth_resolve {
                attachments.push(resolve.clone());
            }
        }
        None => {
            attachments.push(offscreen.clone());
            attachments.push(depth.clone());
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The MSAA depth resolve (subpass `depth_stencil_resolve`) is the only way
    /// to get a readable single-sampled depth buffer for the SSR pass, so the
    /// render pass + resolved depth target must build on the local device at
    /// every supported sample count. Runs the real headless device creation.
    #[test]
    fn msaa_scene_render_pass_resolves_depth() {
        let instance = crate::create_headless_instance();
        let devices = crate::gpu::enumerate_devices(&instance);
        let physical = crate::gpu::select_physical_device(&devices, 0);
        let (device, _) = crate::gpu::create_graphics_context_headless(&physical);
        for aa in [SampleCount::Sample2, SampleCount::Sample4] {
            let render_pass = build_scene_render_pass(&device, aa);
            let allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
            let offscreen = create_offscreen_view(&allocator, [64, 64]);
            let msaa_color = create_msaa_color_view(&allocator, [64, 64], aa)
                .expect("msaa color view must exist under MSAA");
            let depth = create_depth_view(&allocator, [64, 64], aa);
            let depth_resolve = create_depth_resolve_view(&allocator, [64, 64]);
            let framebuffer = create_scene_framebuffer(
                &render_pass,
                &offscreen,
                &depth,
                Some(&msaa_color),
                Some(&depth_resolve),
            );
            assert_eq!(
                framebuffer.attachments().len(),
                4,
                "MSAA framebuffer must carry color, resolve, depth and depth_resolve"
            );
        }
    }
}
