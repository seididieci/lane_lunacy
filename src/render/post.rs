// SPDX-License-Identifier: MIT

//! Post-processing stage for the windowed renderer.
//!
//! The windowed flow renders the scene into an offscreen HDR color target, runs
//! the post-processing composite into the swapchain, and (when BLOOM is on)
//! downsamples the offscreen image through a half/quarter/eighth chain whose
//! bottom level is added back in the composite. The headless snapshot path does
//! not run this stage, so the golden baselines stay deterministic.

use std::sync::Arc;

use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::image::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage, SampleCount};
use vulkano::memory::allocator::{AllocationCreateInfo, StandardMemoryAllocator};
use vulkano::pipeline::graphics::rasterization::CullMode;
use vulkano::pipeline::graphics::vertex_input::VertexInputState;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};

use crate::render::pipeline::{graphics_pipeline, load_stages, Blend, Depth, PipelineSpec};
use crate::shaders;

/// Bloom downsample levels: 1/2, 1/4, 1/8.
const BLOOM_LEVELS: usize = 3;

/// Everything the composite + bloom stages own. Passes and pipelines are bound
/// to the device and swapchain format (persist across resizes); the bloom
/// images/framebuffers are extent-dependent and rebuilt via
/// [`PostResources::create_bloom_images`].
pub struct PostResources {
    /// Composite render pass: one color attachment in the swapchain format.
    pub pass: Arc<RenderPass>,
    /// Fullscreen-triangle composite pipeline (`post.frag`).
    pub pipeline: Arc<GraphicsPipeline>,
    /// Bloom downsample render pass: one R16G16B16A16_SFLOAT attachment.
    pub bloom_pass: Arc<RenderPass>,
    /// Fullscreen-triangle downsample pipeline (`bloom.frag`).
    pub bloom_pipeline: Arc<GraphicsPipeline>,
    /// Linear/clamped sampler shared by the post and bloom passes.
    pub sampler: Arc<Sampler>,
    /// Bloom chain images, level 0 = half res down to level 2 = eighth res.
    pub bloom_views: Vec<Arc<ImageView>>,
    /// One framebuffer per bloom level.
    pub bloom_fbs: Vec<Arc<Framebuffer>>,
}

impl PostResources {
    pub fn new(
        device: &Arc<Device>,
        swapchain_format: Format,
        memory_allocator: &Arc<StandardMemoryAllocator>,
        extent: [u32; 2],
    ) -> Self {
        let pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    format: swapchain_format,
                    samples: 1,
                    load_op: DontCare,
                    store_op: Store,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {},
            }
        )
        .expect("post render pass");
        let subpass = Subpass::from(pass.clone(), 0).unwrap();
        let stages = load_stages(device, shaders::POST_VERT_SPV, shaders::POST_FRAG_SPV);
        let pipeline = graphics_pipeline(
            device,
            &subpass,
            PipelineSpec {
                label: "post pipeline",
                cull_mode: CullMode::None,
                depth: Depth::None,
                blend: Blend::Opaque,
            },
            stages.stages,
            VertexInputState::new(),
            stages.layout,
            SampleCount::Sample1,
        );

        let bloom_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    format: Format::R16G16B16A16_SFLOAT,
                    samples: 1,
                    load_op: DontCare,
                    store_op: Store,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {},
            }
        )
        .expect("bloom render pass");
        let bloom_subpass = Subpass::from(bloom_pass.clone(), 0).unwrap();
        let bloom_stages = load_stages(device, shaders::POST_VERT_SPV, shaders::BLOOM_FRAG_SPV);
        let bloom_pipeline = graphics_pipeline(
            device,
            &bloom_subpass,
            PipelineSpec {
                label: "bloom downsample pipeline",
                cull_mode: CullMode::None,
                depth: Depth::None,
                blend: Blend::Opaque,
            },
            bloom_stages.stages,
            VertexInputState::new(),
            bloom_stages.layout,
            SampleCount::Sample1,
        );

        let sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                lod: 0.0..=0.0,
                ..Default::default()
            },
        )
        .expect("post sampler");

        let mut post = PostResources {
            pass,
            pipeline,
            bloom_pass,
            bloom_pipeline,
            sampler,
            bloom_views: Vec::new(),
            bloom_fbs: Vec::new(),
        };
        post.create_bloom_images(memory_allocator, extent);
        post
    }

    /// Rebuilds the extent-dependent bloom images and framebuffers (window
    /// resize). Passes and pipelines are unaffected.
    pub fn create_bloom_images(
        &mut self,
        memory_allocator: &Arc<StandardMemoryAllocator>,
        extent: [u32; 2],
    ) {
        let mut views = Vec::with_capacity(BLOOM_LEVELS);
        let mut fbs = Vec::with_capacity(BLOOM_LEVELS);
        for level in 0..BLOOM_LEVELS {
            let w = (extent[0] >> (level + 1)).max(1);
            let h = (extent[1] >> (level + 1)).max(1);
            let image = Image::new(
                memory_allocator.clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: Format::R16G16B16A16_SFLOAT,
                    extent: [w, h, 1],
                    usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .expect("bloom image");
            let view = ImageView::new_default(image).expect("bloom view");
            let fb = Framebuffer::new(
                self.bloom_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view.clone()],
                    ..Default::default()
                },
            )
            .expect("bloom framebuffer");
            views.push(view);
            fbs.push(fb);
        }
        self.bloom_views = views;
        self.bloom_fbs = fbs;
    }
}
