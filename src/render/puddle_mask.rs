// SPDX-License-Identifier: MIT

use std::sync::Arc;

use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::image::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage, SampleCount};
use vulkano::memory::allocator::{AllocationCreateInfo, StandardMemoryAllocator};
use vulkano::pipeline::graphics::rasterization::CullMode;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};

use crate::render::pipeline::{graphics_pipeline, load_shaders, Blend, Depth, PipelineSpec};
use crate::shaders;
use crate::vertex::Vertex3d;

pub struct PuddleMaskResources {
    pub pass: Arc<RenderPass>,
    pub pipeline: Arc<GraphicsPipeline>,
    pub sampler: Arc<Sampler>,
    pub mask_view: Arc<ImageView>,
    pub depth_view: Arc<ImageView>,
    pub framebuffer: Arc<Framebuffer>,
}

impl PuddleMaskResources {
    pub fn new(
        device: &Arc<Device>,
        memory_allocator: &Arc<StandardMemoryAllocator>,
        extent: [u32; 2],
    ) -> Self {
        let pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    format: Format::R8_UNORM,
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
        .expect("puddle-mask render pass");
        let subpass = Subpass::from(pass.clone(), 0).unwrap();

        let mesh = load_shaders::<Vertex3d>(
            device,
            shaders::MESH_VERT_SPV,
            shaders::PUDDLE_MASK_FRAG_SPV,
        );
        let pipeline = graphics_pipeline(
            device,
            &subpass,
            PipelineSpec {
                label: "puddle-mask pipeline",
                cull_mode: CullMode::Back,
                depth: Depth::Test { write: true },
                blend: Blend::Opaque,
            },
            mesh.stages,
            mesh.vertex_input,
            mesh.layout,
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
        .expect("puddle-mask sampler");

        let (mask_view, depth_view, framebuffer) =
            create_puddle_mask_targets(&pass, memory_allocator, extent);

        PuddleMaskResources {
            pass,
            pipeline,
            sampler,
            mask_view,
            depth_view,
            framebuffer,
        }
    }

    pub fn resize(&mut self, memory_allocator: &Arc<StandardMemoryAllocator>, extent: [u32; 2]) {
        let (mask_view, depth_view, framebuffer) =
            create_puddle_mask_targets(&self.pass, memory_allocator, extent);
        self.mask_view = mask_view;
        self.depth_view = depth_view;
        self.framebuffer = framebuffer;
    }
}

fn create_puddle_mask_targets(
    pass: &Arc<RenderPass>,
    memory_allocator: &Arc<StandardMemoryAllocator>,
    extent: [u32; 2],
) -> (Arc<ImageView>, Arc<ImageView>, Arc<Framebuffer>) {
    let color = Image::new(
        memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R8_UNORM,
            extent: [extent[0], extent[1], 1],
            usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
    )
    .expect("puddle-mask color image");
    let depth = Image::new(
        memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::D32_SFLOAT,
            extent: [extent[0], extent[1], 1],
            usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
    )
    .expect("puddle-mask depth image");
    let color_view = ImageView::new_default(color).expect("puddle-mask color view");
    let depth_view = ImageView::new_default(depth).expect("puddle-mask depth view");
    let framebuffer = Framebuffer::new(
        pass.clone(),
        FramebufferCreateInfo {
            attachments: vec![color_view.clone(), depth_view.clone()],
            ..Default::default()
        },
    )
    .expect("puddle-mask framebuffer");
    (color_view, depth_view, framebuffer)
}
