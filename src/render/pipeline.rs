// SPDX-License-Identifier: MIT

//! Pipeline factory: removes the six hand-written `GraphicsPipeline::new`
//! blocks that used to duplicate the same stages/vertex-input/viewport/
//! multisample/dynamic-state boilerplate in `SceneResources`.

use std::marker::PhantomData;
use std::sync::Arc;

use vulkano::device::Device;
use vulkano::pipeline::graphics::color_blend::{
    AttachmentBlend, BlendFactor, BlendOp, ColorBlendAttachmentState, ColorBlendState,
};
use vulkano::pipeline::graphics::depth_stencil::{CompareOp, DepthState, DepthStencilState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{CullMode, RasterizationState};
use vulkano::pipeline::graphics::vertex_input::{
    Vertex as VertexTrait, VertexDefinition, VertexInputState,
};
use vulkano::pipeline::graphics::viewport::ViewportState;
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::Subpass;
use vulkano::shader::{ShaderModule, ShaderModuleCreateInfo};

use crate::shaders;

/// Loads a vertex+fragment shader pair and derives the vertex-input state and
/// descriptor-set layout for `V`, so the caller can drop straight into
/// [`graphics_pipeline`].
pub struct ShaderStages<V: VertexTrait> {
    pub stages: Vec<PipelineShaderStageCreateInfo>,
    pub vertex_input: VertexInputState,
    pub layout: Arc<PipelineLayout>,
    pub _vertex: PhantomData<V>,
}

/// Loads `vs`/`fs` (SPIR-V byte arrays) into a shader pair typed by the vertex
/// format `V`.
pub fn load_shaders<V: VertexTrait>(
    device: &Arc<Device>,
    vs: &'static [u8],
    fs: &'static [u8],
) -> ShaderStages<V> {
    let vs = unsafe {
        ShaderModule::new(
            device.clone(),
            ShaderModuleCreateInfo::new(&shaders::spv_words(vs)),
        )
    }
    .expect("vertex shader module");
    let fs = unsafe {
        ShaderModule::new(
            device.clone(),
            ShaderModuleCreateInfo::new(&shaders::spv_words(fs)),
        )
    }
    .expect("fragment shader module");
    let vs_ep = vs.entry_point("main").unwrap();
    let fs_ep = fs.entry_point("main").unwrap();
    let vertex_input = V::per_vertex().definition(&vs_ep).unwrap();
    let stages = vec![
        PipelineShaderStageCreateInfo::new(vs_ep),
        PipelineShaderStageCreateInfo::new(fs_ep),
    ];
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )
    .unwrap();
    ShaderStages {
        stages: stages.into(),
        vertex_input,
        layout,
        _vertex: PhantomData,
    }
}

/// Depth behaviour for a pipeline.
#[derive(Clone, Copy)]
pub enum Depth {
    /// No depth test (2D overlays such as HUD, sky, flare).
    None,
    /// Depth test enabled with `Less`, optionally writing depth (3D geometry).
    Test { write: bool },
}

/// Color blend behaviour for the (single) color attachment.
#[derive(Clone, Copy)]
pub enum Blend {
    /// Opaque: no blending.
    Opaque,
    /// Standard source-alpha-over blending (HUD, dust).
    Alpha,
    /// Additive (`SrcAlpha`, `One`) — rain, lights, flare.
    Additive,
}

/// Everything that distinguishes one of the scene's graphics pipelines.
#[derive(Clone, Copy)]
pub struct PipelineSpec {
    /// Human-readable label used in the pipeline `expect`.
    pub label: &'static str,
    /// Back-face culling mode.
    pub cull_mode: CullMode,
    /// Depth test/write behaviour.
    pub depth: Depth,
    /// Color blending.
    pub blend: Blend,
}

/// Builds a `GraphicsPipeline` from a spec plus the shader stages, vertex input
/// and layout produced by [`load_shaders`]. Every pipeline shares the same
/// viewport dynamic state, single-sample multisampling and triangle lists.
pub fn graphics_pipeline(
    device: &Arc<Device>,
    subpass: &Subpass,
    spec: PipelineSpec,
    stages: Vec<PipelineShaderStageCreateInfo>,
    vertex_input: VertexInputState,
    layout: Arc<PipelineLayout>,
) -> Arc<GraphicsPipeline> {
    let depth_stencil = match spec.depth {
        Depth::None => DepthStencilState {
            depth: None,
            ..Default::default()
        },
        Depth::Test { write } => DepthStencilState {
            depth: Some(DepthState {
                write_enable: write,
                compare_op: CompareOp::Less,
            }),
            ..Default::default()
        },
    };
    let blend_attachment = match spec.blend {
        Blend::Opaque => ColorBlendAttachmentState::default(),
        Blend::Alpha => ColorBlendAttachmentState {
            blend: Some(AttachmentBlend::alpha()),
            ..Default::default()
        },
        Blend::Additive => ColorBlendAttachmentState {
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
    };

    GraphicsPipeline::new(
        device.clone(),
        None,
        GraphicsPipelineCreateInfo {
            stages: stages.into(),
            vertex_input_state: Some(vertex_input),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState::default()),
            rasterization_state: Some(RasterizationState {
                cull_mode: spec.cull_mode,
                ..Default::default()
            }),
            multisample_state: Some(MultisampleState::default()),
            depth_stencil_state: Some(depth_stencil),
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                subpass.num_color_attachments(),
                blend_attachment,
            )),
            dynamic_state: [DynamicState::Viewport].into_iter().collect(),
            subpass: Some(subpass.clone().into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .expect(spec.label)
}
