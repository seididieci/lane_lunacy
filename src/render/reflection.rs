// SPDX-License-Identifier: MIT

//! Planar road reflections: a mirrored-camera pass that renders the scene
//! (sky + world chunks + cars) into a scaled color target, plus the
//! reflection backend selector.
//!
//! The composite pass then samples that target for wet-asphalt puddles, so the
//! road reflects what a mirror laid on the asphalt would show: sky, NPCs,
//! posts and scenery — with no screen-space self-reflection on the road.
//!
//! The backend is abstracted so a future SSR / ray-tracing source can replace
//! the planar target without touching the composite: the post shader just
//! consumes a reflection texture plus a `reflection_method` selector.

use std::sync::Arc;

use glam::Mat4;

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

/// World-space height of the road plane the planar camera mirrors across. The
/// asphalt ribbon sits at ~0.015 (on top of the flat terrain), shoulders at
/// 0.021; a single plane in between keeps puddles aligned with both.
pub const REFLECTION_PLANE_Y: f32 = 0.02;
/// World-space height of the clip plane used by the reflection pass: geometry
/// strictly below it (terrain at 0, verges at 0.016, asphalt at 0.015,
/// shoulders at 0.021) is discarded so the mirrored camera, which sits *under*
/// the road, never draws the road/ground into the reflection.
pub const REFLECTION_CLIP_Y: f32 = 0.03;

/// Reflection backend selector. This is the seam a future RT/SSR backend slots
/// into; `Planar` is the only implemented method right now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReflectionMethod {
    Off,
    Planar,
}

impl ReflectionMethod {
    /// Uniform value shipped to the post shader (see `shaders::REFLECT_*`).
    pub fn uniform(self) -> f32 {
        match self {
            ReflectionMethod::Off => crate::shaders::REFLECT_OFF,
            ReflectionMethod::Planar => crate::shaders::REFLECT_PLANAR,
        }
    }
}

/// Reflection matrix across the horizontal plane `y = plane_y`. Reflects a
/// point `(x, y, z)` to `(x, 2*plane_y - y, z)`.
pub fn reflect_matrix(plane_y: f32) -> Mat4 {
    Mat4::from_cols(
        glam::Vec4::new(1.0, 0.0, 0.0, 0.0),
        glam::Vec4::new(0.0, -1.0, 0.0, 0.0),
        glam::Vec4::new(0.0, 0.0, 1.0, 0.0),
        glam::Vec4::new(0.0, 2.0 * plane_y, 0.0, 1.0),
    )
}

/// Mirrored camera view for the planar reflection pass.
pub fn reflected_view(view: Mat4, plane_y: f32) -> Mat4 {
    view * reflect_matrix(plane_y)
}

/// Everything the planar reflection pass owns. Passes and pipelines persist
/// across resizes; the color/depth images and framebuffer are extent-dependent
/// and rebuilt via [`ReflectionResources::resize`].
pub struct ReflectionResources {
    /// Reflection render pass: one RGBA16F color + one D32 depth attachment,
    /// always single-sampled (the mirrored pass runs at quality-selected scale).
    pub pass: Arc<RenderPass>,
    /// Sky dome pipeline bound to the reflection pass (depth off, like the
    /// scene sky pipeline).
    pub sky_pipeline: Arc<GraphicsPipeline>,
    /// World/car mesh pipeline bound to the reflection pass. Culling is
    /// disabled because mirroring the view flips the triangle winding.
    pub mesh_pipeline: Arc<GraphicsPipeline>,
    /// Linear/clamped sampler for sampling the reflection target in the
    /// composite pass.
    pub sampler: Arc<Sampler>,
    /// Reflection color target (quality-selected resolution).
    pub color_view: Arc<ImageView>,
    /// Reflection depth target (quality-selected resolution).
    pub depth_view: Arc<ImageView>,
    /// Framebuffer over the color+depth targets.
    pub framebuffer: Arc<Framebuffer>,
    /// Reflection resolution divisor vs swapchain extent (1=full, 2=half,
    /// 4=quarter).
    pub scale_div: u32,
}

impl ReflectionResources {
    pub fn new(
        device: &Arc<Device>,
        memory_allocator: &Arc<StandardMemoryAllocator>,
        extent: [u32; 2],
        scale_div: u32,
    ) -> Self {
        let pass = vulkano::single_pass_renderpass!(
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
        .expect("reflection render pass");
        let subpass = Subpass::from(pass.clone(), 0).unwrap();

        let sky = load_shaders::<Vertex3d>(device, shaders::SKY_VERT_SPV, shaders::SKY_FRAG_SPV);
        let sky_pipeline = graphics_pipeline(
            device,
            &subpass,
            PipelineSpec {
                label: "reflection sky pipeline",
                cull_mode: CullMode::None,
                depth: Depth::None,
                blend: Blend::Opaque,
            },
            sky.stages,
            sky.vertex_input,
            sky.layout,
            SampleCount::Sample1,
        );

        let mesh = load_shaders::<Vertex3d>(device, shaders::MESH_VERT_SPV, shaders::MESH_FRAG_SPV);
        let mesh_pipeline = graphics_pipeline(
            device,
            &subpass,
            PipelineSpec {
                label: "reflection mesh pipeline",
                // Mirroring flips winding; the existing cull-back setup would
                // cull everything, so the reflected pass draws double-sided.
                cull_mode: CullMode::None,
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
        .expect("reflection sampler");

        let (color_view, depth_view, framebuffer) =
            create_reflection_targets(&pass, memory_allocator, extent, scale_div);

        ReflectionResources {
            pass,
            sky_pipeline,
            mesh_pipeline,
            sampler,
            color_view,
            depth_view,
            framebuffer,
            scale_div,
        }
    }

    /// Rebuilds the extent-dependent color/depth images and framebuffer
    /// (window resize / quality change).
    pub fn resize(
        &mut self,
        memory_allocator: &Arc<StandardMemoryAllocator>,
        extent: [u32; 2],
        scale_div: u32,
    ) {
        let (color_view, depth_view, framebuffer) =
            create_reflection_targets(&self.pass, memory_allocator, extent, scale_div);
        self.color_view = color_view;
        self.depth_view = depth_view;
        self.framebuffer = framebuffer;
        self.scale_div = scale_div.max(1);
    }
}

/// Builds the color/depth targets and framebuffer for the reflection pass at a
/// quality-selected scale of the target resolution.
fn create_reflection_targets(
    pass: &Arc<RenderPass>,
    memory_allocator: &Arc<StandardMemoryAllocator>,
    extent: [u32; 2],
    scale_div: u32,
) -> (Arc<ImageView>, Arc<ImageView>, Arc<Framebuffer>) {
    let div = scale_div.max(1);
    let refl_extent = [(extent[0] / div).max(1), (extent[1] / div).max(1)];
    let color = Image::new(
        memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R16G16B16A16_SFLOAT,
            extent: [refl_extent[0], refl_extent[1], 1],
            usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
    )
    .expect("reflection color image");
    let depth = Image::new(
        memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::D32_SFLOAT,
            extent: [refl_extent[0], refl_extent[1], 1],
            usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
    )
    .expect("reflection depth image");
    let color_view = ImageView::new_default(color).expect("reflection color view");
    let depth_view = ImageView::new_default(depth).expect("reflection depth view");
    let framebuffer = Framebuffer::new(
        pass.clone(),
        FramebufferCreateInfo {
            attachments: vec![color_view.clone(), depth_view.clone()],
            ..Default::default()
        },
    )
    .expect("reflection framebuffer");
    (color_view, depth_view, framebuffer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn reflect_matrix_mirrors_points_across_the_plane() {
        let r = reflect_matrix(0.02);
        // A point above the road lands equally far below it, same x/z.
        let p = r.transform_point3(Vec3::new(3.0, 4.5, -12.0));
        assert!((p.x - 3.0).abs() < 1e-6);
        assert!((p.y - (2.0 * 0.02 - 4.5)).abs() < 1e-6);
        assert!((p.z + 12.0).abs() < 1e-6);
    }

    #[test]
    fn mirrored_camera_sees_the_reflection_at_the_same_pixel() {
        // A camera above the road and its mirror: for any view direction `d`,
        // the mirrored camera looking along the reflected direction maps to the
        // same view-space direction, so a shared projection samples the same
        // texel. That is the whole planar pixel-correspondence argument.
        let eye = Vec3::new(0.0, 4.0, -10.0);
        let forward = Vec3::new(0.0, -0.3714, 0.9285);
        let view = Mat4::look_at_rh(eye, eye + forward, Vec3::Y);
        let reflected = reflected_view(view, 0.02);

        let d = Vec3::new(0.0, -0.4472, 0.8944);
        let d_view = view.transform_vector3(d);
        let r = reflect_matrix(0.02);
        let reflected_dir = reflected.transform_vector3(r.transform_vector3(d));
        assert!((d_view - reflected_dir).length() < 1e-4);
    }

    #[test]
    fn clip_sentinel_never_discards_and_reflection_plane_clips_below_road() {
        // (0,0,0,-1) is the disabled sentinel: dot is always -1, never > 0.
        let sentinel = [0.0, 0.0, 0.0, -1.0];
        let dot = |p: Vec3| glam::Vec4::new(p.x, p.y, p.z, 1.0).dot(sentinel.into());
        assert!(dot(Vec3::new(0.0, -5.0, 0.0)) <= 0.0);

        // (0,-1,0,CLIP) discards anything under the clip height.
        let clip = [0.0, -1.0, 0.0, REFLECTION_CLIP_Y];
        let under = glam::Vec4::new(0.0, 0.0, 0.0, 1.0).dot(clip.into());
        assert!(under > 0.0, "ground at y=0 must be clipped");
        let car = glam::Vec4::new(0.0, 0.35, 0.0, 1.0).dot(clip.into());
        assert!(car <= 0.0, "traffic at y=0.35 must survive the clip");
    }

    #[test]
    fn planar_projection_of_a_visible_road_point_is_valid() {
        // Replicates the chase-camera setup from `frame.rs` and checks that a
        // road point ahead of the camera maps to a valid planar sample. The
        // mirrored camera shares the main projection, so this MUST hold for the
        // composite to pick up the reflection texel.
        use crate::render::camera::{perspective_vulkan, Camera};
        let cam_forward = Vec3::new(0.0, 0.0, -1.0); // heading 0
        let car_pos = Vec3::new(0.0, 0.0, 0.0);
        let dist = 6.0f32;
        let eye = car_pos - cam_forward * dist + Vec3::new(0.0, 4.0, 0.0);
        let look_at = car_pos + cam_forward * 4.0 + Vec3::new(0.0, 3.6, 0.0);
        let cam = Camera {
            eye,
            forward: (look_at - eye).normalize(),
        };
        let view = cam.view();
        let proj = perspective_vulkan(60.0f32.to_radians(), 16.0 / 9.0, 0.1, 600.0);

        for z in [-10.0f32, -20.0, -40.0, -80.0] {
            for x in [-3.0f32, 0.0, 3.0] {
                let p = Vec3::new(x, 0.02, z);
                let main_clip = proj * view * glam::Vec4::new(p.x, p.y, p.z, 1.0);
                assert!(main_clip.w > 0.0, "road point should be visible");
                let main_uv = main_clip.truncate() / main_clip.w * 0.5 + 0.5;
                assert!(
                    main_uv.x >= 0.0 && main_uv.x <= 1.0 && main_uv.y >= 0.0 && main_uv.y <= 1.0,
                    "road point should be on screen"
                );

                let planar_clip = proj * reflected_view(view, 0.02)
                    * glam::Vec4::new(p.x, p.y, p.z, 1.0);
                assert!(
                    planar_clip.w > 0.0,
                    "planar clip.w must be positive at ({x},{z})"
                );
                let puv = planar_clip.truncate() / planar_clip.w * 0.5 + 0.5;
                assert!(
                    puv.x >= 0.0 && puv.x <= 1.0 && puv.y >= 0.0 && puv.y <= 1.0,
                    "planar uv must be in range at ({x},{z})"
                );
                assert!(
                    (puv - main_uv).length() < 1e-3,
                    "planar uv must match the main-camera uv"
                );
            }
        }
    }
}
