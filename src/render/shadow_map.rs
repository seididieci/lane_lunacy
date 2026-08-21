// SPDX-License-Identifier: MIT

//! Raster sun-shadow mapping: a depth-only pass that renders the world chunks
//! from the sun's point of view, sampled by the mesh shader to darken the
//! direct-sun term.
//!
//! The design mirrors the ray-traced backend's shadow rules exactly:
//! - **Casters** are world chunks only (terrain, walls, rock faces) — the cars
//!   are never drawn into the map, so the player/traffic never cast (same as
//!   the RT cull masks).
//! - **Receivers** are everything the mesh shader draws (world + cars), and the
//!   planar-reflection pass samples the same map, matching RT's shadowed
//!   reflected rays.
//! - The map covers a fixed ortho box around the player (~±120 m lateral,
//!   ±140 m along the light) so near-field walls/trees cast; distant hills fade
//!   behind fog, which hides the map's edge.
//! - At night / when the sun is below the horizon the pass still clears the
//!   depth to far, so receivers sample "fully lit" and moonlight never casts
//!   (RT gates its probes on the sun elevation too).

use std::sync::Arc;

use glam::{Mat4, Vec3};

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

/// Shadow-map resolution (square). Must match `SHADOW_TEXEL` in `mesh.frag.glsl`
/// (1/2048).
pub const SHADOW_MAP_SIZE: u32 = 2048;
/// Half-extent of the shadow ortho box in the light's lateral plane (metres).
const SHADOW_HALF: f32 = 120.0;
/// Half-extent of the box along the light direction (metres).
const SHADOW_DEPTH_HALF: f32 = 140.0;

/// Everything the depth-only shadow pass owns. Fixed 2048² resolution, so the
/// images and framebuffer never need resizing with the window.
pub struct ShadowMapResources {
    /// Depth-only render pass: one D32_SFLOAT attachment.
    pub pass: Arc<RenderPass>,
    /// `shadow.vert/frag` pipeline bound to `pass` (writes depth, no color).
    pub pipeline: Arc<GraphicsPipeline>,
    /// NEAREST/clamped sampler used to read the map in the mesh shader (PCF is
    /// hand-rolled in `mesh.frag.glsl`, so the sampler stays unfiltered).
    pub sampler: Arc<Sampler>,
    /// The shadow depth image view (also `SAMPLED` for the mesh pass).
    pub depth_view: Arc<ImageView>,
    /// Framebuffer over the depth image.
    pub framebuffer: Arc<Framebuffer>,
}

impl ShadowMapResources {
    pub fn new(
        device: &Arc<Device>,
        memory_allocator: &Arc<StandardMemoryAllocator>,
    ) -> Self {
        let pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                depth: {
                    format: Format::D32_SFLOAT,
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
            },
            pass: {
                color: [],
                depth_stencil: { depth },
            }
        )
        .expect("shadow-map render pass");
        let subpass = Subpass::from(pass.clone(), 0).unwrap();

        let shadow =
            load_shaders::<Vertex3d>(device, shaders::SHADOW_VERT_SPV, shaders::SHADOW_FRAG_SPV);
        let pipeline = graphics_pipeline(
            device,
            &subpass,
            PipelineSpec {
                label: "shadow-map pipeline",
                // Back-face culling: only sun-facing surfaces write depth, so
                // the far side of a hill never occludes the side the sun sees.
                cull_mode: CullMode::Back,
                depth: Depth::Test { write: true },
                blend: Blend::Opaque,
            },
            shadow.stages,
            shadow.vertex_input,
            shadow.layout,
            SampleCount::Sample1,
        );

        let sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Nearest,
                min_filter: Filter::Nearest,
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                lod: 0.0..=0.0,
                ..Default::default()
            },
        )
        .expect("shadow-map sampler");

        let depth = Image::new(
            memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::D32_SFLOAT,
                extent: [SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 1],
                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::SAMPLED,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .expect("shadow-map depth image");
        let depth_view = ImageView::new_default(depth).expect("shadow-map depth view");
        let framebuffer = Framebuffer::new(
            pass.clone(),
            FramebufferCreateInfo {
                attachments: vec![depth_view.clone()],
                ..Default::default()
            },
        )
        .expect("shadow-map framebuffer");

        ShadowMapResources {
            pass,
            pipeline,
            sampler,
            depth_view,
            framebuffer,
        }
    }
}

/// Orthographic projection for Vulkan clip space (z in [0, 1]), right-handed —
/// the analogue of `camera::perspective_vulkan` for the shadow box.
fn ortho_vulkan(l: f32, r: f32, b: f32, t: f32, near: f32, far: f32) -> Mat4 {
    // glam 0.29's `orthographic_rh` already maps view depth to [0, 1].
    let mut p = Mat4::orthographic_rh(l, r, b, t, near, far);
    p.y_axis.y *= -1.0;
    p
}

/// Builds the `(view * proj)` matrix for the directional light shadow pass.
/// The ortho box is centered on `center` (the player), spans `SHADOW_HALF` in
/// the light's lateral plane and `SHADOW_DEPTH_HALF` along the light, and its
/// lateral offset is snapped to whole texels so the depth grid stays stable
/// while the camera glides instead of crawling every frame.
pub fn directional_light_view_proj(light_dir: Vec3, center: Vec3) -> Mat4 {
    let l = light_dir.normalize();
    // The light camera sits up-sun and looks back at the scene.
    let view = Mat4::look_at_rh(center + l * SHADOW_DEPTH_HALF, center, Vec3::Y);
    let texel = (2.0 * SHADOW_HALF) / SHADOW_MAP_SIZE as f32;
    let center_light = view.transform_point3(center);
    let snapped = Vec3::new(
        (center_light.x / texel).round() * texel,
        (center_light.y / texel).round() * texel,
        center_light.z,
    );
    let proj = ortho_vulkan(
        snapped.x - SHADOW_HALF,
        snapped.x + SHADOW_HALF,
        snapped.y - SHADOW_HALF,
        snapped.y + SHADOW_HALF,
        0.0,
        2.0 * SHADOW_DEPTH_HALF,
    );
    proj * view
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_centers_on_the_player_and_keeps_ndc_in_range() {
        // A sun near the top of its arc, straight ahead of the car.
        let light = Vec3::new(0.0, 2.75, -2.75).normalize();
        for center in [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(37.2, 0.0, -91.5),
            Vec3::new(-11.3, 0.0, 201.0),
        ] {
            let vp = directional_light_view_proj(light, center);
            let clip = vp * center.extend(1.0);
            let ndc = clip.truncate() / clip.w;
            assert!(
                (ndc.z - 0.5).abs() < 0.1,
                "box center must sit mid-depth, got z={}",
                ndc.z
            );
            // A point ~40m in front of the player stays inside the box.
            let ahead = center + Vec3::new(0.0, 0.0, -40.0);
            let clip_ahead = vp * ahead.extend(1.0);
            let ndc_ahead = clip_ahead.truncate() / clip_ahead.w;
            assert!(
                ndc_ahead.x.abs() <= 1.0 && ndc_ahead.y.abs() <= 1.0,
                "ahead point must fall inside the shadow box"
            );
        }
    }

    #[test]
    fn texel_snapping_keeps_the_grid_stable() {
        let light = Vec3::new(0.5, 2.0, -1.0).normalize();
        let a = directional_light_view_proj(light, Vec3::new(0.0, 0.0, 0.0));
        let b = directional_light_view_proj(light, Vec3::new(3.1, 0.0, -1.7));
        // The projection translation must snap to texel multiples: for a small
        // camera shift within one texel the matrices are identical.
        let texel = (2.0 * SHADOW_HALF) / SHADOW_MAP_SIZE as f32;
        let ca = a.transform_point3(Vec3::new(50.0, 5.0, -80.0));
        let cb = b.transform_point3(Vec3::new(50.0, 5.0, -80.0));
        assert!(
            (ca.x - cb.x).abs() <= texel + 1e-3 && (ca.y - cb.y).abs() <= texel + 1e-3,
            "grid must not crawl more than one texel per small camera move"
        );
    }
}
