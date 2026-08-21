// SPDX-License-Identifier: MIT

//! Single command-buffer recorders for any present target.
//!
//! [`record_frame`] renders straight into a framebuffer (used by both the
//! headless snapshot presenter and the old-style direct path), while
//! [`record_frame_posted`] first renders the scene into an offscreen color
//! target, then runs the bloom chain + post-processing composite into the
//! swapchain, and finally draws the HUD/text on top of the swapchain so it
//! stays flat and unaffected by the post effects. Both paths share
//! [`record_scene_contents`], so sky, scene, particles and flare can never
//! drift apart.

use std::sync::Arc;

use glam::{Mat4, Vec3};

use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, RenderPassBeginInfo,
    SubpassBeginInfo, SubpassContents, SubpassEndInfo,
};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::image::view::ImageView;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint};
use vulkano::render_pass::Framebuffer;

use crate::game::Game;
use crate::render::frame::{traffic_rotation, Frame, SKY_RADIUS};
use crate::render::frame_builder::WorldChunk;
use crate::render::post::PostResources;
use crate::render::puddle_mask::PuddleMaskResources;
use crate::render::raytrace::RayTraceResources;
use crate::render::reflection::{
    reflected_view, ReflectionBackend, ReflectionResources, REFLECTION_CLIP_Y,
};
use crate::render::scene::SceneResources;
use crate::road::road_curve;
use crate::shaders::{BloomParams, PostSettings};
use crate::vertex::{HudVertex, Vertex3d};

/// Linear-HDR luminance above which a source pixel contributes to the bloom
/// glow, and the width of the smooth knee around it (see `BloomParams`).
const BLOOM_THRESHOLD: f32 = 0.8;
const BLOOM_KNEE: f32 = 0.12;

/// Records every draw in the scene into a primary command buffer for the given
/// framebuffer. The caller owns the render pass begin/end, the framebuffer,
/// and the submit/present.
pub fn record_frame(
    scene: &SceneResources,
    game: &Game,
    frame: &Frame,
    world_chunks: &[WorldChunk],
    framebuffer: Arc<Framebuffer>,
    viewport: &Viewport,
) -> Arc<PrimaryAutoCommandBuffer> {
    let mut builder = AutoCommandBufferBuilder::primary(
        scene.command_allocator.clone(),
        scene.queue_family_index,
        CommandBufferUsage::MultipleSubmit,
    )
    .expect("command buffer builder");

    builder
        .begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![Some([0.9, 0.7, 0.5, 1.0].into()), Some(1.0f32.into())],
                ..RenderPassBeginInfo::framebuffer(framebuffer)
            },
            SubpassBeginInfo {
                contents: SubpassContents::Inline,
                ..Default::default()
            },
        )
        .expect("begin render pass")
        .set_viewport(0, [viewport.clone()].into_iter().collect())
        .expect("set viewport");
    record_scene_contents(&mut builder, scene, game, frame, world_chunks);
    record_hud(
        &mut builder,
        scene,
        scene.hud_pipeline.clone(),
        &frame.hud_verts,
    );
    builder
        .end_render_pass(SubpassEndInfo::default())
        .expect("end render pass");
    builder.build().expect("build command buffer")
}

/// Records the windowed path: scene into the offscreen framebuffer, the bloom
/// downsample chain (when `settings.flags` has `POST_BLOOM`), the post
/// composite into the swapchain framebuffer, then the HUD/text pass on top of
/// the swapchain so text stays flat and readable (no bloom glow, no chromatic
/// aberration fringes, no FXAA/grain/vignette softening).
///
/// `offscreen_view` is the scene color target sampled by the composite (and
/// the first bloom downsample); `bloom_views` is `post.bloom_views`; bloom
/// level *n* downsamples level *n-1* (level 0 downsamples the offscreen).
/// `hud_framebuffer` is the swapchain image bound to `post.hud_pass`; its
/// `load_op: Load` composites the text over the post output.
#[allow(clippy::too_many_arguments)]
pub fn record_frame_posted(
    scene: &SceneResources,
    post: &PostResources,
    reflection: &ReflectionResources,
    puddle_mask: &PuddleMaskResources,
    mut raytrace: Option<&mut RayTraceResources>,
    should_record_reflection: bool,
    game: &Game,
    frame: &Frame,
    world_chunks: &[WorldChunk],
    chunk_indices: &[i32],
    scene_framebuffer: Arc<Framebuffer>,
    particle_framebuffer: Arc<Framebuffer>,
    post_framebuffer: Arc<Framebuffer>,
    hud_framebuffer: Arc<Framebuffer>,
    bloom_fbs: &[Arc<Framebuffer>],
    viewport: &Viewport,
    offscreen_view: Arc<ImageView>,
    depth_view: Arc<ImageView>,
    bloom_views: &[Arc<ImageView>],
    settings: &PostSettings,
    image_i: usize,
    timings: &mut crate::profiler::FrameTimings,
) -> Arc<PrimaryAutoCommandBuffer> {
    let mut builder = AutoCommandBufferBuilder::primary(
        scene.command_allocator.clone(),
        scene.queue_family_index,
        CommandBufferUsage::MultipleSubmit,
    )
    .expect("command buffer builder");

    // ---- Scene into offscreen ----
    // Under ray tracing the backend replaces the raster scene (and the puddle
    // mask / planar-reflection passes below) by writing the offscreen directly,
    // so bloom + post + HUD keep reading the same images as the raster path.
    let scene_started = std::time::Instant::now();
    match raytrace.as_deref_mut() {
        Some(rt) => {
            let [w, h] = viewport.extent;
            rt.record(
                &mut builder,
                scene,
                game,
                frame,
                world_chunks,
                chunk_indices,
                image_i,
                offscreen_view.clone(),
                [w as u32, h as u32],
            );

            // ---- Particle overlay (rain / mist / drift dust / night glows) ----
            // The RT backend writes the scene color directly into the offscreen
            // and has no depth buffer, so the CPU particle quads are composited
            // in a dedicated color-only pass that *loads* the RT output. The
            // shader occludes per-pixel against the raygen's linear-depth image
            // (discards fragments behind geometry), so particles never overdraw
            // cars while rain still shows over sky and road. Same draw order as
            // the raster path: mist, then dust (alpha), then rain+lights
            // (additive).
            builder
                .begin_render_pass(
                    RenderPassBeginInfo {
                        // The color attachment is `Load`: keep the RT image.
                        clear_values: vec![None],
                        ..RenderPassBeginInfo::framebuffer(particle_framebuffer)
                    },
                    SubpassBeginInfo {
                        contents: SubpassContents::Inline,
                        ..Default::default()
                    },
                )
                .expect("begin rt particle render pass")
                .set_viewport(0, [viewport.clone()].into_iter().collect())
                .expect("set rt particle viewport");
            let rt_depth = rt.depth_view();
            let depth_sampler = post.depth_sampler.clone();
            if !frame.mist_verts.is_empty() {
                scene.draw_rt_particles(
                    &mut builder,
                    &scene.rt_dust_pipeline,
                    &frame.mist_verts,
                    &frame.uniforms,
                    &frame.headlights,
                    rt_depth.clone(),
                    depth_sampler.clone(),
                );
            }
            if !frame.dust_verts.is_empty() {
                scene.draw_rt_particles(
                    &mut builder,
                    &scene.rt_dust_pipeline,
                    &frame.dust_verts,
                    &frame.uniforms,
                    &frame.headlights,
                    rt_depth.clone(),
                    depth_sampler.clone(),
                );
            }
            if !frame.particle_verts.is_empty() {
                scene.draw_rt_particles(
                    &mut builder,
                    &scene.rt_particle_pipeline,
                    &frame.particle_verts,
                    &frame.uniforms,
                    &frame.headlights,
                    rt_depth,
                    depth_sampler,
                );
            }
            builder
                .end_render_pass(SubpassEndInfo::default())
                .expect("end rt particle render pass");
        }
        None => {
            // One clear value per attachment: the color and depth are `Clear`,
            // the MSAA color resolve and the single-sampled depth resolve
            // targets (present only under 2x/4x) are `DontCare`.
            let scene_clears = match scene_framebuffer.attachments().len() {
                2 => vec![Some([0.9, 0.7, 0.5, 1.0].into()), Some(1.0f32.into())],
                4 => vec![
                    Some([0.9, 0.7, 0.5, 1.0].into()),
                    None,
                    Some(1.0f32.into()),
                    None,
                ],
                n => unreachable!("unexpected scene framebuffer attachment count: {n}"),
            };
            builder
                .begin_render_pass(
                    RenderPassBeginInfo {
                        clear_values: scene_clears,
                        ..RenderPassBeginInfo::framebuffer(scene_framebuffer)
                    },
                    SubpassBeginInfo {
                        contents: SubpassContents::Inline,
                        ..Default::default()
                    },
                )
                .expect("begin scene render pass")
                .set_viewport(0, [viewport.clone()].into_iter().collect())
                .expect("set viewport");
            record_scene_contents(&mut builder, scene, game, frame, world_chunks);
            builder
                .end_render_pass(SubpassEndInfo::default())
                .expect("end scene render pass");
        }
    }
    timings.scene_ms = scene_started.elapsed().as_secs_f32() * 1000.0;

    // ---- Planar road reflections ----
    // Mirrored-camera pass into a quality-scaled target sampled by the
    // composite for wet-asphalt puddles. Runs whenever reflections are enabled;
    // when the flag is off the composite never samples the stale target. The
    // ray-traced backend replaces this pass, so it is skipped here too.
    if raytrace.is_none()
        && settings.flags & crate::shaders::POST_REFLECT != 0
        && settings.wet_fac > 0.001
    {
        let [mw, mh] = puddle_mask.framebuffer.extent();
        let mask_viewport = Viewport {
            offset: [0.0, 0.0],
            extent: [mw as f32, mh as f32],
            depth_range: 0.0..=1.0,
        };
        record_puddle_mask(
            &mut builder,
            scene,
            puddle_mask,
            game,
            frame,
            world_chunks,
            puddle_mask.framebuffer.clone(),
            &mask_viewport,
        );

        if should_record_reflection {
            let [rw, rh] = reflection.framebuffer.extent();
            let reflection_viewport = Viewport {
                offset: [0.0, 0.0],
                extent: [rw as f32, rh as f32],
                depth_range: 0.0..=1.0,
            };
            record_reflection(
                &mut builder,
                scene,
                reflection,
                game,
                frame,
                world_chunks,
                reflection.framebuffer.clone(),
                &reflection_viewport,
                settings.planar_plane_y,
            );
        }
    }

    // ---- Bloom downsample chain ----
    let bloom_started = std::time::Instant::now();
    if settings.flags & crate::shaders::POST_BLOOM != 0 {
        for (level, fb) in bloom_fbs.iter().enumerate() {
            let src = if level == 0 {
                offscreen_view.clone()
            } else {
                bloom_views[level - 1].clone()
            };
            // Each bloom framebuffer is half the resolution of the one above;
            // the viewport must match the framebuffer, not the full window, or
            // the fullscreen triangle only covers its top-left quadrant.
            let [bw, bh] = fb.extent();
            let bloom_viewport = Viewport {
                offset: [0.0, 0.0],
                extent: [bw as f32, bh as f32],
                depth_range: 0.0..=1.0,
            };
            let bloom_params = BloomParams {
                threshold: BLOOM_THRESHOLD,
                knee: BLOOM_KNEE,
                first_pass: u32::from(level == 0),
                _pad: 0.0,
            };
            let bloom_buf = Buffer::from_data(
                scene.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::UNIFORM_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                bloom_params,
            )
            .expect("bloom params buffer");
            let set_layout = post.bloom_pipeline.layout().set_layouts()[0].clone();
            let set = DescriptorSet::new(
                scene.descriptor_set_allocator.clone(),
                set_layout,
                [
                    WriteDescriptorSet::image_view_sampler(0, src, post.sampler.clone()),
                    WriteDescriptorSet::buffer(1, bloom_buf),
                ],
                [],
            )
            .expect("bloom descriptor set");
            builder
                .begin_render_pass(
                    RenderPassBeginInfo {
                        clear_values: vec![None],
                        ..RenderPassBeginInfo::framebuffer(fb.clone())
                    },
                    SubpassBeginInfo {
                        contents: SubpassContents::Inline,
                        ..Default::default()
                    },
                )
                .expect("begin bloom render pass")
                .set_viewport(0, [bloom_viewport].into_iter().collect())
                .expect("set viewport")
                .bind_pipeline_graphics(post.bloom_pipeline.clone())
                .expect("bind bloom pipeline")
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    post.bloom_pipeline.layout().clone(),
                    0,
                    set,
                )
                .expect("bind bloom descriptor sets");
            unsafe {
                builder.draw(3, 1, 0, 0).expect("draw bloom");
            }
            builder
                .end_render_pass(SubpassEndInfo::default())
                .expect("end bloom render pass");
        }
    }
    timings.bloom_ms = bloom_started.elapsed().as_secs_f32() * 1000.0;

    // ---- Post composite into the swapchain ----
    let post_started = std::time::Instant::now();
    let post_buf = Buffer::from_data(
        scene.memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::UNIFORM_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        *settings,
    )
    .expect("post settings buffer");
    let post_set_layout = post.pipeline.layout().set_layouts()[0].clone();
    let post_set = DescriptorSet::new(
        scene.descriptor_set_allocator.clone(),
        post_set_layout,
        [
            WriteDescriptorSet::buffer(0, post_buf),
            WriteDescriptorSet::image_view_sampler(1, offscreen_view, post.sampler.clone()),
            WriteDescriptorSet::image_view_sampler(
                2,
                bloom_views.last().cloned().expect("bloom views"),
                post.sampler.clone(),
            ),
            WriteDescriptorSet::image_view_sampler(
                3,
                depth_view,
                post.depth_sampler.clone(),
            ),
            WriteDescriptorSet::image_view_sampler(
                4,
                reflection.color_view().clone(),
                reflection.sampler().clone(),
            ),
            WriteDescriptorSet::image_view_sampler(
                5,
                puddle_mask.mask_view.clone(),
                puddle_mask.sampler.clone(),
            ),
        ],
        [],
    )
    .expect("post descriptor set");
    builder
        .begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![None],
                ..RenderPassBeginInfo::framebuffer(post_framebuffer)
            },
            SubpassBeginInfo {
                contents: SubpassContents::Inline,
                ..Default::default()
            },
        )
        .expect("begin post render pass")
        .set_viewport(0, [viewport.clone()].into_iter().collect())
        .expect("set viewport")
        .bind_pipeline_graphics(post.pipeline.clone())
        .expect("bind post pipeline")
        .bind_descriptor_sets(
            PipelineBindPoint::Graphics,
            post.pipeline.layout().clone(),
            0,
            post_set,
        )
        .expect("bind post descriptor sets");
    unsafe {
        builder.draw(3, 1, 0, 0).expect("draw post");
    }
    builder
        .end_render_pass(SubpassEndInfo::default())
        .expect("end post render pass");
    timings.post_ms = post_started.elapsed().as_secs_f32() * 1000.0;

    // ---- HUD/text flat pass on top of the post output ----
    // `load_op: Load` keeps the post composite already written to the
    // swapchain image; nothing is cleared. Drawn at 1x against the swapchain
    // format so bloom/chroma/FXAA/grain/vignette never touch the text.
    let hud_started = std::time::Instant::now();
    builder
        .begin_render_pass(
            RenderPassBeginInfo {
                // The attachment is `Load` (composites over the post output),
                // so the entry is `None`: nothing is cleared.
                clear_values: vec![None],
                ..RenderPassBeginInfo::framebuffer(hud_framebuffer)
            },
            SubpassBeginInfo {
                contents: SubpassContents::Inline,
                ..Default::default()
            },
        )
        .expect("begin hud render pass")
        .set_viewport(0, [viewport.clone()].into_iter().collect())
        .expect("set viewport");
    record_hud(
        &mut builder,
        scene,
        post.hud_pipeline.clone(),
        &frame.hud_verts,
    );
    builder
        .end_render_pass(SubpassEndInfo::default())
        .expect("end hud render pass");
    timings.hud_ms = hud_started.elapsed().as_secs_f32() * 1000.0;
    builder.build().expect("build command buffer")
}

/// Records sky, 3D scene, particles and flare draws between the render pass
/// begin and end. The caller owns begin/end and the viewport; the HUD/text
/// pass is recorded separately by [`record_hud`] so it can be drawn on top of
/// the post composite in the windowed path.
fn record_scene_contents(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    scene: &SceneResources,
    game: &Game,
    frame: &Frame,
    world_chunks: &[WorldChunk],
) {
    let Frame {
        uniforms,
        headlights,
        sky_uniform,
        particle_verts,
        dust_verts,
        mist_verts,
        flare_verts,
        ..
    } = frame;

    // ---- Sky dome (background) ----
    // Drawn first with depth disabled so the 3D scene overdraws it.
    builder
        .bind_pipeline_graphics(scene.sky_pipeline.clone())
        .expect("bind sky pipeline");

    let sky_buf = Buffer::from_data(
        scene.memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::UNIFORM_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        *sky_uniform,
    )
    .expect("sky uniform buffer");
    let sky_set_layout = scene.sky_pipeline.layout().set_layouts()[0].clone();
    let sky_set = DescriptorSet::new(
        scene.descriptor_set_allocator.clone(),
        sky_set_layout,
        [
            WriteDescriptorSet::buffer(0, sky_buf),
            WriteDescriptorSet::image_view_sampler(
                1,
                scene.cloud_a_view.clone(),
                scene.mesh_sampler.clone(),
            ),
            WriteDescriptorSet::image_view_sampler(
                2,
                scene.cloud_b_view.clone(),
                scene.mesh_sampler.clone(),
            ),
        ],
        [],
    )
    .expect("sky descriptor set");
    let sky_index_count = scene.sky_dome_indices.len() as u32;
    builder
        .bind_descriptor_sets(
            PipelineBindPoint::Graphics,
            scene.sky_pipeline.layout().clone(),
            0,
            sky_set,
        )
        .expect("bind sky descriptor sets")
        .bind_vertex_buffers(0, scene.sky_dome_vertices.clone())
        .expect("bind sky vertex buffers")
        .bind_index_buffer(scene.sky_dome_indices.clone())
        .expect("bind sky index buffer");
    unsafe {
        builder
            .draw_indexed(sky_index_count, 1, 0, 0, 0)
            .expect("draw sky");
    }

    // ---- 3D scene ----
    builder
        .bind_pipeline_graphics(scene.mesh_pipeline.clone())
        .expect("bind mesh pipeline");

    let draw = |builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
                vertices: Subbuffer<[Vertex3d]>,
                indices: Subbuffer<[u32]>,
                texture: Arc<ImageView>,
                model: Mat4| {
        let index_count = indices.len() as u32;
        let mvp = scene.mvp_buffer(model, uniforms, headlights, [0.0, 0.0, 0.0, -1.0]);
        let set_layout = scene.mesh_pipeline.layout().set_layouts()[0].clone();
        let set = DescriptorSet::new(
            scene.descriptor_set_allocator.clone(),
            set_layout,
            [
                WriteDescriptorSet::buffer(0, mvp.clone()),
                WriteDescriptorSet::image_view_sampler(1, texture, scene.mesh_sampler.clone()),
            ],
            [],
        )
        .expect("descriptor set");
        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                scene.mesh_pipeline.layout().clone(),
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

    for (world_vertices, world_indices) in world_chunks {
        draw(
            builder,
            world_vertices.clone(),
            world_indices.clone(),
            scene.world_texture_view.clone(),
            Mat4::IDENTITY,
        );
    }
    // player car
    draw(
        builder,
        scene.car_vertices.clone(),
        scene.car_indices.clone(),
        scene.car_texture_view.clone(),
        Mat4::from_scale_rotation_translation(
            Vec3::ONE,
            glam::Quat::from_rotation_y(-game.vehicle.heading),
            Vec3::new(game.player_world_x(), 0.03, game.player_world_z()),
        ),
    );
    // traffic
    for (idx, t) in game.traffic.iter().enumerate() {
        let tvx = road_curve(t.distance) + t.lane;
        let traffic_rot = traffic_rotation(t.lane, t.distance);
        let (traffic_vertices, traffic_indices, _anchors) =
            &scene.traffic_meshes[idx % scene.traffic_meshes.len()];
        draw(
            builder,
            traffic_vertices.clone(),
            traffic_indices.clone(),
            scene.car_texture_view.clone(),
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
    // Mist is drawn first (big, soft, background) with alpha blending; drift
    // dust and rain composite on top.
    if !mist_verts.is_empty() {
        scene.draw_particles(
            builder,
            &scene.dust_pipeline,
            mist_verts,
            uniforms,
            headlights,
        );
    }
    if !dust_verts.is_empty() {
        scene.draw_particles(
            builder,
            &scene.dust_pipeline,
            dust_verts,
            uniforms,
            headlights,
        );
    }
    if !particle_verts.is_empty() {
        scene.draw_particles(
            builder,
            &scene.particle_pipeline,
            particle_verts,
            uniforms,
            headlights,
        );
    }

    // ---- Sun lens flare ----
    // Quads are baked into the CPU Frame (NDC positions, fan layout,
    // intensity); we only upload and draw them here.
    if !flare_verts.is_empty() {
        let flare_set_layout = scene.flare_pipeline.layout().set_layouts()[0].clone();
        let flare_set = DescriptorSet::new(
            scene.descriptor_set_allocator.clone(),
            flare_set_layout,
            [
                WriteDescriptorSet::image_view_sampler(
                    0,
                    scene.flare_core_view.clone(),
                    scene.flare_sampler.clone(),
                ),
                WriteDescriptorSet::image_view_sampler(
                    1,
                    scene.flare_streak_view.clone(),
                    scene.flare_sampler.clone(),
                ),
                WriteDescriptorSet::image_view_sampler(
                    2,
                    scene.flare_ring_view.clone(),
                    scene.flare_sampler.clone(),
                ),
            ],
            [],
        )
        .expect("flare descriptor set");
        let flare_buf = Buffer::from_iter(
            scene.memory_allocator.clone(),
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
            .bind_pipeline_graphics(scene.flare_pipeline.clone())
            .expect("bind flare pipeline")
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                scene.flare_pipeline.layout().clone(),
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
}

/// Records the mirrored-camera planar reflection pass: sky dome + world
/// chunks + player + traffic into the reflection target, with geometry below
/// the road plane clipped out and culling disabled (mirroring flips winding).
/// Particles and the lens flare are intentionally omitted — the flare is a
/// screen-space artifact and rain/mist would render inconsistently from the
/// mirrored camera.
#[allow(clippy::too_many_arguments)]
fn record_reflection(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    scene: &SceneResources,
    reflection: &ReflectionResources,
    game: &Game,
    frame: &Frame,
    world_chunks: &[WorldChunk],
    framebuffer: Arc<Framebuffer>,
    viewport: &Viewport,
    plane_y: f32,
) {
    let reflect_view = reflected_view(frame.uniforms.view, plane_y);
    let mut uniforms = frame.uniforms;
    uniforms.view = reflect_view;

    // Mirror the sky dome around the road plane too, so it centers on the
    // mirrored camera and the reflection shows a seamless sky.
    let eye = frame.uniforms.eye;
    let mirrored_eye = Vec3::new(eye.x, 2.0 * plane_y - eye.y, eye.z);
    let mut sky_uniform = frame.sky_uniform;
    sky_uniform.model = Mat4::from_scale_rotation_translation(
        Vec3::splat(SKY_RADIUS),
        glam::Quat::IDENTITY,
        mirrored_eye,
    )
    .to_cols_array_2d();
    sky_uniform.view = reflect_view.to_cols_array_2d();
    sky_uniform.projection = frame.uniforms.proj.to_cols_array_2d();

    let clip_plane = [0.0, -1.0, 0.0, REFLECTION_CLIP_Y];

    builder
        .begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![
                    Some(frame.uniforms.fog_color.into()),
                    Some(1.0f32.into()),
                ],
                ..RenderPassBeginInfo::framebuffer(framebuffer)
            },
            SubpassBeginInfo {
                contents: SubpassContents::Inline,
                ..Default::default()
            },
        )
        .expect("begin reflection render pass")
        .set_viewport(0, [viewport.clone()].into_iter().collect())
        .expect("set reflection viewport");

    // ---- Sky dome ----
    builder
        .bind_pipeline_graphics(reflection.sky_pipeline.clone())
        .expect("bind reflection sky pipeline");
    let sky_buf = Buffer::from_data(
        scene.memory_allocator.clone(),
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
    .expect("reflection sky uniform buffer");
    let sky_set_layout = reflection.sky_pipeline.layout().set_layouts()[0].clone();
    let sky_set = DescriptorSet::new(
        scene.descriptor_set_allocator.clone(),
        sky_set_layout,
        [
            WriteDescriptorSet::buffer(0, sky_buf),
            WriteDescriptorSet::image_view_sampler(
                1,
                scene.cloud_a_view.clone(),
                scene.mesh_sampler.clone(),
            ),
            WriteDescriptorSet::image_view_sampler(
                2,
                scene.cloud_b_view.clone(),
                scene.mesh_sampler.clone(),
            ),
        ],
        [],
    )
    .expect("reflection sky descriptor set");
    let sky_index_count = scene.sky_dome_indices.len() as u32;
    builder
        .bind_descriptor_sets(
            PipelineBindPoint::Graphics,
            reflection.sky_pipeline.layout().clone(),
            0,
            sky_set,
        )
        .expect("bind reflection sky descriptor sets")
        .bind_vertex_buffers(0, scene.sky_dome_vertices.clone())
        .expect("bind reflection sky vertex buffers")
        .bind_index_buffer(scene.sky_dome_indices.clone())
        .expect("bind reflection sky index buffer");
    unsafe {
        builder
            .draw_indexed(sky_index_count, 1, 0, 0, 0)
            .expect("draw reflection sky");
    }

    // ---- 3D scene (world, player, traffic) ----
    builder
        .bind_pipeline_graphics(reflection.mesh_pipeline.clone())
        .expect("bind reflection mesh pipeline");
    let draw = |builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
                vertices: Subbuffer<[Vertex3d]>,
                indices: Subbuffer<[u32]>,
                texture: Arc<ImageView>,
                model: Mat4| {
        let index_count = indices.len() as u32;
        let mvp = scene.mvp_buffer(model, &uniforms, &frame.headlights, clip_plane);
        let set_layout = reflection.mesh_pipeline.layout().set_layouts()[0].clone();
        let set = DescriptorSet::new(
            scene.descriptor_set_allocator.clone(),
            set_layout,
            [
                WriteDescriptorSet::buffer(0, mvp),
                WriteDescriptorSet::image_view_sampler(1, texture, scene.mesh_sampler.clone()),
            ],
            [],
        )
        .expect("reflection descriptor set");
        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                reflection.mesh_pipeline.layout().clone(),
                0,
                set,
            )
            .expect("bind reflection descriptor sets")
            .bind_vertex_buffers(0, vertices)
            .expect("bind reflection vertex buffers")
            .bind_index_buffer(indices)
            .expect("bind reflection index buffer");
        unsafe {
            builder
                .draw_indexed(index_count, 1, 0, 0, 0)
                .expect("draw reflection indexed");
        }
    };

    for (world_vertices, world_indices) in world_chunks {
        draw(
            builder,
            world_vertices.clone(),
            world_indices.clone(),
            scene.world_texture_view.clone(),
            Mat4::IDENTITY,
        );
    }
    draw(
        builder,
        scene.car_vertices.clone(),
        scene.car_indices.clone(),
        scene.car_texture_view.clone(),
        Mat4::from_scale_rotation_translation(
            Vec3::ONE,
            glam::Quat::from_rotation_y(-game.vehicle.heading),
            Vec3::new(game.player_world_x(), 0.03, game.player_world_z()),
        ),
    );
    for (idx, t) in game.traffic.iter().enumerate() {
        let tvx = road_curve(t.distance) + t.lane;
        let traffic_rot = traffic_rotation(t.lane, t.distance);
        let (traffic_vertices, traffic_indices, _anchors) =
            &scene.traffic_meshes[idx % scene.traffic_meshes.len()];
        draw(
            builder,
            traffic_vertices.clone(),
            traffic_indices.clone(),
            scene.car_texture_view.clone(),
            Mat4::from_scale_rotation_translation(
                Vec3::ONE,
                traffic_rot,
                Vec3::new(tvx, 0.35, -t.distance),
            ),
        );
    }

    builder
        .end_render_pass(SubpassEndInfo::default())
        .expect("end reflection render pass");
}

/// Records a dedicated puddle-mask pass from the main camera. The pass writes
/// an asphalt-only static puddle mask into an R8 texture sampled by post.
#[allow(clippy::too_many_arguments)]
fn record_puddle_mask(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    scene: &SceneResources,
    puddle_mask: &PuddleMaskResources,
    game: &Game,
    frame: &Frame,
    world_chunks: &[WorldChunk],
    framebuffer: Arc<Framebuffer>,
    viewport: &Viewport,
) {
    builder
        .begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![Some([0.0, 0.0, 0.0, 1.0].into()), Some(1.0f32.into())],
                ..RenderPassBeginInfo::framebuffer(framebuffer)
            },
            SubpassBeginInfo {
                contents: SubpassContents::Inline,
                ..Default::default()
            },
        )
        .expect("begin puddle-mask render pass")
        .set_viewport(0, [viewport.clone()].into_iter().collect())
        .expect("set puddle-mask viewport")
        .bind_pipeline_graphics(puddle_mask.pipeline.clone())
        .expect("bind puddle-mask pipeline");

    let draw = |builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
                vertices: Subbuffer<[Vertex3d]>,
                indices: Subbuffer<[u32]>,
                model: Mat4| {
        let index_count = indices.len() as u32;
        let mvp = scene.mvp_buffer(
            model,
            &frame.uniforms,
            &frame.headlights,
            [0.0, 0.0, 0.0, -1.0],
        );
        let set_layout = puddle_mask.pipeline.layout().set_layouts()[0].clone();
        let set = DescriptorSet::new(
            scene.descriptor_set_allocator.clone(),
            set_layout,
            [WriteDescriptorSet::buffer(0, mvp)],
            [],
        )
        .expect("puddle-mask descriptor set");
        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                puddle_mask.pipeline.layout().clone(),
                0,
                set,
            )
            .expect("bind puddle-mask descriptor sets")
            .bind_vertex_buffers(0, vertices)
            .expect("bind puddle-mask vertex buffers")
            .bind_index_buffer(indices)
            .expect("bind puddle-mask index buffer");
        unsafe {
            builder
                .draw_indexed(index_count, 1, 0, 0, 0)
                .expect("draw puddle-mask indexed");
        }
    };

    for (world_vertices, world_indices) in world_chunks {
        draw(
            builder,
            world_vertices.clone(),
            world_indices.clone(),
            Mat4::IDENTITY,
        );
    }
    draw(
        builder,
        scene.car_vertices.clone(),
        scene.car_indices.clone(),
        Mat4::from_scale_rotation_translation(
            Vec3::ONE,
            glam::Quat::from_rotation_y(-game.vehicle.heading),
            Vec3::new(game.player_world_x(), 0.03, game.player_world_z()),
        ),
    );
    for (idx, t) in game.traffic.iter().enumerate() {
        let tvx = road_curve(t.distance) + t.lane;
        let traffic_rot = traffic_rotation(t.lane, t.distance);
        let (traffic_vertices, traffic_indices, _anchors) =
            &scene.traffic_meshes[idx % scene.traffic_meshes.len()];
        draw(
            builder,
            traffic_vertices.clone(),
            traffic_indices.clone(),
            Mat4::from_scale_rotation_translation(
                Vec3::ONE,
                traffic_rot,
                Vec3::new(tvx, 0.35, -t.distance),
            ),
        );
    }

    builder
        .end_render_pass(SubpassEndInfo::default())
        .expect("end puddle-mask render pass");
}

/// Records the HUD/text draw: binds `hud_pipeline` (either the scene pass's
/// pipeline or `post.hud_pipeline` for the flat post-post pass), uploads the
/// CPU-built `HudVertex` quads and draws them. The descriptor set is shared
/// (same shaders, same layout) and lives on `SceneResources`.
fn record_hud(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    scene: &SceneResources,
    hud_pipeline: Arc<GraphicsPipeline>,
    hud_verts: &[HudVertex],
) {
    if hud_verts.is_empty() {
        // No HUD (e.g. F4 clean-screen): skip the draw entirely. Vulkano's
        // `Buffer::from_iter` panics on an empty iterator.
        return;
    }
    builder
        .bind_pipeline_graphics(hud_pipeline.clone())
        .expect("bind hud pipeline")
        .bind_descriptor_sets(
            PipelineBindPoint::Graphics,
            hud_pipeline.layout().clone(),
            0,
            scene.hud_descriptor_set.clone(),
        )
        .expect("bind hud descriptor set");
    let hud_vertex_count = hud_verts.len() as u32;
    let hud_buf = Buffer::from_iter(
        scene.memory_allocator.clone(),
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
}
