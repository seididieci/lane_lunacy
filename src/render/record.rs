// SPDX-License-Identifier: MIT

//! Single command-buffer recorder for any present target.
//!
//! `record_frame` turns a CPU `Frame` + `SceneResources` + world chunks into a
//! ready-to-submit primary command buffer for a given framebuffer. Both the
//! windowed `Renderer` and the headless snapshot presenter call it, so the
//! two targets can never drift apart: same sky, scene, particles, flare, HUD.

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
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
use vulkano::render_pass::Framebuffer;

use crate::game::Game;
use crate::render::frame::{traffic_rotation, Frame};
use crate::render::scene::SceneResources;
use crate::road::road_curve;
use crate::vertex::Vertex3d;

/// Records every draw in the scene into a primary command buffer for the given
/// framebuffer. The caller owns the render pass begin/end, the framebuffer,
/// and the submit/present.
pub fn record_frame(
    scene: &SceneResources,
    game: &Game,
    frame: &Frame,
    world_chunks: &[(Subbuffer<[Vertex3d]>, Subbuffer<[u32]>)],
    framebuffer: Arc<Framebuffer>,
    viewport: &Viewport,
) -> Arc<PrimaryAutoCommandBuffer> {
    let Frame {
        view,
        proj,
        lights,
        fog_color,
        wet_fac,
        headlight_pos,
        headlight_dir,
        traffic_head_pos,
        traffic_head_dir,
        traffic_head_state,
        sky_uniform,
        particle_verts,
        dust_verts,
        flare_verts,
        hud_verts,
        ..
    } = frame;

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
        let mvp = scene.mvp_buffer(
            model,
            *view,
            *proj,
            lights,
            *wet_fac,
            *fog_color,
            *headlight_pos,
            *headlight_dir,
            *traffic_head_pos,
            *traffic_head_dir,
            *traffic_head_state,
        );
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
            &mut builder,
            world_vertices.clone(),
            world_indices.clone(),
            scene.world_texture_view.clone(),
            Mat4::IDENTITY,
        );
    }
    // player car
    draw(
        &mut builder,
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
            &mut builder,
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
    if !dust_verts.is_empty() {
        scene.draw_particles(
            &mut builder,
            &scene.dust_pipeline,
            dust_verts,
            *view,
            *proj,
            lights,
            *wet_fac,
            *fog_color,
            *headlight_pos,
            *headlight_dir,
            *traffic_head_pos,
            *traffic_head_dir,
            *traffic_head_state,
        );
    }
    if !particle_verts.is_empty() {
        scene.draw_particles(
            &mut builder,
            &scene.particle_pipeline,
            particle_verts,
            *view,
            *proj,
            lights,
            *wet_fac,
            *fog_color,
            *headlight_pos,
            *headlight_dir,
            *traffic_head_pos,
            *traffic_head_dir,
            *traffic_head_state,
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

    // ---- HUD ----
    builder
        .bind_pipeline_graphics(scene.hud_pipeline.clone())
        .expect("bind hud pipeline")
        .bind_descriptor_sets(
            PipelineBindPoint::Graphics,
            scene.hud_pipeline.layout().clone(),
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

    builder
        .end_render_pass(SubpassEndInfo::default())
        .expect("end render pass");
    builder.build().expect("build command buffer")
}
