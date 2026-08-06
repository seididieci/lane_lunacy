// SPDX-License-Identifier: MIT

//! Offscreen snapshot rendering: the headless "programmatic eye".
//!
//! Renders one fully-determined frame of a `Game` into an offscreen color image
//! (no window, no swapchain) and reads the pixels back to the CPU as an
//! `image::RgbaImage`. The scene is recorded with the same `record_frame`
//! recorder the windowed renderer uses, so a snapshot is pixel-identical to
//! what the window would show for the same state.

use std::sync::Arc;
use std::time::Duration;

use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyImageToBufferInfo,
};
use vulkano::device::{Device, Queue};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo};
use vulkano::sync::{self, GpuFuture};

use crate::font::FontAtlas;
use crate::game::Game;
use crate::hud::build_hud_tree;
use crate::mesh::build_world_chunk;
use crate::render::frame::build_frame;
use crate::render::record::record_frame;
use crate::render::scene::SceneResources;
use crate::render::{WORLD_CHUNKS_AHEAD, WORLD_CHUNKS_BEHIND, WORLD_CHUNK_LEN};
use crate::ui::Ui;

/// Renders one deterministic frame of `game` offscreen and returns the pixels
/// as an RGBA image (PNG-friendly channel order).
pub fn render_snapshot(
    device: Arc<Device>,
    queue: Arc<Queue>,
    game: &Game,
    font_atlas: &FontAtlas,
    seed: u64,
    width: u32,
    height: u32,
) -> image::RgbaImage {
    let render_pass = vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                format: Format::R16G16B16A16_SFLOAT,
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },
            depth: { format: Format::D32_SFLOAT, samples: 1, load_op: Clear, store_op: DontCare },
        },
        pass: {
            color: [color],
            depth_stencil: { depth },
        }
    )
    .expect("snapshot render pass");

    let scene = SceneResources::new(
        device.clone(),
        queue.clone(),
        render_pass.clone(),
        font_atlas,
        seed,
    );

    // Offscreen color attachment that can also serve as the copy source for
    // the host readback. The format matches the windowed swapchain
    // (R16G16B16A16_SFLOAT), so the offscreen render is exactly what the window
    // shows and probes can measure true linear luminance.
    let color_image = Image::new(
        scene.memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R16G16B16A16_SFLOAT,
            extent: [width, height, 1],
            usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
    )
    .expect("snapshot color image");
    let depth_image = Image::new(
        scene.memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::D32_SFLOAT,
            extent: [width, height, 1],
            usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
    )
    .expect("snapshot depth image");

    let framebuffer = Framebuffer::new(
        render_pass,
        FramebufferCreateInfo {
            attachments: vec![
                ImageView::new_default(color_image.clone()).expect("snapshot color view"),
                ImageView::new_default(depth_image).expect("snapshot depth view"),
            ],
            ..Default::default()
        },
    )
    .expect("snapshot framebuffer");

    let viewport = Viewport {
        offset: [0.0, 0.0],
        extent: [width as f32, height as f32],
        depth_range: 0.0..=1.0,
    };

    // World chunks anchored at the player's current chunk, exactly like the
    // windowed renderer keeps them.
    let mut world_chunks = Vec::new();
    let anchor_chunk = (game.vehicle.distance / WORLD_CHUNK_LEN).floor() as i32;
    for rel in -WORLD_CHUNKS_BEHIND..=WORLD_CHUNKS_AHEAD {
        let chunk_idx = anchor_chunk + rel;
        let start_s = chunk_idx as f32 * WORLD_CHUNK_LEN;
        let (wv, wi) = build_world_chunk(start_s, WORLD_CHUNK_LEN);
        let world_vertices = Buffer::from_iter(
            scene.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            wv,
        )
        .expect("world chunk vertices");
        let world_indices = Buffer::from_iter(
            scene.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            wi,
        )
        .expect("world chunk indices");
        world_chunks.push((world_vertices, world_indices));
    }

    // Deterministic CPU frame: zero dt (no sky drift, no camera smoothing),
    // freshly-constructed particle systems, and the playing HUD.
    let aspect = width as f32 / height as f32;
    let mut sky_time = 0.0;
    let mut camera_heading = 0.0;
    let mut rain = crate::render::particles::RainSystem::new();
    let mut dust = crate::render::particles::DustSystem::new();
    let mut hud_root = build_hud_tree(game);
    let hud_verts = Ui::new().build(&mut hud_root, font_atlas, aspect, 0.0);

    let frame = build_frame(
        game,
        Duration::ZERO,
        aspect,
        &mut sky_time,
        &mut camera_heading,
        &mut rain,
        &mut dust,
        &scene.player_anchors,
        &scene.traffic_anchors,
        hud_verts,
    );

    let command_buffer = record_frame(&scene, game, &frame, &world_chunks, framebuffer, &viewport);

    let readback = Buffer::new_slice::<u8>(
        scene.memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        (width as u64) * (height as u64) * 4 * 2,
    )
    .expect("snapshot readback buffer");

    let mut copy_builder = AutoCommandBufferBuilder::primary(
        scene.command_allocator.clone(),
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .expect("snapshot copy builder");
    copy_builder
        .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
            color_image,
            readback.clone(),
        ))
        .expect("copy snapshot image to buffer");
    let copy_cb = copy_builder.build().expect("build snapshot copy buffer");

    sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_execute(queue.clone(), copy_cb)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    let guard = readback.read().expect("read snapshot pixels");
    let mut pixels = Vec::with_capacity((width as usize) * (height as usize) * 4);
    // R16G16B16A16_SFLOAT is two little-endian half-float bytes per channel.
    for chunk in guard.chunks_exact(8) {
        let r = half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32();
        let g = half::f16::from_bits(u16::from_le_bytes([chunk[2], chunk[3]])).to_f32();
        let b = half::f16::from_bits(u16::from_le_bytes([chunk[4], chunk[5]])).to_f32();
        pixels.push(linear_to_srgb_u8(r));
        pixels.push(linear_to_srgb_u8(g));
        pixels.push(linear_to_srgb_u8(b));
        pixels.push(255);
    }
    image::RgbaImage::from_raw(width, height, pixels).expect("snapshot pixel size")
}

/// Converts a linear float color channel (0..1, as stored in the HDR
/// framebuffer) to an 8-bit sRGB-encoded byte for PNG output.
fn linear_to_srgb_u8(linear: f32) -> u8 {
    let c = linear.clamp(0.0, 1.0);
    let s = if c <= 0.0031308 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0).round() as u8
}

#[cfg(test)]
mod tests {
    use super::linear_to_srgb_u8;

    #[test]
    fn black_and_white_map_to_extremes() {
        assert_eq!(linear_to_srgb_u8(0.0), 0);
        assert_eq!(linear_to_srgb_u8(1.0), 255);
    }

    #[test]
    fn mid_gray_is_encoded_to_srgb() {
        // 0.5 linear encodes to ~0.735 sRGB -> 188.
        assert_eq!(linear_to_srgb_u8(0.5), 188);
    }

    #[test]
    fn out_of_range_is_clamped() {
        assert_eq!(linear_to_srgb_u8(-0.5), 0);
        assert_eq!(linear_to_srgb_u8(2.0), 255);
    }
}
