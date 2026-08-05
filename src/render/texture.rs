// SPDX-License-Identifier: MIT

use std::sync::Arc;

use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, BlitImageInfo, CommandBufferUsage, CopyBufferToImageInfo,
};
use vulkano::device::Queue;
use vulkano::image::view::ImageView;
use vulkano::image::sampler::Filter;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::{
    AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator,
};
use vulkano::sync::{self, GpuFuture};

use crate::vertex::Vertex3d;

pub fn make_mesh_buffers(
    memory_allocator: Arc<StandardMemoryAllocator>,
    vertices: Vec<Vertex3d>,
    indices: Vec<u32>,
) -> (Subbuffer<[Vertex3d]>, Subbuffer<[u32]>) {
    let vertex_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        vertices,
    )
    .expect("mesh vertices");
    let index_buffer = Buffer::from_iter(
        memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::INDEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        indices,
    )
    .expect("mesh indices");
    (vertex_buffer, index_buffer)
}

pub fn upload_rgba8_texture(
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_allocator: Arc<StandardCommandBufferAllocator>,
    queue: Arc<Queue>,
    width: u32,
    height: u32,
    pixels: Vec<u8>,
) -> Arc<ImageView> {
    let staging = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        pixels,
    )
    .expect("texture staging buffer");

    let texture_image = Image::new(
        memory_allocator,
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: vulkano::format::Format::R8G8B8A8_UNORM,
            extent: [width, height, 1],
            usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
    )
    .expect("texture image");

    let mut upload_builder = AutoCommandBufferBuilder::primary(
        command_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .expect("texture upload builder");
    upload_builder
        .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(staging, texture_image.clone()))
        .expect("copy texture to image");
    let upload_cb = upload_builder
        .build()
        .expect("build texture upload command buffer");
    sync::now(queue.device().clone())
        .then_execute(queue, upload_cb)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    ImageView::new_default(texture_image).expect("texture image view")
}

/// Uploads an RGBA texture and generates a full mipmap chain via linear blits.
///
/// Returns the image view (covering all levels) plus the mip level count so the
/// caller can clamp the sampler's LOD range. Mipmapping pre-filters the texture
/// so sub-pixel (far-away) quads sample a smooth, averaged level instead of
/// aliasing into shimmering texels.
pub fn upload_rgba8_texture_mipmapped(
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_allocator: Arc<StandardCommandBufferAllocator>,
    queue: Arc<Queue>,
    width: u32,
    height: u32,
    pixels: Vec<u8>,
) -> (Arc<ImageView>, u32) {
    let mip_levels = (width.max(height) as f32).log2().floor() as u32 + 1;
    let staging = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        pixels,
    )
    .expect("texture staging buffer");

    let texture_image = Image::new(
        memory_allocator,
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: vulkano::format::Format::R8G8B8A8_UNORM,
            extent: [width, height, 1],
            usage: ImageUsage::TRANSFER_DST | ImageUsage::TRANSFER_SRC | ImageUsage::SAMPLED,
            mip_levels,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
    )
    .expect("texture image");

    let mut upload_builder = AutoCommandBufferBuilder::primary(
        command_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .expect("texture upload builder");
    upload_builder
        .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(staging, texture_image.clone()))
        .expect("copy texture to image");

    // Blit each level down to the next, one at a time, so every level is in a
    // clean layout when it becomes the source of the following blit.
    for m in 0..mip_levels - 1 {
        let sw = (width >> m).max(1);
        let sh = (height >> m).max(1);
        let mut info = BlitImageInfo::images(texture_image.clone(), texture_image.clone());
        info.filter = Filter::Linear;
        let region = &mut info.regions[0];
        let mut src_layers = texture_image.subresource_layers();
        src_layers.mip_level = m;
        let mut dst_layers = texture_image.subresource_layers();
        dst_layers.mip_level = m + 1;
        region.src_subresource = src_layers;
        region.src_offsets = [[0, 0, 0], [sw, sh, 1]];
        region.dst_subresource = dst_layers;
        region.dst_offsets = [[0, 0, 0], [(sw >> 1).max(1), (sh >> 1).max(1), 1]];
        upload_builder.blit_image(info).expect("blit mip level");
    }

    let upload_cb = upload_builder
        .build()
        .expect("build texture upload command buffer");
    sync::now(queue.device().clone())
        .then_execute(queue, upload_cb)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    (
        ImageView::new_default(texture_image).expect("texture image view"),
        mip_levels,
    )
}
