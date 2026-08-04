// SPDX-License-Identifier: MIT

use vulkano::buffer::BufferContents;
use vulkano::pipeline::graphics::vertex_input::Vertex;

#[derive(BufferContents, Vertex, Default, Clone, Copy, Debug)]
#[repr(C)]
pub struct Vertex3d {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub normal: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub color: [f32; 3],
    #[format(R32G32_SFLOAT)]
    pub tex_coord: [f32; 2],
    #[format(R32_SFLOAT)]
    pub material: f32,
}

#[derive(BufferContents, Vertex, Default, Clone, Copy, Debug)]
#[repr(C)]
pub struct HudVertex {
    #[format(R32G32_SFLOAT)]
    pub position: [f32; 2],
    #[format(R32G32B32A32_SFLOAT)]
    pub color: [f32; 4],
    #[format(R32G32_SFLOAT)]
    pub uv: [f32; 2],
}
