// SPDX-License-Identifier: MIT

use vulkano::buffer::BufferContents;

pub const MESH_VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spv/mesh.vert.spv"));
pub const MESH_FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spv/mesh.frag.spv"));
pub const HUD_VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spv/hud.vert.spv"));
pub const HUD_FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spv/hud.frag.spv"));

pub fn spv_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[derive(BufferContents, Clone, Copy, Debug)]
#[repr(C)]
pub struct MVP {
    pub model: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub projection: [[f32; 4]; 4],
    pub light_dir: [f32; 4],
}
