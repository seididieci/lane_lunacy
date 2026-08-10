// SPDX-License-Identifier: MIT

use vulkano::buffer::BufferContents;

pub const MESH_VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spv/mesh.vert.spv"));
pub const MESH_FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spv/mesh.frag.spv"));
pub const HUD_VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spv/hud.vert.spv"));
pub const HUD_FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spv/hud.frag.spv"));
pub const SKY_VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spv/sky.vert.spv"));
pub const SKY_FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spv/sky.frag.spv"));
pub const PARTICLE_VERT_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/spv/particle.vert.spv"));
pub const PARTICLE_FRAG_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/spv/particle.frag.spv"));
pub const FLARE_VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spv/flare.vert.spv"));
pub const FLARE_FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spv/flare.frag.spv"));
pub const POST_VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spv/post.vert.spv"));
pub const POST_FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spv/post.frag.spv"));
pub const BLOOM_FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spv/bloom.frag.spv"));

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
    pub fog_color: [f32; 4],
    pub light_state: [f32; 4],
    pub headlight_pos: [f32; 4],
    pub headlight_dir: [f32; 4],
    pub traffic_head_pos: [[f32; 4]; 16],
    pub traffic_head_dir: [[f32; 4]; 16],
    pub traffic_head_state: [[f32; 4]; 16],
    /// Street-lamp projectors (downward warm pools). state = [warm.r, warm.g,
    /// warm.b, strength]. Mirrors `MAX_LAMPS` in `render/frame.rs`.
    pub lamp_pos: [[f32; 4]; 16],
    pub lamp_dir: [[f32; 4]; 16],
    pub lamp_state: [[f32; 4]; 16],
    /// Terrain (grass/verge) day/night tint: [tint.rgb, unused].
    pub terrain_state: [f32; 4],
}

#[derive(BufferContents, Clone, Copy, Debug)]
#[repr(C)]
pub struct SkyUniform {
    pub model: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub projection: [[f32; 4]; 4],
    pub time: f32,
    pub _pad: [f32; 3],
    pub zenith: [f32; 4],
    pub horizon: [f32; 4],
    pub cloud_tint: [f32; 4],
    pub light_dir: [f32; 4],
    pub cloud_amount: f32,
    pub sun_state: [f32; 4],
}

/// Per-FX bits for [`PostSettings::flags`]. Mirrors the `FLAG_*` consts in
/// `post.frag.glsl`.
pub const POST_FXAA: u32 = 1 << 0;
pub const POST_BLOOM: u32 = 1 << 1;
pub const POST_VIGNETTE: u32 = 1 << 2;
pub const POST_GRAIN: u32 = 1 << 3;
pub const POST_SATURATION: u32 = 1 << 4;
pub const POST_CHROMA: u32 = 1 << 5;

/// UBO for the post-processing pass. `flags` gates each effect; the float
/// factors are the fixed per-effect intensities; `texel_x/y` are the inverse
/// framebuffer size (for FXAA/chroma); `time` drives the animated grain.
#[derive(BufferContents, Clone, Copy, Debug)]
#[repr(C)]
pub struct PostSettings {
    pub flags: u32,
    pub time: f32,
    pub vignette_strength: f32,
    pub grain_amount: f32,
    pub saturation_boost: f32,
    pub bloom_strength: f32,
    pub chroma_strength: f32,
    pub texel_x: f32,
    pub texel_y: f32,
    pub _pad: [f32; 3],
}

/// Linear-HDR luminance gate for the bloom downsample pass. `threshold`/`knee`
/// define a soft knee applied only on the first downsample (`first_pass != 0`),
/// so only bright sources (sun, headlights, taillights) feed the glow while the
/// sky and road stay out of it. Mirrors `BloomParams` in `bloom.frag.glsl`.
#[derive(BufferContents, Clone, Copy, Debug)]
#[repr(C)]
pub struct BloomParams {
    pub threshold: f32,
    pub knee: f32,
    pub first_pass: u32,
    pub _pad: f32,
}
