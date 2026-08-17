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
    /// Camera eye position in world space (wet-road specular in `mesh.frag`).
    /// Kept after `fog_color` so the particle shaders' shorter MVP block (which
    /// reads only up to `fog_color`) keeps its offsets.
    pub camera_pos: [f32; 4],
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
/// Camera rain-droplet lens effect (wet lens), gated by `wet_fac`.
pub const POST_RAINDROPS: u32 = 1 << 6;
/// Screen-space puddle reflections on the wet asphalt, gated by `wet_fac`.
pub const POST_REFLECT: u32 = 1 << 7;

/// UBO for the post-processing pass. `flags` gates each effect; the float
/// factors are the fixed per-effect intensities; `texel_x/y` are the inverse
/// framebuffer size (for FXAA/chroma); `time` drives the animated grain and
/// the rain droplets; `wet_fac` drives the wet-lens droplets and the puddle
/// reflections. `inv_view_proj`, `eye` and `fog_color` feed the screen-space
/// reflection pass (world-position reconstruction from the depth attachment).
///
/// Layout must mirror the `PostSettings` block in `post.frag.glsl` exactly
/// (std140): the scalar fields pad to the 16-byte alignment the `mat4` needs.
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
    pub wet_fac: f32,
    /// Puddle-reflection quality uniform: 0 = off, 1 = low, 2 = high. Lives in
    /// what used to be padding, so the std140 layout is unchanged.
    pub puddle_quality: f32,
    pub _pad: [f32; 5],
    /// Inverse of (projection * view): maps a depth sample back to world space.
    pub inv_view_proj: [[f32; 4]; 4],
    /// (projection * view): projects SSR ray samples back to screen space.
    pub view_proj: [[f32; 4]; 4],
    /// Camera eye position in world space (ray origin for reflections).
    pub eye: [f32; 4],
    /// Horizon fog color (SSR miss fallback and far-fade tint).
    pub fog_color: [f32; 4],
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
