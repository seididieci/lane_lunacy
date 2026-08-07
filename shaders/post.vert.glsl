#version 450
// SPDX-License-Identifier: MIT
// Fullscreen triangle for the post-processing pass. Emits a clip-space triangle
// from gl_VertexIndex (no vertex buffer) covering the whole framebuffer, plus
// matching UVs: (0,0) at the top-left of the framebuffer, matching the first
// texel of the offscreen scene image so the passthrough is identity.
layout(location = 0) out vec2 v_uv;
void main() {
    uint idx = gl_VertexIndex;
    vec2 pos = vec2(float((idx << 1) & 2u), float(idx & 2u));
    v_uv = pos;
    gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
}
