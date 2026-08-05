#version 450
// SPDX-License-Identifier: MIT
layout(location = 0) in vec2 position;
layout(location = 1) in vec4 color;
layout(location = 2) in vec2 uv;
layout(location = 3) in float kind;
layout(location = 0) out vec4 v_color;
layout(location = 1) out vec2 v_uv;
layout(location = 2) out float v_kind;
void main() {
    v_color = color;
    v_uv = uv;
    v_kind = kind;
    // Same y-up NDC convention as the HUD (see hud.vert.glsl).
    gl_Position = vec4(position.x, -position.y, 0.0, 1.0);
}
