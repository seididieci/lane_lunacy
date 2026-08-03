#version 450
// SPDX-License-Identifier: MIT
layout(location = 0) in vec2 position;
layout(location = 1) in vec3 color;
layout(location = 2) in vec2 uv;
layout(location = 0) out vec3 v_color;
layout(location = 1) out vec2 v_uv;
void main() {
    v_color = color;
    v_uv = uv;
    gl_Position = vec4(position, 0.0, 1.0);
}
