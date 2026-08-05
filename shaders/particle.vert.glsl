#version 450
// SPDX-License-Identifier: MIT
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec4 color;

layout(location = 0) out vec2 v_uv;
layout(location = 1) out vec4 v_color;
layout(location = 2) out float v_depth;

layout(set = 0, binding = 0) uniform MVP {
    mat4 model;
    mat4 view;
    mat4 projection;
    vec4 light_dir;
    vec4 fog_color;
};

void main() {
    v_uv = uv;
    v_color = color;
    vec4 view_pos = view * vec4(position, 1.0);
    v_depth = -view_pos.z;
    gl_Position = projection * view_pos;
}
