#version 450
// SPDX-License-Identifier: MIT
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;
layout(location = 3) in vec2 tex_coord;

layout(location = 0) out vec3 v_normal;
layout(location = 1) out vec3 v_color;
layout(location = 2) out vec2 v_uv;
layout(location = 3) out float v_depth;

layout(set = 0, binding = 0) uniform MVP {
    mat4 model;
    mat4 view;
    mat4 projection;
    vec4 light_dir;
};

void main() {
    v_normal = mat3(model) * normal;
    v_color = color;
    v_uv = tex_coord;
    vec4 view_pos = view * model * vec4(position, 1.0);
    v_depth = -view_pos.z;
    gl_Position = projection * view_pos;
}
