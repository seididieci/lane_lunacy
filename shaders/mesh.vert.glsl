#version 450
// SPDX-License-Identifier: MIT
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;
layout(location = 3) in vec2 tex_coord;
layout(location = 4) in float material;

layout(location = 0) out vec3 v_normal;
layout(location = 1) out vec3 v_color;
layout(location = 2) out vec2 v_uv;
layout(location = 3) out float v_depth;
layout(location = 4) out float v_material;
layout(location = 5) out vec3 v_world_pos;

layout(set = 0, binding = 0) uniform MVP {
    mat4 model;
    mat4 view;
    mat4 projection;
    vec4 light_dir;
    vec4 fog_color;
    vec4 light_state;
    vec4 headlight_pos;
    vec4 headlight_dir;
};

void main() {
    v_normal = mat3(model) * normal;
    v_color = color;
    v_uv = tex_coord;
    v_material = material;
    vec4 world_pos = model * vec4(position, 1.0);
    v_world_pos = world_pos.xyz;
    vec4 view_pos = view * world_pos;
    v_depth = -view_pos.z;
    gl_Position = projection * view_pos;
}
