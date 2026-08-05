#version 450
// SPDX-License-Identifier: MIT
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;
layout(location = 3) in vec2 tex_coord;
layout(location = 4) in float material;

layout(location = 0) out vec3 v_dir;

layout(set = 0, binding = 0) uniform SkyUniform {
    mat4 model;
    mat4 view;
    mat4 projection;
    float time;
    vec3 padding;
    vec4 zenith;
    vec4 horizon;
    vec4 cloud_tint;
    vec4 light_dir;
    float cloud_amount;
    vec3 padding2;
};

void main() {
    v_dir = normalize(mat3(model) * position);
    vec4 world = model * vec4(position, 1.0);
    gl_Position = projection * view * world;
}
