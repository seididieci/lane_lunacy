#version 450
// SPDX-License-Identifier: MIT
layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec4 v_color;
layout(location = 2) in float v_depth;
layout(location = 3) in float v_variant;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 1) uniform sampler2D sprite;

layout(set = 0, binding = 0) uniform MVP {
    mat4 model;
    mat4 view;
    mat4 projection;
    vec4 light_dir;
    vec4 fog_color;
};

// Horizontal sprite atlas: cell 0 = rain gaussian, cells 1..=3 = cloud shapes.
const float CELL_W = 0.25;

void main() {
    vec2 cell_uv = vec2(v_uv.x * CELL_W + v_variant * CELL_W, v_uv.y);
    vec4 tex = texture(sprite, cell_uv);
    vec3 col = tex.rgb * v_color.rgb;
    float alpha = tex.a * v_color.a;
    // Fade with the same fog ramp as the road so distant streaks melt into
    // the sky horizon instead of popping out.
    float fog = smoothstep(100.0, 600.0, v_depth);
    col = mix(col, fog_color.rgb, fog);
    alpha *= 1.0 - fog;
    f_color = vec4(col, alpha);
}
