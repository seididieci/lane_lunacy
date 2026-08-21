#version 450
// SPDX-License-Identifier: MIT
// Particle fragment shader for the ray-traced overlay pass. Same sprite/fog
// shading as `particle.frag.glsl`, plus a per-pixel occlusion test against the
// ray-traced primary depth: fragments whose distance from the eye is *behind*
// a closer RT hit are discarded, so rain/mist/dust no longer overdraw cars.
layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec4 v_color;
layout(location = 2) in float v_depth;
layout(location = 3) in float v_variant;
layout(location = 4) in vec3 v_view_pos;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 1) uniform sampler2D sprite;
layout(set = 0, binding = 2) uniform sampler2D rt_depth;

layout(set = 0, binding = 0) uniform MVP {
    mat4 model;
    mat4 view;
    mat4 projection;
    vec4 light_dir;
    vec4 fog_color;
};

// Horizontal sprite atlas: cell 0 = rain gaussian, cells 1..=3 = cloud shapes.
const float CELL_W = 0.25;

// Occlusion bias in metres. Keeps surface-resting particles (lamp glows on
// cars, mist hugging the road) from self-occluding, while still culling
// drops a few centimetres behind geometry.
const float OCCL_BIAS_M = 0.08;

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

    // Occlude against the ray-traced primary depth (linear eye distance).
    vec2 depth_uv = gl_FragCoord.xy / vec2(textureSize(rt_depth, 0));
    float occl_d = texture(rt_depth, depth_uv).r;
    float my_d = length(v_view_pos);
    if (occl_d < my_d - OCCL_BIAS_M) {
        discard;
    }

    f_color = vec4(col, alpha);
}
