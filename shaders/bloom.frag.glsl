#version 450
// SPDX-License-Identifier: MIT
// Bloom downsample pass: bilinear 2x2 average into a half-resolution image.
// Applied repeatedly (half -> quarter -> eighth) each cheap blur builds softness
// for the bloom composite. The first pass gates the source by a soft-knee on
// luminance so only bright sources (sun, headlights, taillights) feed the glow;
// later passes just average the gated image down.
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform sampler2D src;

layout(set = 0, binding = 1) uniform BloomParams {
    float threshold;
    float knee;
    uint first_pass;
    float _pad0;
};

const vec3 LUMA = vec3(0.299, 0.587, 0.114);

void main() {
    vec2 texel = 1.0 / textureSize(src, 0);
    vec2 o = texel * 0.5;
    vec4 a = texture(src, v_uv - o);
    vec4 b = texture(src, v_uv + vec2(o.x, -o.y));
    vec4 c = texture(src, v_uv + vec2(-o.x, o.y));
    vec4 d = texture(src, v_uv + o);
    vec4 color = (a + b + c + d) * 0.25;

    if (first_pass != 0u) {
        float l = dot(color.rgb, LUMA);
        color.rgb *= smoothstep(threshold - knee, threshold + knee, l);
    }

    f_color = vec4(color.rgb, 1.0);
}
