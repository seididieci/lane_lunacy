#version 450
// SPDX-License-Identifier: MIT
// Post-processing composite pass. Reads the HDR offscreen scene image (and, when
// BLOOM is on, the lowest-downsampled bloom image) and applies the enabled FX
// chain. All effects are gated by bits in the PostSettings `flags` uniform, so
// with everything off this is an identity passthrough.
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform PostSettings {
    uint flags;
    float time;
    float vignette_strength;
    float grain_amount;
    float saturation_boost;
    float bloom_strength;
    float chroma_strength;
    float texel_x;
    float texel_y;
    float _pad0;
    float _pad1;
    float _pad2;
};

layout(set = 0, binding = 1) uniform sampler2D scene;
layout(set = 0, binding = 2) uniform sampler2D bloom;

const uint FLAG_FXAA = 1u << 0;
const uint FLAG_BLOOM = 1u << 1;
const uint FLAG_VIGNETTE = 1u << 2;
const uint FLAG_GRAIN = 1u << 3;
const uint FLAG_SATURATION = 1u << 4;
const uint FLAG_CHROMA = 1u << 5;

const vec3 LUMA = vec3(0.299, 0.587, 0.114);

float luma(vec3 c) {
    return dot(c, LUMA);
}

// Deterministic hash for film grain; animates with `time`.
float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// FXAA-style edge blending: on a detected edge, blend the center sample toward
// its two neighbors across the dominant edge direction.
vec3 fxaa(vec3 color, vec2 uv) {
    vec2 inv = vec2(texel_x, texel_y);
    vec3 e = texture(scene, uv + vec2(-1.0, -1.0) * inv).rgb;
    vec3 f = texture(scene, uv + vec2(0.0, -1.0) * inv).rgb;
    vec3 g = texture(scene, uv + vec2(1.0, -1.0) * inv).rgb;
    vec3 b = texture(scene, uv + vec2(-1.0, 0.0) * inv).rgb;
    vec3 d = texture(scene, uv + vec2(1.0, 0.0) * inv).rgb;
    vec3 h = texture(scene, uv + vec2(-1.0, 1.0) * inv).rgb;
    vec3 i = texture(scene, uv + vec2(0.0, 1.0) * inv).rgb;
    vec3 j = texture(scene, uv + vec2(1.0, 1.0) * inv).rgb;

    float lC = luma(color);
    float lE = luma(e);
    float lF = luma(f);
    float lG = luma(g);
    float lB = luma(b);
    float lD = luma(d);
    float lH = luma(h);
    float lI = luma(i);
    float lJ = luma(j);

    // Horizontal vs vertical edge energy from the middle row / middle column.
    float edgeH = abs(lB + lD - 2.0 * lC) + abs(lE + lG - 2.0 * lF) + abs(lH + lJ - 2.0 * lI);
    float edgeV = abs(lF + lI - 2.0 * lC) + abs(lE + lH - 2.0 * lB) + abs(lG + lJ - 2.0 * lD);
    float edge = edgeH + edgeV;
    if (edge < 0.01) {
        return color;
    }

    // dir -> 1 blends horizontally (across a vertical edge), dir -> 0 blends
    // vertically (across a horizontal edge).
    float dir = 0.5 + 0.5 * clamp((edgeV - edgeH) / max(edge, 1e-6), -1.0, 1.0);
    vec3 blend = mix((f + i) * 0.5, (b + d) * 0.5, dir);
    float alpha = clamp(edge, 0.0, 0.75) * 0.35;
    return mix(color, blend, alpha);
}

void main() {
    vec2 uv = v_uv;
    vec3 color = texture(scene, uv).rgb;

    if ((flags & FLAG_CHROMA) != 0u) {
        // Chromatic aberration: shift red/blue samples radially from center.
        vec2 dir = (uv - 0.5) * 2.0;
        vec2 off = dir * chroma_strength;
        float r = texture(scene, uv + off).r;
        float bl = texture(scene, uv - off).b;
        color = vec3(r, color.g, bl);
    }

    if ((flags & FLAG_FXAA) != 0u) {
        color = fxaa(color, uv);
    }

    if ((flags & FLAG_BLOOM) != 0u) {
        color += texture(bloom, uv).rgb * bloom_strength;
    }

    if ((flags & FLAG_SATURATION) != 0u) {
        float l = luma(color);
        color = mix(vec3(l), color, saturation_boost);
    }

    if ((flags & FLAG_VIGNETTE) != 0u) {
        vec2 ndc = (uv - 0.5) * 2.0;
        float d = dot(ndc, ndc) * 0.5;
        color *= 1.0 - vignette_strength * smoothstep(0.4, 1.6, d);
    }

    if ((flags & FLAG_GRAIN) != 0u) {
        float n = hash12(uv * 1920.0 + vec2(time * 17.0, time * 13.0));
        color += (n - 0.5) * grain_amount;
    }

    f_color = vec4(color, 1.0);
}
