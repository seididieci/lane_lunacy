#version 450
// SPDX-License-Identifier: MIT

layout(location = 4) in float v_material;
layout(location = 5) in vec3 v_world_pos;
layout(location = 0) out vec4 f_color;

const float ROAD_HALF = 4.8;
const float SHOULDER_W = 0.55;

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float value_noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    vec2 a = i + vec2(0.0, 0.0);
    vec2 b = i + vec2(1.0, 0.0);
    vec2 c = i + vec2(0.0, 1.0);
    vec2 d = i + vec2(1.0, 1.0);
    float v = mix(
        mix(hash12(a * 7.13 + vec2(1.7, 3.1)), hash12(b * 7.13 + vec2(1.7, 3.1)), u.x),
        mix(hash12(c * 7.13 + vec2(1.7, 3.1)), hash12(d * 7.13 + vec2(1.7, 3.1)), u.x),
        u.y);
    return v;
}

float puddle_noise(vec2 p) {
    float amp = 0.55;
    float v = 0.0;
    float norm = 0.0;
    vec2 q = p;
    for (int i = 0; i < 2; ++i) {
        v += amp * value_noise(q);
        norm += amp;
        q = q * 2.13 + vec2(7.3, 3.7);
        amp *= 0.55;
    }
    return v / norm;
}

float road_center_x(float s) {
    return 12.0 * sin(s * 0.02);
}

float road_lateral(float x, float s) {
    return x - road_center_x(s);
}

float road_surface_height(float s, float lat) {
    float d = abs(lat);
    if (d <= ROAD_HALF) {
        return 0.015;
    }
    if (d <= ROAD_HALF + SHOULDER_W) {
        return 0.021;
    }
    return 0.0;
}

float static_puddle_mask(vec3 world_pos, float material) {
    // Asphalt-only: world atlas slots 0..2.
    if (material < 0.0 || material >= 3.0) {
        return 0.0;
    }
    float s = -world_pos.z;
    float lat = road_lateral(world_pos.x, s);
    float half_road = ROAD_HALF + SHOULDER_W;
    if (abs(lat) > half_road) {
        return 0.0;
    }
    float road_y = road_surface_height(s, lat);
    if (abs(world_pos.y - road_y) > 0.065) {
        return 0.0;
    }

    vec2 q = vec2(s * 0.11, lat * 0.45);
    vec2 w = 0.35 * vec2(
        puddle_noise(q + vec2(0.0, 1.7)),
        puddle_noise(q + vec2(5.3, 2.9)));
    float n = puddle_noise(q + w);
    float pat = smoothstep(0.48, 0.60, n);
    if (pat <= 0.001) {
        return 0.0;
    }
    float edge = smoothstep(half_road, half_road - 0.7, abs(lat));
    return clamp(pat * edge, 0.0, 1.0);
}

void main() {
    float m = static_puddle_mask(v_world_pos, v_material);
    f_color = vec4(m, 0.0, 0.0, 1.0);
}
