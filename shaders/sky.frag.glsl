#version 450
// SPDX-License-Identifier: MIT
layout(location = 0) in vec3 v_dir;
layout(location = 0) out vec4 f_color;

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

layout(set = 0, binding = 1) uniform sampler2D clouds_a;
layout(set = 0, binding = 2) uniform sampler2D clouds_b;

void main() {
    vec3 dir = normalize(v_dir);
    float t = clamp(dir.y, 0.0, 1.0);

    // Golden-hour gradient: warm band hugging the horizon, cooler zenith.
    vec3 sky = mix(horizon.rgb, zenith.rgb, smoothstep(0.0, 0.28, t));

    // Spherical mapping: horizontal wraps around seamlessly.
    float u = atan(dir.x, dir.z) * (1.0 / 6.28318530718) + 0.5;
    float v = t;

    // Two cloud layers scroll independently so the sky drifts over time.
    float amount = cloud_amount;

    // Layer A = cumulus (lower, denser), Layer B = cirrus (high, thin wisps).
    vec2 uv_a = vec2(u + time * 0.0017, v + time * 0.00015);
    vec2 uv_b = vec2(u - time * 0.0012 + 0.37, v + time * 0.00010);

    float cov_a = texture(clouds_a, uv_a * 2.2).a;
    float cov_b = texture(clouds_b, uv_b * 5.0).a;
    cov_a = smoothstep(0.12, 0.7, cov_a);
    cov_b = smoothstep(0.1, 0.65, cov_b);

    // Coverage scales with weather (clear/partly/dramatic), capped so the sky
    // can never wall out.
    float horizon_band = 1.0 - smoothstep(0.25, 1.0, t);
    float ca = cov_a * mix(0.10, 0.48, amount) * mix(0.35, 1.0, horizon_band);
    float cb = cov_b * mix(0.05, 0.20, amount) * mix(0.8, 0.4, t);
    float cloud_alpha = min(ca + cb, 0.55);

    // Sunlit low clouds warm gold, high wisps cool.
    vec3 cool = mix(cloud_tint.rgb, zenith.rgb, 0.45);
    vec3 cloud_col = mix(cloud_tint.rgb, cool, smoothstep(0.05, 0.35, t));
    vec3 col = mix(sky, cloud_col, cloud_alpha);

    // Subtle golden halo around the sun direction, strongest near the horizon.
    vec3 sun = normalize(light_dir.xyz);
    float sun_h = sun.y;
    float glow = pow(max(dot(dir, sun), 0.0), 6.0);
    glow *= smoothstep(0.0, 0.25, 1.0 - abs(t - sun_h));
    vec3 glow_col = mix(horizon.rgb, vec3(1.0, 0.82, 0.55), 0.5);
    col += glow * glow_col * 0.18;

    // Soft haze toward the horizon so the dome edge reads cleanly.
    col = mix(col, horizon.rgb, (1.0 - t) * 0.14);

    f_color = vec4(col, 1.0);
}
