#version 450
// SPDX-License-Identifier: MIT
layout(location = 0) in vec3 v_dir;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform SkyUniform {
    mat4 model;
    mat4 view;
    mat4 projection;
    float time;
    vec4 zenith;
    vec4 horizon;
    vec4 cloud_tint;
    vec4 light_dir;
    float cloud_amount;
};

layout(set = 0, binding = 1) uniform sampler2D clouds_a;
layout(set = 0, binding = 2) uniform sampler2D clouds_b;

void main() {
    vec3 dir = normalize(v_dir);
    float t = clamp(dir.y, 0.0, 1.0);
    float amount = cloud_amount;

    // Spherical mapping: horizontal wraps around seamlessly.
    float u = atan(dir.x, dir.z) * (1.0 / 6.28318530718) + 0.5;
    float v = t;

    // Two cloud layers scroll independently so the sky drifts over time.
    vec2 uv_a = vec2(u + time * 0.0017, v + time * 0.00015);
    vec2 uv_b = vec2(u - time * 0.0012 + 0.37, v + time * 0.00010);

    // Layer A = cumulus (lower, denser), Layer B = cirrus (high, thin wisps).
    // The horizontal repeat must be an integer number of tile wraps across the
    // full azimuth, or the pattern is discontinuous at the u=0/1 seam.
    float raw_a = texture(clouds_a, uv_a * vec2(2.0, 1.6)).a;
    float raw_b = texture(clouds_b, uv_b * 4.0).a;

    // ---- Weather mood ----
    // Threshold curve: below ~0.10 cloud_amount the sky stays genuinely clear,
    // then coverage builds smoothly toward a full wall-out. Drives both the
    // gradient and the cloud mass.
    float cover = smoothstep(0.10, 1.0, amount);

    // Base gradient reacts to weather: bright golden when clear, a desaturated
    // grey overcast when clouded over, and dimmed overall by the cloud cover.
    vec3 sky_clear = mix(horizon.rgb, zenith.rgb, smoothstep(0.0, 0.28, t));
    vec3 oc_horizon = vec3(0.60, 0.60, 0.63);
    vec3 oc_zenith = vec3(0.28, 0.31, 0.38);
    vec3 sky_overcast = mix(oc_horizon, oc_zenith, smoothstep(0.0, 0.4, t));
    vec3 sky = mix(sky_clear, sky_overcast, cover);
    sky *= mix(1.0, 0.78, cover);

    // In clear weather the camera only sees a narrow upper-horizon slice; if
    // thresholds are too permissive, one unlucky tile bank can look like a
    // single giant cloud blanket. Tighten the cut in clear and open it up as
    // weather gets heavier.
    float a_lo = mix(0.58, 0.12, cover);
    float a_hi = mix(0.86, 0.70, cover);
    float b_lo = mix(0.50, 0.10, cover);
    float b_hi = mix(0.82, 0.65, cover);
    float cov_a = smoothstep(a_lo, a_hi, raw_a);
    float cov_b = smoothstep(b_lo, b_hi, raw_b);

    // ---- Cloud mass ----
    // CLEAR keeps mostly wisps; CLOUDY/RAIN progressively unlock denser low
    // cloud mass. This avoids the "one always-on cloud" look in CLEAR while
    // preserving dramatic overcast in RAIN.
    float horizon_band = 1.0 - smoothstep(0.25, 1.0, t);
    float low_cloud_weight = smoothstep(0.20, 0.65, cover);
    float ca = cov_a * mix(0.01, 1.0, cover) * mix(0.5, 1.0, horizon_band)
        * mix(0.10, 1.0, low_cloud_weight);
    float cb = cov_b * mix(0.03, 0.45, cover) * mix(0.9, 0.4, t);
    float cloud_alpha = min(ca + cb, mix(0.16, 0.88, cover));

    // Sunlit clouds glow near-white against the warm sky so they read clearly;
    // under overcast they shift to dark slate masses.
    vec3 warm = mix(vec3(1.0, 0.97, 0.90), cloud_tint.rgb, 0.25);
    vec3 overcast = vec3(0.34, 0.37, 0.44);
    vec3 cloud_col = mix(warm, overcast, cover);
    cloud_col = mix(cloud_col, zenith.rgb, smoothstep(0.05, 0.35, t) * 0.3);
    vec3 col = mix(sky, cloud_col, cloud_alpha);

    // Golden halo around the sun, muted by cloud cover.
    vec3 sun = normalize(light_dir.xyz);
    float sun_h = sun.y;
    float glow = pow(max(dot(dir, sun), 0.0), 6.0);
    glow *= smoothstep(0.0, 0.25, 1.0 - abs(t - sun_h));
    glow *= mix(1.0, 0.12, cover);
    vec3 glow_col = mix(horizon.rgb, vec3(1.0, 0.82, 0.55), 0.5);
    col += glow * glow_col * 0.18;

    // Soft haze toward the horizon so the dome edge reads cleanly.
    vec3 haze = mix(horizon.rgb, oc_horizon, cover);
    col = mix(col, haze, (1.0 - t) * 0.12);

    f_color = vec4(col, 1.0);
}
