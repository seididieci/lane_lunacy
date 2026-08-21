#version 460
#extension GL_EXT_ray_tracing : require
// SPDX-License-Identifier: MIT

// Ray generation for the ray-traced renderer. Fires one primary ray per pixel,
// then (for wet asphalt) a single reflected ray as a second sequential depth-0
// trace, so the pipeline needs no recursion and every result is fully shaded by
// the same closest-hit shader that colors the primary hit. The finished color
// lands in the storage image `rt_out`, which the recorder then copies into the
// offscreen color target so bloom + post + HUD keep working unchanged.

struct RTShade {
    vec4 color;
    vec4 normal;
    vec4 world_pos;
    vec4 albedo;
    vec4 uv;
    vec4 extra;
};

// Single 96-byte `RTShade` payload shared by the primary, reflected and shadow
// probe rays (hit/miss stages declare the same struct). The shadow probe reuses
// the payload as a 1-float channel: the miss shader writes 1.0 (unoccluded) /
// the shadow any-hit writes 0.0 into `albedo.x`, and the raygen reads it back
// right after the probe trace.
layout(location = 0) rayPayloadEXT RTShade rtp;

layout(set = 0, binding = 0, std140) uniform RtUniforms {
    mat4 inv_view_proj;
    vec4 eye;
    vec4 fog_color;
    vec4 zenith;
    vec4 horizon;
    vec4 cloud_tint;
    vec4 light_dir;
    float cloud_amount;
    float _pad1;
    float _pad2;
    float _pad3;
    vec4 sun_state;
    float time;
    float _pad4;
    float _pad5;
    float _pad6;
};

layout(set = 0, binding = 2, rgba16f) uniform image2D rt_out;

// Primary-ray depth as linear eye distance (metres), used by the RT particle
// overlay pass to occlude rain/mist/dust behind geometry. Written per pixel,
// compared in the same metres space the particle quads use, so occlusion stays
// exact at every range (NDC depth would collapse near `far`).
layout(set = 0, binding = 10, r32f) uniform image2D depth_out;

layout(set = 1, binding = 0) uniform accelerationStructureEXT tlas;

// Ray-traced sun/moon shadows. The probe traces toward `light_dir` with a cull
// mask that only chunk (world) instances carry, so the car statics never cast.
// The probe is gated on the sun/moon elevation (`sun_state.x`) so it only runs
// while the light actually contributes direct lighting.
// Cull mask for the shadow probes: matches `CHUNK_INSTANCE_MASK 0x01` in
// raytrace.rs (bit 0 set). The car statics clear bit 0 (`0xfe`), so shadow rays
// only ever touch world geometry.
const int SHADOW_RAY_MASK = 0x01;
// Angular soften for the shadow edge: the probe ray jitters within a small disc
// perpendicular to the sun ray, so the per-pixel result lands somewhere on a
// soft penumbra instead of a hard binary edge. The 2× supersampled dispatch
// (SUPERSAMPLE=2 in raytrace.rs) blends neighbouring pixels further, so edges
// read soft without extra rays.
const float SHADOW_SOFT_RADIUS = 0.006;
// Deterministic per-pixel hash so the jitter is stable frame to frame.
float shadow_hash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// One shadow probe: 1.0 when the sun/moon reaches `pos`, 0.0 when chunk
// geometry blocks it. The cull mask strips the car statics (their instance mask
// clears bit 0), so player/traffic cars never cast a shadow. Result is carried
// in `rtp.albedo.x` (the probe is the last trace before `main` reads it back).
//
// SBT indexing note: vulkano's `ShaderBindingTable` re-packs the pipeline groups
// into per-section tables, so records are NOT addressed by group index. With our
// group order [0 rgen | 1 rmiss, 2 shadow-miss | 3 rchit, 4 shadow-anyhit] the
// miss section is [rmiss=0, shadow-miss=1] and the hit section is
// [rchit=0, shadow-anyhit=1]. Every TLAS instance leaves its SBT record offset
// at 0, so `sbtRecordOffset` selects the hit record directly.
float shadow_factor(vec3 pos, vec3 normal, vec2 jitter) {
    vec3 l = normalize(light_dir.xyz);
    vec3 t = normalize(cross(normal, l));
    vec3 b = normalize(cross(t, l));
    vec3 soft = normalize(l + (t * jitter.x + b * jitter.y) * SHADOW_SOFT_RADIUS);
    rtp.albedo = vec4(1.0, 0.0, 0.0, 0.0);
    rtp.color = vec4(0.0);
    traceRayEXT(
        tlas,
        gl_RayFlagsSkipClosestHitShaderEXT,
        SHADOW_RAY_MASK,      // cull statics (cars never cast shadows)
        1,                    // hit record 1 in the hit section = shadow any-hit
        1,                    // stride (instances use SBT record offset 0)
        1,                    // miss record 1 in the miss section = shadow miss
        pos + normal * 0.05,
        0.001,
        soft,
        2000.0,
        0);                   // payload location 0 (the shared RTShade)
    return rtp.albedo.x;
}

void main() {
    uvec3 launch = gl_LaunchIDEXT;
    uvec3 size = gl_LaunchSizeEXT;
    // NDC in Vulkan clip space: the app's `perspective_vulkan` already negates
    // the projection y axis (y-down NDC), so framebuffer row 0 (top) maps to -1.
    vec2 ndc = vec2(
        2.0 * ((float(launch.x) + 0.5) / float(size.x)) - 1.0,
        2.0 * ((float(launch.y) + 0.5) / float(size.y)) - 1.0);
    vec4 far_pt = inv_view_proj * vec4(ndc, 1.0, 1.0);
    vec3 dir = normalize(far_pt.xyz / far_pt.w - eye.xyz);

    // Per-pixel ray divergence (angular size of one pixel) — the exact analogue
    // of the raster path's screen-space derivatives. The hit shader divides the
    // world-space pixel footprint by this (via the surface's incidence cosine)
    // to pick the texture mip level, so RT filtering matches raster on any
    // surface orientation instead of a ground-only heuristic.
    vec4 far_x = inv_view_proj * vec4(ndc + vec2(2.0 / float(size.x), 0.0), 1.0, 1.0);
    vec3 dir_x = normalize(far_x.xyz / far_x.w - eye.xyz);
    vec4 far_y = inv_view_proj * vec4(ndc + vec2(0.0, 2.0 / float(size.y)), 1.0, 1.0);
    vec3 dir_y = normalize(far_y.xyz / far_y.w - eye.xyz);
    float pixel_angle = max(length(cross(dir, dir_x)), length(cross(dir, dir_y)));
    pixel_angle = max(pixel_angle, 1e-6);

    rtp.color = vec4(1.0, 0.0, 1.0, 1.0);
    rtp.extra = vec4(0.0, 0.0, pixel_angle, 0.0);
    traceRayEXT(tlas, gl_RayFlagsOpaqueEXT, 0xFF, 0, 0, 0, eye.xyz, 0.001, dir, 2000.0, 0);

    vec4 primary_col = rtp.color;
    vec4 primary_normal = rtp.normal;
    vec4 primary_pos = rtp.world_pos;
    vec4 primary_albedo = rtp.albedo;
    float reflectivity = rtp.extra.x;

    // Ray-traced sun/moon shadow: one probe per pixel toward the light, culled
    // to chunk instances only (the car statics clear the shadow bit, so they
    // never cast). The probe only fires when the sun/moon actually contributes
    // light (`shadow_vis` stays 1.0 at night, and the diffuse term is ~0 then).
    float shadow_vis = 1.0;
    vec2 jitter = vec2(
        2.0 * shadow_hash(gl_LaunchIDEXT.xy) - 1.0,
        2.0 * shadow_hash(gl_LaunchIDEXT.xy + vec2(5.3, 7.1)) - 1.0);
    if (primary_col.w > 0.5 && sun_state.x > 0.001) {
        shadow_vis = shadow_factor(primary_pos.xyz, primary_normal.xyz, jitter);
    }

    if (primary_col.w > 0.5 && reflectivity > 0.0) {
        rtp.color = vec4(0.0);
        rtp.normal = vec4(0.0);
        rtp.world_pos = vec4(0.0);
        rtp.albedo = vec4(0.0);
        rtp.uv = vec4(0.0);
        rtp.extra = vec4(0.0, 0.0, pixel_angle, 0.0);
        vec3 refl_dir = reflect(dir, primary_normal.xyz);
        vec3 origin = primary_pos.xyz + primary_normal.xyz * 0.01;
        traceRayEXT(tlas, gl_RayFlagsOpaqueEXT, 0xFF, 0, 0, 0, origin, 0.001, refl_dir, 2000.0, 0);
        // The reflected surface is shadowed exactly like a primary hit: the cull
        // mask strips the car statics, so reflected world geometry (trees/walls)
        // staying in shade crosses into the puddle as shade too.
        float refl_vis = 1.0;
        if (rtp.color.w > 0.5 && sun_state.x > 0.001) {
            refl_vis = shadow_factor(rtp.world_pos.xyz, rtp.normal.xyz, jitter + 0.5);
        }
        vec3 reflected = rtp.color.rgb;
        vec3 refl_sun = rtp.albedo.rgb;
        vec3 refl_shaded = reflected - refl_sun + refl_sun * refl_vis;
        // Mirror the raster composite's puddle blend: mix at `blend` then
        // darken by `1 - 0.22 * blend` so puddles read wet, not glowing.
        float blend = reflectivity;
        vec3 base = primary_col.rgb - primary_albedo.rgb + primary_albedo.rgb * shadow_vis;
        rtp.color.rgb = mix(base, refl_shaded, blend);
        rtp.color.rgb *= 1.0 - 0.22 * blend;
    } else {
        rtp.color.rgb = primary_col.rgb - primary_albedo.rgb + primary_albedo.rgb * shadow_vis;
    }

    imageStore(rt_out, ivec2(launch.xy), vec4(rtp.color.rgb, 1.0));

    // Occlusion depth for the particle overlay: linear distance from the eye
    // to the primary hit. A miss (sky) stores a far sentinel so particles
    // always draw over open sky.
    float occl_d = length(primary_pos.xyz - eye.xyz);
    if (primary_col.w < 0.5) {
        occl_d = 1e30;
    }
    imageStore(depth_out, ivec2(launch.xy), vec4(occl_d, 0.0, 0.0, 1.0));
}
