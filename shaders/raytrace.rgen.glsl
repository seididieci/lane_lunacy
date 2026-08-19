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

    rtp.color = vec4(1.0, 0.0, 1.0, 1.0);
    rtp.extra = vec4(0.0);
    traceRayEXT(tlas, gl_RayFlagsOpaqueEXT, 0xFF, 0, 0, 0, eye.xyz, 0.001, dir, 2000.0, 0);

    vec4 primary_col = rtp.color;
    vec4 primary_normal = rtp.normal;
    vec4 primary_pos = rtp.world_pos;
    float reflectivity = rtp.extra.x;

    if (primary_col.w > 0.5 && reflectivity > 0.0) {
        rtp.color = vec4(0.0);
        rtp.normal = vec4(0.0);
        rtp.world_pos = vec4(0.0);
        rtp.albedo = vec4(0.0);
        rtp.uv = vec4(0.0);
        rtp.extra = vec4(0.0);
        vec3 refl_dir = reflect(dir, primary_normal.xyz);
        vec3 origin = primary_pos.xyz + primary_normal.xyz * 0.01;
        traceRayEXT(tlas, gl_RayFlagsOpaqueEXT, 0xFF, 0, 0, 0, origin, 0.001, refl_dir, 2000.0, 0);
        // Mirror the raster composite's puddle blend: mix at `blend` then
        // darken by `1 - 0.22 * blend` so puddles read wet, not glowing.
        float blend = reflectivity;
        rtp.color.rgb = mix(primary_col.rgb, rtp.color.rgb, blend);
        rtp.color.rgb *= 1.0 - 0.22 * blend;
    } else {
        rtp.color = primary_col;
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
