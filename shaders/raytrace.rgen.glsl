#version 460
#extension GL_EXT_ray_tracing : require
// SPDX-License-Identifier: MIT

// Ray generation for the ray-traced renderer. Fires one primary ray per pixel,
// then (for wet asphalt) a single reflected ray as a second sequential depth-0
// trace, so the pipeline needs no recursion and every result is fully shaded by
// the same closest-hit shader that colors the primary hit. The finished color
// lands in the storage image `rt_out`, which the recorder then copies into the
// offscreen color target so bloom + post + HUD keep working unchanged.

layout(location = 0) rayPayloadEXT vec4 rtp;

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

layout(set = 1, binding = 0) uniform accelerationStructureEXT tlas;

void main() {
    uvec3 launch = gl_LaunchIDEXT;
    uvec3 size = gl_LaunchSizeEXT;
    // NDC with y up (the app's perspective_vulkan negates the projection y
    // axis), so framebuffer row 0 (top) maps to +1.
    vec2 ndc = vec2(
        2.0 * ((float(launch.x) + 0.5) / float(size.x)) - 1.0,
        1.0 - 2.0 * ((float(launch.y) + 0.5) / float(size.y)));
    vec4 far_pt = inv_view_proj * vec4(ndc, 1.0, 1.0);
    vec3 dir = normalize(far_pt.xyz / far_pt.w - eye.xyz);

    rtp = vec4(1.0, 0.0, 1.0, 1.0);
    traceRayEXT(tlas, gl_RayFlagsOpaqueEXT, 0xFF, 0, 0, 0, eye.xyz, 0.001, dir, 2000.0, 0);

    imageStore(rt_out, ivec2(launch.xy), rtp);
}
