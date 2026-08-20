#version 460
#extension GL_EXT_ray_tracing : require
// SPDX-License-Identifier: MIT

// Shadow any-hit shader (ray traced sun/moon shadows). Used only by the raygen's
// per-pixel shadow probes, which are traced with
// `GL_RAY_FLAG_SKIP_CLOSEST_HIT_SHADER_EXT` and a cull mask that only chunk
// (world) instances carry; the car statics are masked out so the probes never
// even intersect them. This shader therefore only sees world geometry and writes
// `albedo.x = 0.0` (occluded) before terminating on the very first such hit
// ("anything opaque between the point and the sun"). The miss shader
// (`raytrace.rsmiss.glsl`) reports a clean miss with `albedo.x = 1.0`.

struct RTShade {
    vec4 color;
    vec4 normal;
    vec4 world_pos;
    vec4 albedo;
    vec4 uv;
    vec4 extra;
};

rayPayloadInEXT RTShade rtp;

void main() {
    rtp.albedo.x = 0.0;
    terminateRayEXT;
}