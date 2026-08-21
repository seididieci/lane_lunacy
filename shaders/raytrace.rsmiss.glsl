#version 460
#extension GL_EXT_ray_tracing : require
// SPDX-License-Identifier: MIT

// Shadow-ray miss shader (run only by the raygen's per-pixel sun/moon shadow
// probes). The probe's cull mask strips the car statics, so a miss means the
// point is NOT occluded by world geometry (walls, trees, terrain): this shader
// writes `albedo.x = 1.0` as the probe's "unoccluded" reading. The shadow
// any-hit shader (`raytrace.rshad.glsl`) writes `0.0` and terminates on the
// first chunk hit, and the raygen reads `rtp.albedo.x` back.

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
    rtp.albedo.x = 1.0;
}