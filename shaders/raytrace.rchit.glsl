#version 460
#extension GL_EXT_ray_tracing : require
// SPDX-License-Identifier: MIT

// Closest-hit shader for the ray-traced renderer. Rebuilds the triangle from
// the compact per-frame vertex pool (vec4-aligned, 3 vec4 per vertex: position
// / normal / uv+material), interpolates with the default hit attributes, then
// runs the exact same material + lighting rules as `mesh.frag.glsl`: world
// texture atlas vs the car colormap, terrain day/night tint, ambient + sun
// diffuse, wet-asphalt sheen, player headlight cone, oncoming traffic beams,
// street-lamp pools and the long fog ramp. The payload flags whether the
// surface reflects (wet asphalt) so the raygen can fire the secondary ray.

// Vertex pool caps. Must match the Rust constants in `render/raytrace.rs`.
#define VERT_CAP 14000000
#define INDEX_CAP 7000000
#define SLOT_CAP 256

layout(location = 0) rayPayloadEXT vec4 rtp;

layout(set = 0, binding = 1, std140) uniform MVP {
    mat4 model;
    mat4 view;
    mat4 projection;
    vec4 light_dir;
    vec4 fog_color;
    vec4 camera_pos;
    vec4 light_state;
    vec4 headlight_pos;
    vec4 headlight_dir;
    vec4 traffic_head_pos[16];
    vec4 traffic_head_dir[16];
    vec4 traffic_head_state[16];
    vec4 lamp_pos[16];
    vec4 lamp_dir[16];
    vec4 lamp_state[16];
    vec4 terrain_state;
    vec4 clip_plane;
};

struct RtSlot {
    uint vertex_base; // in vec4 units
    uint vertex_count;
    uint index_base;  // in uint units
    uint index_count;
};

layout(set = 0, binding = 3, std430) readonly buffer VertexBuf {
    vec4 verts[VERT_CAP];
};
layout(set = 0, binding = 4, std430) readonly buffer IndexBuf {
    uint indices[INDEX_CAP];
};
layout(set = 0, binding = 5, std430) readonly buffer SlotBuf {
    RtSlot slots[SLOT_CAP];
};

layout(set = 0, binding = 2, rgba16f) uniform image2D rt_out;

layout(set = 0, binding = 6) uniform sampler2D world_tex;
layout(set = 0, binding = 7) uniform sampler2D car_tex;

void main() {
    uint slot = gl_InstanceCustomIndexEXT;
    RtSlot s = slots[slot];
    uint vi = s.index_base + gl_PrimitiveID * 3u;
    uint i0 = indices[vi + 0u];
    uint i1 = indices[vi + 1u];
    uint i2 = indices[vi + 2u];

    // 3 vec4 per vertex, matching `render/raytrace.rs::pack`:
    //   verts[base + i*3 + 0] = position.xyz, uv.x
    //   verts[base + i*3 + 1] = normal.xyz,  material
    //   verts[base + i*3 + 2] = color.xyz,   uv.y
    vec3 p0 = verts[s.vertex_base + i0 * 3u + 0u].xyz;
    vec3 p1 = verts[s.vertex_base + i1 * 3u + 0u].xyz;
    vec3 p2 = verts[s.vertex_base + i2 * 3u + 0u].xyz;
    vec3 n0 = verts[s.vertex_base + i0 * 3u + 1u].xyz;
    vec3 n1 = verts[s.vertex_base + i1 * 3u + 1u].xyz;
    vec3 n2 = verts[s.vertex_base + i2 * 3u + 1u].xyz;
    vec3 c0 = verts[s.vertex_base + i0 * 3u + 2u].xyz;
    vec3 c1 = verts[s.vertex_base + i1 * 3u + 2u].xyz;
    vec3 c2 = verts[s.vertex_base + i2 * 3u + 2u].xyz;
    float uvx0 = verts[s.vertex_base + i0 * 3u + 0u].w;
    float uvx1 = verts[s.vertex_base + i1 * 3u + 0u].w;
    float uvx2 = verts[s.vertex_base + i2 * 3u + 0u].w;
    vec2 uv0 = vec2(uvx0, verts[s.vertex_base + i0 * 3u + 2u].w);
    vec2 uv1 = vec2(uvx1, verts[s.vertex_base + i1 * 3u + 2u].w);
    vec2 uv2 = vec2(uvx2, verts[s.vertex_base + i2 * 3u + 2u].w);
    float m0 = verts[s.vertex_base + i0 * 3u + 1u].w;
    float m1 = verts[s.vertex_base + i1 * 3u + 1u].w;
    float m2 = verts[s.vertex_base + i2 * 3u + 1u].w;

    // Barycentric coordinates, computed geometrically in object space (the
    // vertex pool stores object-space positions; the default hit attribute
    // built-in is not exposed as a bare expression by this glslang).
    vec3 h = gl_ObjectRayOriginEXT + gl_HitTEXT * gl_ObjectRayDirectionEXT;
    vec3 e1 = p1 - p0;
    vec3 e2 = p2 - p0;
    vec3 h0 = h - p0;
    float d00 = dot(e1, e1);
    float d01 = dot(e1, e2);
    float d11 = dot(e2, e2);
    float d20 = dot(h0, e1);
    float d21 = dot(h0, e2);
    float denom = d00 * d11 - d01 * d01;
    // Weight of vertex 1 and vertex 2 respectively; vertex 0 carries the rest.
    float v = (d11 * d20 - d01 * d21) / denom;
    float w = (d00 * d21 - d01 * d20) / denom;
    float w0 = 1.0 - v - w;

    vec3 v_color = c0 * w0 + c1 * v + c2 * w;
    vec2 v_uv = uv0 * w0 + uv1 * v + uv2 * w;
    float v_material = m0 * w0 + m1 * v + m2 * w;
    vec3 v_world_pos = gl_WorldRayOriginEXT + gl_HitTEXT * gl_WorldRayDirectionEXT;
    // Rigid (unscaled) instances only, so the upper-left 3x3 of the instance
    // transform is the rotation alone.
    vec3 n = normalize(mat3(gl_ObjectToWorldEXT) * (n0 * w0 + n1 * v + n2 * w));

    // ---- Identical material rules to mesh.frag ----
    vec3 diff_dir = normalize(light_dir.xyz);
    float diff = max(dot(n, diff_dir), 0.0);
    float ambient = light_state.x;
    float sun_intensity = light_state.y;
    float wet_fac = light_state.z;
    float night_fac = light_state.w;
    float wet_cine = smoothstep(0.15, 1.0, wet_fac);

    vec3 tex_col;
    if (v_material >= 90.0) {
        tex_col = texture(car_tex, v_uv).rgb;
    } else {
        float atlas_u = v_material * (1.0 / 6.0);
        vec2 uv = vec2(fract(v_uv.x) * (1.0 / 6.0) + atlas_u, fract(v_uv.y));
        tex_col = texture(world_tex, uv).rgb;
        float luma = dot(tex_col, vec3(0.299, 0.587, 0.114));
        tex_col = mix(tex_col, vec3(luma), 0.35);
        if (v_material >= 3.0 && v_material < 4.0) {
            tex_col = mix(vec3(0.5), tex_col, 0.35);
        }
        if (v_material >= 4.0 && v_material < 5.0) {
            tex_col = mix(vec3(0.85), tex_col, 0.25);
        }
    }
    vec3 albedo = v_color * tex_col;
    if (v_material >= 3.0 && v_material < 90.0) {
        albedo *= terrain_state.xyz;
    }
    vec3 lit = albedo * (ambient + diff * sun_intensity * 0.85);

    // Wet asphalt: darken + glossy sun/moon sheen (mirrors mesh.frag).
    float wet_look = 0.0;
    if (v_material >= 0.0 && v_material < 3.0) {
        wet_look = wet_cine;
    }
    if (wet_look > 0.0) {
        lit *= mix(1.0, 0.82, wet_look);
        vec3 V = normalize(camera_pos.xyz - v_world_pos);
        vec3 H = normalize(diff_dir + V);
        float ndoth = max(dot(n, H), 0.0);
        float spec_hi = pow(ndoth, 128.0);
        float spec_lo = pow(ndoth, 24.0);
        float grazing = pow(1.0 - max(dot(n, V), 0.0), 2.0);
        lit += vec3(1.0)
            * (spec_hi * 0.5 + spec_lo * 0.35)
            * sun_intensity
            * wet_look
            * (0.4 + 0.6 * grazing);
    }

    // Player headlight cone.
    vec3 to_light = headlight_pos.xyz - v_world_pos;
    float head_dist = length(to_light);
    vec3 L = to_light / max(head_dist, 1e-4);
    float spot = dot(-L, normalize(headlight_dir.xyz));
    float head_inner_core = mix(0.97, 0.945, wet_cine);
    float head_outer_core = mix(0.90, 0.845, wet_cine);
    float head_inner_skirt = mix(0.97, 0.925, wet_cine);
    float head_outer_skirt = mix(0.90, 0.80, wet_cine);
    float head_decay = mix(0.06, 0.024, wet_cine);
    float head_core = smoothstep(head_outer_core, head_inner_core, spot);
    float head_skirt = smoothstep(head_outer_skirt, head_inner_skirt, spot);
    float head = (head_core + head_skirt * (0.35 * wet_cine)) * exp(-head_dist * head_decay);
    head *= mix(1.0, 1.45, wet_cine);
    float near_head_fade = mix(1.0, smoothstep(1.4, 4.2, head_dist), wet_cine);
    head *= near_head_fade;
    head = min(head, 1.6);
    head *= night_fac;
    lit += albedo * head * 0.85;

    // Oncoming traffic headlight projectors.
    for (int i = 0; i < 16; ++i) {
        vec3 lp = traffic_head_pos[i].xyz;
        vec3 ld = normalize(traffic_head_dir[i].xyz);
        float strength = traffic_head_state[i].x;
        if (strength <= 0.001) {
            continue;
        }
        float max_dist = traffic_head_state[i].y;
        float cos_inner = traffic_head_state[i].z;
        float cos_outer = traffic_head_state[i].w;
        float traffic_dist = max_dist * mix(1.0, 1.65, wet_cine);
        float traffic_inner = mix(cos_inner, cos_inner - 0.05, wet_cine);
        float traffic_outer = mix(cos_outer, cos_outer - 0.12, wet_cine);
        float traffic_gain = mix(1.0, 1.55, wet_cine);
        vec3 to_l = lp - v_world_pos;
        float dist = length(to_l);
        if (dist <= 1e-4 || dist >= traffic_dist) {
            continue;
        }
        vec3 Ll = to_l / dist;
        float cone = dot(-Ll, ld);
        float beam = smoothstep(traffic_outer, traffic_inner, cone);
        float fall = (1.0 - dist / traffic_dist);
        float dist_falloff = fall * fall;
        float near_traffic_fade = mix(1.0, smoothstep(1.6, 5.4, dist), wet_cine);
        float road_mask = 1.0 - smoothstep(0.24, 2.2, abs(v_world_pos.y - 0.02));
        float traffic_head =
            beam * dist_falloff * strength * traffic_gain * near_traffic_fade * night_fac * road_mask;
        traffic_head = min(traffic_head, 1.4);
        lit += albedo * traffic_head * 0.80;
    }

    // Street-lamp pools.
    for (int i = 0; i < 16; ++i) {
        float strength = lamp_state[i].w;
        if (strength <= 0.001) {
            continue;
        }
        vec3 lp = lamp_pos[i].xyz;
        vec3 ld = normalize(lamp_dir[i].xyz);
        vec3 lamp_col = lamp_state[i].rgb;
        vec3 to_l = lp - v_world_pos;
        float dist = length(to_l);
        float max_dist = 24.0;
        if (dist <= 1e-4 || dist >= max_dist) {
            continue;
        }
        vec3 Ll = to_l / dist;
        float cone = dot(-Ll, ld);
        float beam = smoothstep(0.80, 0.90, cone);
        float fall = (1.0 - dist / max_dist);
        float dist_falloff = fall * fall;
        float road_mask = 1.0 - smoothstep(0.24, 2.2, abs(v_world_pos.y - 0.02));
        float lamp_light = beam * dist_falloff * strength * night_fac * road_mask;
        lamp_light = min(lamp_light, 0.9);
        lit += albedo * lamp_col * lamp_light * 0.8;
    }

    // Long fog ramp (identical to mesh.frag).
    float dist = gl_HitTEXT;
    float fog = smoothstep(100.0, 600.0, dist);
    vec3 final_col = mix(lit, fog_color.rgb, fog);

    rtp = vec4(0.0, 0.0, 1.0, 1.0);
    imageStore(rt_out, ivec2(gl_LaunchIDEXT.xy), vec4(0.0, 1.0, 0.0, 1.0));
}