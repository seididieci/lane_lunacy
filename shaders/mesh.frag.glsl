#version 450
// SPDX-License-Identifier: MIT
layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec3 v_color;
layout(location = 2) in vec2 v_uv;
layout(location = 3) in float v_depth;
layout(location = 4) in float v_material;
layout(location = 5) in vec3 v_world_pos;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform MVP {
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

layout(set = 0, binding = 1) uniform sampler2D tex;

void main() {
    // Planar-reflection clip plane (world space): discard fragments strictly
    // below the road surface so the mirrored camera (positioned under the road)
    // never draws the ground into the reflection. The ordinary scene passes
    // keep `clip_plane = (0,0,0,-1)`, which never triggers.
    if (dot(vec4(v_world_pos, 1.0), clip_plane) > 0.0) {
        discard;
    }
    vec3 n = normalize(v_normal);
    float diff = max(dot(n, normalize(light_dir.xyz)), 0.0);
    float ambient = light_state.x;
    float sun_intensity = light_state.y;
    float wet_fac = light_state.z;
    float night_fac = light_state.w;
    float wet_cine = smoothstep(0.15, 1.0, wet_fac);
    vec3 tex_col;
    // Cars (material 99) use the car colormap directly.
    if (v_material >= 90.0) {
        tex_col = texture(tex, v_uv).rgb;
    } else {
        // World texture atlas, one row of 6 slots:
        // 0=asphalt base, 1=asphalt worn, 2=asphalt cracked, 3=grass,
        // 4=foliage, 5=rock.
        float atlas_u = v_material * (1.0 / 6.0);
        vec2 uv = vec2(fract(v_uv.x) * (1.0 / 6.0) + atlas_u, fract(v_uv.y));
        tex_col = texture(tex, uv).rgb;
        // Reduce noisy contrast around local luma (keeps overall brightness).
        float luma = dot(tex_col, vec3(0.299, 0.587, 0.114));
        tex_col = mix(tex_col, vec3(luma), 0.35);
        // Grass gets flattened toward mid-grey so it reads soft and clean.
        if (v_material >= 3.0 && v_material < 4.0) {
            tex_col = mix(vec3(0.5), tex_col, 0.35);
        }
        // Foliage keeps its own green tint in v_color, using the tile only for
        // low-contrast canopy texture (mixed toward mid-grey). Rock (slot 5)
        // must stay clear of this branch.
        if (v_material >= 4.0 && v_material < 5.0) {
            tex_col = mix(vec3(0.85), tex_col, 0.25);
        }
    }
    vec3 albedo = v_color * tex_col;
    // Terrain (grass/verge, material 3) reacts to the day/night cycle like the
    // sky: identity by day, cool under moonlight, warm at dawn/dusk. Asphalt
    // stays pure.
    if (v_material >= 3.0 && v_material < 90.0) {
        albedo *= terrain_state.xyz;
    }
    vec3 lit = albedo * (ambient + diff * sun_intensity * 0.85);

    // Wet asphalt: darken the tarmac and add a glossy sun/moon specular with a
    // broad low-exponent sheen, so rain-soaked asphalt glints under any light.
    // Only the asphalt slots (0..2, including the shoulders) pick this up.
    float wet_look = 0.0;
    if (v_material >= 0.0 && v_material < 3.0) {
        wet_look = wet_cine;
    }
    if (wet_look > 0.0) {
        lit *= mix(1.0, 0.82, wet_look);
        vec3 V = normalize(camera_pos.xyz - v_world_pos);
        vec3 L = normalize(light_dir.xyz);
        vec3 H = normalize(L + V);
        float ndoth = max(dot(n, H), 0.0);
        float spec_hi = pow(ndoth, 128.0);
        float spec_lo = pow(ndoth, 24.0);
        // Fresnel-ish pickup at grazing angles (how the road reads from the
        // chase camera), so the sheen is strongest just ahead of the car.
        float grazing = pow(1.0 - max(dot(n, V), 0.0), 2.0);
        lit += vec3(1.0)
            * (spec_hi * 0.5 + spec_lo * 0.35)
            * sun_intensity
            * wet_look
            * (0.4 + 0.6 * grazing);
    }

    // Headlight cone cast from the player car, scaled with night darkness so
    // only the harder difficulties actually need to switch them on.
    vec3 to_light = headlight_pos.xyz - v_world_pos;
    float head_dist = length(to_light);
    vec3 L = to_light / max(head_dist, 1e-4);
    // L points fragment->light; cone axis points light->forward.
    float spot = dot(-L, normalize(headlight_dir.xyz));
    // Dual-layer wet cone: narrow bright core + wider faint skirt ("both"
    // presets together) so the beam keeps definition while retaining cinematic
    // peripheral spread in rain.
    float head_inner_core = mix(0.97, 0.945, wet_cine);
    float head_outer_core = mix(0.90, 0.845, wet_cine);
    float head_inner_skirt = mix(0.97, 0.925, wet_cine);
    float head_outer_skirt = mix(0.90, 0.80, wet_cine);
    float head_decay = mix(0.06, 0.024, wet_cine);
    float head_core = smoothstep(head_outer_core, head_inner_core, spot);
    float head_skirt = smoothstep(head_outer_skirt, head_inner_skirt, spot);
    float head = (head_core + head_skirt * (0.35 * wet_cine)) * exp(-head_dist * head_decay);
    head *= mix(1.0, 1.45, wet_cine);
    // In heavy rain, keep reach cinematic but avoid a blown-out patch right
    // under the camera by softly suppressing the nearest meters.
    float near_head_fade = mix(1.0, smoothstep(1.4, 4.2, head_dist), wet_cine);
    head *= near_head_fade;
    head = min(head, 1.6);
    head *= night_fac;
    lit += albedo * head * 0.85;

    // Oncoming traffic headlight projectors (uniform road light pools).
    // state = [strength, max_dist, cos_inner, cos_outer]
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
        // Ll points fragment->light; projector axis is light->forward.
        float cone = dot(-Ll, ld);
        float beam = smoothstep(traffic_outer, traffic_inner, cone);
        float fall = (1.0 - dist / traffic_dist);
        float dist_falloff = fall * fall;
        float near_traffic_fade = mix(1.0, smoothstep(1.6, 5.4, dist), wet_cine);
        // Strongest near the road plane, with a soft vertical fade up car bodies.
        float road_mask = 1.0 - smoothstep(0.24, 2.2, abs(v_world_pos.y - 0.02));
        float traffic_head =
            beam * dist_falloff * strength * traffic_gain * near_traffic_fade * night_fac * road_mask;
        traffic_head = min(traffic_head, 1.4);
        lit += albedo * traffic_head * 0.80;
    }

    // Street-lamp projectors: fixed downward warm pools on the road, gated by
    // night darkness like the headlights. state = [warm.r, warm.g, warm.b,
    // strength]; strength <= 0.001 means the lamp is off (day).
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
        // Lamp beams point straight down; the cone is wide enough to paint a
        // soft ~4m pool under the luminaire.
        float cone = dot(-Ll, ld);
        float beam = smoothstep(0.80, 0.90, cone);
        float fall = (1.0 - dist / max_dist);
        float dist_falloff = fall * fall;
        float road_mask = 1.0 - smoothstep(0.24, 2.2, abs(v_world_pos.y - 0.02));
        float lamp_light =
            beam * dist_falloff * strength * night_fac * road_mask;
        // Keep the pools subtle so headlights stay the dominant night light.
        lamp_light = min(lamp_light, 0.9);
        lit += albedo * lamp_col * lamp_light * 0.8;
    }

    // Long, gentle fog ramp that reaches full opacity exactly at the far clip
    // plane, so distant geometry fades into the same color as the sky horizon
    // instead of forming a visible band.
    float fog = smoothstep(100.0, 600.0, v_depth);
    vec3 final_col = mix(lit, fog_color.rgb, fog);
    f_color = vec4(final_col, 1.0);
}
