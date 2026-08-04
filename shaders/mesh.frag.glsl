#version 450
// SPDX-License-Identifier: MIT
layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec3 v_color;
layout(location = 2) in vec2 v_uv;
layout(location = 3) in float v_depth;
layout(location = 4) in float v_material;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform MVP {
    mat4 model;
    mat4 view;
    mat4 projection;
    vec4 light_dir;
};

layout(set = 0, binding = 1) uniform sampler2D tex;

void main() {
    vec3 n = normalize(v_normal);
    float diff = max(dot(n, normalize(light_dir.xyz)), 0.0);
    float ambient = 0.48;
    vec3 tex_col;
    // Cars (material 4) use the car colormap directly.
    if (v_material >= 3.5) {
        tex_col = texture(tex, v_uv).rgb;
    } else {
        // World texture atlas, one row of 4 slots:
        // 0=asphalt base, 1=asphalt worn, 2=asphalt cracked, 3=grass.
        float atlas_u = v_material * 0.25;
        vec2 uv = vec2(fract(v_uv.x) * 0.25 + atlas_u, fract(v_uv.y));
        tex_col = texture(tex, uv).rgb;
        // Reduce noisy contrast around local luma (keeps overall brightness).
        float luma = dot(tex_col, vec3(0.299, 0.587, 0.114));
        tex_col = mix(tex_col, vec3(luma), 0.35);
        // Grass gets flattened toward mid-grey so it reads soft and clean.
        if (v_material >= 3.0) {
            tex_col = mix(vec3(0.5), tex_col, 0.35);
        }
    }
    vec3 albedo = v_color * tex_col;
    vec3 lit = albedo * (ambient + diff * 0.85);

    float fog = smoothstep(45.0, 260.0, v_depth);
    vec3 fog_color = vec3(0.48, 0.63, 0.8);
    vec3 final_col = mix(lit, fog_color, fog);
    f_color = vec4(final_col, 1.0);
}
