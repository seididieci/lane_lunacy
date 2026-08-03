#version 450
// SPDX-License-Identifier: MIT
layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec3 v_color;
layout(location = 2) in vec2 v_uv;
layout(location = 3) in float v_depth;
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
    vec3 albedo = v_color * texture(tex, v_uv).rgb;
    vec3 lit = albedo * (ambient + diff * 0.85);

    float fog = smoothstep(45.0, 260.0, v_depth);
    vec3 fog_color = vec3(0.48, 0.63, 0.8);
    vec3 final_col = mix(lit, fog_color, fog);
    f_color = vec4(final_col, 1.0);
}
