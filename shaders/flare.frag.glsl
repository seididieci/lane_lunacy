#version 450
// SPDX-License-Identifier: MIT
layout(location = 0) in vec4 v_color;
layout(location = 1) in vec2 v_uv;
layout(location = 2) in float v_kind;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform sampler2D core;
layout(set = 0, binding = 1) uniform sampler2D streak;
layout(set = 0, binding = 2) uniform sampler2D ring;

void main() {
    vec4 tex;
    if (v_kind > 2.5) {
        tex = texture(ring, v_uv);
    } else if (v_kind > 1.5) {
        tex = texture(streak, v_uv);
    } else {
        tex = texture(core, v_uv);
    }
    float alpha = tex.a * v_color.a;
    f_color = vec4(tex.rgb * v_color.rgb, alpha);
}
