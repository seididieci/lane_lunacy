#version 450
// SPDX-License-Identifier: MIT
layout(location = 0) in vec4 v_color;
layout(location = 1) in vec2 v_uv;
layout(location = 2) in float v_kind;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform sampler2D core;
layout(set = 0, binding = 1) uniform sampler2D streak;

void main() {
    vec4 tex = v_kind > 1.5 ? texture(streak, v_uv) : texture(core, v_uv);
    float alpha = tex.a * v_color.a;
    f_color = vec4(tex.rgb * v_color.rgb, alpha);
}
