#version 450
// SPDX-License-Identifier: MIT
layout(location = 0) in vec4 v_color;
layout(location = 1) in vec2 v_uv;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform sampler2D atlas;

void main() {
    if (v_uv.x < 0.0) {
        f_color = vec4(v_color.rgb, v_color.a);
    } else {
        float coverage = texture(atlas, v_uv).r;
        if (coverage < 0.02) {
            discard;
        }
        f_color = vec4(v_color.rgb, v_color.a * coverage);
    }
}
