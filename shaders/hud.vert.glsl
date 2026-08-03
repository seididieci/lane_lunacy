#version 450
// SPDX-License-Identifier: MIT
layout(location = 0) in vec2 position;
layout(location = 1) in vec4 color;
layout(location = 2) in vec2 uv;
layout(location = 0) out vec4 v_color;
layout(location = 1) out vec2 v_uv;
void main() {
    v_color = color;
    v_uv = uv;
    // Flip Y: Vulkan's NDC maps +y to the bottom of the framebuffer, but the
    // HUD/menu use y-up coordinates (matching the 3D scene's corrected projection).
    gl_Position = vec4(position.x, -position.y, 0.0, 1.0);
}
