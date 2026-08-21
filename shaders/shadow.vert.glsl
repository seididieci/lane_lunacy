#version 450
// SPDX-License-Identifier: MIT

// Shadow-map vertex shader: renders the world chunks from the sun's point of
// view into a depth-only target, so the mesh shader can shadow-test receivers.
// Only position is needed; the pass stores depth alone (no color attachment).

layout(location = 0) in vec3 position;

layout(set = 0, binding = 0) uniform ShadowVP {
    mat4 view_proj;
};

void main() {
    gl_Position = view_proj * vec4(position, 1.0);
}
