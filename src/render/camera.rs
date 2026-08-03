// SPDX-License-Identifier: MIT

use glam::{Mat4, Vec3};

pub struct Camera {
    pub eye: Vec3,
    pub forward: Vec3,
}

impl Camera {
    pub fn view(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye, self.eye + self.forward, Vec3::Y)
    }
}

/// Perspective matrix for Vulkan clip space (z in [0, 1]), right-handed view space.
pub fn perspective_vulkan(fovy: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let mut p = Mat4::perspective_rh(fovy, aspect, near, far);
    p.y_axis.y *= -1.0;
    p
}
