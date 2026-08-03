// SPDX-License-Identifier: MIT

pub const ROAD_HALF: f32 = 4.8;
pub const CAR_HALF_W: f32 = 0.9;
pub const CAR_LEN: f32 = 4.0;

pub fn road_curve(s: f32) -> f32 {
    12.0 * (s * 0.02).sin()
}

pub fn road_tangent(s: f32) -> f32 {
    0.24 * (s * 0.02).cos()
}
