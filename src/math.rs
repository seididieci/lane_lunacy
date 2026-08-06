// SPDX-License-Identifier: MIT

//! Shared scalar math used across the game and render layers.
//!
//! These mirror the GLSL builtins of the same names so CPU-side scene math and
//! shader math stay in lockstep. They used to be copy-pasted into several
//! modules; keeping single definitions guarantees they never drift.

/// GLSL-style `smoothstep(edge0, edge1, x)`: 0 below `edge0`, 1 above `edge1`,
/// smooth cubic in between. Handles inverted (`edge0 > edge1`) intervals.
pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// GLSL-style `mix(a, b, t)`: linear interpolation between `a` and `b`.
pub fn mix(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoothstep_clamps_and_is_monotonic() {
        assert_eq!(smoothstep(0.0, 1.0, -1.0), 0.0);
        assert_eq!(smoothstep(0.0, 1.0, 2.0), 1.0);
        assert!((smoothstep(0.0, 1.0, 0.5) - 0.5).abs() < 1e-6);
        let v: Vec<f32> = (0..=10).map(|i| smoothstep(0.0, 1.0, i as f32 / 10.0)).collect();
        assert!(v.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn smoothstep_handles_inverted_edges() {
        // Game uses `smoothstep(0.06, -0.02, ...)` (high -> low) frequently.
        assert_eq!(smoothstep(0.06, -0.02, 0.5), 0.0);
        assert_eq!(smoothstep(0.06, -0.02, -0.5), 1.0);
    }

    #[test]
    fn mix_interpolates_and_clamps_extremes() {
        assert_eq!(mix(0.0, 10.0, 0.0), 0.0);
        assert_eq!(mix(0.0, 10.0, 1.0), 10.0);
        assert_eq!(mix(0.0, 10.0, 0.25), 2.5);
    }
}
