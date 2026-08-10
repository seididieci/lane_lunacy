// SPDX-License-Identifier: MIT

//! Continuous roadside terrain: rolling hills plus deterministic cliff walls,
//! expressed as a pure function of world coordinates so the chunk mesh, the
//! trees rooted on it, and the gameplay speed hook all agree and chunk rebuilds
//! stay stable. This is continuous terrain, not a discrete placed object, so it
//! deliberately does not implement `RoadsideObject`.
//!
//! The road corridor (asphalt, shoulders, verges, marker posts at ~5.8m and
//! lamp poles at ~6.5m) stays flat inside `RISE_START`; terrain climbs beyond.

use crate::math::{mix, smoothstep};
use crate::road::ROAD_HALF;
use crate::world::hash01;

/// Lateral distance from the road center at which terrain starts rising. Keeps
/// the posts (ROAD_HALF+1.0) and lamp poles (ROAD_HALF+1.7) on flat ground.
pub const RISE_START: f32 = ROAD_HALF + 1.9;
/// Lateral band over which the hill ramp goes 0 → 1.
const RISE_SPAN: f32 = 8.0;
/// Full hill amplitude (metres), reached where the ramp saturates.
const HILL_AMP: f32 = 3.0;
/// Lattice cell size (metres) of the world-coordinate value noise.
const NOISE_CELL: f32 = 14.0;
/// Cliff walls are rolled per `CLIFF_BLOCK`-long block of world-`s`.
const CLIFF_BLOCK: f32 = 130.0;
/// Fraction of blocks (per side) that contain a cliff wall.
const CLIFF_CHANCE: f32 = 0.28;
/// Fraction of a block over which the wall fades in/out at its ends.
const CLIFF_FADE: f32 = 0.18;
/// Hard ceiling on terrain height, so geometry and lighting stay bounded.
const MAX_HEIGHT: f32 = 25.0;
/// Steepness (rise/m) that maps to `terrain_gradient == 1` (a full canyon).
const GRADIENT_REF: f32 = 2.0;
/// How much top speed the terrain gradient can reduce (moderate: -25%).
const SPEED_REDUCTION: f32 = 0.25;

/// A deterministic cliff wall on one side of the road: it rises steeply from
/// `RISE_START` to `height` metres at `wall_lateral` metres of lateral offset.
/// `height` is already faded in/out across the block so cliffs don't pop.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cliff {
    pub wall_lateral: f32,
    pub height: f32,
}

/// Deterministic cliff profile for one side (`+1` right, `-1` left) at a
/// world-`s`. Pure function of world coordinates.
pub fn cliff_profile(s: f32, side: f32) -> Option<Cliff> {
    let block = (s / CLIFF_BLOCK).floor();
    let salt = if side >= 0.0 { 0.0 } else { 777.0 };
    let r = hash01(block * 1.7 + salt);
    if r >= CLIFF_CHANCE {
        return None;
    }
    let wall_lateral = RISE_START + 1.0 + hash01(block * 2.3 + salt * 0.1) * 2.5;
    let height = 7.0 + hash01(block * 3.1 + salt * 0.7) * 8.0;
    let t = (s - block * CLIFF_BLOCK) / CLIFF_BLOCK;
    let fade = smoothstep(0.0, CLIFF_FADE, t) * (1.0 - smoothstep(1.0 - CLIFF_FADE, 1.0, t));
    if fade <= 1e-4 {
        return None;
    }
    Some(Cliff {
        wall_lateral,
        height: height * fade,
    })
}

/// Terrain elevation at a world point. Zero inside the flat roadside zone,
/// rolling hills beyond it, and steep deterministic cliff walls where a block
/// profile is active. Continuous in `s` and `lateral` (hills ramp up smoothly,
/// cliff walls fade across block ends), so adjacent chunks never show seams.
pub fn terrain_height(s: f32, lateral: f32) -> f32 {
    let d = lateral.abs();
    if d <= RISE_START {
        return 0.0;
    }
    let side = if lateral >= 0.0 { 1.0 } else { -1.0 };
    let ramp = smoothstep(RISE_START, RISE_START + RISE_SPAN, d);
    let hill = (0.5 + 0.5 * value_noise(s, d)) * HILL_AMP * ramp;
    let mut h = hill;
    if let Some(c) = cliff_profile(s, side) {
        if d < c.wall_lateral {
            let u = (d - RISE_START) / (c.wall_lateral - RISE_START);
            let wall = u * u * c.height;
            h = h.max(wall);
        }
    }
    h.clamp(0.0, MAX_HEIGHT)
}

/// Outward terrain slope (rise per metre) at a lateral offset, used to pick the
/// cell material (grass vs rock) and to derive the gameplay gradient.
pub fn terrain_slope(s: f32, lateral: f32) -> f32 {
    let d = lateral.abs().max(0.001);
    let inner = terrain_height(s, d - 0.5);
    let outer = terrain_height(s, d + 0.5);
    (outer - inner).max(0.0)
}

/// Normalized steepness of the terrain adjacent to the road (0 = open rolling
/// ground, 1 = a steep canyon wall), driving the vehicle speed hook.
pub fn terrain_gradient(s: f32) -> f32 {
    let mut g = 0.0f32;
    for d in [RISE_START + 0.5, RISE_START + 1.5, RISE_START + 2.5] {
        g = g.max(terrain_slope(s, d)).max(terrain_slope(s, -d));
    }
    (g / GRADIENT_REF).clamp(0.0, 1.0)
}

/// Speed multiplier derived from the local terrain gradient: 1.0 on open
/// ground, down to `1.0 - SPEED_REDUCTION` in steep canyon sections.
pub fn speed_factor(s: f32) -> f32 {
    1.0 - SPEED_REDUCTION * terrain_gradient(s)
}

/// World-coordinate value noise in [0, 1] over `(s, lateral)` space. Seamless
/// across chunk boundaries by construction (the lattice is anchored on absolute
/// world coordinates) and deterministic.
fn value_noise(s: f32, d: f32) -> f32 {
    let x = s / NOISE_CELL;
    let y = d / NOISE_CELL;
    let x0 = x.floor();
    let y0 = y.floor();
    let fx = x - x0;
    let fy = y - y0;
    let a = hash01(x0 * 0.7 + y0 * 1.3);
    let b = hash01((x0 + 1.0) * 0.7 + y0 * 1.3);
    let c = hash01(x0 * 0.7 + (y0 + 1.0) * 1.3);
    let e = hash01((x0 + 1.0) * 0.7 + (y0 + 1.0) * 1.3);
    let sx = smoothstep(0.0, 1.0, fx);
    let sy = smoothstep(0.0, 1.0, fy);
    mix(mix(a, b, sx), mix(c, e, sx), sy)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terrain_is_flat_inside_the_roadside_zone() {
        // Posts (~5.8m), lamp poles (~6.5m) and the whole road corridor sit on
        // flat ground.
        for lateral in [0.0, 3.0, ROAD_HALF + 1.0, ROAD_HALF + 1.7, RISE_START] {
            assert_eq!(terrain_height(100.0, lateral), 0.0, "lateral {lateral}");
        }
    }

    #[test]
    fn terrain_is_non_negative_and_bounded() {
        let mut s = 0.0;
        while s < 520.0 {
            for d in (0..40).map(|i| i as f32 * 6.0) {
                let h = terrain_height(s, d);
                assert!((0.0..=MAX_HEIGHT).contains(&h), "s={s} d={d} h={h}");
                assert_eq!(terrain_height(s, -d), h, "symmetric in |lateral|");
            }
            s += 37.0;
        }
    }

    #[test]
    fn terrain_height_is_deterministic() {
        for (s, d) in [(0.0, 10.0), (123.4, 33.0), (260.0, 80.0), (1000.0, 5.0)] {
            assert_eq!(terrain_height(s, d), terrain_height(s, d));
        }
    }

    #[test]
    fn cliffs_rise_steeply_when_a_block_profile_is_active() {
        let mut found = false;
        let mut s = 0.0;
        while s < 2600.0 {
            for side in [-1.0, 1.0] {
                if let Some(c) = cliff_profile(s, side) {
                    let mid = (s / CLIFF_BLOCK).floor() * CLIFF_BLOCK + CLIFF_BLOCK * 0.5;
                    let h = terrain_height(mid, side * (c.wall_lateral - 0.2));
                    assert!(h > 5.0, "cliff wall must rise steeply, got {h} for {:?}", c);
                    assert!(c.wall_lateral > RISE_START && c.wall_lateral <= RISE_START + 3.6);
                    assert!(c.height <= MAX_HEIGHT);
                    found = true;
                }
            }
            s += CLIFF_BLOCK * 0.5;
        }
        assert!(found, "some block must contain a cliff");
    }

    #[test]
    fn speed_factor_slows_in_canyons_and_stays_in_range() {
        let mut slowest = f32::MAX;
        let mut s = 0.0;
        while s < 2600.0 {
            let f = speed_factor(s);
            assert!(
                (1.0 - SPEED_REDUCTION..=1.0).contains(&f),
                "s={s} factor {f}"
            );
            slowest = slowest.min(f);
            s += 6.5;
        }
        assert!(
            slowest < 1.0 - SPEED_REDUCTION * 0.5,
            "steep canyon sections must slow the car meaningfully"
        );
    }

    #[test]
    fn gradient_and_profile_are_deterministic() {
        for s in [0.0, 130.0, 500.0, 1234.0] {
            assert_eq!(terrain_gradient(s), terrain_gradient(s));
            assert_eq!(speed_factor(s), speed_factor(s));
            assert_eq!(cliff_profile(s, 1.0), cliff_profile(s, 1.0));
        }
    }
}
