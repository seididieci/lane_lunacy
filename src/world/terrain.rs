// SPDX-License-Identifier: MIT

//! Continuous roadside terrain: rolling hills, foothills that climb toward the
//! horizon, and rounded mountain ridges, all expressed as a pure function of
//! world coordinates so the chunk mesh, the trees rooted on it, and the
//! gameplay speed hook all agree and chunk rebuilds stay stable. Continuous
//! terrain, not a discrete placed object, so it deliberately does not
//! implement `RoadsideObject`.
//!
//! The road corridor (asphalt, shoulders, verges, marker posts at ~5.8m and
//! lamp poles at ~6.5m) stays flat inside `RISE_START`; terrain climbs beyond
//! into a mountainous landscape. Mountains are rounded ridges — a steep but
//! smooth near face, a rounded crest that undulates along the road, and a
//! gentle far slope that blends back into the foothills — so they read as
//! mountains, not vertical walls with flat tops.

use crate::math::{mix, smoothstep};
use crate::road::ROAD_HALF;
use crate::world::hash01;

/// Lateral distance from the road center at which terrain starts rising. Keeps
/// the posts (ROAD_HALF+1.0) and lamp poles (ROAD_HALF+1.7) on flat ground.
pub const RISE_START: f32 = ROAD_HALF + 1.9;
/// Lateral band over which the valley wall (hill ramp) goes 0 → 1.
const RISE_SPAN: f32 = 10.0;
/// Full rolling-hill amplitude (metres), reached where the ramp saturates.
const HILL_AMP: f32 = 5.0;
/// Lattice cell sizes (metres) of the two hill-noise octaves.
const NOISE_CELL: f32 = 14.0;
const NOISE_CELL_2: f32 = 28.0;
/// Weight of the second (broader) hill octave; the rest is the first octave.
const HILL_OCTAVE_2: f32 = 0.5;
/// Metres the ground gains toward the horizon, turning the sides into a
/// mountain backdrop instead of flat land.
const FOOTHILL_RISE: f32 = 18.0;
/// Lateral span over which the foothill rise goes 0 → full.
const FOOTHILL_START: f32 = 40.0;
const FOOTHILL_END: f32 = 160.0;
/// Mountain ridges are rolled per `MOUNTAIN_BLOCK`-long block of world-`s`.
const MOUNTAIN_BLOCK: f32 = 130.0;
/// Fraction of blocks (per side) that contain a mountain ridge.
const MOUNTAIN_CHANCE: f32 = 0.35;
/// Fraction of a block over which the ridge fades in/out at its ends.
const MOUNTAIN_FADE: f32 = 0.18;
/// Cell (metres) of the 1D noise that makes ridge crests undulate along the
/// road, so a ridge reads as a range rather than a uniform wall.
const RIDGE_WAVE_CELL: f32 = 40.0;
/// Hard ceiling on terrain height, so geometry and lighting stay bounded.
pub const MAX_HEIGHT: f32 = 32.0;
/// Steepness (rise/m) that maps to `terrain_gradient == 1` (a full canyon).
const GRADIENT_REF: f32 = 2.0;
/// How much top speed the terrain gradient can reduce (moderate: -25%).
const SPEED_REDUCTION: f32 = 0.25;

/// A deterministic mountain ridge on one side of the road: a rounded crest
/// `crest_height` metres tall at `crest_lateral` metres of lateral offset,
/// sloping down over `crest_span` metres on its far side. `crest_height` is
/// already faded in/out across the block so ridges don't pop.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mountain {
    pub crest_lateral: f32,
    pub crest_height: f32,
    pub crest_span: f32,
}

/// Deterministic mountain profile for one side (`+1` right, `-1` left) at a
/// world-`s`. Pure function of world coordinates.
pub fn mountain_profile(s: f32, side: f32) -> Option<Mountain> {
    let block = (s / MOUNTAIN_BLOCK).floor();
    let salt = if side >= 0.0 { 0.0 } else { 777.0 };
    let r = hash01(block * 1.7 + salt);
    if r >= MOUNTAIN_CHANCE {
        return None;
    }
    let crest_lateral = RISE_START + 5.0 + hash01(block * 2.3 + salt * 0.1) * 7.0;
    let crest_height = 12.0 + hash01(block * 3.1 + salt * 0.7) * 10.0;
    let crest_span = crest_height * 0.9;
    let t = (s - block * MOUNTAIN_BLOCK) / MOUNTAIN_BLOCK;
    let fade = smoothstep(0.0, MOUNTAIN_FADE, t) * (1.0 - smoothstep(1.0 - MOUNTAIN_FADE, 1.0, t));
    if fade <= 1e-4 {
        return None;
    }
    Some(Mountain {
        crest_lateral,
        crest_height: crest_height * fade,
        crest_span,
    })
}

/// Terrain elevation at a world point. Inside the flat roadside zone it matches
/// `road_height(s)` exactly. Beyond that, hills, foothills, and mountains rise
/// relative to the road surface. The entire terrain is offset by `road_height(s)`,
/// so the road corridor follows its undulating path. Continuous in `s` and
/// `lateral` (hills and foothills ramp smoothly, ridges fade across block ends),
/// so adjacent chunks never show seams.
pub fn terrain_height(s: f32, lateral: f32) -> f32 {
    let d = lateral.abs();
    if d <= RISE_START {
        return road_height(s);
    }
    let side = if lateral >= 0.0 { 1.0 } else { -1.0 };
    let ramp = smoothstep(RISE_START, RISE_START + RISE_SPAN, d);
    let hills = hills_noise(s, d) * HILL_AMP * ramp;
    let foothill = FOOTHILL_RISE * smoothstep(FOOTHILL_START, FOOTHILL_END, d);
    let ridge = ridge_height(s, side, d);
    // The road elevation plus the relative terrain profile
    (hills + foothill).max(ridge).clamp(0.0, MAX_HEIGHT) + road_height(s)
}

/// Outward terrain slope (rise per metre) on the side of the road the point is
/// on, used to pick the cell material (grass vs rock) and to derive the
/// gameplay gradient.
pub fn terrain_slope(s: f32, lateral: f32) -> f32 {
    let sign = if lateral >= 0.0 { 1.0 } else { -1.0 };
    let d = lateral.abs().max(0.001);
    let inner = terrain_height(s, sign * (d - 0.5));
    let outer = terrain_height(s, sign * (d + 0.5));
    (outer - inner).max(0.0)
}

/// Road elevation wave along the road's length. Creates a continuous, smooth
/// undulation that is consistent across chunk boundaries. The amplitude is
/// limited by `MAX_HEIGHT` to ensure the whole scene stays bounded.
pub fn road_height(s: f32) -> f32 {
    ROAD_WAVE_FREQ * s.sin().abs() * ROAD_WAVE_AMPLITUDE
}

/// Wave frequency for road elevation (1/wavelength).
const ROAD_WAVE_FREQ: f32 = 0.05; // wavelength ~62 units (meters)
/// Wave amplitude for road elevation (max height change from road center).
const ROAD_WAVE_AMPLITUDE: f32 = 8.0; // +/- 8m elevation change

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

/// Two-octave rolling-hill noise in [0, 1]. Seamless across chunk boundaries
/// by construction (the lattices are anchored on absolute world coordinates)
/// and deterministic.
fn hills_noise(s: f32, d: f32) -> f32 {
    let n1 = value_noise(s, d, NOISE_CELL, 0.7, 1.3);
    let n2 = value_noise(s, d, NOISE_CELL_2, 1.9, 3.7);
    n1 * (1.0 - HILL_OCTAVE_2) + n2 * HILL_OCTAVE_2
}

/// Height contributed by an active mountain ridge at `(s, lateral)`: a steep
/// but smooth near face rising to the crest, then a gentle far slope back down
/// to the foothills. Both profiles meet the crest with zero slope, so the ridge
/// is a rounded peak (never a flat-topped wall). The crest height undulates
/// along the road so the ridge reads as a mountain range.
fn ridge_height(s: f32, side: f32, d: f32) -> f32 {
    let Some(m) = mountain_profile(s, side) else {
        return 0.0;
    };
    let wave = 0.85 + 0.3 * (0.5 + 0.5 * ridge_wave_noise(s));
    let h = m.crest_height * wave;
    if d <= m.crest_lateral {
        let near_span = m.crest_lateral - RISE_START;
        smoothstep(0.0, 1.0, (d - RISE_START) / near_span) * h
    } else {
        (1.0 - smoothstep(0.0, 1.0, (d - m.crest_lateral) / m.crest_span)) * h
    }
}

/// One-dimensional value noise in [0, 1] along `s`, used to modulate ridge
/// crests. Continuous and deterministic.
fn ridge_wave_noise(s: f32) -> f32 {
    let x = s / RIDGE_WAVE_CELL;
    let x0 = x.floor();
    let fx = x - x0;
    let a = hash01(x0 * 0.9);
    let b = hash01((x0 + 1.0) * 0.9);
    mix(a, b, smoothstep(0.0, 1.0, fx))
}

/// World-coordinate value noise in [0, 1] over `(s, lateral)` space. Seamless
/// across chunk boundaries by construction (the lattice is anchored on absolute
/// world coordinates) and deterministic. `xsalt`/`ysalt` seed each octave's
/// lattice so octaves don't coincide.
fn value_noise(s: f32, d: f32, cell: f32, xsalt: f32, ysalt: f32) -> f32 {
    let x = s / cell;
    let y = d / cell;
    let x0 = x.floor();
    let y0 = y.floor();
    let fx = x - x0;
    let fy = y - y0;
    let a = hash01(x0 * xsalt + y0 * ysalt);
    let b = hash01((x0 + 1.0) * xsalt + y0 * ysalt);
    let c = hash01(x0 * xsalt + (y0 + 1.0) * ysalt);
    let e = hash01((x0 + 1.0) * xsalt + (y0 + 1.0) * ysalt);
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
        // flat ground relative to the road surface, which is now elevated by road_height(s).
        for lateral in [0.0, 3.0, ROAD_HALF + 1.0, ROAD_HALF + 1.7, RISE_START] {
            assert!(
                (terrain_height(100.0, lateral) - road_height(100.0)).abs() < 0.001,
                "lateral {lateral}"
            );
        }
    }

    #[test]
    fn terrain_is_non_negative_and_bounded() {
        let mut s = 0.0;
        while s < 520.0 {
            for d in (0..40).map(|i| i as f32 * 6.0) {
                for lateral in [d, -d] {
                    let h = terrain_height(s, lateral);
                    assert!((0.0..=MAX_HEIGHT).contains(&h), "s={s} d={d} h={h}");
                }
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
    fn hills_are_visible_and_foothills_climb_toward_the_horizon() {
        // Rolling hills exist right off the corridor: some sample well above
        // the road plane.
        let mut max_near = 0.0f32;
        let mut s = 0.0;
        while s < 1040.0 {
            max_near = max_near.max(terrain_height(s, 20.0));
            s += 13.0;
        }
        assert!(max_near > 2.0, "hills should be visible off the corridor");

        // The far ground climbs toward the horizon (foothill rise saturates).
        for s in [0.0, 130.0, 500.0] {
            assert!(
                terrain_height(s, 160.0) >= 18.0,
                "far terrain must climb into a mountain backdrop, s={s}"
            );
        }
    }

    #[test]
    fn mountains_rise_as_rounded_ridges_not_walls() {
        let mut found = false;
        let mut s = 0.0;
        while s < 2600.0 {
            for side in [-1.0, 1.0] {
                if let Some(m) = mountain_profile(s, side) {
                    let mid = (s / MOUNTAIN_BLOCK).floor() * MOUNTAIN_BLOCK + MOUNTAIN_BLOCK * 0.5;
                    let crest = terrain_height(mid, side * m.crest_lateral);
                    let near = terrain_height(mid, side * (m.crest_lateral - 0.5));
                    let far = terrain_height(mid, side * (m.crest_lateral + m.crest_span));
                    // The near face rises steeply up to the crest...
                    assert!(
                        crest > 5.0 && crest > near,
                        "near face must rise to a crest, got crest={crest} near={near}"
                    );
                    assert!(m.crest_lateral > RISE_START && m.crest_lateral <= RISE_START + 12.0);
                    assert!(m.crest_height <= MAX_HEIGHT);
                    // ...and the far side descends (a ridge, not a plateau wall).
                    assert!(
                        far < crest * 0.5,
                        "far side must descend, got far={far} crest={crest}"
                    );
                    found = true;
                }
            }
            s += MOUNTAIN_BLOCK * 0.5;
        }
        assert!(found, "some block must contain a mountain");
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
            assert_eq!(mountain_profile(s, 1.0), mountain_profile(s, 1.0));
        }
    }
}
