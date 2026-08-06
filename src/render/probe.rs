// SPDX-License-Identifier: MIT

//! Scene probes: cheap, deterministic measurements of a rendered frame.
//!
//! Two kinds feed the "programmatic eye":
//! - **CPU probes** are derived from the pure `Frame` + `Game` (no pixels):
//!   projected sun position, flare intensity, how much road the headlights
//!   cover, and the wet/night factors.
//! - **GPU probes** are measured from the rendered linear (HDR) pixels:
//!   sky-top and road-center luminance, the sun disc's peak luminance, and the
//!   lens-flare bloom around the sun.
//!
//! Probes let a CI run compare two renders (windowed vs snapshot, or a commit
//! before/after a refactor) without eyeballing PNGs: identical scene state must
//! produce identical probe values.

use glam::Vec3;

use crate::game::Game;
use crate::render::frame::Frame;
use crate::road::road_curve;

/// CPU-side probes, all derived from scene math (no GPU, no pixels).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CpuProbe {
    /// Sun position in NDC (y-up) when the flare is visible, else `None`.
    pub sun_ndc: Option<[f32; 2]>,
    /// Lens flare strength (0..1-ish) of the sun this frame.
    pub flare_intensity: f32,
    /// Fraction of sampled road points ahead of the player inside the
    /// headlight cone (0 = dark road, 1 = fully lit).
    pub projector_road_coverage: f32,
    /// Rain intensity 0..1 (how wet the road is).
    pub wet_fac: f32,
    /// Night darkness 0..1 (0 = full day, 1 = full night).
    pub night_fac: f32,
}

/// GPU-side probes measured from rendered pixels (linear HDR values).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuProbe {
    /// Average linear luminance of the top sixth of the frame (the sky).
    pub sky_top_lum: f32,
    /// Average linear luminance of the road-center band ahead of the camera.
    pub road_center_lum: f32,
    /// Peak linear luminance inside the sun disc (0 when the sun is hidden).
    pub sun_disc_max_lum: f32,
    /// Peak linear luminance in the flare bloom ring around the sun (0 when
    /// hidden). Excludes the sun disc itself.
    pub flare_bloom_lum: f32,
}

/// Everything measured for one rendered frame, plus the scenario it belongs to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Probe {
    pub cpu: CpuProbe,
    pub gpu: GpuProbe,
}

const COS_OUTER: f32 = 0.90;
const HEADLIGHT_REACH: f32 = 40.0;
const ROAD_SAMPLE_STEP: f32 = 5.0;

/// Computes the CPU probes for a frame.
pub fn compute_cpu(game: &Game, frame: &Frame) -> CpuProbe {
    CpuProbe {
        sun_ndc: frame.sun_ndc,
        flare_intensity: frame.flare_intensity,
        projector_road_coverage: projector_road_coverage(game),
        wet_fac: frame.uniforms.wet_fac,
        night_fac: frame.night_fac,
    }
}

/// Fraction of sampled points along the road ahead of the player that fall
/// inside the player headlight cone (same geometry the mesh shader uses).
fn projector_road_coverage(game: &Game) -> f32 {
    let head_pos = Vec3::new(game.player_world_x(), 0.9, game.player_world_z());
    let axis = Vec3::new(
        game.vehicle.heading.sin(),
        -0.15,
        -game.vehicle.heading.cos(),
    )
    .normalize();
    let mut lit = 0u32;
    let mut total = 0u32;
    let mut s = ROAD_SAMPLE_STEP;
    while s <= HEADLIGHT_REACH {
        let p = Vec3::new(road_curve(s), 0.02, -s);
        let to_light = head_pos - p;
        let dist = to_light.length();
        if dist > 1e-4 && dist <= HEADLIGHT_REACH {
            let spot = (-to_light / dist).dot(axis);
            if spot >= COS_OUTER {
                lit += 1;
            }
        }
        total += 1;
        s += ROAD_SAMPLE_STEP;
    }
    if total == 0 {
        0.0
    } else {
        lit as f32 / total as f32
    }
}

/// Computes the GPU probes from linear HDR RGBA pixels (`width * height * 4`
/// floats, R G B A per pixel) and the projected sun position.
pub fn compute_gpu(
    linear_rgba: &[f32],
    width: u32,
    height: u32,
    sun_ndc: Option<[f32; 2]>,
) -> GpuProbe {
    let w = width as usize;
    let h = height as usize;
    let lum = |x: usize, y: usize| {
        let i = (y * w + x) * 4;
        0.2126 * linear_rgba[i] + 0.7152 * linear_rgba[i + 1] + 0.0722 * linear_rgba[i + 2]
    };

    // Sky top: top sixth of the frame.
    let sky_rows = h / 6;
    let sky_top_lum = avg_region(linear_rgba, w, h, 0, sky_rows, 0, w);

    // Road center: a band just below mid-frame, centered on the road ahead.
    let road_lo = (h as f32 * 0.72) as usize;
    let road_hi = (h as f32 * 0.90) as usize;
    let road_center_lum = avg_region(linear_rgba, w, h, road_lo, road_hi, w / 4, 3 * w / 4);

    // Sun disc + flare bloom, only meaningful when the sun is on-screen.
    let (mut sun_disc_max_lum, mut flare_bloom_lum) = (0.0f32, 0.0f32);
    if let Some(ndc) = sun_ndc {
        let cx = (((ndc[0] + 1.0) * 0.5) * w as f32) as i64;
        let cy = (((1.0 - ndc[1]) * 0.5) * h as f32) as i64;
        let disc_r = (w.min(h) as f32 * 0.025).max(1.0) as i64;
        let bloom_r = (w.min(h) as f32 * 0.08).max(2.0) as i64;
        for dy in -bloom_r..=bloom_r {
            for dx in -bloom_r..=bloom_r {
                let (x, y) = (cx + dx, cy + dy);
                if x < 0 || y < 0 || x >= w as i64 || y >= h as i64 {
                    continue;
                }
                let r2 = dx * dx + dy * dy;
                if r2 <= disc_r * disc_r {
                    sun_disc_max_lum = sun_disc_max_lum.max(lum(x as usize, y as usize));
                } else if r2 <= bloom_r * bloom_r {
                    flare_bloom_lum = flare_bloom_lum.max(lum(x as usize, y as usize));
                }
            }
        }
    }

    GpuProbe {
        sky_top_lum,
        road_center_lum,
        sun_disc_max_lum,
        flare_bloom_lum,
    }
}

/// Average luminance over a pixel rectangle `[x0,x1) x [y0,y1)` (clamped).
fn avg_region(
    linear_rgba: &[f32],
    w: usize,
    h: usize,
    y0: usize,
    y1: usize,
    x0: usize,
    x1: usize,
) -> f32 {
    let mut sum = 0.0f64;
    let mut n = 0u64;
    for y in y0.min(h)..y1.min(h) {
        for x in x0.min(w)..x1.min(w) {
            let i = (y * w + x) * 4;
            sum += (0.2126 * linear_rgba[i]
                + 0.7152 * linear_rgba[i + 1]
                + 0.0722 * linear_rgba[i + 2]) as f64;
            n += 1;
        }
    }
    if n == 0 {
        0.0
    } else {
        (sum / n as f64) as f32
    }
}

/// Renders `probe` as a compact JSON object (hand-rolled; no serde dep).
pub fn to_json(probe: &Probe) -> String {
    let sun = match probe.cpu.sun_ndc {
        Some([x, y]) => format!(r#"[{x:.6},{y:.6}]"#),
        None => "null".to_string(),
    };
    format!(
        "{{\n  \"sun_ndc\": {sun},\n  \"flare_intensity\": {fi:.6},\n  \"projector_road_coverage\": {prc:.6},\n  \"wet_fac\": {wf:.6},\n  \"night_fac\": {nf:.6},\n  \"sky_top_lum\": {stl:.6},\n  \"road_center_lum\": {rcl:.6},\n  \"sun_disc_max_lum\": {sdml:.6},\n  \"flare_bloom_lum\": {fbl:.6}\n}}",
        fi = probe.cpu.flare_intensity,
        prc = probe.cpu.projector_road_coverage,
        wf = probe.cpu.wet_fac,
        nf = probe.cpu.night_fac,
        stl = probe.gpu.sky_top_lum,
        rcl = probe.gpu.road_center_lum,
        sdml = probe.gpu.sun_disc_max_lum,
        fbl = probe.gpu.flare_bloom_lum,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Weather;
    use crate::render::frame::build_frame;
    use crate::render::particles::{DustSystem, RainSystem};
    use std::time::Duration;

    fn deterministic_game(time: f32, weather: Weather) -> Game {
        let mut game = Game::new();
        game.set_start_hour(time);
        game.set_weather(weather);
        game.set_seed(42);
        game
    }

    fn frame_for(game: &Game) -> Frame {
        let mut sky_time = 0.0;
        let mut camera_heading = 0.0;
        let mut rain = RainSystem::new();
        let mut dust = DustSystem::new();
        let anchors = crate::model::CarLightAnchors {
            lateral: 0.8,
            long_half: 1.84,
            headlight_y: 0.48,
            taillight_y: 0.48,
        };
        build_frame(
            game,
            Duration::ZERO,
            16.0 / 9.0,
            &mut sky_time,
            &mut camera_heading,
            &mut rain,
            &mut dust,
            &anchors,
            &[anchors],
            Vec::new(),
        )
    }

    #[test]
    fn wet_and_night_factors_track_weather_and_clock() {
        let noon = compute_cpu(
            &deterministic_game(12.0, Weather::Clear),
            &frame_for(&deterministic_game(12.0, Weather::Clear)),
        );
        let night_rain = compute_cpu(
            &deterministic_game(0.0, Weather::Rain),
            &frame_for(&deterministic_game(0.0, Weather::Rain)),
        );
        assert!(noon.wet_fac < 0.01, "clear noon should be dry");
        assert!(noon.night_fac < 0.5, "noon should be bright");
        assert!(night_rain.wet_fac > 0.5, "rain should be wet");
        assert!(night_rain.night_fac > 0.5, "midnight should be dark");
    }

    #[test]
    fn road_coverage_is_full_in_daylight_and_always_in_bounds() {
        let noon = compute_cpu(
            &deterministic_game(12.0, Weather::Clear),
            &frame_for(&deterministic_game(12.0, Weather::Clear)),
        );
        // Headlights switch off in daylight, but the probe measures the cone
        // geometry, which still covers the road ahead.
        assert!((0.0..=1.0).contains(&noon.projector_road_coverage));
        assert!(noon.projector_road_coverage > 0.5);
    }

    #[test]
    fn gpu_probes_find_the_bright_sun_disc() {
        let w = 100u32;
        let h = 100u32;
        // Black frame with a bright white sun at NDC (0, 0) = center.
        let mut px = vec![0.0f32; (w * h * 4) as usize];
        let c = (50 * 100 + 50) * 4;
        px[c] = 8.0;
        px[c + 1] = 8.0;
        px[c + 2] = 8.0;
        px[c + 3] = 1.0;
        let gpu = compute_gpu(&px, w, h, Some([0.0, 0.0]));
        assert!(
            gpu.sun_disc_max_lum > 5.0,
            "sun disc must catch the bright pixel"
        );
        assert!(gpu.sky_top_lum < 0.1, "black sky stays dark");
        assert!(gpu.road_center_lum < 0.1);
    }

    #[test]
    fn hidden_sun_yields_zero_sun_probes() {
        let w = 100u32;
        let h = 100u32;
        let px = vec![0.0f32; (w * h * 4) as usize];
        let gpu = compute_gpu(&px, w, h, None);
        assert_eq!(gpu.sun_disc_max_lum, 0.0);
        assert_eq!(gpu.flare_bloom_lum, 0.0);
    }

    #[test]
    fn json_serializes_all_fields() {
        let probe = Probe {
            cpu: CpuProbe {
                sun_ndc: Some([0.5, -0.25]),
                flare_intensity: 0.8,
                projector_road_coverage: 0.75,
                wet_fac: 0.3,
                night_fac: 0.9,
            },
            gpu: GpuProbe {
                sky_top_lum: 0.1,
                road_center_lum: 0.2,
                sun_disc_max_lum: 3.0,
                flare_bloom_lum: 1.5,
            },
        };
        let s = to_json(&probe);
        for needle in [
            "sun_ndc",
            "[0.500000,-0.250000]",
            "flare_intensity",
            "projector_road_coverage",
            "wet_fac",
            "night_fac",
            "sky_top_lum",
            "road_center_lum",
            "sun_disc_max_lum",
            "flare_bloom_lum",
        ] {
            assert!(s.contains(needle), "missing {needle} in {s}");
        }
    }
}
