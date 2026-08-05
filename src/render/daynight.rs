// SPDX-License-Identifier: MIT

//! Day/night sky palette and lighting computation (pure, unit-testable).
//!
//! Everything here is a function of the sun elevation, weather cover, and the
//! difficulty's night darkness. It feeds the sky uniforms, the mesh lights,
//! and the lens-flare gating in `render/mod.rs`.

// Sun sweeps along this fixed azimuth in world space (keeps today's look).
const SUN_AZ_X: f32 = 0.25;
const SUN_AZ_Z: f32 = 0.4;

const DAY_ZENITH: [f32; 3] = [0.18, 0.42, 0.83];
const DAY_HORIZON: [f32; 3] = [0.55, 0.70, 0.92];
const DAY_CLOUD_TINT: [f32; 3] = [1.0, 0.97, 0.92];
const NIGHT_ZENITH: [f32; 3] = [0.02, 0.03, 0.07];
const NIGHT_HORIZON: [f32; 3] = [0.04, 0.05, 0.09];
const NIGHT_CLOUD_TINT: [f32; 3] = [0.22, 0.24, 0.32];
const OVERCAST_HORIZON: [f32; 3] = [0.60, 0.60, 0.63];
const DUSK_WARM: [f32; 3] = [0.85, 0.45, 0.22];

/// Faint moonlight direction used whenever the sun is below the horizon.
const MOON_DIR: [f32; 3] = [-0.3, 0.5, -0.35];

/// Sky colors passed to the sky dome shader (alpha is unused there).
pub struct SkyPalette {
    pub zenith: [f32; 4],
    pub horizon: [f32; 4],
    pub cloud_tint: [f32; 4],
    /// Horizon color the mesh fog blends into (already weather-dimmed).
    pub fog_color: [f32; 4],
}

/// Per-frame lighting values for the mesh and particle shaders.
pub struct Lights {
    /// Sun (day) or moon (night) direction in world space.
    pub light_dir: [f32; 4],
    /// Ambient base added to every fragment (0.48 day .. ~0.06 night).
    pub ambient: f32,
    /// Direct-light multiplier: sunlight by day, faint moonlight at night.
    pub sun_intensity: f32,
    /// 0 at night .. 1 at midday (drives stars / overcast in the sky shader).
    pub day_fac: f32,
    /// Effective night darkness 0..1 (already scaled by difficulty).
    pub night_fac: f32,
}

/// Computes the sky palettes and lights for a frame.
pub fn compute(sun_elevation: f32, cover: f32, night_fac: f32) -> (SkyPalette, Lights) {
    let day_fac = smoothstep(-0.02, 0.08, sun_elevation);
    let night_curve = 1.0 - day_fac;

    // Warm tint on the horizon at dawn/dusk (sun low but above the horizon).
    let dusk = smoothstep(0.22, 0.0, sun_elevation) * smoothstep(-0.12, -0.02, sun_elevation);

    let zenith = mix3(mix3(DAY_ZENITH, NIGHT_ZENITH, night_curve), DUSK_WARM, dusk * 0.25);
    let horizon = mix3(mix3(DAY_HORIZON, NIGHT_HORIZON, night_curve), DUSK_WARM, dusk * 0.55);
    let cloud_tint = mix3(DAY_CLOUD_TINT, NIGHT_CLOUD_TINT, night_curve);

    // Fog mirrors the sky shader's horizon at t=0, keeping the weather dim and
    // overcast shift so the road melts into the horizon exactly.
    let dim = 1.0 - 0.22 * cover;
    let fog_color = mix3(horizon, OVERCAST_HORIZON, cover).map(|c| c * dim);

    let (light_dir, sun_intensity) = if sun_elevation > 0.0 {
        let d = vec3_normalize([SUN_AZ_X, sun_elevation, SUN_AZ_Z]);
        (d, smoothstep(0.0, 0.12, sun_elevation))
    } else {
        (MOON_DIR, night_fac * 0.14)
    };

    let ambient = mix(0.48, 0.06, night_fac);

    (
        SkyPalette {
            zenith: [zenith[0], zenith[1], zenith[2], 1.0],
            horizon: [horizon[0], horizon[1], horizon[2], 1.0],
            cloud_tint: [cloud_tint[0], cloud_tint[1], cloud_tint[2], 1.0],
            fog_color: [fog_color[0], fog_color[1], fog_color[2], 1.0],
        },
        Lights {
            light_dir: [light_dir[0], light_dir[1], light_dir[2], 0.0],
            ambient,
            sun_intensity,
            day_fac,
            night_fac,
        },
    )
}

fn mix(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn mix3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [mix(a[0], b[0], t), mix(a[1], b[1], t), mix(a[2], b[2], t)]
}

fn vec3_normalize(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if l <= 0.0 {
        [0.0, 1.0, 0.0]
    } else {
        [v[0] / l, v[1] / l, v[2] / l]
    }
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noon_sun_points_up_and_day_fac_peaks() {
        let (_, lights) = compute(1.0, 0.0, 0.0);
        assert!(lights.light_dir[1] > 0.7, "sun high at noon");
        assert!(lights.day_fac > 0.99);
        assert!(lights.ambient > 0.4, "bright ambient by day");
    }

    #[test]
    fn midnight_uses_moonlight_and_dark_ambient() {
        let (_, lights) = compute(-1.0, 0.0, 1.0);
        assert_eq!(lights.light_dir, [MOON_DIR[0], MOON_DIR[1], MOON_DIR[2], 0.0]);
        assert!(lights.day_fac < 0.01);
        assert!(lights.sun_intensity < 0.15, "moonlight is faint");
        assert!(lights.ambient < 0.1, "night ambient is dark");
    }

    #[test]
    fn night_sky_is_darker_than_day() {
        let (day, _) = compute(1.0, 0.0, 0.0);
        let (night, _) = compute(-1.0, 0.0, 1.0);
        assert!(night.zenith[0] < day.zenith[0]);
        assert!(night.fog_color[0] < day.fog_color[0]);
    }

    #[test]
    fn weather_dims_the_fog_color() {
        let (clear, _) = compute(0.6, 0.0, 0.0);
        let (rainy, _) = compute(0.6, 1.0, 0.0);
        assert!(rainy.fog_color[0] < clear.fog_color[0]);
    }
}
