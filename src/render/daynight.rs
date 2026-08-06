// SPDX-License-Identifier: MIT

//! Day/night sky palette and lighting computation (pure, unit-testable).
//!
//! Everything here is a function of the sun elevation, the time of day, the
//! weather cover, and the difficulty's night darkness. It feeds the sky
//! uniforms, the mesh lights, and the lens-flare gating in `render/mod.rs`.

// The sun's azimuth at sunrise and sunset (radians, `atan2(x, z)` convention).
// Across the daylight hours the sun sweeps east -> south -> west between these
// two bearings, so its position is locked to the time of day.
const SUN_AZ_SUNRISE: f32 = 55.0_f32.to_radians();
const SUN_AZ_SUNSET: f32 = 305.0_f32.to_radians();

// Horizontal magnitude of the sun direction. The ratio of the vertical
// elevation factor to this length sets the sun's angle above the horizon; it
// is tuned so the sun peaks at ~20° elevation, keeping it inside the chase
// camera's vertical FOV whenever it is in front of the car (and the moon
// mirrored at night).
const SUN_HORIZONTAL: f32 = 2.75;

const DAY_ZENITH: [f32; 3] = [0.18, 0.42, 0.83];
const DAY_HORIZON: [f32; 3] = [0.55, 0.70, 0.92];
const DAY_CLOUD_TINT: [f32; 3] = [1.0, 0.97, 0.92];
const NIGHT_ZENITH: [f32; 3] = [0.02, 0.03, 0.07];
const NIGHT_HORIZON: [f32; 3] = [0.04, 0.05, 0.09];
const NIGHT_CLOUD_TINT: [f32; 3] = [0.22, 0.24, 0.32];
const OVERCAST_HORIZON: [f32; 3] = [0.60, 0.60, 0.63];
const DUSK_WARM: [f32; 3] = [0.85, 0.45, 0.22];

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

/// Sun azimuth in radians for any hour of the day, sweeping east -> west over
/// the daylight span and extrapolating (wrapped) through the night. The moon is
/// the antipode of this same sweep.
fn solar_azimuth(hours: f32, day_fraction: f32) -> f32 {
    let day_hours = day_fraction * 24.0;
    let sunrise = (24.0 - day_hours) * 0.5;
    let t = (hours - sunrise) / day_hours;
    (SUN_AZ_SUNRISE + (SUN_AZ_SUNSET - SUN_AZ_SUNRISE) * t).rem_euclid(std::f32::consts::TAU)
}

/// World-space direction toward the sun for a frame (the day branch of
/// `compute`). Also used at startup to hint where the sun sits in the sky.
pub fn sun_direction(sun_elevation: f32, hours: f32, day_fraction: f32) -> [f32; 3] {
    let az = solar_azimuth(hours, day_fraction);
    vec3_normalize([
        az.sin() * SUN_HORIZONTAL,
        sun_elevation,
        az.cos() * SUN_HORIZONTAL,
    ])
}

/// Computes the sky palettes and lights for a frame.
///
/// `hours` is the in-game time of day (0..24) and `day_fraction` the share of
/// the cycle with the sun above the horizon; together they place the sun, and
/// the moon (its antipode), at a position locked to the day/night cycle.
pub fn compute(
    sun_elevation: f32,
    hours: f32,
    day_fraction: f32,
    cover: f32,
    night_fac: f32,
) -> (SkyPalette, Lights) {
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

    let (light_dir, sun_intensity) = if sun_elevation >= 0.0 {
        let d = sun_direction(sun_elevation, hours, day_fraction);
        (d, smoothstep(0.0, 0.12, sun_elevation))
    } else {
        // The moon is the sun's antipode: mirrored azimuth, mirrored elevation.
        let sun_az = solar_azimuth(hours, day_fraction);
        let moon_az = (sun_az + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU);
        let d = vec3_normalize([
            moon_az.sin() * SUN_HORIZONTAL,
            -sun_elevation,
            moon_az.cos() * SUN_HORIZONTAL,
        ]);
        (d, night_fac * 0.14)
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

    // EasyArcade's day/night tuning: 82% daylight.
    const DAY_FRACTION: f32 = 0.82;

    fn day_hours() -> f32 {
        DAY_FRACTION * 24.0
    }

    fn sunrise() -> f32 {
        (24.0 - day_hours()) * 0.5
    }

    fn sunset() -> f32 {
        sunrise() + day_hours()
    }

    #[test]
    fn noon_sun_points_up_and_day_fac_peaks() {
        let (_, lights) = compute(1.0, 12.0, DAY_FRACTION, 0.0, 0.0);
        let expected = 1.0_f32.atan2(SUN_HORIZONTAL);
        let elev = (lights.light_dir[0] * lights.light_dir[0]
            + lights.light_dir[2] * lights.light_dir[2])
            .sqrt()
            .atan2(lights.light_dir[1]);
        assert!(
            (elev - (std::f32::consts::FRAC_PI_2 - expected)).abs() < 1e-3,
            "sun at its ~20° noon angle above the horizon"
        );
        assert!(lights.day_fac > 0.99);
        assert!(lights.ambient > 0.4, "bright ambient by day");
    }

    #[test]
    fn midnight_uses_moonlight_and_dark_ambient() {
        let (_, lights) = compute(-1.0, 0.0, DAY_FRACTION, 0.0, 1.0);
        // The moon mirrors the sun's low arc as its antipode, so it sits at the
        // same ~20° elevation at midnight.
        let horiz =
            (lights.light_dir[0] * lights.light_dir[0] + lights.light_dir[2] * lights.light_dir[2])
                .sqrt();
        let elev = lights.light_dir[1].atan2(horiz);
        let expected = 1.0_f32.atan2(SUN_HORIZONTAL);
        assert!(
            (elev - expected).abs() < 1e-3,
            "moon at the sun's noon elevation at midnight"
        );
        assert!(lights.day_fac < 0.01);
        assert!(lights.sun_intensity < 0.15, "moonlight is faint");
        assert!(lights.ambient < 0.1, "night ambient is dark");
    }

    #[test]
    fn night_sky_is_darker_than_day() {
        let (day, _) = compute(1.0, 12.0, DAY_FRACTION, 0.0, 0.0);
        let (night, _) = compute(-1.0, 0.0, DAY_FRACTION, 0.0, 1.0);
        assert!(night.zenith[0] < day.zenith[0]);
        assert!(night.fog_color[0] < day.fog_color[0]);
    }

    #[test]
    fn weather_dims_the_fog_color() {
        let (clear, _) = compute(0.6, 12.0, DAY_FRACTION, 0.0, 0.0);
        let (rainy, _) = compute(0.6, 12.0, DAY_FRACTION, 1.0, 0.0);
        assert!(rainy.fog_color[0] < clear.fog_color[0]);
    }

    #[test]
    fn sun_azimuth_sweeps_east_to_south_to_west() {
        let (_, rise) = compute(0.0, sunrise(), DAY_FRACTION, 0.0, 0.0);
        let (_, noon) = compute(1.0, 12.0, DAY_FRACTION, 0.0, 0.0);
        let (_, set) = compute(0.0, sunset(), DAY_FRACTION, 0.0, 0.0);
        let az = |l: &Lights| {
            l.light_dir[0]
                .atan2(l.light_dir[2])
                .rem_euclid(std::f32::consts::TAU)
        };
        assert!((az(&rise) - SUN_AZ_SUNRISE).abs() < 1e-3, "sun rises in the east");
        assert!((az(&noon) - std::f32::consts::PI).abs() < 1e-3, "sun crosses the south at noon");
        assert!((az(&set) - SUN_AZ_SUNSET).abs() < 1e-3, "sun sets in the west");
    }

    #[test]
    fn moon_is_the_suns_antipode() {
        // Slightly after sunset the moon rises opposite where the sun just set.
        let (_, dusk) = compute(-0.1, sunset() + 0.5, DAY_FRACTION, 0.0, 0.5);
        assert!(dusk.light_dir[1] > 0.0, "moon rising above the horizon");
        let dusk_az = dusk.light_dir[0]
            .atan2(dusk.light_dir[2])
            .rem_euclid(std::f32::consts::TAU);
        let expected =
            (solar_azimuth(sunset() + 0.5, DAY_FRACTION) + std::f32::consts::PI)
                .rem_euclid(std::f32::consts::TAU);
        assert!((dusk_az - expected).abs() < 1e-3, "moon azimuth mirrors the sun");

        // Midnight: moon at the top of its (low) arc, antipodal to the
        // extrapolated sun.
        let (_, midnight) = compute(-1.0, 0.0, DAY_FRACTION, 0.0, 1.0);
        let elev = (midnight.light_dir[0] * midnight.light_dir[0]
            + midnight.light_dir[2] * midnight.light_dir[2])
            .sqrt()
            .atan2(midnight.light_dir[1]);
        assert!(elev > 0.0, "moon up at midnight");
        let midnight_az = midnight.light_dir[0]
            .atan2(midnight.light_dir[2])
            .rem_euclid(std::f32::consts::TAU);
        let expected = (solar_azimuth(0.0, DAY_FRACTION) + std::f32::consts::PI)
            .rem_euclid(std::f32::consts::TAU);
        assert!((midnight_az - expected).abs() < 1e-3, "midnight moon mirrors the sun");
    }

    #[test]
    fn elevation_curve_is_preserved_while_the_sun_sweeps() {
        // The azimuth sweep must not change the elevation curve: for a given
        // elevation factor the sun's elevation angle stays the same, so only
        // the horizontal direction rotates.
        let (_, rise) = compute(0.0, sunrise(), DAY_FRACTION, 0.0, 0.0);
        let (_, noon) = compute(1.0, 12.0, DAY_FRACTION, 0.0, 0.0);
        let (_, set) = compute(0.0, sunset(), DAY_FRACTION, 0.0, 0.0);
        let elev = |l: &Lights| {
            let horiz =
                (l.light_dir[0] * l.light_dir[0] + l.light_dir[2] * l.light_dir[2]).sqrt();
            l.light_dir[1].atan2(horiz)
        };
        let expected_noon = 1.0_f32.atan2(SUN_HORIZONTAL);
        assert!(
            (elev(&noon) - expected_noon).abs() < 1e-3,
            "noon elevation angle unchanged from the fixed-azimuth look"
        );
        assert!(elev(&rise).abs() < 1e-3, "sun on the horizon at sunrise");
        assert!(elev(&set).abs() < 1e-3, "sun on the horizon at sunset");
    }
}
