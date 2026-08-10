// SPDX-License-Identifier: MIT

//! Pure CPU per-frame scene description.
//!
//! `Frame` carries everything a frame needs to be drawn once — the view/proj
//! matrices, day/night lights, sky uniform, headlight projectors, particle
//! quads, flare quads — without any vulkano types. `build_frame` computes it
//! from the `Game` + mutable particle state + model anchors, so the exact same
//! scene math drives both the windowed renderer and the headless snapshot path.

use std::time::Duration;

use glam::{Mat4, Vec3};

use crate::game::Game;
use crate::math::smoothstep;
use crate::mesh::{lamp_head_pos, roadside_lamps};
use crate::model::CarLightAnchors;
use crate::render::camera::{perspective_vulkan, Camera};
use crate::render::daynight::{self, Lights};
use crate::render::flare;
use crate::render::particles::{
    build_headlights, build_lamp_glows, build_taillights, drift_intensity, DustSystem, MistSystem,
    RainSystem,
};
use crate::road::{road_curve, road_tangent, CAR_HALF_W, CAR_LEN};
use crate::shaders::SkyUniform;
use crate::surface::material_at;
use crate::vertex::{FlareVertex, HudVertex, ParticleVertex};

pub const SKY_RADIUS: f32 = 550.0;
pub const MAX_TRAFFIC_HEADLIGHTS: usize = 16;
/// Street-lamp projector slots. Two lamps per spacing interval on a ~400m
/// window fit comfortably inside 16.
pub const MAX_LAMPS: usize = 16;

/// Camera + lighting context shared by every draw in a frame. One bundle keeps
/// the recorder and the `SceneResources` draw methods from threading half a
/// dozen `Mat4`/`Lights`/scalar parameters through every call site.
#[derive(Clone, Copy)]
pub struct FrameUniforms {
    pub view: Mat4,
    pub proj: Mat4,
    pub lights: Lights,
    pub wet_fac: f32,
    pub fog_color: [f32; 4],
}

/// Headlight projector payload for one frame: the player's cone plus the
/// oncoming/same-direction traffic cones and the street-lamp pools, packed
/// exactly as the mesh and particle shaders expect them.
#[derive(Clone, Copy)]
pub struct Headlights {
    pub pos: [f32; 4],
    pub dir: [f32; 4],
    pub traffic_pos: [[f32; 4]; MAX_TRAFFIC_HEADLIGHTS],
    pub traffic_dir: [[f32; 4]; MAX_TRAFFIC_HEADLIGHTS],
    pub traffic_state: [[f32; 4]; MAX_TRAFFIC_HEADLIGHTS],
    /// Street-lamp projectors: `state` = [warm.r, warm.g, warm.b, strength].
    pub lamp_pos: [[f32; 4]; MAX_LAMPS],
    pub lamp_dir: [[f32; 4]; MAX_LAMPS],
    pub lamp_state: [[f32; 4]; MAX_LAMPS],
}

/// One fully-computed frame, ready to be recorded into a command buffer. All
/// scene math (camera, day/night, projectors, particles, flare) is baked here
/// on the CPU so both presenters can share a single recording pass.
pub struct Frame {
    pub aspect: f32,
    pub uniforms: FrameUniforms,
    pub eye: Vec3,
    pub cam_forward: Vec3,
    pub night_fac: f32,
    pub sky_uniform: SkyUniform,
    pub headlights: Headlights,
    pub particle_verts: Vec<ParticleVertex>,
    pub dust_verts: Vec<ParticleVertex>,
    /// Local low-hanging mist quads (camera-following ground haze).
    pub mist_verts: Vec<ParticleVertex>,
    /// Mist intensity 0..1 this frame (weather + dawn/dusk driven). Exposed
    /// for CPU probes / tests.
    pub mist_intensity: f32,
    pub flare_verts: Vec<FlareVertex>,
    /// Projected sun NDC position and flare intensity, when the flare is
    /// visible. Exposed for CPU probes / tests.
    pub sun_ndc: Option<[f32; 2]>,
    pub flare_intensity: f32,
    /// Kept for the HUD pass.
    pub hud_verts: Vec<HudVertex>,
}

/// Persistent per-frame state shared by every presenter: the smoothed camera
/// heading, the sky clock, and the particle systems. Pure CPU (no vulkano
/// types), so `build_frame` stays deterministic and testable.
pub struct FrameState {
    pub sky_time: f32,
    pub camera_heading: f32,
    pub rain: RainSystem,
    pub dust: DustSystem,
    pub mist: MistSystem,
}

impl Default for FrameState {
    fn default() -> Self {
        FrameState {
            sky_time: 0.0,
            camera_heading: 0.0,
            rain: RainSystem::new(),
            dust: DustSystem::new(),
            mist: MistSystem::new(),
        }
    }
}

impl FrameState {
    /// Deterministic variant for the headless snapshot path: the particle
    /// systems seed from the scenario seed so the render is reproducible.
    pub fn with_seed(seed: u64) -> Self {
        FrameState {
            sky_time: 0.0,
            camera_heading: 0.0,
            rain: RainSystem::with_seed(seed),
            dust: DustSystem::with_seed(seed),
            mist: MistSystem::with_seed(seed),
        }
    }
}

/// Low-hanging mist amount (0..1) for a frame. Weather cover drives the bulk
/// so CLEAR skies stay clear; a low-sun term peaks near the horizon and dies
/// out by mid-morning, giving faint banks at dawn/dusk. Zero at noon and at
/// mid-night (when the sun is at its most extreme elevation).
fn mist_intensity(cloud_amount: f32, sun_elevation: f32) -> f32 {
    let cover = smoothstep(0.55, 0.95, cloud_amount);
    let low_sun = 1.0 - smoothstep(0.0, 0.9, sun_elevation.abs());
    (cover * 0.5 + low_sun * 0.25).clamp(0.0, 1.0)
}

/// Rotation aligning a traffic car to its lane direction. Cars on the right
/// (lane > 0) drive toward -Z; oncoming cars (lane < 0) face +Z.
pub(crate) fn traffic_rotation(lane: f32, distance: f32) -> glam::Quat {
    if lane > 0.0 {
        glam::Quat::from_rotation_y(f32::atan2(-road_tangent(distance), 1.0))
    } else {
        glam::Quat::from_rotation_y(f32::atan2(road_tangent(distance), -1.0))
    }
}

/// Computes the full frame for a `Game` state.
///
/// `state` is the presenter's persistent smoothed camera + particle state
/// (updated here). `player_anchors` and `traffic_anchors` are the per-model
/// lamp anchors from the loaded GLB meshes. `hud_verts` are the UI quads drawn
/// last.
pub fn build_frame(
    game: &Game,
    dt: Duration,
    aspect: f32,
    state: &mut FrameState,
    player_anchors: &CarLightAnchors,
    traffic_anchors: &[CarLightAnchors],
    hud_verts: Vec<HudVertex>,
) -> Frame {
    let FrameState {
        sky_time,
        camera_heading,
        rain,
        dust,
        mist,
    } = state;

    let proj = perspective_vulkan(60.0f32.to_radians(), aspect, 0.1, 600.0);

    // Day/night lighting: the sun sweeps through the day, giving way to faint
    // moonlight at night. The palettes also mirror the weather cover so the fog
    // color matches the sky horizon exactly.
    let cover = {
        let t = ((game.cloud_amount() - 0.10) / 0.90).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    };
    let (palette, lights) = daynight::compute(
        game.sun_elevation(),
        game.time_of_day(),
        game.difficulty.tuning().day_fraction,
        cover,
        game.night_fac(),
    );
    let fog_color = palette.fog_color;
    let wet_fac = game.rain_intensity();
    let night_fac = lights.night_fac;

    let car_pos = Vec3::new(game.player_world_x(), 0.0, game.player_world_z());
    let dt_secs = dt.as_secs_f32().min(0.05);
    *sky_time += dt_secs;
    let diff = game.vehicle.heading - *camera_heading;
    *camera_heading += diff * (dt_secs * 3.0).min(1.0);
    let cam_forward = Vec3::new(camera_heading.sin(), 0.0, -camera_heading.cos());
    let eye = car_pos - cam_forward * 8.0 + Vec3::new(0.0, 4.0, 0.0);
    let look_at = car_pos + cam_forward * 4.0 + Vec3::new(0.0, 3.6, 0.0);
    let cam = Camera {
        eye,
        forward: (look_at - eye).normalize(),
    };
    let view = cam.view();

    let sky_uniform = SkyUniform {
        model: Mat4::from_scale_rotation_translation(
            Vec3::splat(SKY_RADIUS),
            glam::Quat::IDENTITY,
            eye,
        )
        .to_cols_array_2d(),
        view: view.to_cols_array_2d(),
        projection: proj.to_cols_array_2d(),
        time: *sky_time,
        _pad: [0.0; 3],
        zenith: palette.zenith,
        horizon: palette.horizon,
        cloud_tint: palette.cloud_tint,
        light_dir: lights.light_dir,
        cloud_amount: game.cloud_amount(),
        sun_state: [
            game.sun_elevation(),
            lights.day_fac,
            game.time_of_day(),
            lights.sun_intensity,
        ],
    };

    // Headlight cone from the player car, aimed slightly down the road.
    let head_forward = Vec3::new(game.vehicle.heading.sin(), 0.0, -game.vehicle.heading.cos());
    let headlight_pos = [
        game.player_world_x() + head_forward.x * 2.5,
        0.9,
        game.player_world_z() + head_forward.z * 2.5,
        1.0,
    ];
    let headlight_dir = [head_forward.x, -0.15, head_forward.z, 0.0];

    // Traffic headlight anchors are used in two ways:
    // 1) projected onto asphalt in the mesh shader for uniform road light
    //    (ALL cars, so same-direction cars light the road ahead too),
    // 2) as visible lamp sprites in the particle pass (oncoming only; the
    //    rear of same-direction cars shows red taillights instead).
    let mut oncoming_head_lights = Vec::with_capacity(game.traffic.len() * 2);
    let mut traffic_projectors = Vec::with_capacity(game.traffic.len() * 2);
    if night_fac > 0.02 {
        for (idx, t) in game.traffic.iter().enumerate() {
            let tvx = road_curve(t.distance) + t.lane;
            let rot = traffic_rotation(t.lane, t.distance);
            let anchors = &traffic_anchors[idx % traffic_anchors.len()];
            let car_pos = Vec3::new(tvx, 0.35, -t.distance);
            let forward = (rot * Vec3::new(0.0, 0.0, -1.0)).normalize();
            let left = car_pos
                + rot * Vec3::new(-anchors.lateral, anchors.headlight_y, -anchors.long_half);
            let right =
                car_pos + rot * Vec3::new(anchors.lateral, anchors.headlight_y, -anchors.long_half);
            traffic_projectors.push((left, forward));
            traffic_projectors.push((right, forward));
            if t.lane < 0.0 {
                oncoming_head_lights.push((left, forward));
                oncoming_head_lights.push((right, forward));
            }
        }
    }
    let mut traffic_head_pos = [[0.0; 4]; MAX_TRAFFIC_HEADLIGHTS];
    let mut traffic_head_dir = [[0.0; 4]; MAX_TRAFFIC_HEADLIGHTS];
    let mut traffic_head_state = [[0.0; 4]; MAX_TRAFFIC_HEADLIGHTS];
    for (i, (center, forward)) in traffic_projectors
        .iter()
        .take(MAX_TRAFFIC_HEADLIGHTS)
        .enumerate()
    {
        traffic_head_pos[i] = [center.x, center.y, center.z, 1.0];
        traffic_head_dir[i] = [forward.x, -0.13, forward.z, 0.0];
        // [strength, max_dist, cos_inner, cos_outer]
        traffic_head_state[i] = [0.95, 20.0, 0.965, 0.90];
    }

    // Street lamps: warm downward pools from the luminaire heads, placed by the
    // same deterministic list as the chunk mesh. The closest lamps ahead fill
    // the fixed pool; pools fade in with night darkness like the headlights.
    let mut lamp_head_positions = Vec::with_capacity(MAX_LAMPS);
    let mut lamp_pos_arr = [[0.0; 4]; MAX_LAMPS];
    let mut lamp_dir_arr = [[0.0; 4]; MAX_LAMPS];
    let mut lamp_state_arr = [[0.0; 4]; MAX_LAMPS];
    if night_fac > 0.02 {
        for (i, (lamp_s, side)) in
            roadside_lamps(game.vehicle.distance, game.vehicle.distance + 400.0)
                .into_iter()
                .take(MAX_LAMPS)
                .enumerate()
        {
            let head = lamp_head_pos(lamp_s, side);
            lamp_pos_arr[i] = [head[0], head[1], head[2], 1.0];
            lamp_dir_arr[i] = [0.0, -1.0, 0.0, 0.0];
            // [warm.r, warm.g, warm.b, strength]
            lamp_state_arr[i] = [1.0, 0.82, 0.55, 0.8];
            lamp_head_positions.push(Vec3::new(head[0], head[1], head[2]));
        }
    }

    // Particles (rain + drift dust).
    let mut particle_verts = Vec::new();
    let mut dust_verts = Vec::new();
    let rain_intensity = wet_fac;
    if rain_intensity > 0.0 {
        let cam_right = cam_forward.cross(Vec3::Y);
        rain.update(dt_secs, eye, game.vehicle.speed);
        particle_verts = rain.build_vertices(eye, cam_right, rain_intensity);
    }

    // Drift dust: puffs kicked up on hard steering/sideslip, launch, and
    // (minimally) just driving over dustier surfaces, scaled by the road
    // material under the car.
    {
        let cam_right = cam_forward.cross(Vec3::Y);
        let v = &game.vehicle;
        let lateral_v = v.speed * (v.heading.sin() - road_tangent(v.distance) * v.heading.cos());
        let profile = material_at(v.distance, v.offset).dust_profile();
        let drift = drift_intensity(v.speed, lateral_v, v.steer, v.throttle, profile.emission);
        let p_rot = glam::Quat::from_rotation_y(-v.heading);
        let base = Vec3::new(game.player_world_x(), 0.12, game.player_world_z());
        let rear = [
            base + p_rot * Vec3::new(-(CAR_HALF_W + 0.25), 0.0, CAR_LEN * 0.5 + 0.1),
            base + p_rot * Vec3::new(CAR_HALF_W + 0.25, 0.0, CAR_LEN * 0.5 + 0.1),
        ];
        dust.update(dt_secs, drift, &profile, rear, cam_forward);
        dust_verts.append(&mut dust.build_vertices(eye, cam_right));
    }

    // Low-hanging mist: a camera-following bank of soft puffs at road level,
    // driven by the weather cover plus a dawn/dusk boost. The far sky keeps its
    // tile-based dome (task 2); this is the local "hybrid" layer.
    let mist_intensity = mist_intensity(game.cloud_amount(), game.sun_elevation());
    let mut mist_verts = Vec::new();
    if mist_intensity > 0.0 {
        let cam_right = cam_forward.cross(Vec3::Y);
        // Anchor the volume at a road-level point under the camera so the bank
        // stays grounded, and cull against the eye.
        mist.update(dt_secs, Vec3::new(eye.x, 0.0, eye.z));
        mist_verts = mist.build_vertices(
            eye,
            cam_right,
            mist_intensity,
            Vec3::new(fog_color[0], fog_color[1], fog_color[2]),
        );
    }

    // Night traffic/player lights (additive, camera-facing).
    if night_fac > 0.02 {
        let cam_right = cam_forward.cross(Vec3::Y);
        let mut tail_centers = Vec::with_capacity(game.traffic.len() * 2 + 2);

        // The player car's own rear taillights, from its model anchors.
        let player_rot = glam::Quat::from_rotation_y(-game.vehicle.heading);
        let p_base = Vec3::new(game.player_world_x(), 0.03, game.player_world_z());
        tail_centers.push(
            p_base
                + player_rot
                    * Vec3::new(
                        -player_anchors.lateral,
                        player_anchors.taillight_y,
                        player_anchors.long_half,
                    ),
        );
        tail_centers.push(
            p_base
                + player_rot
                    * Vec3::new(
                        player_anchors.lateral,
                        player_anchors.taillight_y,
                        player_anchors.long_half,
                    ),
        );

        for (idx, t) in game.traffic.iter().enumerate() {
            let tvx = road_curve(t.distance) + t.lane;
            let rot = traffic_rotation(t.lane, t.distance);
            let anchors = &traffic_anchors[idx % traffic_anchors.len()];
            let car_pos = Vec3::new(tvx, 0.35, -t.distance);
            if t.lane > 0.0 {
                // Same direction: red taillights at the rear (facing away).
                tail_centers.push(
                    car_pos
                        + rot * Vec3::new(-anchors.lateral, anchors.taillight_y, anchors.long_half),
                );
                tail_centers.push(
                    car_pos
                        + rot * Vec3::new(anchors.lateral, anchors.taillight_y, anchors.long_half),
                );
            }
        }
        particle_verts.append(&mut build_taillights(
            &tail_centers,
            eye,
            cam_right,
            night_fac,
        ));
        particle_verts.append(&mut build_headlights(
            &oncoming_head_lights,
            eye,
            cam_right,
            night_fac,
        ));
        // Street-lamp lantern glows, a touch dimmer than headlights so the
        // pools stay the main signal.
        particle_verts.append(&mut build_lamp_glows(
            &lamp_head_positions,
            eye,
            cam_right,
            night_fac * 0.7,
        ));
    }

    // Sun lens flare: project the sun (a world direction) into NDC and fan
    // additive sprites along the sun->screen-center axis, faded by brightness,
    // cloud cover, and how far off-screen the sun is.
    let mut sun_ndc = None;
    let mut flare_intensity = 0.0;
    let mut flare_verts = Vec::new();
    if lights.sun_intensity > 0.0 {
        let sun_dir = Vec3::new(
            lights.light_dir[0],
            lights.light_dir[1],
            lights.light_dir[2],
        );
        let view_dir = view.transform_vector3(sun_dir);
        if view_dir.z < 0.0 {
            let clip = proj * view_dir.extend(1.0);
            // Projection is Vulkan y-down; flip y so flare positions use the
            // shader's y-up NDC convention (same as the HUD).
            let ndc = [clip.x / clip.w, -clip.y / clip.w];
            let off = (ndc[0].abs().max(ndc[1].abs()) - 0.9).max(0.0);
            let off_fade = 1.0 / (1.0 + off * 8.0);
            let intensity = lights.sun_intensity * (1.0 - 0.9 * cover) * off_fade;
            flare_verts = flare::build_flare_verts(ndc, aspect, intensity);
            if intensity > 0.001 {
                sun_ndc = Some(ndc);
                flare_intensity = intensity;
            }
        }
    }

    Frame {
        aspect,
        uniforms: FrameUniforms {
            view,
            proj,
            lights,
            wet_fac,
            fog_color,
        },
        eye,
        cam_forward,
        night_fac,
        sky_uniform,
        headlights: Headlights {
            pos: headlight_pos,
            dir: headlight_dir,
            traffic_pos: traffic_head_pos,
            traffic_dir: traffic_head_dir,
            traffic_state: traffic_head_state,
            lamp_pos: lamp_pos_arr,
            lamp_dir: lamp_dir_arr,
            lamp_state: lamp_state_arr,
        },
        particle_verts,
        dust_verts,
        mist_verts,
        mist_intensity,
        flare_verts,
        sun_ndc,
        flare_intensity,
        hud_verts,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Weather;

    fn anchors() -> CarLightAnchors {
        CarLightAnchors {
            lateral: 0.8,
            long_half: 1.84,
            headlight_y: 0.48,
            taillight_y: 0.48,
        }
    }

    fn deterministic_game(time: f32, weather: Weather) -> Game {
        let mut game = Game::new();
        game.set_start_hour(time);
        game.set_weather(weather);
        game.traffic.clear();
        game
    }

    fn frame_for(game: &Game) -> Frame {
        let mut state = FrameState::default();
        build_frame(
            game,
            Duration::from_secs_f32(1.0 / 60.0),
            16.0 / 9.0,
            &mut state,
            &anchors(),
            &[anchors()],
            Vec::new(),
        )
    }

    #[test]
    fn noon_clear_produces_sun_flare_and_no_night_lights() {
        let game = deterministic_game(12.0, Weather::Clear);
        let frame = frame_for(&game);
        assert!(frame.sun_ndc.is_some(), "sun should be visible at noon");
        assert!(frame.flare_intensity > 0.5, "sun flare strong at noon");
        assert!(frame.uniforms.lights.sun_intensity > 0.9);
        assert_eq!(frame.night_fac, 0.0);
        assert!(frame.particle_verts.is_empty(), "no taillights by day");
    }

    #[test]
    fn midnight_rain_turns_on_night_lights_and_dust_fades() {
        let game = deterministic_game(0.0, Weather::Rain);
        let frame = frame_for(&game);
        assert!(frame.night_fac > 0.3, "night darkness active");
        assert!(frame.uniforms.wet_fac > 0.9, "rain fully wet");
        // Taillights/headlight discs are only oncoming + same-direction traffic
        // which we cleared, so the player's own taillights remain (2 centers).
        assert!(!frame.particle_verts.is_empty(), "player taillights lit");
        // At midnight the moon (antipode of the sun) can be up, but its flare
        // must be far dimmer than high-noon sunlight.
        assert!(
            frame.flare_intensity < 0.05,
            "moon flare must be weak, got {}",
            frame.flare_intensity
        );
    }

    #[test]
    fn midnight_lamps_fill_the_pool_and_glow() {
        let game = deterministic_game(0.0, Weather::Clear);
        let frame = frame_for(&game);
        assert!(frame.night_fac > 0.3, "night darkness active");
        // Pools are filled from the deterministic roadside_lamps list.
        let active = frame
            .headlights
            .lamp_state
            .iter()
            .filter(|s| s[3] > 0.0)
            .count();
        assert!(active > 0, "street lamps must light the road at night");
        assert_eq!(frame.headlights.lamp_dir[0], [0.0, -1.0, 0.0, 0.0]);
        // Glow sprites are baked into the particle pass: warm lantern discs
        // (green channel ~0.85), distinct from red taillights and cool-white
        // headlight discs.
        let lamp_glow = |v: &crate::vertex::ParticleVertex| {
            v.color[0] > 0.9
                && v.color[1] > 0.8
                && v.color[1] < 0.9
                && v.color[2] > 0.5
                && v.color[2] < 0.7
        };
        assert!(
            frame.particle_verts.iter().any(lamp_glow),
            "warm lamp glow sprites must be present at night"
        );
        // Every lamp's pool position sits at its luminaire head, off the road.
        for (i, state) in frame.headlights.lamp_state.iter().enumerate() {
            if state[3] > 0.0 {
                let p = frame.headlights.lamp_pos[i];
                assert!(p[1] > 4.0, "lamp head is elevated");
            }
        }
    }

    #[test]
    fn noon_turns_street_lamps_off() {
        let game = deterministic_game(12.0, Weather::Clear);
        let frame = frame_for(&game);
        assert_eq!(frame.night_fac, 0.0);
        assert!(
            frame.headlights.lamp_state.iter().all(|s| s[3] == 0.0),
            "lamps must be off by day"
        );
        assert!(
            !frame.particle_verts.iter().any(|v| v.color[0] > 0.9
                && v.color[1] > 0.8
                && v.color[1] < 0.9
                && v.color[2] < 0.7),
            "no lamp glow by day"
        );
    }

    #[test]
    fn rain_intensity_scales_with_weather() {
        let clear = deterministic_game(12.0, Weather::Clear);
        let rain = deterministic_game(12.0, Weather::Rain);
        assert_eq!(frame_for(&clear).uniforms.wet_fac, 0.0);
        assert!(frame_for(&rain).uniforms.wet_fac > 0.9);
    }

    #[test]
    fn mist_tracks_weather_and_low_sun() {
        // Noon clear: no clouds, high sun -> no mist.
        let noon = frame_for(&deterministic_game(12.0, Weather::Clear));
        assert_eq!(noon.mist_intensity, 0.0);
        assert!(noon.mist_verts.is_empty(), "clear noon stays mist-free");

        // Full rain -> a dense bank regardless of the clock.
        let rain = frame_for(&deterministic_game(12.0, Weather::Rain));
        assert!(rain.mist_intensity >= 0.5, "rain mist dense");
        assert!(!rain.mist_verts.is_empty(), "rain mist rendered");

        // Clear dusk (sun near the horizon, EASY) -> faint but present.
        let dusk = frame_for(&deterministic_game(18.0, Weather::Clear));
        assert!(
            (0.02..0.5).contains(&dusk.mist_intensity),
            "dusk mist is subtle: {}",
            dusk.mist_intensity
        );
        assert!(!dusk.mist_verts.is_empty(), "dusk mist rendered");
    }

    #[test]
    fn mist_intensity_peaks_at_the_horizon_and_zeroes_at_extremes() {
        assert_eq!(mist_intensity(0.15, 1.0), 0.0, "clear noon");
        assert_eq!(mist_intensity(0.15, -1.0), 0.0, "clear midnight");
        assert!(mist_intensity(1.0, 1.0) >= 0.5, "rain is misty by day");
        assert!(mist_intensity(0.15, 0.2) > 0.15, "dawn/dusk boost");
        assert!(mist_intensity(0.15, 0.2) > mist_intensity(0.15, 0.8));
    }
}
