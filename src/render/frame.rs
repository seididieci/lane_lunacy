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
use crate::model::CarLightAnchors;
use crate::render::camera::{perspective_vulkan, Camera};
use crate::render::daynight::{self, Lights};
use crate::render::flare;
use crate::render::particles::{
    build_headlights, build_taillights, drift_intensity, DustSystem, RainSystem,
};
use crate::road::{road_curve, road_tangent, CAR_HALF_W, CAR_LEN};
use crate::shaders::SkyUniform;
use crate::surface::material_at;
use crate::vertex::{FlareVertex, HudVertex, ParticleVertex};

pub const SKY_RADIUS: f32 = 550.0;
pub const MAX_TRAFFIC_HEADLIGHTS: usize = 16;

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
/// oncoming/same-direction traffic cones, packed exactly as the mesh and
/// particle shaders expect them.
#[derive(Clone, Copy)]
pub struct Headlights {
    pub pos: [f32; 4],
    pub dir: [f32; 4],
    pub traffic_pos: [[f32; 4]; MAX_TRAFFIC_HEADLIGHTS],
    pub traffic_dir: [[f32; 4]; MAX_TRAFFIC_HEADLIGHTS],
    pub traffic_state: [[f32; 4]; MAX_TRAFFIC_HEADLIGHTS],
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
    pub flare_verts: Vec<FlareVertex>,
    /// Projected sun NDC position and flare intensity, when the flare is
    /// visible. Exposed for CPU probes / tests.
    pub sun_ndc: Option<[f32; 2]>,
    pub flare_intensity: f32,
    /// Kept for the HUD pass.
    pub hud_verts: Vec<HudVertex>,
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
/// `sky_time`/`camera_heading` are the renderer's persistent smoothed camera
/// state (updated here); `rain`/`dust` are the persistent particle systems.
/// `player_anchors` and `traffic_anchors` are the per-model lamp anchors from
/// the loaded GLB meshes. `hud_verts` are the UI quads drawn last.
pub fn build_frame(
    game: &Game,
    dt: Duration,
    aspect: f32,
    sky_time: &mut f32,
    camera_heading: &mut f32,
    rain: &mut RainSystem,
    dust: &mut DustSystem,
    player_anchors: &CarLightAnchors,
    traffic_anchors: &[CarLightAnchors],
    hud_verts: Vec<HudVertex>,
) -> Frame {
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
        },
        particle_verts,
        dust_verts,
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
        let mut sky_time = 0.0;
        let mut camera_heading = 0.0;
        let mut rain = RainSystem::new();
        let mut dust = DustSystem::new();
        build_frame(
            game,
            Duration::from_secs_f32(1.0 / 60.0),
            16.0 / 9.0,
            &mut sky_time,
            &mut camera_heading,
            &mut rain,
            &mut dust,
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
    fn rain_intensity_scales_with_weather() {
        let clear = deterministic_game(12.0, Weather::Clear);
        let rain = deterministic_game(12.0, Weather::Rain);
        assert_eq!(frame_for(&clear).uniforms.wet_fac, 0.0);
        assert!(frame_for(&rain).uniforms.wet_fac > 0.9);
    }
}
