// SPDX-License-Identifier: MIT

use crate::game::difficulty::DifficultyTuning;
use crate::game::vehicle::Vehicle;
use crate::road::{road_curve, CAR_HALF_W, CAR_LEN};

pub struct Traffic {
    pub distance: f32,
    pub lane: f32,
    pub speed: f32,
}

pub fn rebuild_traffic(tuning: &DifficultyTuning, vehicle_distance: f32) -> Vec<Traffic> {
    let base_distance = vehicle_distance + tuning.init_start;
    let mut traffic = Vec::new();
    for k in 0..tuning.traffic_count {
        let lane = if k % 2 == 0 {
            -tuning.lane_offset
        } else {
            tuning.lane_offset
        };
        let speed_roll = ((k as f32 * 1.31).sin() * 0.5 + 0.5).clamp(0.0, 1.0);
        traffic.push(Traffic {
            distance: base_distance + k as f32 * tuning.init_gap,
            lane,
            speed: tuning.speed_base + speed_roll * tuning.speed_var,
        });
    }
    traffic
}

pub fn update_traffic(
    traffic: &mut [Traffic],
    vehicle: &Vehicle,
    tuning: &DifficultyTuning,
    dt: f32,
) {
    for t in traffic.iter_mut() {
        let dir = if t.lane > 0.0 { 1.0 } else { -1.0 };
        t.distance += t.speed * dir * dt;
    }

    for idx in 0..traffic.len() {
        if traffic[idx].distance < vehicle.distance - tuning.respawn_behind {
            let mut spawn_distance =
                vehicle.distance + tuning.respawn_ahead + idx as f32 * tuning.respawn_gap;
            let mut lane = if idx % 2 == 0 {
                -tuning.lane_offset
            } else {
                tuning.lane_offset
            };

            let mut attempt = 0;
            while attempt < 8 {
                let too_close = traffic.iter().enumerate().any(|(j, other)| {
                    j != idx
                        && (other.distance - spawn_distance).abs() < tuning.respawn_min_gap
                        && (other.lane - lane).abs() < 0.7
                }) || traffic.iter().any(|other| {
                    (other.lane - lane).abs() >= 0.7
                        && (other.distance - spawn_distance).abs() < tuning.wall_gap
                });
                if !too_close {
                    break;
                }
                spawn_distance += tuning.respawn_retry_step;
                if attempt % 2 == 1 {
                    lane = -lane;
                }
                attempt += 1;
            }

            let speed_roll =
                ((vehicle.distance * 0.013 + idx as f32 * 1.37).sin() * 0.5 + 0.5).clamp(0.0, 1.0);
            traffic[idx].distance = spawn_distance;
            traffic[idx].lane = lane;
            traffic[idx].speed = tuning.speed_base + speed_roll * tuning.speed_var;
        }
    }
}

pub fn check_collision(traffic: &[Traffic], vehicle: &Vehicle, tuning: &DifficultyTuning) -> bool {
    let pvx = road_curve(vehicle.distance) + vehicle.offset;
    traffic.iter().any(|t| {
        let tvx = road_curve(t.distance) + t.lane;
        (t.distance - vehicle.distance).abs() < CAR_LEN * tuning.collision_long_mul
            && (tvx - pvx).abs() < CAR_HALF_W * tuning.collision_lat_mul
    })
}
