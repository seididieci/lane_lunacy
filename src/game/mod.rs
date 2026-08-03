// SPDX-License-Identifier: MIT

use std::time::Duration;

use crate::game::difficulty::DifficultyTuning;
use crate::game::traffic::{rebuild_traffic, update_traffic, Traffic, check_collision};
use crate::game::vehicle::Vehicle;
use crate::input::Input;
use crate::road::road_curve;

pub mod difficulty;
pub mod traffic;
pub mod vehicle;

pub use difficulty::DifficultyLevel;

pub struct Game {
    pub vehicle: Vehicle,
    pub traffic: Vec<Traffic>,
    pub wrecks: u32,
    pub wreck_timer: f32,
    pub game_over: bool,
    pub speed_kmh: f32,
    pub difficulty: DifficultyLevel,
}

impl Game {
    pub fn new() -> Self {
        let mut game = Game {
            vehicle: Vehicle::new(),
            traffic: Vec::new(),
            wrecks: 0,
            wreck_timer: 0.0,
            game_over: false,
            speed_kmh: 0.0,
            difficulty: DifficultyLevel::EasyArcade,
        };
        game.rebuild_traffic();
        game
    }

    pub fn set_difficulty(&mut self, difficulty: DifficultyLevel) {
        if self.difficulty == difficulty {
            return;
        }
        self.difficulty = difficulty;
        self.wreck_timer = 0.0;
        self.wrecks = 0;
        self.game_over = false;
        self.rebuild_traffic();
    }

    fn rebuild_traffic(&mut self) {
        let tuning = self.difficulty.tuning();
        self.traffic = rebuild_traffic(&tuning, self.vehicle.distance);
    }

    pub fn update(&mut self, dt: Duration, input: &Input) {
        if self.game_over {
            return;
        }
        let tuning: DifficultyTuning = self.difficulty.tuning();
        let dt = dt.as_secs_f32().min(0.05);

        self.vehicle.update(dt, input);
        update_traffic(&mut self.traffic, &self.vehicle, &tuning, dt);

        if self.wreck_timer <= 0.0 {
            if check_collision(&self.traffic, &self.vehicle, &tuning) {
                self.wrecks += 1;
                self.wreck_timer = tuning.wreck_cooldown;
                self.vehicle.speed = 0.0;
                if self.wrecks >= tuning.wreck_limit {
                    self.game_over = true;
                }
            }
        } else {
            self.wreck_timer -= dt;
        }

        self.speed_kmh = self.vehicle.speed * 3.6;
    }

    pub fn player_world_x(&self) -> f32 {
        road_curve(self.vehicle.distance) + self.vehicle.offset
    }

    pub fn player_world_z(&self) -> f32 {
        -self.vehicle.distance
    }
}
