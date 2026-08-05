// SPDX-License-Identifier: MIT

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::game::difficulty::DifficultyTuning;
use crate::game::traffic::{rebuild_traffic, update_traffic, Traffic, check_collision};
use crate::game::vehicle::{
    Vehicle, RED_ZONE_START, PERFECT_LO, PERFECT_HI, BOOST_DURATION, RED_SHIFT_HEAT_KICK,
};
use crate::input::Input;
use crate::road::road_curve;

pub mod difficulty;
pub mod traffic;
pub mod vehicle;
pub mod weather;

pub use difficulty::DifficultyLevel;
pub use weather::Weather;

const SCORE_SPEED_WEIGHT: f32 = 0.05;
const PERFECT_SHIFT_BONUS: u32 = 250;
const PERFECT_SHIFT_POPUP_TIME: f32 = 1.2;
const COAST_STOP_SPEED: f32 = 0.5;
const WEATHER_CYCLE_SPEED: f32 = 0.04;

pub struct Game {
    pub vehicle: Vehicle,
    pub traffic: Vec<Traffic>,
    pub wrecks: u32,
    pub wreck_timer: f32,
    pub game_over: bool,
    pub engine_blown: bool,
    pub engine_heat: f32,
    pub perfect_shift_timer: f32,
    pub speed_kmh: f32,
    pub difficulty: DifficultyLevel,
    pub weather: Weather,
    weather_phase: f32,
    pub score: u32,
    pub bonus_score: u32,
    pub best_score: u32,
    pub avg_speed: f32,
    pub time: f32,
}

impl Game {
    pub fn new() -> Self {
        let mut game = Game {
            vehicle: Vehicle::new(),
            traffic: Vec::new(),
            wrecks: 0,
            wreck_timer: 0.0,
            game_over: false,
            engine_blown: false,
            engine_heat: 0.0,
            perfect_shift_timer: 0.0,
            speed_kmh: 0.0,
            difficulty: DifficultyLevel::EasyArcade,
            weather: Weather::Auto,
            weather_phase: random_weather_phase(),
            score: 0,
            bonus_score: 0,
            best_score: 0,
            avg_speed: 0.0,
            time: 0.0,
        };
        game.rebuild_traffic();
        game
    }

    pub fn restart(&mut self) {
        self.vehicle.reset();
        self.wrecks = 0;
        self.wreck_timer = 0.0;
        self.game_over = false;
        self.engine_blown = false;
        self.engine_heat = 0.0;
        self.perfect_shift_timer = 0.0;
        self.speed_kmh = 0.0;
        self.score = 0;
        self.bonus_score = 0;
        self.avg_speed = 0.0;
        self.time = 0.0;
        self.weather_phase = random_weather_phase();
        self.rebuild_traffic();
    }

    pub fn set_weather(&mut self, weather: Weather) {
        self.weather = weather;
    }

    /// Effective cloud coverage (0..1) for the sky. `Auto` animates a slow
    /// cycle whose start is randomized per run.
    pub fn cloud_amount(&self) -> f32 {
        match self.weather {
            Weather::Auto => {
                let c = 0.5 + 0.5 * (self.time * WEATHER_CYCLE_SPEED + self.weather_phase).sin();
                c.clamp(0.12, 0.9)
            }
            w => w.cloud_amount(),
        }
    }

    pub fn set_difficulty(&mut self, difficulty: DifficultyLevel) {
        if self.difficulty == difficulty {
            return;
        }
        self.difficulty = difficulty;
        self.wreck_timer = 0.0;
        self.wrecks = 0;
        self.game_over = false;
        self.engine_blown = false;
        self.engine_heat = 0.0;
        self.perfect_shift_timer = 0.0;
        self.bonus_score = 0;
        self.rebuild_traffic();
    }

    fn rebuild_traffic(&mut self) {
        let tuning = self.difficulty.tuning();
        self.traffic = rebuild_traffic(&tuning, self.vehicle.distance);
    }

    pub fn update(&mut self, dt: Duration, input: &Input) {
        let dt = dt.as_secs_f32().min(0.05);
        if self.game_over {
            return;
        }
        let tuning: DifficultyTuning = self.difficulty.tuning();

        if self.engine_blown {
            // Engine dead: the car only coasts to a stop while the world keeps
            // moving; collisions no longer matter.
            self.vehicle.update(dt, input, false);
            update_traffic(&mut self.traffic, &self.vehicle, &tuning, dt);
            if self.vehicle.speed <= COAST_STOP_SPEED {
                self.game_over = true;
                self.best_score = self.best_score.max(self.score);
            }
            self.update_stats(dt);
            return;
        }

        let gear_before = self.vehicle.gear;
        self.vehicle.update(dt, input, true);
        update_traffic(&mut self.traffic, &self.vehicle, &tuning, dt);

        // Judge a gear change at the pre-shift ratio.
        if self.vehicle.gear > gear_before {
            let frac = self.vehicle.rpm_frac_for(gear_before);
            if frac >= PERFECT_LO && frac <= PERFECT_HI {
                let bonus =
                    (PERFECT_SHIFT_BONUS as f32 * self.difficulty.score_multiplier()) as u32;
                self.bonus_score += bonus;
                self.perfect_shift_timer = PERFECT_SHIFT_POPUP_TIME;
                self.vehicle.boost = BOOST_DURATION;
            } else if frac > RED_ZONE_START {
                self.engine_heat += RED_SHIFT_HEAT_KICK;
            }
        }

        if self.wreck_timer <= 0.0 {
            if check_collision(&self.traffic, &self.vehicle, &tuning) {
                self.wrecks += 1;
                self.wreck_timer = tuning.wreck_cooldown;
                self.vehicle.speed = 0.0;
                if self.wrecks >= tuning.wreck_limit {
                    self.game_over = true;
                    self.best_score = self.best_score.max(self.score);
                }
            }
        } else {
            self.wreck_timer -= dt;
        }

        // Engine heat: builds while revving in the danger zone, decays out of it.
        let frac = self.vehicle.rpm_frac();
        if frac >= RED_ZONE_START {
            self.engine_heat += tuning.engine_heat_rate * dt;
        } else {
            self.engine_heat -= tuning.engine_heat_cool * dt;
        }
        self.engine_heat = self.engine_heat.clamp(0.0, 1.0);
        if self.engine_heat >= 1.0 {
            self.engine_blown = true;
            self.vehicle.boost = 0.0;
            println!("Engine blown!");
        }

        self.update_stats(dt);
    }

    fn update_stats(&mut self, dt: f32) {
        self.perfect_shift_timer = (self.perfect_shift_timer - dt).max(0.0);
        self.time += dt;
        self.avg_speed = if self.time > 0.0 {
            self.vehicle.distance / self.time
        } else {
            0.0
        };
        // Base score grows with distance/speed; perfect-shift bonuses persist.
        let base = (self.vehicle.distance
            * (self.avg_speed * SCORE_SPEED_WEIGHT)
            * self.difficulty.score_multiplier()) as u32;
        self.score = base + self.bonus_score;
        self.speed_kmh = self.vehicle.speed * 3.6;
    }

    pub fn player_world_x(&self) -> f32 {
        road_curve(self.vehicle.distance) + self.vehicle.offset
    }

    pub fn player_world_z(&self) -> f32 {
        -self.vehicle.distance
    }
}

/// Random phase in [0, 2π) for the Auto weather cycle, derived from the clock.
fn random_weather_phase() -> f32 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before epoch")
        .as_nanos() as u64;
    let r = (nanos ^ (nanos >> 32)) & 0x00FF_FFFF;
    (r as f32 / 16_777_215.0) * std::f32::consts::TAU
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::vehicle::redline_speed;
    use crate::input::Input;

    fn dt() -> Duration {
        Duration::from_secs_f32(1.0 / 60.0)
    }

    fn pinned_in_red(game: &mut Game) {
        // Park the vehicle in the danger zone of gear 1 (over-revving).
        game.vehicle.gear = 1;
        game.vehicle.speed = (redline_speed(1) + redline_speed(2)) / 2.0;
    }

    #[test]
    fn sustained_red_zone_blows_the_engine_then_coasts_to_game_over() {
        let mut game = Game::new();
        game.traffic.clear();
        pinned_in_red(&mut game);
        let input = Input {
            throttle: true,
            ..Input::default()
        };

        for _ in 0..600 {
            game.update(dt(), &input);
            if game.engine_blown {
                break;
            }
        }
        assert!(game.engine_blown, "engine must blow from sustained red zone");
        assert!(!game.game_over, "car still coasts after the blow");

        // Coasting: the world keeps updating until the car stops.
        let mut ticks = 0;
        while !game.game_over {
            game.update(dt(), &input);
            ticks += 1;
            assert!(ticks < 1200, "car never came to a stop");
        }
        assert!(game.vehicle.speed <= COAST_STOP_SPEED, "car rolled to a stop");
        assert!(game.best_score >= game.score);
    }

    #[test]
    fn normal_driving_does_not_blow_the_engine() {
        let mut game = Game::new();
        game.traffic.clear();
        // Cruise well below the red zone.
        game.vehicle.speed = 8.0;
        for _ in 0..600 {
            game.update(dt(), &Input::default());
        }
        assert!(!game.engine_blown);
        assert_eq!(game.engine_heat, 0.0);
    }

    #[test]
    fn perfect_shift_awards_bonus_and_boost() {
        let mut game = Game::new();
        game.traffic.clear();
        // Reach the perfect-shift band in gear 1, then shift up.
        game.vehicle.gear = 1;
        game.vehicle.speed = (PERFECT_LO + PERFECT_HI) / 2.0 * redline_speed(1);
        let input = Input {
            gear_up: true,
            ..Input::default()
        };
        let before = game.score;
        game.update(dt(), &input);
        assert!(game.vehicle.gear > 1);
        assert!(game.perfect_shift_timer > 0.0, "popup timer set");
        assert!(game.vehicle.boost > 0.0, "boost granted");
        assert!(game.score > before, "score bonus awarded");
    }

    #[test]
    fn red_zone_shift_does_not_reward() {
        let mut game = Game::new();
        game.traffic.clear();
        game.vehicle.gear = 1;
        game.vehicle.speed = redline_speed(1);
        let input = Input {
            gear_up: true,
            ..Input::default()
        };
        game.update(dt(), &input);
        assert_eq!(game.perfect_shift_timer, 0.0);
        assert_eq!(game.vehicle.boost, 0.0);
        assert!(game.engine_heat > 0.0, "red-zone shift heats the engine");
    }
}
