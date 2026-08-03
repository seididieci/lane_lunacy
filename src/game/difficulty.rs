// SPDX-License-Identifier: MIT

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DifficultyLevel {
    EasyArcade,
    Normal,
    Hard,
}

#[derive(Clone, Copy)]
pub struct DifficultyTuning {
    pub traffic_count: usize,
    pub init_start: f32,
    pub init_gap: f32,
    pub lane_offset: f32,
    pub speed_base: f32,
    pub speed_var: f32,
    pub respawn_behind: f32,
    pub respawn_ahead: f32,
    pub respawn_gap: f32,
    pub respawn_min_gap: f32,
    pub respawn_retry_step: f32,
    pub wall_gap: f32,
    pub collision_long_mul: f32,
    pub collision_lat_mul: f32,
    pub wreck_cooldown: f32,
    pub wreck_limit: u32,
}

impl DifficultyLevel {
    pub fn tuning(self) -> DifficultyTuning {
        match self {
            DifficultyLevel::EasyArcade => DifficultyTuning {
                traffic_count: 4,
                init_start: 95.0,
                init_gap: 74.0,
                lane_offset: 2.6,
                speed_base: 8.0,
                speed_var: 4.0,
                respawn_behind: 30.0,
                respawn_ahead: 210.0,
                respawn_gap: 36.0,
                respawn_min_gap: 44.0,
                respawn_retry_step: 34.0,
                wall_gap: 20.0,
                collision_long_mul: 0.72,
                collision_lat_mul: 1.15,
                wreck_cooldown: 0.9,
                wreck_limit: 7,
            },
            DifficultyLevel::Normal => DifficultyTuning {
                traffic_count: 5,
                init_start: 85.0,
                init_gap: 58.0,
                lane_offset: 2.5,
                speed_base: 9.5,
                speed_var: 4.8,
                respawn_behind: 26.0,
                respawn_ahead: 175.0,
                respawn_gap: 30.0,
                respawn_min_gap: 36.0,
                respawn_retry_step: 28.0,
                wall_gap: 12.0,
                collision_long_mul: 0.8,
                collision_lat_mul: 1.25,
                wreck_cooldown: 1.0,
                wreck_limit: 6,
            },
            DifficultyLevel::Hard => DifficultyTuning {
                traffic_count: 6,
                init_start: 75.0,
                init_gap: 46.0,
                lane_offset: 2.4,
                speed_base: 11.0,
                speed_var: 5.8,
                respawn_behind: 22.0,
                respawn_ahead: 145.0,
                respawn_gap: 22.0,
                respawn_min_gap: 30.0,
                respawn_retry_step: 22.0,
                wall_gap: 6.0,
                collision_long_mul: 0.9,
                collision_lat_mul: 1.35,
                wreck_cooldown: 1.2,
                wreck_limit: 5,
            },
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            DifficultyLevel::EasyArcade => "EASY",
            DifficultyLevel::Normal => "NORMAL",
            DifficultyLevel::Hard => "HARD",
        }
    }

    pub fn score_multiplier(self) -> f32 {
        match self {
            DifficultyLevel::EasyArcade => 1.0,
            DifficultyLevel::Normal => 1.5,
            DifficultyLevel::Hard => 2.0,
        }
    }
}
