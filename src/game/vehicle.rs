// SPDX-License-Identifier: MIT

use crate::input::Input;
use crate::road::{road_tangent, CAR_HALF_W, ROAD_HALF};

pub struct Vehicle {
    pub distance: f32,
    pub offset: f32,
    pub speed: f32,
    pub steer: f32,
    pub gear: u32,
    pub heading: f32,
}

impl Vehicle {
    pub fn new() -> Self {
        Vehicle {
            distance: 0.0,
            offset: 0.0,
            speed: 0.0,
            steer: 0.0,
            gear: 1,
            heading: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.distance = 0.0;
        self.offset = 0.0;
        self.speed = 0.0;
        self.steer = 0.0;
        self.gear = 1;
        self.heading = 0.0;
    }

    pub fn update(&mut self, dt: f32, input: &Input) {
        if input.gear_up {
            self.gear = (self.gear + 1).min(5);
        }
        if input.gear_down {
            self.gear = (self.gear as i32 - 1).max(1) as u32;
        }

        let target = input.steer;
        let turn_rate = 5.0 / (1.0 + self.speed * 0.02);
        self.steer += (target - self.steer) * (dt * turn_rate).min(1.0);

        let authority = (1.7 - self.speed * 0.0126).max(0.5);
        self.heading += self.steer * authority * dt;

        let travel = self.speed * dt;
        let cos_h = self.heading.cos();
        let sin_h = self.heading.sin();
        let tan = road_tangent(self.distance);

        self.distance += travel * cos_h;
        self.offset += travel * (sin_h - tan * cos_h);
        self.offset = self
            .offset
            .clamp(-(ROAD_HALF - CAR_HALF_W), ROAD_HALF - CAR_HALF_W);

        let gear = (self.gear as usize).min(5);
        let max_v = GEAR_MAX[gear];
        if input.throttle {
            self.speed += GEAR_ACCEL[gear] * dt;
        } else if input.brake {
            self.speed -= 14.0 * dt;
        } else {
            self.speed -= 3.0 * dt;
        }
        self.speed = self.speed.clamp(0.0, max_v);
    }
}

// top speed (m/s) and acceleration (m/s^2) per gear index (0 = neutral/unused)
const GEAR_MAX: [f32; 6] = [0.0, 18.0, 35.0, 55.0, 80.0, 95.0];
const GEAR_ACCEL: [f32; 6] = [0.0, 9.0, 7.0, 6.0, 5.0, 4.0];
