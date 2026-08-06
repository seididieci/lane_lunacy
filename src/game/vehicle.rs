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
    /// Remaining seconds of perfect-shift acceleration boost.
    pub boost: f32,
    /// Whether the player was holding throttle this frame (drives launch dust).
    pub throttle: bool,
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
            boost: 0.0,
            throttle: false,
        }
    }

    pub fn reset(&mut self) {
        self.distance = 0.0;
        self.offset = 0.0;
        self.speed = 0.0;
        self.steer = 0.0;
        self.gear = 1;
        self.heading = 0.0;
        self.boost = 0.0;
        self.throttle = false;
    }

    /// RPM at the current speed and gear. Idles at standstill, reaches
    /// `REDLINE_RPM` at `redline_speed(gear)`.
    pub fn rpm(&self) -> f32 {
        let frac = self.rpm_frac();
        IDLE_RPM + frac * (REDLINE_RPM - IDLE_RPM)
    }

    /// Normalized revs in 0..=1 against the redline.
    pub fn rpm_frac(&self) -> f32 {
        let gear = (self.gear as usize).min(5);
        let redline = redline_speed(gear);
        if redline <= 0.0 {
            return 0.0;
        }
        (self.speed / redline).clamp(0.0, 1.0)
    }

    /// RPM fraction with `gear` replaced by the given gear (used to judge a
    /// gear change at the pre-shift ratio, before the gear increments).
    pub fn rpm_frac_for(&self, gear: u32) -> f32 {
        let gear = (gear as usize).min(5);
        let redline = redline_speed(gear);
        if redline <= 0.0 {
            return 0.0;
        }
        (self.speed / redline).clamp(0.0, 1.0)
    }

    /// Update drivetrain + steering. When `drivetrain_live` is false the engine
    /// is dead: throttle/brake/gear are ignored and the car only coasts to a
    /// stop on drag (used after an engine blow).
    pub fn update(&mut self, dt: f32, input: &Input, drivetrain_live: bool) {
        if drivetrain_live {
            if input.gear_up {
                self.gear = (self.gear + 1).min(5);
            }
            if input.gear_down {
                self.gear = (self.gear as i32 - 1).max(1) as u32;
            }
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
        let redline = redline_speed(gear);
        // The actual speed clamp is gear 1-4's rev limiter; 5th gear is capped
        // at its own top speed (the redline reference sits far beyond it).
        let speed_limit = if gear >= 5 { GEAR_MAX[5] } else { redline };
        if drivetrain_live {
            let boost_mul = if self.boost > 0.0 { BOOST_ACCEL_MUL } else { 1.0 };
            self.boost = (self.boost - dt).max(0.0);
            if input.throttle {
                // Soft rev limiter: above the gear's natural top speed the
                // engine gradually loses power, so the needle creeps toward the
                // redline instead of slamming into it.
                let overrev = (redline - GEAR_MAX[gear]).max(1.0);
                let limiter_scale =
                    ((redline - self.speed) / overrev).clamp(0.0, 1.0);
                self.speed += GEAR_ACCEL[gear] * boost_mul * limiter_scale * dt;
            } else if input.brake {
                self.speed -= BRAKE_DECEL * dt;
            } else {
                self.speed -= DRAG_DECEL * dt;
            }
        } else {
            self.speed -= COAST_DECEL * dt;
        }
        self.speed = self.speed.clamp(0.0, speed_limit);
        self.throttle = drivetrain_live && input.throttle;
    }
}

// top speed (m/s) and acceleration (m/s^2) per gear index (0 = neutral/unused)
const GEAR_MAX: [f32; 6] = [0.0, 18.0, 35.0, 55.0, 80.0, 95.0];
const GEAR_ACCEL: [f32; 6] = [0.0, 9.0, 7.0, 6.0, 5.0, 4.0];

const BRAKE_DECEL: f32 = 14.0;
const DRAG_DECEL: f32 = 3.0;
/// Strong deceleration used while the engine is dead (after a blow), so the
/// car rolls to a stop promptly.
const COAST_DECEL: f32 = 10.0;

/// Fraction of `GEAR_MAX` a redline-speed reference sits past a gear's top speed.
pub const LIMITER_FRAC: f32 = 1.12;
/// 5th gear is an overdrive: its redline reference sits this far past its top
/// speed so the revs can never reach the red zone at top speed.
const GEAR5_OVERDRIVE: f32 = 1.5;

/// Speed (m/s) at which a gear's RPM equals the redline.
pub fn redline_speed(gear: usize) -> f32 {
    if gear >= 5 {
        GEAR_MAX[5] * GEAR5_OVERDRIVE
    } else {
        GEAR_MAX[gear] * LIMITER_FRAC
    }
}

/// Rev range, in RPM.
pub const IDLE_RPM: f32 = 900.0;
pub const REDLINE_RPM: f32 = 8000.0;

/// Normalized revs at which the danger zone begins (frac >= this => heat builds).
pub const RED_ZONE_START: f32 = 0.90;
/// Perfect-shift band: shifting up while frac is in [PERFECT_LO, PERFECT_HI].
pub const PERFECT_LO: f32 = 0.78;
pub const PERFECT_HI: f32 = 0.90;

/// Perfect-shift reward: seconds of acceleration boost and its multiplier.
pub const BOOST_DURATION: f32 = 1.2;
pub const BOOST_ACCEL_MUL: f32 = 1.6;
/// Extra heat kick when the player shifts up while in the red zone.
pub const RED_SHIFT_HEAT_KICK: f32 = 0.08;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::Input;

    fn idle_input() -> Input {
        Input::default()
    }

    #[test]
    fn rpm_idles_at_standstill_and_redlines_at_the_limiter() {
        let mut v = Vehicle::new();
        v.update(0.0, &idle_input(), true);
        assert!((v.rpm() - IDLE_RPM).abs() < 1.0);

        // In 1st gear the limiter is GEAR_MAX[1] * LIMITER_FRAC.
        v.speed = redline_speed(1);
        v.gear = 1;
        assert!((v.rpm() - REDLINE_RPM).abs() < 1.0);
        assert!((v.rpm_frac() - 1.0).abs() < 1e-3);
    }

    #[test]
    fn fifth_gear_cannot_reach_the_red_zone() {
        let mut v = Vehicle::new();
        v.gear = 5;
        v.speed = GEAR_MAX[5];
        assert!(v.rpm_frac() < RED_ZONE_START);
        assert!(v.rpm() < REDLINE_RPM);
    }

    #[test]
    fn soft_limiter_decays_acceleration_past_gear_top_speed() {
        let mut v = Vehicle::new();
        v.gear = 1;
        v.speed = GEAR_MAX[1];
        let input = Input {
            throttle: true,
            ..Input::default()
        };
        // Gear-1 limiter is GEAR_MAX[1] * LIMITER_FRAC; the car must not exceed it.
        for _ in 0..300 {
            v.update(0.016, &input, true);
            assert!(v.speed <= redline_speed(1) + 1e-3);
        }
        assert!(v.speed >= GEAR_MAX[1]);
    }

    #[test]
    fn boost_decays_over_time() {
        let mut v = Vehicle::new();
        v.boost = BOOST_DURATION;
        v.update(BOOST_DURATION / 2.0, &idle_input(), true);
        assert!(v.boost > 0.0);
        v.update(BOOST_DURATION, &idle_input(), true);
        assert_eq!(v.boost, 0.0);
    }

    #[test]
    fn dead_drivetrain_coasts_to_a_stop() {
        let mut v = Vehicle::new();
        v.speed = 40.0;
        let input = Input {
            throttle: true,
            ..Input::default()
        };
        // With the engine dead, throttle is ignored: the car only slows down.
        for _ in 0..400 {
            v.update(0.016, &input, false);
        }
        assert_eq!(v.speed, 0.0);
    }
}
