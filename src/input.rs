// SPDX-License-Identifier: MIT

#[derive(Default)]
pub struct Input {
    pub throttle: bool,
    pub brake: bool,
    pub left: bool,
    pub right: bool,
    pub steer: f32,
    pub gear_up: bool,
    pub gear_down: bool,
}

impl Input {
    pub fn sync_keyboard_steer(&mut self) {
        self.steer = match (self.left, self.right) {
            (true, false) => -1.0,
            (false, true) => 1.0,
            _ => 0.0,
        };
    }
}
