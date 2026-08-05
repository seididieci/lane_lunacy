// SPDX-License-Identifier: MIT

use crate::font::{ICON_CHIP, ICON_LOGOUT, ICON_PLAY, ICON_SPEEDOMETER, ICON_STEERING, ICON_WEATHER};
use crate::game::{DifficultyLevel, Weather};
use crate::ui::{Align, Button, Column, HAlign, Insets, Node, Overlay, Panel, Spacer, Text};

const BACKDROP: [f32; 4] = [0.02, 0.03, 0.05, 0.55];
const TITLE_COLOR: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
const ROW_COLOR: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
const START_COLOR: [f32; 4] = [0.3, 1.0, 0.4, 1.0];
const EXIT_COLOR: [f32; 4] = [1.0, 0.45, 0.45, 1.0];
const FOOT_COLOR: [f32; 4] = [0.72, 0.78, 0.82, 1.0];

const TITLE_EM: f32 = 72.0;
const ROW_EM: f32 = 30.0;
const FOOT_EM: f32 = 18.0;
const ROW_GAP: f32 = 18.0;
const SECTION_GAP: f32 = 64.0;
const CARD_PAD: Insets = Insets::new(64.0, 48.0, 64.0, 48.0);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MenuRow {
    Gpu,
    Difficulty,
    Weather,
    Start,
    Exit,
}

impl MenuRow {
    /// The row above this one (clamped at the top).
    pub fn previous(self) -> Self {
        match self {
            MenuRow::Gpu => MenuRow::Gpu,
            MenuRow::Difficulty => MenuRow::Gpu,
            MenuRow::Weather => MenuRow::Difficulty,
            MenuRow::Start => MenuRow::Weather,
            MenuRow::Exit => MenuRow::Start,
        }
    }

    /// The row below this one (clamped at the bottom).
    pub fn next(self) -> Self {
        match self {
            MenuRow::Gpu => MenuRow::Difficulty,
            MenuRow::Difficulty => MenuRow::Weather,
            MenuRow::Weather => MenuRow::Start,
            MenuRow::Start => MenuRow::Exit,
            MenuRow::Exit => MenuRow::Exit,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MenuState {
    pub gpu_index: usize,
    pub difficulty: DifficultyLevel,
    pub weather: Weather,
    pub cursor: MenuRow,
    difficulty_changed: bool,
}

impl MenuState {
    pub fn new(gpu_index: usize, weather: Weather) -> Self {
        MenuState {
            gpu_index,
            difficulty: DifficultyLevel::EasyArcade,
            weather,
            cursor: MenuRow::Gpu,
            difficulty_changed: false,
        }
    }

    pub fn open_for_pause(&mut self) {
        self.cursor = MenuRow::Start;
        self.difficulty_changed = false;
    }

    /// Returns true if the difficulty was changed while this menu was open, and
    /// resets the flag so the change is only acted upon once.
    pub fn take_difficulty_changed(&mut self) -> bool {
        let changed = self.difficulty_changed;
        self.difficulty_changed = false;
        changed
    }

    pub fn cycle_gpu(&mut self, delta: i32, device_count: usize) {
        if device_count == 0 {
            return;
        }
        self.gpu_index = (self.gpu_index as i32 + delta).rem_euclid(device_count as i32) as usize;
    }

    pub fn cycle_difficulty(&mut self, delta: i32) {
        let levels = [
            DifficultyLevel::EasyArcade,
            DifficultyLevel::Normal,
            DifficultyLevel::Hard,
        ];
        let cur = levels
            .iter()
            .position(|l| *l == self.difficulty)
            .unwrap_or(0);
        let next = (cur as i32 + delta).rem_euclid(levels.len() as i32) as usize;
        self.difficulty = levels[next];
        self.difficulty_changed = true;
    }

    pub fn cycle_weather(&mut self, delta: i32) {
        let states = [Weather::Auto, Weather::Clear, Weather::Cloudy, Weather::Rain];
        let cur = states
            .iter()
            .position(|w| *w == self.weather)
            .unwrap_or(0);
        let next = (cur as i32 + delta).rem_euclid(states.len() as i32) as usize;
        self.weather = states[next];
    }
}

fn gpu_label(index: usize, names: &[String]) -> String {
    let name = names.get(index).map(|s| s.as_str()).unwrap_or("?");
    const MAX_CHARS: usize = 40;
    let trimmed: String = name.chars().take(MAX_CHARS).collect();
    if name.chars().count() > MAX_CHARS {
        format!("{} GPU  [{}]  {}...", ICON_CHIP, index, trimmed)
    } else {
        format!("{} GPU  [{}]  {}", ICON_CHIP, index, trimmed)
    }
}

/// Builds the menu widget tree for the given state.
pub(crate) fn build_menu_tree(menu: &MenuState, gpu_names: &[String]) -> Node {
    let title = format!("{}  LANE LUNACY", ICON_STEERING);
    let gpu_t = gpu_label(menu.gpu_index, gpu_names);
    let mode_t = format!("{}  MODE  {}", ICON_SPEEDOMETER, menu.difficulty.label());
    let weather_t = format!("{}  WEATHER  {}", ICON_WEATHER, menu.weather.label());
    let foot = "UP/DOWN MOVE, LEFT/RIGHT CHANGE, ENTER CONFIRM";

    let rows = Column::new(
        vec![
            Node::new(
                Button::new(gpu_t, ROW_EM, ROW_COLOR, 0).focused(menu.cursor == MenuRow::Gpu),
            ),
            Node::new(
                Button::new(mode_t, ROW_EM, ROW_COLOR, 1)
                    .focused(menu.cursor == MenuRow::Difficulty),
            ),
            Node::new(
                Button::new(weather_t, ROW_EM, ROW_COLOR, 2)
                    .focused(menu.cursor == MenuRow::Weather),
            ),
            Node::new(
                Button::new(format!("{}  START", ICON_PLAY), ROW_EM, START_COLOR, 3)
                    .focused(menu.cursor == MenuRow::Start),
            ),
            Node::new(
                Button::new(format!("{}  EXIT", ICON_LOGOUT), ROW_EM, EXIT_COLOR, 4)
                    .focused(menu.cursor == MenuRow::Exit),
            ),
        ],
        ROW_GAP,
        HAlign::Center,
    );

    let card = Column::new(
        vec![
            Node::new(Text::new(title, TITLE_EM, TITLE_COLOR).aligned(HAlign::Center)),
            Node::new(Spacer::new(0.0, SECTION_GAP)),
            Node::new(rows),
            Node::new(Spacer::new(0.0, SECTION_GAP)),
            Node::new(Text::new(foot, FOOT_EM, FOOT_COLOR).aligned(HAlign::Center)),
        ],
        0.0,
        HAlign::Center,
    );

    Node::new(
        Overlay::new().child(
            Align::Center,
            Node::new(Panel::wrap(BACKDROP, CARD_PAD, Node::new(card))),
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::font::FontAtlas;
    use crate::ui::{Point, Ui};

    #[test]
    fn menu_builds_vertices_and_centers_rows() {
        let menu = MenuState::new(0, Weather::Auto);
        let names = vec!["Test GPU".to_string()];
        let atlas = FontAtlas::load();
        let ui = Ui::new();

        let mut root = build_menu_tree(&menu, &names);
        let verts = ui.build(&mut root, &atlas, 16.0 / 9.0, 0.0);
        assert!(!verts.is_empty());

        // The START button (id 3) must be hit-testable somewhere on the canvas
        // center line, wherever the rows land.
        let canvas = ui.virtual_size(16.0 / 9.0);
        let cx = canvas.w / 2.0;
        let mut start_hit = false;
        let mut y = 0.0;
        while y <= canvas.h {
            if let Some(hit) = ui.hit_test(&root, Point::new(cx, y)) {
                if hit.id == 3 {
                    start_hit = true;
                    break;
                }
            }
            y += 8.0;
        }
        assert!(start_hit, "START button must be hit-testable on the center line");
    }
}
