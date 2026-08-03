// SPDX-License-Identifier: MIT

use crate::font::{FontAtlas, ICON_CHIP, ICON_LOGOUT, ICON_PLAY, ICON_SPEEDOMETER, ICON_STEERING};
use crate::game::DifficultyLevel;
use crate::hud::{draw_text, push_panel_color, text_bounds, text_width, PANEL_PAD};
use crate::vertex::HudVertex;

const HIGHLIGHT: [f32; 4] = [0.35, 0.5, 0.7, 0.35];
const BACKDROP: [f32; 4] = [0.02, 0.03, 0.05, 0.55];
const TITLE_COLOR: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
const ROW_COLOR: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
const START_COLOR: [f32; 4] = [0.3, 1.0, 0.4, 1.0];
const EXIT_COLOR: [f32; 4] = [1.0, 0.45, 0.45, 1.0];
const FOOT_COLOR: [f32; 4] = [0.72, 0.78, 0.82, 1.0];

const ROW_EM: f32 = 0.055;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MenuRow {
    Gpu,
    Difficulty,
    Start,
    Exit,
}

impl MenuRow {
    /// The row above this one (clamped at the top).
    pub fn previous(self) -> Self {
        match self {
            MenuRow::Gpu => MenuRow::Gpu,
            MenuRow::Difficulty => MenuRow::Gpu,
            MenuRow::Start => MenuRow::Difficulty,
            MenuRow::Exit => MenuRow::Start,
        }
    }

    /// The row below this one (clamped at the bottom).
    pub fn next(self) -> Self {
        match self {
            MenuRow::Gpu => MenuRow::Difficulty,
            MenuRow::Difficulty => MenuRow::Start,
            MenuRow::Start => MenuRow::Exit,
            MenuRow::Exit => MenuRow::Exit,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MenuState {
    pub gpu_index: usize,
    pub difficulty: DifficultyLevel,
    pub cursor: MenuRow,
    difficulty_changed: bool,
}

impl MenuState {
    pub fn new(gpu_index: usize) -> Self {
        MenuState {
            gpu_index,
            difficulty: DifficultyLevel::EasyArcade,
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

fn push_menu_row(
    out: &mut Vec<HudVertex>,
    atlas: &FontAtlas,
    aspect: f32,
    text: &str,
    y: f32,
    em: f32,
    color: [f32; 4],
    focused: bool,
) {
    let px_to_ndc_x = em / atlas.raster_px / aspect;
    let w = text_width(atlas, text, px_to_ndc_x);
    let x = -w / 2.0;
    if focused {
        if let Some((min_x, top, max_x, bottom)) = text_bounds(atlas, text, x, y, em, aspect) {
            push_panel_color(
                out,
                min_x - PANEL_PAD,
                top + PANEL_PAD,
                (max_x - min_x) + 2.0 * PANEL_PAD,
                (top - bottom) + 2.0 * PANEL_PAD,
                HIGHLIGHT,
            );
        }
    }
    draw_text(out, atlas, text, x, y, em, aspect, color);
}

pub fn build_menu_vertices(
    menu: &MenuState,
    gpu_names: &[String],
    atlas: &FontAtlas,
    aspect: f32,
) -> Vec<HudVertex> {
    let mut out: Vec<HudVertex> = Vec::new();

    // Backdrop card
    push_panel_color(&mut out, -0.55, 0.78, 1.1, 1.33, BACKDROP);

    // Title
    let title = format!("{}  LANE LUNACY", ICON_STEERING);
    let title_em = 0.15;
    let tw = text_width(atlas, &title, title_em / atlas.raster_px / aspect);
    draw_text(
        &mut out,
        atlas,
        &title,
        -tw / 2.0,
        0.68,
        title_em,
        aspect,
        TITLE_COLOR,
    );

    // Rows
    let gpu_t = gpu_label(menu.gpu_index, gpu_names);
    let mode_t = format!("{}  MODE  {}", ICON_SPEEDOMETER, menu.difficulty.label());

    push_menu_row(
        &mut out,
        atlas,
        aspect,
        &gpu_t,
        0.22,
        ROW_EM,
        ROW_COLOR,
        menu.cursor == MenuRow::Gpu,
    );
    push_menu_row(
        &mut out,
        atlas,
        aspect,
        &mode_t,
        0.12,
        ROW_EM,
        ROW_COLOR,
        menu.cursor == MenuRow::Difficulty,
    );
    push_menu_row(
        &mut out,
        atlas,
        aspect,
        &format!("{}  START", ICON_PLAY),
        0.02,
        ROW_EM,
        START_COLOR,
        menu.cursor == MenuRow::Start,
    );
    push_menu_row(
        &mut out,
        atlas,
        aspect,
        &format!("{}  EXIT", ICON_LOGOUT),
        -0.08,
        ROW_EM,
        EXIT_COLOR,
        menu.cursor == MenuRow::Exit,
    );

    // Footer hint
    let foot = "UP/DOWN MOVE, LEFT/RIGHT CHANGE, ENTER CONFIRM";
    let foot_em = 0.032;
    let fw = text_width(atlas, foot, foot_em / atlas.raster_px / aspect);
    draw_text(
        &mut out,
        atlas,
        foot,
        -fw / 2.0,
        -0.34,
        foot_em,
        aspect,
        FOOT_COLOR,
    );

    out
}
