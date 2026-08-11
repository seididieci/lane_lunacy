// SPDX-License-Identifier: MIT

use crate::font::{
    ICON_CHIP, ICON_LOGOUT, ICON_PLAY, ICON_SPEEDOMETER, ICON_STEERING, ICON_WEATHER,
};
use crate::game::{DifficultyLevel, Weather};
use crate::mesh::TerrainDetail;
use crate::ui::{Align, Button, Column, HAlign, Insets, Node, Overlay, Panel, Spacer, Text};

const BACKDROP: [f32; 4] = [0.02, 0.03, 0.05, 0.55];
const TITLE_COLOR: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
const ROW_COLOR: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
const START_COLOR: [f32; 4] = [0.3, 1.0, 0.4, 1.0];
const EXIT_COLOR: [f32; 4] = [1.0, 0.45, 0.45, 1.0];
const DIM_COLOR: [f32; 4] = [0.5, 0.55, 0.6, 1.0];
const FOOT_COLOR: [f32; 4] = [0.72, 0.78, 0.82, 1.0];

const TITLE_EM: f32 = 72.0;
const ROW_EM: f32 = 30.0;
const FOOT_EM: f32 = 18.0;
const ROW_GAP: f32 = 18.0;
const SECTION_GAP: f32 = 64.0;
const CARD_PAD: Insets = Insets::new(64.0, 48.0, 64.0, 48.0);

/// Which menu screen is currently open. `Main` is the title/pause entry point;
/// `Settings` is a submenu reached from it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MenuScreen {
    Main,
    Settings,
}

/// Rows of the main menu (title screen and pause menu). `Mode`/`Weather` are
/// value rows cycled with Left/Right and committed immediately; the others are
/// activated with Enter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MenuRow {
    Start,
    Mode,
    Weather,
    Settings,
    Exit,
}

impl MenuRow {
    /// The row above this one (clamped at the top).
    pub fn previous(self) -> Self {
        match self {
            MenuRow::Start => MenuRow::Start,
            MenuRow::Mode => MenuRow::Start,
            MenuRow::Weather => MenuRow::Mode,
            MenuRow::Settings => MenuRow::Weather,
            MenuRow::Exit => MenuRow::Settings,
        }
    }

    /// The row below this one (clamped at the bottom).
    pub fn next(self) -> Self {
        match self {
            MenuRow::Start => MenuRow::Mode,
            MenuRow::Mode => MenuRow::Weather,
            MenuRow::Weather => MenuRow::Settings,
            MenuRow::Settings => MenuRow::Exit,
            MenuRow::Exit => MenuRow::Exit,
        }
    }
}

/// Rows of the settings submenu.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SettingsRow {
    Gpu,
    Antialias,
    TerrainDetail,
    Fxaa,
    Bloom,
    Vignette,
    Grain,
    Saturation,
    ChromaticAberration,
    Apply,
    Back,
}

impl SettingsRow {
    /// The row above this one (clamped at the top).
    pub fn previous(self) -> Self {
        match self {
            SettingsRow::Gpu => SettingsRow::Gpu,
            SettingsRow::Antialias => SettingsRow::Gpu,
            SettingsRow::TerrainDetail => SettingsRow::Antialias,
            SettingsRow::Fxaa => SettingsRow::TerrainDetail,
            SettingsRow::Bloom => SettingsRow::Fxaa,
            SettingsRow::Vignette => SettingsRow::Bloom,
            SettingsRow::Grain => SettingsRow::Vignette,
            SettingsRow::Saturation => SettingsRow::Grain,
            SettingsRow::ChromaticAberration => SettingsRow::Saturation,
            SettingsRow::Apply => SettingsRow::ChromaticAberration,
            SettingsRow::Back => SettingsRow::Apply,
        }
    }

    /// The row below this one (clamped at the bottom).
    pub fn next(self) -> Self {
        match self {
            SettingsRow::Gpu => SettingsRow::Antialias,
            SettingsRow::Antialias => SettingsRow::TerrainDetail,
            SettingsRow::TerrainDetail => SettingsRow::Fxaa,
            SettingsRow::Fxaa => SettingsRow::Bloom,
            SettingsRow::Bloom => SettingsRow::Vignette,
            SettingsRow::Vignette => SettingsRow::Grain,
            SettingsRow::Grain => SettingsRow::Saturation,
            SettingsRow::Saturation => SettingsRow::ChromaticAberration,
            SettingsRow::ChromaticAberration => SettingsRow::Apply,
            SettingsRow::Apply => SettingsRow::Back,
            SettingsRow::Back => SettingsRow::Back,
        }
    }
}

/// Antialiasing modes offered by the ANTIALIASING row. Stored as an index into
/// the capability-gated `supported` list, so the row only cycles modes the
/// selected GPU actually supports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AaMode {
    Off,
    X2,
    X4,
}

impl AaMode {
    pub fn label(self) -> &'static str {
        match self {
            AaMode::Off => "OFF",
            AaMode::X2 => "MSAA 2x",
            AaMode::X4 => "MSAA 4x",
        }
    }
}

/// The ten user-adjustable settings. Staged in the menu, committed in one shot
/// by the APPLY row; `PartialEq` lets the app show the APPLY row as enabled only
/// when the staged values differ from what is in effect.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SettingsState {
    pub gpu_index: usize,
    pub difficulty: DifficultyLevel,
    pub weather: Weather,
    /// Index into the capability-gated `AaMode` list.
    pub antialias: usize,
    pub terrain_detail: TerrainDetail,
    pub fxaa: bool,
    pub bloom: bool,
    pub vignette: bool,
    pub grain: bool,
    pub saturation: bool,
    pub chroma: bool,
}

impl Default for SettingsState {
    fn default() -> Self {
        SettingsState {
            gpu_index: 0,
            difficulty: DifficultyLevel::EasyArcade,
            weather: Weather::Auto,
            antialias: 0,
            terrain_detail: TerrainDetail::Medium,
            fxaa: false,
            bloom: false,
            vignette: false,
            grain: false,
            saturation: false,
            chroma: false,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MenuState {
    pub screen: MenuScreen,
    pub main_cursor: MenuRow,
    pub settings_cursor: SettingsRow,
    pub settings: SettingsState,
}

impl MenuState {
    pub fn new(gpu_index: usize, weather: Weather) -> Self {
        MenuState {
            screen: MenuScreen::Main,
            main_cursor: MenuRow::Start,
            settings_cursor: SettingsRow::Gpu,
            settings: SettingsState {
                gpu_index,
                weather,
                ..SettingsState::default()
            },
        }
    }

    /// Opens the pause menu on the main screen with START selected.
    pub fn open_for_pause(&mut self) {
        self.screen = MenuScreen::Main;
        self.main_cursor = MenuRow::Start;
        self.settings_cursor = SettingsRow::Gpu;
    }

    /// Enters the settings submenu.
    pub fn open_settings(&mut self) {
        self.screen = MenuScreen::Settings;
        self.settings_cursor = SettingsRow::Gpu;
    }

    /// Returns to the main menu.
    pub fn back_to_main(&mut self) {
        self.screen = MenuScreen::Main;
    }

    pub fn cycle_gpu(&mut self, delta: i32, device_count: usize) {
        if device_count == 0 {
            return;
        }
        self.settings.gpu_index =
            (self.settings.gpu_index as i32 + delta).rem_euclid(device_count as i32) as usize;
    }

    pub fn cycle_difficulty(&mut self, delta: i32) {
        let levels = [
            DifficultyLevel::EasyArcade,
            DifficultyLevel::Normal,
            DifficultyLevel::Hard,
        ];
        let cur = levels
            .iter()
            .position(|l| *l == self.settings.difficulty)
            .unwrap_or(0);
        let next = (cur as i32 + delta).rem_euclid(levels.len() as i32) as usize;
        self.settings.difficulty = levels[next];
    }

    pub fn cycle_terrain_detail(&mut self, delta: i32) {
        let levels = [
            TerrainDetail::Low,
            TerrainDetail::Medium,
            TerrainDetail::High,
        ];
        let cur = levels
            .iter()
            .position(|l| *l == self.settings.terrain_detail)
            .unwrap_or(1);
        let next = (cur as i32 + delta).rem_euclid(levels.len() as i32) as usize;
        self.settings.terrain_detail = levels[next];
    }

    pub fn cycle_weather(&mut self, delta: i32) {
        let states = [
            Weather::Auto,
            Weather::Clear,
            Weather::Cloudy,
            Weather::Rain,
        ];
        let cur = states
            .iter()
            .position(|w| *w == self.settings.weather)
            .unwrap_or(0);
        let next = (cur as i32 + delta).rem_euclid(states.len() as i32) as usize;
        self.settings.weather = states[next];
    }

    /// Cycles the ANTIALIASING row over the supported modes only, skipping any
    /// mode the GPU does not offer.
    pub fn cycle_antialias(&mut self, delta: i32, supported: &[AaMode]) {
        if supported.is_empty() {
            self.settings.antialias = 0;
            return;
        }
        let cur = self
            .settings
            .antialias
            .min(supported.len().saturating_sub(1));
        self.settings.antialias = (cur as i32 + delta).rem_euclid(supported.len() as i32) as usize;
    }

    /// Clamps the ANTIALIASING index when the supported list shrinks (e.g. the
    /// staged GPU changed to one with fewer MSAA modes).
    pub fn clamp_antialias(&mut self, supported: &[AaMode]) {
        if supported.is_empty() {
            self.settings.antialias = 0;
            return;
        }
        self.settings.antialias = self
            .settings
            .antialias
            .min(supported.len().saturating_sub(1));
    }

    /// Toggles the boolean FX row `row` (no-op for non-FX rows).
    pub fn toggle_fx(&mut self, row: SettingsRow) {
        match row {
            SettingsRow::Fxaa => self.settings.fxaa = !self.settings.fxaa,
            SettingsRow::Bloom => self.settings.bloom = !self.settings.bloom,
            SettingsRow::Vignette => self.settings.vignette = !self.settings.vignette,
            SettingsRow::Grain => self.settings.grain = !self.settings.grain,
            SettingsRow::Saturation => self.settings.saturation = !self.settings.saturation,
            SettingsRow::ChromaticAberration => self.settings.chroma = !self.settings.chroma,
            _ => {}
        }
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

fn on_off(on: bool) -> &'static str {
    if on {
        "ON"
    } else {
        "OFF"
    }
}

/// Builds the widget tree for the currently open menu screen.
pub(crate) fn build_menu_tree(
    menu: &MenuState,
    gpu_names: &[String],
    supported_aa: &[AaMode],
    dirty: bool,
) -> Node {
    match menu.screen {
        MenuScreen::Main => build_main_tree(menu),
        MenuScreen::Settings => build_settings_tree(menu, gpu_names, supported_aa, dirty),
    }
}

fn build_main_tree(menu: &MenuState) -> Node {
    let title = format!("{}  LANE LUNACY", ICON_STEERING);
    let foot = "UP/DOWN MOVE, LEFT/RIGHT CHANGE, ENTER CONFIRM";

    let rows = Column::new(
        vec![
            Node::new(
                Button::new(format!("{}  START", ICON_PLAY), ROW_EM, START_COLOR, 0)
                    .focused(menu.main_cursor == MenuRow::Start),
            ),
            Node::new(
                Button::new(
                    format!(
                        "{}  MODE  {}",
                        ICON_SPEEDOMETER,
                        menu.settings.difficulty.label()
                    ),
                    ROW_EM,
                    ROW_COLOR,
                    1,
                )
                .focused(menu.main_cursor == MenuRow::Mode),
            ),
            Node::new(
                Button::new(
                    format!(
                        "{}  WEATHER  {}",
                        ICON_WEATHER,
                        menu.settings.weather.label()
                    ),
                    ROW_EM,
                    ROW_COLOR,
                    2,
                )
                .focused(menu.main_cursor == MenuRow::Weather),
            ),
            Node::new(
                Button::new("SETTINGS", ROW_EM, ROW_COLOR, 3)
                    .focused(menu.main_cursor == MenuRow::Settings),
            ),
            Node::new(
                Button::new(format!("{}  EXIT", ICON_LOGOUT), ROW_EM, EXIT_COLOR, 4)
                    .focused(menu.main_cursor == MenuRow::Exit),
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

    Node::new(Overlay::new().child(
        Align::Center,
        Node::new(Panel::wrap(BACKDROP, CARD_PAD, Node::new(card))),
    ))
}

fn build_settings_tree(
    menu: &MenuState,
    gpu_names: &[String],
    supported_aa: &[AaMode],
    dirty: bool,
) -> Node {
    let s = &menu.settings;
    let title = format!("{}  SETTINGS", ICON_SPEEDOMETER);
    let gpu_t = gpu_label(s.gpu_index, gpu_names);
    let aa_label = supported_aa
        .get(s.antialias)
        .map(|m| m.label())
        .unwrap_or("OFF");
    let aa_t = format!("ANTIALIASING  {}", aa_label);
    let terrain_t = format!("TERRAIN DETAIL  {}", s.terrain_detail.label());
    let foot = "UP/DOWN MOVE, LEFT/RIGHT CHANGE, ENTER CONFIRM";
    let apply_color = if dirty { ROW_COLOR } else { DIM_COLOR };

    let focused = |row: SettingsRow| menu.settings_cursor == row;
    let rows = Column::new(
        vec![
            Node::new(Button::new(gpu_t, ROW_EM, ROW_COLOR, 10).focused(focused(SettingsRow::Gpu))),
            Node::new(
                Button::new(aa_t, ROW_EM, ROW_COLOR, 11).focused(focused(SettingsRow::Antialias)),
            ),
            Node::new(
                Button::new(terrain_t, ROW_EM, ROW_COLOR, 12)
                    .focused(focused(SettingsRow::TerrainDetail)),
            ),
            Node::new(
                Button::new(format!("FXAA  {}", on_off(s.fxaa)), ROW_EM, ROW_COLOR, 13)
                    .focused(focused(SettingsRow::Fxaa)),
            ),
            Node::new(
                Button::new(format!("BLOOM  {}", on_off(s.bloom)), ROW_EM, ROW_COLOR, 14)
                    .focused(focused(SettingsRow::Bloom)),
            ),
            Node::new(
                Button::new(
                    format!("VIGNETTE  {}", on_off(s.vignette)),
                    ROW_EM,
                    ROW_COLOR,
                    15,
                )
                .focused(focused(SettingsRow::Vignette)),
            ),
            Node::new(
                Button::new(format!("GRAIN  {}", on_off(s.grain)), ROW_EM, ROW_COLOR, 16)
                    .focused(focused(SettingsRow::Grain)),
            ),
            Node::new(
                Button::new(
                    format!("SATURATION  {}", on_off(s.saturation)),
                    ROW_EM,
                    ROW_COLOR,
                    17,
                )
                .focused(focused(SettingsRow::Saturation)),
            ),
            Node::new(
                Button::new(
                    format!("CHROMATIC  {}", on_off(s.chroma)),
                    ROW_EM,
                    ROW_COLOR,
                    18,
                )
                .focused(focused(SettingsRow::ChromaticAberration)),
            ),
            Node::new(
                Button::new("APPLY", ROW_EM, apply_color, 19).focused(focused(SettingsRow::Apply)),
            ),
            Node::new(
                Button::new("BACK", ROW_EM, ROW_COLOR, 20).focused(focused(SettingsRow::Back)),
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

    Node::new(Overlay::new().child(
        Align::Center,
        Node::new(Panel::wrap(BACKDROP, CARD_PAD, Node::new(card))),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::font::FontAtlas;
    use crate::ui::{Point, Ui};

    fn names() -> Vec<String> {
        vec!["Test GPU".to_string()]
    }

    fn hit_test_id(root: &Node, ui: &Ui, id: u64) -> bool {
        let canvas = ui.virtual_size(16.0 / 9.0);
        let cx = canvas.w / 2.0;
        let mut y = 0.0;
        while y <= canvas.h {
            if let Some(hit) = ui.hit_test(root, Point::new(cx, y)) {
                if hit.id == id {
                    return true;
                }
            }
            y += 8.0;
        }
        false
    }

    #[test]
    fn main_menu_builds_and_centers_start_first() {
        let menu = MenuState::new(0, Weather::Auto);
        let supported = [AaMode::Off];
        let atlas = FontAtlas::load();
        let ui = Ui::new();

        let mut root = build_menu_tree(&menu, &names(), &supported, false);
        let verts = ui.build(&mut root, &atlas, 16.0 / 9.0, 0.0);
        assert!(!verts.is_empty());

        // START (id 0) is the first row, so it must be hit-testable on the
        // canvas center line.
        assert!(
            hit_test_id(&root, &ui, 0),
            "START button must be hit-testable on the center line"
        );
    }

    #[test]
    fn settings_screen_builds_all_rows() {
        let mut menu = MenuState::new(0, Weather::Auto);
        menu.open_settings();
        let supported = [AaMode::Off, AaMode::X2, AaMode::X4];
        let atlas = FontAtlas::load();
        let ui = Ui::new();

        let mut root = build_menu_tree(&menu, &names(), &supported, true);
        let verts = ui.build(&mut root, &atlas, 16.0 / 9.0, 0.0);
        assert!(!verts.is_empty());

        // Every settings row must be present and hit-testable on the center line.
        for id in 10..=20 {
            assert!(
                hit_test_id(&root, &ui, id),
                "settings row id {id} must be hit-testable"
            );
        }
    }

    #[test]
    fn main_menu_shows_mode_and_weather_rows() {
        let menu = MenuState::new(0, Weather::Auto);
        let supported = [AaMode::Off];
        let atlas = FontAtlas::load();
        let ui = Ui::new();

        let mut root = build_menu_tree(&menu, &names(), &supported, false);
        let verts = ui.build(&mut root, &atlas, 16.0 / 9.0, 0.0);
        assert!(!verts.is_empty());

        // START, MODE, WEATHER, SETTINGS and EXIT all present on the center line.
        for id in 0..=4 {
            assert!(
                hit_test_id(&root, &ui, id),
                "main menu row id {id} must be hit-testable"
            );
        }
    }

    #[test]
    fn difficulty_and_weather_cycle_through_all_states() {
        let mut menu = MenuState::new(0, Weather::Auto);
        let all = [
            DifficultyLevel::EasyArcade,
            DifficultyLevel::Normal,
            DifficultyLevel::Hard,
            DifficultyLevel::EasyArcade,
        ];
        for level in all {
            assert_eq!(menu.settings.difficulty, level);
            menu.cycle_difficulty(1);
        }
        let weathers = [
            Weather::Auto,
            Weather::Clear,
            Weather::Cloudy,
            Weather::Rain,
            Weather::Auto,
        ];
        for weather in weathers {
            assert_eq!(menu.settings.weather, weather);
            menu.cycle_weather(1);
        }
    }

    #[test]
    fn main_cursor_clamps_at_ends() {
        assert_eq!(MenuRow::Start.previous(), MenuRow::Start);
        assert_eq!(MenuRow::Exit.next(), MenuRow::Exit);
        assert_eq!(MenuRow::Start.next(), MenuRow::Mode);
        assert_eq!(MenuRow::Mode.next(), MenuRow::Weather);
        assert_eq!(MenuRow::Weather.next(), MenuRow::Settings);
        assert_eq!(MenuRow::Settings.next(), MenuRow::Exit);
        assert_eq!(MenuRow::Settings.previous(), MenuRow::Weather);
    }

    #[test]
    fn settings_cursor_clamps_at_ends() {
        assert_eq!(SettingsRow::Gpu.previous(), SettingsRow::Gpu);
        assert_eq!(SettingsRow::Back.next(), SettingsRow::Back);
        assert_eq!(SettingsRow::Back.previous(), SettingsRow::Apply);
        assert_eq!(SettingsRow::Gpu.next(), SettingsRow::Antialias);
    }

    #[test]
    fn aa_cycle_skips_unsupported_modes() {
        // Only OFF and 4x supported: cycling wraps and must never land on 2x.
        let supported = [AaMode::Off, AaMode::X4];
        let mut menu = MenuState::new(0, Weather::Auto);
        assert_eq!(menu.settings.antialias, 0);

        menu.cycle_antialias(1, &supported);
        assert_eq!(menu.settings.antialias, 1);
        assert_eq!(supported[menu.settings.antialias], AaMode::X4);

        menu.cycle_antialias(1, &supported);
        assert_eq!(menu.settings.antialias, 0);
        assert_eq!(supported[menu.settings.antialias], AaMode::Off);
    }

    #[test]
    fn terrain_detail_cycles_through_all_levels() {
        let mut menu = MenuState::new(0, Weather::Auto);
        assert_eq!(menu.settings.terrain_detail, TerrainDetail::Medium);
        menu.cycle_terrain_detail(1);
        assert_eq!(menu.settings.terrain_detail, TerrainDetail::High);
        menu.cycle_terrain_detail(1);
        assert_eq!(menu.settings.terrain_detail, TerrainDetail::Low);
        menu.cycle_terrain_detail(1);
        assert_eq!(menu.settings.terrain_detail, TerrainDetail::Medium);
        menu.cycle_terrain_detail(-1);
        assert_eq!(menu.settings.terrain_detail, TerrainDetail::Low);
    }

    #[test]
    fn clamp_antialias_shrinks_index() {
        let mut menu = MenuState::new(0, Weather::Auto);
        menu.settings.antialias = 2;
        menu.clamp_antialias(&[AaMode::Off]);
        assert_eq!(menu.settings.antialias, 0);
    }

    #[test]
    fn fx_toggles_flip_bool_rows() {
        let mut menu = MenuState::new(0, Weather::Auto);
        menu.toggle_fx(SettingsRow::Bloom);
        assert!(menu.settings.bloom);
        menu.toggle_fx(SettingsRow::Bloom);
        assert!(!menu.settings.bloom);
        // Non-FX rows are no-ops.
        menu.toggle_fx(SettingsRow::Apply);
        assert!(!menu.settings.fxaa);
    }
}
