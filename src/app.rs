// SPDX-License-Identifier: MIT

use std::sync::Arc;
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowAttributes};

use vulkano::device::physical::PhysicalDevice;
use vulkano::image::{SampleCount, SampleCounts};
use vulkano::instance::Instance;
use vulkano::swapchain::Surface;

use crate::font::FontAtlas;
use crate::game::{Game, Weather};
use crate::gpu::{create_graphics_context, enumerate_devices, select_physical_device};
use crate::hud::build_hud_tree;
use crate::input::Input;
use crate::menu::{
    build_menu_tree, AaMode, MenuRow, MenuScreen, MenuState, SettingsRow, SettingsState,
};
use crate::render::daynight;
use crate::render::{FxSettings, Renderer};
use crate::ui::Ui;

enum AppMode {
    Menu,
    Playing,
}

pub struct App {
    instance: Arc<Instance>,
    window: Option<Arc<Window>>,
    surface: Option<Arc<Surface>>,
    renderer: Option<Renderer>,
    gpu_names: Vec<String>,
    active_gpu_index: usize,
    /// The settings currently in effect (committed by APPLY). `menu.settings`
    /// holds the staged values; the APPLY row is enabled only when they differ.
    applied: SettingsState,
    /// Antialiasing modes the currently-applied GPU supports.
    supported_aa: Vec<AaMode>,
    seed: u64,
    menu: MenuState,
    mode: AppMode,
    game: Game,
    input: Input,
    ui: Ui,
    font_atlas: FontAtlas,
    ui_clock: f32,
    previous: Instant,
}

impl App {
    pub fn new(
        instance: Arc<Instance>,
        gpu_index: usize,
        weather: Weather,
        start_hour: Option<f32>,
        seed: u64,
    ) -> Self {
        let mut game = Game::new();
        game.set_weather(weather);
        game.set_seed(seed);
        if let Some(hour) = start_hour {
            game.set_start_hour(hour);
            // Hint where the sun sits at spawn so the flare can be lined up.
            let day_fraction = game.difficulty.tuning().day_fraction;
            let dir =
                daynight::sun_direction(game.sun_elevation(), game.time_of_day(), day_fraction);
            let az = dir[0].atan2(dir[2]).to_degrees().rem_euclid(360.0);
            let elev = dir[1]
                .atan2((dir[0] * dir[0] + dir[2] * dir[2]).sqrt())
                .to_degrees();
            println!(
                "--time {hour}: sun azimuth {az:.0}°, elevation {elev:.0}° (az 0° = +Z, 90° = +X)"
            );
        }
        Self {
            instance,
            window: None,
            surface: None,
            renderer: None,
            gpu_names: Vec::new(),
            active_gpu_index: gpu_index,
            applied: SettingsState {
                gpu_index,
                weather,
                ..SettingsState::default()
            },
            supported_aa: vec![AaMode::Off],
            seed,
            menu: MenuState::new(gpu_index, weather),
            mode: AppMode::Menu,
            game,
            input: Input::default(),
            ui: Ui::new(),
            font_atlas: FontAtlas::load(),
            ui_clock: 0.0,
            previous: Instant::now(),
        }
    }

    fn init_if_needed(&mut self, event_loop: &ActiveEventLoop) {
        if self.renderer.is_some() {
            return;
        }

        let window = Arc::new(
            event_loop
                .create_window(
                    WindowAttributes::default()
                        .with_title("Lane Lunacy")
                        .with_inner_size(LogicalSize::new(1280, 720)),
                )
                .expect("failed to create window"),
        );

        let surface = Surface::from_window(self.instance.clone(), window.clone())
            .expect("failed to create surface");

        let devices = enumerate_devices(&self.instance);
        self.gpu_names = devices
            .iter()
            .map(|d| d.properties().device_name.clone())
            .collect();
        let max_index = self.gpu_names.len().saturating_sub(1);
        if self.menu.settings.gpu_index > max_index {
            self.menu.settings.gpu_index = max_index;
        }
        let physical = select_physical_device(&devices, self.menu.settings.gpu_index);
        self.supported_aa = supported_aa_modes(&physical);
        self.menu.clamp_antialias(&self.supported_aa);
        self.applied.antialias = self.menu.settings.antialias;
        let (device, queue) = create_graphics_context(surface.clone(), &physical);
        let renderer = Renderer::new(
            device,
            queue,
            surface.clone(),
            window.clone(),
            &physical,
            &self.font_atlas,
            self.seed,
            aa_samples(self.supported_aa[self.applied.antialias]),
        );

        self.active_gpu_index = self.menu.settings.gpu_index;
        self.renderer = Some(renderer);
        self.surface = Some(surface);
        self.window = Some(window);
        self.previous = Instant::now();

        println!(
            "Controls: W / Up = throttle | S / Down = brake | A / Left & D / Right = steer | E = gear up | Q = gear down | R = restart | ESC = pause menu"
        );
    }

    fn toggle_menu(&mut self) {
        match self.mode {
            AppMode::Menu => self.resume_game(),
            AppMode::Playing => {
                self.menu.open_for_pause();
                self.mode = AppMode::Menu;
            }
        }
    }

    fn close_menu(&mut self) {
        self.resume_game();
    }

    fn resume_game(&mut self) {
        // Settings are committed exclusively by the APPLY row; START just
        // resumes the run with whatever is currently in effect.
        self.mode = AppMode::Playing;
    }

    /// Commits the staged settings in one shot. Difficulty/weather apply to the
    /// live game (difficulty restarts the run), and GPU switches the backend;
    /// an AA-only change rebuilds the current backend in place.
    fn apply_settings(&mut self) {
        if self.menu.settings == self.applied {
            return;
        }
        if self.menu.settings.difficulty != self.applied.difficulty {
            self.game.set_difficulty(self.menu.settings.difficulty);
            self.game.restart();
            println!("Run restarted (difficulty changed)");
        }
        self.game.set_weather(self.menu.settings.weather);
        let gpu_changed = self
            .menu
            .settings
            .gpu_index
            .min(self.gpu_names.len().saturating_sub(1))
            != self.active_gpu_index;
        self.switch_gpu();
        if !gpu_changed {
            if let Some(renderer) = &mut self.renderer {
                renderer.set_aa(aa_samples(self.supported_aa[self.menu.settings.antialias]));
            }
        }
        self.applied = self.menu.settings;
    }

    /// Switches the graphics backend to `menu.settings.gpu_index` when it
    /// differs from the active GPU. The window surface is reused and the run
    /// keeps its state. Capability support for the target GPU is re-detected
    /// here and the staged AA index is clamped to it.
    fn switch_gpu(&mut self) {
        let chosen = self
            .menu
            .settings
            .gpu_index
            .min(self.gpu_names.len().saturating_sub(1));
        if chosen == self.active_gpu_index {
            return;
        }
        let Some(surface) = self.surface.clone() else {
            return;
        };
        let Some(window) = self.window.clone() else {
            return;
        };

        // Tear down the old backend first. Its swapchain still owns the window, and
        // Vulkan refuses to create a new swapchain for that surface until the old one
        // is destroyed (VK_ERROR_NATIVE_WINDOW_IN_USE_KHR).
        if let Some(old) = self.renderer.take() {
            old.wait_idle();
        }

        let devices = enumerate_devices(&self.instance);
        let physical = select_physical_device(&devices, chosen);
        self.supported_aa = supported_aa_modes(&physical);
        self.menu.clamp_antialias(&self.supported_aa);
        let (device, queue) = create_graphics_context(surface.clone(), &physical);
        let renderer = Renderer::new(
            device,
            queue,
            surface.clone(),
            window,
            &physical,
            &self.font_atlas,
            self.seed,
            aa_samples(self.supported_aa[self.menu.settings.antialias]),
        );
        self.active_gpu_index = chosen;
        self.renderer = Some(renderer);
    }

    fn handle_menu_key(&mut self, event_loop: &ActiveEventLoop, kb: &KeyEvent, press: bool) {
        if !press {
            return;
        }
        match self.menu.screen {
            MenuScreen::Main => self.handle_main_key(event_loop, kb),
            MenuScreen::Settings => self.handle_settings_key(kb),
        }
    }

    fn handle_main_key(&mut self, event_loop: &ActiveEventLoop, kb: &KeyEvent) {
        match kb.physical_key {
            PhysicalKey::Code(KeyCode::ArrowUp) | PhysicalKey::Code(KeyCode::KeyW) => {
                self.menu.main_cursor = self.menu.main_cursor.previous();
            }
            PhysicalKey::Code(KeyCode::ArrowDown) | PhysicalKey::Code(KeyCode::KeyS) => {
                self.menu.main_cursor = self.menu.main_cursor.next();
            }
            PhysicalKey::Code(KeyCode::Enter) => match self.menu.main_cursor {
                MenuRow::Start => self.close_menu(),
                MenuRow::Settings => self.menu.open_settings(),
                MenuRow::Exit => event_loop.exit(),
            },
            _ => {}
        }
    }

    fn handle_settings_key(&mut self, kb: &KeyEvent) {
        let device_count = self.gpu_names.len();
        match kb.physical_key {
            PhysicalKey::Code(KeyCode::ArrowUp) | PhysicalKey::Code(KeyCode::KeyW) => {
                self.menu.settings_cursor = self.menu.settings_cursor.previous();
            }
            PhysicalKey::Code(KeyCode::ArrowDown) | PhysicalKey::Code(KeyCode::KeyS) => {
                self.menu.settings_cursor = self.menu.settings_cursor.next();
            }
            PhysicalKey::Code(KeyCode::ArrowLeft) | PhysicalKey::Code(KeyCode::KeyA) => {
                self.cycle_settings_row(-1, device_count);
            }
            PhysicalKey::Code(KeyCode::ArrowRight) | PhysicalKey::Code(KeyCode::KeyD) => {
                self.cycle_settings_row(1, device_count);
            }
            PhysicalKey::Code(KeyCode::Enter) => match self.menu.settings_cursor {
                SettingsRow::Apply => self.apply_settings(),
                SettingsRow::Back => self.menu.back_to_main(),
                _ => {}
            },
            _ => {}
        }
    }

    /// Left/Right handler for the settings rows: cycles GPU/mode/weather/AA and
    /// toggles the FX rows. APPLY/BACK have no value to cycle.
    fn cycle_settings_row(&mut self, delta: i32, device_count: usize) {
        match self.menu.settings_cursor {
            SettingsRow::Gpu => self.menu.cycle_gpu(delta, device_count),
            SettingsRow::Mode => self.menu.cycle_difficulty(delta),
            SettingsRow::Weather => self.menu.cycle_weather(delta),
            SettingsRow::Antialias => self.menu.cycle_antialias(delta, &self.supported_aa),
            SettingsRow::Fxaa
            | SettingsRow::Bloom
            | SettingsRow::Vignette
            | SettingsRow::Grain
            | SettingsRow::Saturation
            | SettingsRow::ChromaticAberration => self.menu.toggle_fx(self.menu.settings_cursor),
            SettingsRow::Apply | SettingsRow::Back => {}
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        event_loop.set_control_flow(ControlFlow::Poll);
        self.init_if_needed(event_loop);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let Some(window) = &self.window else {
            return;
        };
        if window.id() != window_id {
            return;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(_) => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.recreate = true;
                }
            }
            WindowEvent::KeyboardInput { event: kb, .. } => {
                let pressed = kb.state == ElementState::Pressed;
                let press = pressed && !kb.repeat;
                match kb.physical_key {
                    PhysicalKey::Code(KeyCode::Escape) if press => {
                        // In the settings submenu Esc steps back to the main
                        // menu; elsewhere it toggles the menu itself.
                        if matches!(self.mode, AppMode::Menu)
                            && self.menu.screen == MenuScreen::Settings
                        {
                            self.menu.back_to_main();
                        } else {
                            self.toggle_menu();
                        }
                    }
                    _ if matches!(self.mode, AppMode::Menu) => {
                        self.handle_menu_key(event_loop, &kb, press);
                    }
                    _ => match kb.physical_key {
                        PhysicalKey::Code(KeyCode::KeyW) | PhysicalKey::Code(KeyCode::ArrowUp) => {
                            self.input.throttle = pressed;
                        }
                        PhysicalKey::Code(KeyCode::KeyS)
                        | PhysicalKey::Code(KeyCode::ArrowDown) => {
                            self.input.brake = pressed;
                        }
                        PhysicalKey::Code(KeyCode::KeyA)
                        | PhysicalKey::Code(KeyCode::ArrowLeft) => {
                            self.input.left = pressed;
                            self.input.sync_keyboard_steer();
                        }
                        PhysicalKey::Code(KeyCode::KeyD)
                        | PhysicalKey::Code(KeyCode::ArrowRight) => {
                            self.input.right = pressed;
                            self.input.sync_keyboard_steer();
                        }
                        PhysicalKey::Code(KeyCode::KeyE) => {
                            if press {
                                self.input.gear_up = true;
                            }
                        }
                        PhysicalKey::Code(KeyCode::KeyQ) => {
                            if press {
                                self.input.gear_down = true;
                            }
                        }
                        PhysicalKey::Code(KeyCode::KeyR) if press => {
                            self.game.restart();
                            println!("Run restarted");
                        }
                        _ => {}
                    },
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let Some(renderer) = &mut self.renderer else {
            return;
        };

        let now = Instant::now();
        let dt = now.duration_since(self.previous);
        self.previous = now;
        self.ui_clock += dt.as_secs_f32();

        if matches!(self.mode, AppMode::Playing) {
            self.game.update(dt, &self.input);
            self.input.gear_up = false;
            self.input.gear_down = false;
        }

        let aspect = self.window.as_ref().map_or(16.0 / 9.0, |w| {
            let size = w.inner_size();
            if size.height == 0 {
                16.0 / 9.0
            } else {
                size.width as f32 / size.height as f32
            }
        });
        let mut root = match &self.mode {
            AppMode::Menu => build_menu_tree(
                &self.menu,
                &self.gpu_names,
                &self.supported_aa,
                self.menu.settings != self.applied,
            ),
            AppMode::Playing => build_hud_tree(&self.game),
        };
        let hud_verts = self
            .ui
            .build(&mut root, &self.font_atlas, aspect, self.ui_clock);
        let fx = FxSettings {
            fxaa: self.applied.fxaa,
            bloom: self.applied.bloom,
            vignette: self.applied.vignette,
            grain: self.applied.grain,
            saturation: self.applied.saturation,
            chroma: self.applied.chroma,
        };
        renderer.render(&self.game, dt, &hud_verts, &fx);
    }
}

/// MSAA modes the physical device supports, from the intersection of the
/// color and depth framebuffer sample-count capabilities. Sample 1 (AA off)
/// is always available.
fn supported_aa_modes(physical: &Arc<PhysicalDevice>) -> Vec<AaMode> {
    let props = physical.properties();
    let supported = props.framebuffer_color_sample_counts & props.framebuffer_depth_sample_counts;
    let mut modes = vec![AaMode::Off];
    if supported.intersects(SampleCounts::SAMPLE_2) {
        modes.push(AaMode::X2);
    }
    if supported.intersects(SampleCounts::SAMPLE_4) {
        modes.push(AaMode::X4);
    }
    modes
}

fn aa_samples(mode: AaMode) -> SampleCount {
    match mode {
        AaMode::Off => SampleCount::Sample1,
        AaMode::X2 => SampleCount::Sample2,
        AaMode::X4 => SampleCount::Sample4,
    }
}
