// SPDX-License-Identifier: MIT

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Fullscreen, Window, WindowAttributes};

#[cfg(target_os = "linux")]
use winit::platform::wayland::WindowAttributesExtWayland;

use vulkano::device::physical::PhysicalDevice;
use vulkano::image::{SampleCount, SampleCounts};
use vulkano::instance::Instance;
use vulkano::swapchain::Surface;

use crate::audio::{
    available_backends, enumerate_output_devices, AudioCapture, AudioEngine, AudioSettings,
    SharedCapture, Sfx,
};
use crate::cli::PresentMode;
use crate::debug::DebugStats;
use crate::font::FontAtlas;
use crate::game::{Game, Weather};
use crate::gpu::{create_graphics_context, enumerate_devices, select_physical_device};
use crate::hud::build_hud_tree;
use crate::input::Input;
use crate::menu::{
    build_menu_tree, AaMode, AudioRow, GraphicsRow, MenuRow, MenuScreen, MenuState, SettingsRow,
    SettingsState,
};
use crate::profiler::SessionProfiler;
use crate::render::daynight;
use crate::render::{FxSettings, Renderer};
use crate::ui::Ui;
use crate::world::terrain::{terrain_height, terrain_slope, RISE_START};

enum AppMode {
    Menu,
    Playing,
}

/// Windowed size: 90% of 1920×1080. The window starts at this size (restored
/// after leaving fullscreen); the default launch is borderless fullscreen.
const WINDOWED_SIZE: LogicalSize<f64> = LogicalSize::new(1728.0, 972.0);

/// Sway `app_id` used by the `for_window`/criteria float rules. A normal
/// xdg-shell toplevel has no native "float me" hint, so on sway the windowed
/// mode is floated via IPC (see [`sway_float_register`]/[`sway_float_window`]).
const SWAY_APP_ID: &str = "lane_lunacy";

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
    /// Names of the audio backends (cpal hosts) compiled into this build.
    audio_backends: Vec<String>,
    /// Names of the staged backend's output devices, deduplicated.
    audio_devices: Vec<String>,
    /// Index of the staged backend's default output device, if any.
    audio_default: Option<usize>,
    /// Audio backend. `None` when no usable output device exists.
    audio: Option<AudioEngine>,
    /// Previous-frame audio event state for edge-triggered SFX.
    audio_prev_wrecks: u32,
    audio_prev_blown: bool,
    audio_prev_perfect_timer: f32,
    audio_prev_gear: u32,
    /// Monotonically increasing frame counter for the audio param thread.
    audio_frame: u32,
    /// `--audio-capture <path.csv>`: per-frame game RPM vs audio pitch capture.
    audio_capture: Option<SharedCapture>,
    /// `--auto-drive <seconds>`: remaining seconds of scripted driving; the app
    /// exits when it reaches zero.
    auto_drive_remaining: Option<f32>,
    seed: u64,
    menu: MenuState,
    mode: AppMode,
    game: Game,
    input: Input,
    ui: Ui,
    font_atlas: FontAtlas,
    ui_clock: f32,
    previous: Instant,
    /// F3: dev-only diagnostics overlay (FPS, timings, mesh volume).
    debug_visible: bool,
    debug: DebugStats,
    /// `false` (default) starts borderless fullscreen; `--windowed` sets this.
    windowed: bool,
    /// `--profile <path.csv>`: records per-frame timings and writes a Markdown
    /// report on close. `None` keeps the hot path untouched.
    profiler: Option<SessionProfiler>,
    /// `--present <mode>`: swapchain present mode requested at startup and on
    /// every GPU switch.
    present_mode: PresentMode,
    /// `--fps-limit <N>`: optional frame-rate cap enforced as an idle sleep, so
    /// the per-frame work (`total_ms`) is measured without the cap wait.
    fps_limit: Option<u32>,
    window_capture: Option<PathBuf>,
    window_capture_armed: bool,
    capture_dir: Option<PathBuf>,
    rockwall_view: bool,
    auto_start: bool,
    forced_camera_heading: Option<f32>,
    capture_seq: u64,
    /// Monotonic frame counter for the profiler rows.
    profile_frame_idx: u64,
    /// End of the previous `about_to_wait`, for measuring event-loop idle time.
    profile_frame_end: Option<Instant>,
}

impl App {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        instance: Arc<Instance>,
        gpu_index: usize,
        weather: Weather,
        start_hour: Option<f32>,
        seed: u64,
        windowed: bool,
        // `--debug`: start with the F3 debug HUD enabled.
        debug: bool,
        // `--raytrace`: start with the ray-traced backend enabled.
        raytrace: bool,
        profile: Option<PathBuf>,
        present_mode: PresentMode,
        fps_limit: Option<u32>,
        window_capture: Option<PathBuf>,
        capture_dir: Option<PathBuf>,
        audio_capture: Option<PathBuf>,
        auto_drive: Option<f32>,
        audio_device: Option<usize>,
        rockwall_view: bool,
        auto_start: bool,
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
        let mut app = Self {
            instance,
            window: None,
            surface: None,
            renderer: None,
            gpu_names: Vec::new(),
            active_gpu_index: gpu_index,
            applied: SettingsState {
                gpu_index,
                weather,
                raytrace,
                ..SettingsState::default()
            },
            supported_aa: vec![AaMode::Off],
            audio_backends: available_backends().into_iter().map(|(_, n)| n).collect(),
            audio_devices: Vec::new(),
            audio_default: None,
            audio: None,
            audio_prev_wrecks: 0,
            audio_prev_blown: false,
            audio_prev_perfect_timer: 0.0,
            audio_prev_gear: 1,
            audio_frame: 0,
            audio_capture: audio_capture.and_then(|p| {
                AudioCapture::open(&p)
                    .inspect_err(|e| {
                        eprintln!("audio capture: failed to open {}: {e}", p.display());
                    })
                    .ok()
                    .map(Arc::new)
            }),
            auto_drive_remaining: auto_drive,
            seed,
            menu: MenuState {
                settings: SettingsState {
                    raytrace,
                    audio: AudioSettings {
                        device_index: audio_device.unwrap_or(0),
                        ..AudioSettings::default()
                    },
                    ..SettingsState::default()
                },
                ..MenuState::new(gpu_index, weather)
            },
            mode: AppMode::Menu,
            game,
            input: Input::default(),
            ui: Ui::new(),
            font_atlas: FontAtlas::load(),
            ui_clock: 0.0,
            previous: Instant::now(),
            debug_visible: debug,
            debug: DebugStats::default(),
            windowed,
            profiler: profile.and_then(|p| {
                SessionProfiler::open(&p)
                    .inspect_err(|e| {
                        eprintln!("profiler: failed to open {}: {e}", p.display());
                    })
                    .ok()
            }),
            present_mode,
            fps_limit,
            window_capture,
            window_capture_armed: false,
            capture_dir,
            rockwall_view,
            auto_start,
            forced_camera_heading: None,
            capture_seq: 0,
            profile_frame_idx: 0,
            profile_frame_end: None,
        };
        if app.auto_start {
            // `--auto-start`: skip the menu and start the session at the spawn
            // (normal chase camera, deterministic world from the seed). Lets
            // capture/profile runs hit the live scene instead of the title.
            app.mode = AppMode::Playing;
        }
        if app.rockwall_view {
            app.activate_rockwall_view();
        }
        app
    }

    fn activate_rockwall_view(&mut self) {
        // Deterministic A/B probe: skip the menu, park the car at the steepest
        // nearby roadside wall, and yaw 90° toward it so captures frame the
        // rock face directly (isolates wall-tiling/relief artifacts quickly).
        self.mode = AppMode::Playing;

        let (target_s, side) = pick_rockwall_probe();
        let old_s = self.game.vehicle.distance;
        self.game.vehicle.distance = target_s;
        let shoulder = crate::road::ROAD_HALF - crate::road::CAR_HALF_W - 0.25;
        self.game.vehicle.offset = side * shoulder.max(0.0) * 0.8;
        self.game.vehicle.heading = side * std::f32::consts::FRAC_PI_2;
        self.forced_camera_heading = Some(self.game.vehicle.heading);
        self.game.vehicle.speed = 0.0;
        self.game.vehicle.steer = 0.0;
        self.game.vehicle.boost = 0.0;
        self.game.vehicle.throttle = false;
        self.game.vehicle.height = terrain_height(self.game.vehicle.distance, 0.0);

        println!(
            "rockwall-view: s={:.1}, side={}, heading={:.0}deg",
            target_s,
            if side >= 0.0 { "right" } else { "left" },
            self.game.vehicle.heading.to_degrees()
        );

        // Keep traffic around the probe location instead of the spawn origin.
        let delta = target_s - old_s;
        for t in &mut self.game.traffic {
            t.distance += delta;
        }
    }

    fn init_if_needed(&mut self, event_loop: &ActiveEventLoop) {
        if self.renderer.is_some() {
            return;
        }

        // `--windowed`: register the sway float rule before the window appears
        // so the fixed 90%-FHD size is honored (a normal toplevel can't float
        // itself on sway). No-op when sway/swaymsg are absent.
        if self.windowed {
            sway_float_register();
        }

        let mut attrs = WindowAttributes::default()
            .with_title("Lane Lunacy")
            .with_inner_size(WINDOWED_SIZE);
        #[cfg(target_os = "linux")]
        {
            attrs = attrs.with_name(SWAY_APP_ID, "Lane Lunacy");
        }
        if !self.windowed {
            attrs = attrs.with_fullscreen(Some(Fullscreen::Borderless(None)));
        }
        let window = Arc::new(
            event_loop
                .create_window(attrs)
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
        // Default every effect on at launch, gated by what the GPU supports:
        // the best MSAA mode (list is built ascending Off/2x/4x) and all post
        // effects, which every Vulkan device can run. The user can dial any of
        // them down in SETTINGS -> APPLY.
        self.menu.settings = SettingsState {
            antialias: self.supported_aa.len().saturating_sub(1),
            fxaa: true,
            bloom: true,
            vignette: true,
            grain: true,
            saturation: true,
            chroma: true,
            ..self.menu.settings
        };
        self.menu.clamp_antialias(&self.supported_aa);
        self.applied = self.menu.settings;
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
            self.present_mode.to_vulkan(),
        );

        self.active_gpu_index = self.menu.settings.gpu_index;
        self.renderer = Some(renderer);
        if let (Some(renderer), Some(heading)) = (&mut self.renderer, self.forced_camera_heading) {
            renderer.set_camera_heading(heading);
        }
        if let (Some(renderer), Some(path)) = (&mut self.renderer, self.window_capture.clone()) {
            renderer.request_window_capture(path);
            self.window_capture_armed = true;
        }
        self.surface = Some(surface);
        self.window = Some(window);
        self.previous = Instant::now();

        // Audio: enumerate the staged backend's output devices and open the
        // selected one. The engine is optional -- no device means the game
        // runs silently.
        let (audio_devices, audio_default) =
            enumerate_output_devices(self.menu.settings.audio.backend);
        self.audio_devices = audio_devices;
        self.audio_default = audio_default;
        let audio = AudioEngine::init(self.menu.settings.audio, self.audio_capture.clone());
        match &audio {
            Some(_) => {
                for (i, name) in self.audio_devices.iter().enumerate() {
                    let tag = if self.audio_default == Some(i) { " (default)" } else { "" };
                    println!("audio device {i}: {name}{tag}");
                }
            }
            None => {
                println!("audio: no output device available; running silent");
            }
        }
        if let Some(cap) = &self.audio_capture {
            println!("audio capture enabled: {}", cap.path().display());
        }
        self.audio = audio;

        println!(
            "Controls: W / Up = throttle | S / Down = brake | A / Left & D / Right = steer | E = gear up | Q = gear down | R = restart | F3 / F4 = debug HUD | F10 = capture frame | F11 = fullscreen | ESC = pause menu"
        );
        if let Some(dir) = &self.capture_dir {
            println!("runtime capture enabled: F10 -> {}/shot_*.png", dir.display());
        }
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

    /// F11: toggles borderless fullscreen <-> the fixed 90%-FHD windowed size.
    /// The swapchain is rebuilt automatically by the `Resized -> recreate`
    /// path when the window changes size.
    fn toggle_fullscreen(&mut self) {
        let Some(window) = &self.window else {
            return;
        };
        if window.fullscreen().is_some() {
            window.set_fullscreen(None);
            let _ = window.request_inner_size(WINDOWED_SIZE);
            // The window exists now, so a direct criteria float applies. No-op
            // on non-sway compositors.
            sway_float_window();
        } else {
            window.set_fullscreen(Some(Fullscreen::Borderless(None)));
        }
    }

    fn close_menu(&mut self) {
        self.resume_game();
    }

    fn resume_game(&mut self) {
        // START resumes with what's in effect. MODE/WEATHER on this screen are
        // committed as you change them; the SETTINGS screen keeps the staged
        // APPLY model for GPU/AA/post effects.
        self.mode = AppMode::Playing;
    }

    /// Commits the staged graphics settings in one shot. GPU switches the
    /// backend; an AA/FX-only change rebuilds the current backend in place.
    /// Difficulty and weather are set on the main menu and are already in
    /// effect by the time APPLY can be reached. Only the graphics subset is
    /// committed; the audio subset has its own APPLY.
    fn apply_settings(&mut self) {
        if self.menu.settings.graphics_equal(&self.applied) {
            return;
        }
        let gpu_changed = self
            .menu
            .settings
            .gpu_index
            .min(self.gpu_names.len().saturating_sub(1))
            != self.active_gpu_index;
        self.switch_gpu();
        if let Some(renderer) = &mut self.renderer {
            // Applied even on a GPU switch: a fresh backend starts at the Medium
            // default, so it must be brought in line with the staged value.
            renderer.set_terrain_detail(self.menu.settings.terrain_detail);
            if !gpu_changed {
                renderer.set_aa(aa_samples(self.supported_aa[self.menu.settings.antialias]));
            }
        }
        self.applied.copy_graphics_from(&self.menu.settings);
    }

    /// Commits the staged audio settings. A different BACKEND or DEVICE
    /// reopens the output stream; volumes/toggles are applied to the live
    /// players. A short test tone confirms the new device/volume took effect.
    fn apply_audio_settings(&mut self) {
        if self.menu.settings.audio_equal(&self.applied) {
            return;
        }
        let mut audio = self.menu.settings.audio;
        if let Some(engine) = &mut self.audio {
            let output_changed =
                audio.backend != engine.active_backend() || audio.device_index != engine.active_device();
            if output_changed
                && engine
                    .switch_output(audio.backend, audio.device_index)
                    .is_err()
            {
                println!(
                    "audio: failed to open backend {} device [{}], keeping the current one",
                    audio.backend, audio.device_index
                );
                audio.backend = engine.active_backend();
                audio.device_index = engine.active_device();
            }
            engine.apply(audio);
            engine.play_sfx(Sfx::Test);
        }
        self.applied.audio = audio;
    }

    /// Feeds the current vehicle state into the engine sound every frame and
    /// fires one-shot SFX on event edges (wreck, blown engine, perfect shift,
    /// gear change). Idles the engine while paused in a menu.
    fn update_audio(&mut self, dt: f32) {
        let Some(audio) = &self.audio else {
            return;
        };
        self.audio_frame += 1;
        if matches!(self.mode, AppMode::Playing) {
            audio.set_engine(
                self.game.vehicle.speed,
                self.game.vehicle.gear,
                dt,
                self.audio_frame,
                self.game.engine_blown,
            );
            if self.game.wrecks > self.audio_prev_wrecks {
                audio.play_sfx(Sfx::Wreck);
            }
            if self.game.engine_blown && !self.audio_prev_blown {
                audio.play_sfx(Sfx::Blow);
            }
            if self.game.perfect_shift_timer > 0.0 && self.audio_prev_perfect_timer == 0.0 {
                audio.play_sfx(Sfx::PerfectShift);
            }
            if self.game.vehicle.gear != self.audio_prev_gear {
                audio.play_sfx(Sfx::Gear);
            }
            self.audio_prev_wrecks = self.game.wrecks;
            self.audio_prev_blown = self.game.engine_blown;
            self.audio_prev_perfect_timer = self.game.perfect_shift_timer;
            self.audio_prev_gear = self.game.vehicle.gear;
        } else {
            audio.set_engine(0.0, 1, dt, self.audio_frame, self.game.engine_blown);
        }

        // Diagnostic capture: pair this frame's game state with the audio
        // engine's live pitch, and drain the audio-thread trace into the CSV.
        if let Some(cap) = &self.audio_capture {
            if cap.is_enabled() {
                let (arpm, ahz) = audio.engine_state();
                cap.record_frame(
                    self.ui_clock,
                    self.audio_frame,
                    dt,
                    self.game.vehicle.speed,
                    self.game.vehicle.gear,
                    self.game.vehicle.rpm_frac(),
                    arpm,
                    ahz,
                    audio.engine_sample_idx(),
                );
            }
        }
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
            self.present_mode.to_vulkan(),
        );
        self.active_gpu_index = chosen;
        self.renderer = Some(renderer);
        if let (Some(renderer), Some(heading)) = (&mut self.renderer, self.forced_camera_heading) {
            renderer.set_camera_heading(heading);
        }
    }

    fn handle_menu_key(&mut self, event_loop: &ActiveEventLoop, kb: &KeyEvent, press: bool) {
        if !press {
            return;
        }
        match self.menu.screen {
            MenuScreen::Main => self.handle_main_key(event_loop, kb),
            MenuScreen::Settings => self.handle_settings_key(kb),
            MenuScreen::Graphics => self.handle_graphics_key(kb),
            MenuScreen::Audio => self.handle_audio_key(kb),
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
            PhysicalKey::Code(KeyCode::ArrowLeft) | PhysicalKey::Code(KeyCode::KeyA) => {
                self.cycle_main_row(-1);
            }
            PhysicalKey::Code(KeyCode::ArrowRight) | PhysicalKey::Code(KeyCode::KeyD) => {
                self.cycle_main_row(1);
            }
            PhysicalKey::Code(KeyCode::Enter) => match self.menu.main_cursor {
                MenuRow::Start => self.close_menu(),
                MenuRow::Settings => self.menu.open_settings(),
                MenuRow::Exit => event_loop.exit(),
                MenuRow::Mode | MenuRow::Weather => {}
            },
            _ => {}
        }
    }

    /// Left/Right on the main menu: cycles the MODE/WEATHER value rows and
    /// commits them to the live game right away (difficulty restarts the run,
    /// weather applies live), keeping the staged/effective states in sync.
    fn cycle_main_row(&mut self, delta: i32) {
        match self.menu.main_cursor {
            MenuRow::Mode => {
                self.menu.cycle_difficulty(delta);
                self.game.set_difficulty(self.menu.settings.difficulty);
                self.game.restart();
                println!("Run restarted (difficulty changed)");
            }
            MenuRow::Weather => {
                self.menu.cycle_weather(delta);
                self.game.set_weather(self.menu.settings.weather);
            }
            MenuRow::Start | MenuRow::Settings | MenuRow::Exit => {}
        }
        self.applied.difficulty = self.menu.settings.difficulty;
        self.applied.weather = self.menu.settings.weather;
    }

    fn handle_settings_key(&mut self, kb: &KeyEvent) {
        match kb.physical_key {
            PhysicalKey::Code(KeyCode::ArrowUp) | PhysicalKey::Code(KeyCode::KeyW) => {
                self.menu.settings_cursor = self.menu.settings_cursor.previous();
            }
            PhysicalKey::Code(KeyCode::ArrowDown) | PhysicalKey::Code(KeyCode::KeyS) => {
                self.menu.settings_cursor = self.menu.settings_cursor.next();
            }
            PhysicalKey::Code(KeyCode::Enter) => match self.menu.settings_cursor {
                SettingsRow::Graphics => self.menu.open_graphics(),
                SettingsRow::Audio => self.menu.open_audio(),
                SettingsRow::Back => self.menu.back_to_main(),
            },
            _ => {}
        }
    }

    fn handle_graphics_key(&mut self, kb: &KeyEvent) {
        let device_count = self.gpu_names.len();
        match kb.physical_key {
            PhysicalKey::Code(KeyCode::ArrowUp) | PhysicalKey::Code(KeyCode::KeyW) => {
                self.menu.graphics_cursor = self.menu.graphics_cursor.previous();
            }
            PhysicalKey::Code(KeyCode::ArrowDown) | PhysicalKey::Code(KeyCode::KeyS) => {
                self.menu.graphics_cursor = self.menu.graphics_cursor.next();
            }
            PhysicalKey::Code(KeyCode::ArrowLeft) | PhysicalKey::Code(KeyCode::KeyA) => {
                self.cycle_graphics_row(-1, device_count);
            }
            PhysicalKey::Code(KeyCode::ArrowRight) | PhysicalKey::Code(KeyCode::KeyD) => {
                self.cycle_graphics_row(1, device_count);
            }
            PhysicalKey::Code(KeyCode::Enter) => match self.menu.graphics_cursor {
                GraphicsRow::Apply => self.apply_settings(),
                GraphicsRow::Back => self.menu.back(),
                _ => {}
            },
            _ => {}
        }
    }

    fn handle_audio_key(&mut self, kb: &KeyEvent) {
        let device_count = self.audio_devices.len();
        let backend_count = self.audio_backends.len();
        match kb.physical_key {
            PhysicalKey::Code(KeyCode::ArrowUp) | PhysicalKey::Code(KeyCode::KeyW) => {
                self.menu.audio_cursor = self.menu.audio_cursor.previous(backend_count);
            }
            PhysicalKey::Code(KeyCode::ArrowDown) | PhysicalKey::Code(KeyCode::KeyS) => {
                self.menu.audio_cursor = self.menu.audio_cursor.next(backend_count);
            }
            PhysicalKey::Code(KeyCode::ArrowLeft) | PhysicalKey::Code(KeyCode::KeyA) => {
                self.cycle_audio_row(-1, backend_count, device_count);
            }
            PhysicalKey::Code(KeyCode::ArrowRight) | PhysicalKey::Code(KeyCode::KeyD) => {
                self.cycle_audio_row(1, backend_count, device_count);
            }
            PhysicalKey::Code(KeyCode::Enter) => match self.menu.audio_cursor {
                AudioRow::Apply => self.apply_audio_settings(),
                AudioRow::Back => self.menu.back(),
                _ => {}
            },
            _ => {}
        }
    }

    /// Left/Right handler for the graphics rows: cycles GPU/AA and toggles the
    /// FX rows. APPLY/BACK have no value to cycle.
    fn cycle_graphics_row(&mut self, delta: i32, device_count: usize) {
        match self.menu.graphics_cursor {
            GraphicsRow::Gpu => self.menu.cycle_gpu(delta, device_count),
            GraphicsRow::Antialias => self.menu.cycle_antialias(delta, &self.supported_aa),
            GraphicsRow::TerrainDetail => self.menu.cycle_terrain_detail(delta),
            GraphicsRow::Fxaa
            | GraphicsRow::Bloom
            | GraphicsRow::Vignette
            | GraphicsRow::Grain
            | GraphicsRow::Saturation
            | GraphicsRow::ChromaticAberration
            | GraphicsRow::RainFx
            | GraphicsRow::Shadows
            | GraphicsRow::Raytrace => self.menu.toggle_fx(self.menu.graphics_cursor),
            GraphicsRow::Reflect => self.menu.cycle_puddles(delta),
            GraphicsRow::Apply | GraphicsRow::Back => {}
        }
    }

    /// Left/Right handler for the audio rows: cycles the backend/device, steps
    /// the volume sliders and toggles the FX/music switches. Cycling the
    /// backend re-enumerates its devices and clamps the staged device index.
    /// APPLY/BACK have no value to cycle.
    fn cycle_audio_row(&mut self, delta: i32, backend_count: usize, device_count: usize) {
        match self.menu.audio_cursor {
            AudioRow::Backend => {
                self.menu.cycle_audio_backend(delta, backend_count);
                let (devices, default) =
                    enumerate_output_devices(self.menu.settings.audio.backend);
                self.audio_devices = devices;
                self.audio_default = default;
                if self.menu.settings.audio.device_index >= self.audio_devices.len() {
                    self.menu.settings.audio.device_index = 0;
                }
            }
            AudioRow::Device => self.menu.cycle_audio_device(delta, device_count),
            AudioRow::Master | AudioRow::Music | AudioRow::Sfx => {
                self.menu.adjust_volume(self.menu.audio_cursor, delta)
            }
            AudioRow::FxEnabled | AudioRow::MusicEnabled => {
                self.menu.toggle_audio(self.menu.audio_cursor)
            }
            AudioRow::Apply | AudioRow::Back => {}
        }
    }
}

fn pick_rockwall_probe() -> (f32, f32) {
    let mut best_s = 260.0f32;
    let mut best_side = 1.0f32;
    let mut best = -1.0f32;
    let mut s = 80.0f32;
    while s <= 5200.0 {
        for side in [-1.0f32, 1.0f32] {
            let mut score = 0.0f32;
            for d in [RISE_START + 1.0, RISE_START + 3.0, RISE_START + 6.0, RISE_START + 9.0] {
                // Average across a short forward/backward window to avoid
                // one-sample spikes and favor persistent steep walls.
                let s0 = terrain_slope((s - 3.0).max(0.0), side * d);
                let s1 = terrain_slope(s, side * d);
                let s2 = terrain_slope(s + 3.0, side * d);
                score = score.max((s0 + s1 + s2) / 3.0);
            }
            if score > best {
                best = score;
                best_s = s;
                best_side = side;
            }
        }
        s += 4.0;
    }
    (best_s, best_side)
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
                    PhysicalKey::Code(KeyCode::F11) if press => self.toggle_fullscreen(),
                    PhysicalKey::Code(KeyCode::F10) if press => {
                        self.request_runtime_capture();
                    }
                    PhysicalKey::Code(KeyCode::F4) if press => {
                        self.debug_visible = !self.debug_visible;
                        println!(
                            "debug HUD {}",
                            if self.debug_visible { "on" } else { "off" }
                        );
                    }
                    PhysicalKey::Code(KeyCode::F3) if press => {
                        self.debug_visible = !self.debug_visible;
                        println!(
                            "debug HUD {}",
                            if self.debug_visible { "on" } else { "off" }
                        );
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
        let frame_started = Instant::now();
        let now = Instant::now();
        let dt = now.duration_since(self.previous);
        self.previous = now;
        self.ui_clock += dt.as_secs_f32();
        self.debug.sample_frame(dt.as_secs_f32());

        let mut timings = crate::profiler::FrameTimings::default();
        timings.frame_idx = self.profile_frame_idx;
        timings.elapsed_s = self.ui_clock;
        timings.dt_ms = dt.as_secs_f32() * 1000.0;
        timings.idle_ms = self
            .profile_frame_end
            .map(|end| now.duration_since(end).as_secs_f32() * 1000.0)
            .unwrap_or(0.0);
        self.profile_frame_idx += 1;

        // `--auto-drive`: script the driver and start/keep playing so the real
        // vsync-paced render loop and cpal audio drive the capture, then exit.
        if self.auto_drive_remaining.is_some() {
            if matches!(self.mode, AppMode::Menu) {
                self.game.set_difficulty(crate::game::DifficultyLevel::EasyArcade);
                self.resume_game();
            }
            self.input.throttle = true;
            self.input.brake = false;
            self.input.steer = 0.0;
            // Shift up only when the current gear has revved into the perfect
            // shift band (0.78..0.90), so each gear sweeps the high-RPM range.
            let gear = self.game.vehicle.gear;
            self.input.gear_up = gear < 5
                && self.game.vehicle.rpm_frac_for(gear) >= crate::game::vehicle::PERFECT_LO;
            self.input.gear_down = false;
            if let Some(remaining) = &mut self.auto_drive_remaining {
                *remaining -= dt.as_secs_f32();
                if *remaining <= 0.0 {
                    _event_loop.exit();
                }
            }
        }

        let sim_started = Instant::now();
        if matches!(self.mode, AppMode::Playing) {
            self.game.update(dt, &self.input);
            self.input.gear_up = false;
            self.input.gear_down = false;
        }
        // Publish the same dt the physics used (clamped to 50 ms) so the audio's
        // first-order hold interpolates the actual speed trajectory.
        let dt_secs = dt.as_secs_f32().min(0.05);
        self.update_audio(dt_secs);
        timings.sim_ms = sim_started.elapsed().as_secs_f32() * 1000.0;

        let Some(renderer) = &mut self.renderer else {
            return;
        };

        let aspect = self.window.as_ref().map_or(16.0 / 9.0, |w| {
            let size = w.inner_size();
            if size.height == 0 {
                16.0 / 9.0
            } else {
                size.width as f32 / size.height as f32
            }
        });
        let mut root = match &self.mode {
            AppMode::Menu => {
                let graphics_dirty = !self.menu.settings.graphics_equal(&self.applied);
                let audio_dirty = !self.menu.settings.audio_equal(&self.applied);
                build_menu_tree(
                    &self.menu,
                    &self.gpu_names,
                    &self.supported_aa,
                    &self.audio_backends,
                    &self.audio_devices,
                    self.audio_default,
                    graphics_dirty,
                    audio_dirty,
                )
            }
            AppMode::Playing => {
                build_hud_tree(&self.game, self.debug_visible.then_some(&self.debug))
            }
        };
        let ui_started = Instant::now();
        let hud_verts = self
            .ui
            .build(&mut root, &self.font_atlas, aspect, self.ui_clock);
        self.debug.hud_verts = hud_verts.len();
        timings.ui_ms = ui_started.elapsed().as_secs_f32() * 1000.0;
        let fx = FxSettings {
            fxaa: self.applied.fxaa,
            bloom: self.applied.bloom,
            vignette: self.applied.vignette,
            grain: self.applied.grain,
            saturation: self.applied.saturation,
            chroma: self.applied.chroma,
            rain_fx: self.applied.rain_fx,
            puddle_quality: self.applied.puddles.uniform(),
            shadows: self.applied.shadows,
            raytrace: self.applied.raytrace,
        };
        let render_started = Instant::now();
        let capture_done = renderer.render(&self.game, dt, &hud_verts, &fx, &mut timings);
        if self.window_capture_armed && capture_done {
            println!("window capture completed; exiting");
            self.window_capture_armed = false;
            _event_loop.exit();
        }
        self.debug
            .sample_cpu(render_started.elapsed().as_secs_f32() * 1000.0);

        let ws = renderer.world_stats();
        self.debug.world_chunks = ws.chunk_count;
        self.debug.world_verts = ws.world_verts;
        self.debug.world_tris = ws.world_tris;
        self.debug.chunk_rebuild_ms = ws.last_rebuild_ms;
        self.debug.chunks_rebuilt = ws.chunks_rebuilt;
        self.debug.chunks_pending = ws.chunks_pending;
        self.debug.chunks_cached = ws.chunks_cached;
        self.debug.particles = ws.particles;
        self.debug.distance = self.game.vehicle.distance;
        self.debug.chunk_index =
            (self.game.vehicle.distance / crate::render::WORLD_CHUNK_LEN).floor() as i32;
        self.debug.terrain_factor = crate::world::terrain::speed_factor(self.game.vehicle.distance);

        timings.total_ms = frame_started.elapsed().as_secs_f32() * 1000.0;
        self.profile_frame_end = Some(Instant::now());
        if let Some(profiler) = &mut self.profiler {
            profiler.push(timings);
        }
        // `--fps-limit <N>`: pad the frame to 1/N s with an idle sleep. It runs
        // after `total_ms` and `profile_frame_end` are recorded, so the cap wait
        // lands in the next frame's `idle_ms` and the profiler still measures
        // real work only (`total_ms`/`submit_ms` unchanged).
        if let Some(limit) = self.fps_limit {
            let target = std::time::Duration::from_secs_f64(1.0 / limit as f64);
            let elapsed = frame_started.elapsed();
            if elapsed < target {
                std::thread::sleep(target - elapsed);
            }
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        // Profiling close: flush the CSV, write the Markdown report, and list
        // the generated files so the terminal shows where the session data is.
        if let Some(profiler) = self.profiler.take() {
            let files = profiler.close();
            println!("profiler session closed; generated:");
            for f in &files {
                println!("  {}", f.display());
            }
        }
        if let Some(cap) = &self.audio_capture {
            let path = cap.close();
            println!("audio capture closed; wrote {}", path.display());
        }
    }
}

impl App {
    fn request_runtime_capture(&mut self) {
        let Some(dir) = &self.capture_dir else {
            println!("capture disabled: start with --capture-dir <dir>");
            return;
        };
        let Some(renderer) = &mut self.renderer else {
            return;
        };
        self.capture_seq = self.capture_seq.saturating_add(1);
        let stamp = (self.ui_clock.max(0.0) * 1000.0).round() as u64;
        let file = format!("shot_{:04}_t{:08}ms.png", self.capture_seq, stamp);
        let path = dir.join(file);
        renderer.request_window_capture(path.clone());
        println!("capture requested: {}", path.display());
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

/// Registers a sway `for_window` rule before the window is created, so a
/// `--windowed` launch is floated from the moment it appears. Best-effort:
/// silently no-ops when swaymsg isn't present or there's no sway session.
fn sway_float_register() {
    let _ = std::process::Command::new("swaymsg")
        .args([
            "-t",
            "command",
            &format!("for_window [app_id=\"{SWAY_APP_ID}\"] floating enable"),
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn();
}

/// Floats an already-existing window (F11 back to windowed) via sway criteria.
/// Best-effort: no-ops on non-sway compositors.
fn sway_float_window() {
    let _ = std::process::Command::new("swaymsg")
        .args([
            "-t",
            "command",
            &format!("[app_id=\"{SWAY_APP_ID}\"] floating enable"),
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn();
}
