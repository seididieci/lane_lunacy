// SPDX-License-Identifier: MIT

use std::sync::Arc;
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowAttributes};

use vulkano::instance::Instance;
use vulkano::swapchain::Surface;

use crate::game::Game;
use crate::gpu::{create_graphics_context, enumerate_devices, select_physical_device};
use crate::input::Input;
use crate::menu::{MenuRow, MenuState};
use crate::render::Renderer;

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
    menu: MenuState,
    mode: AppMode,
    game: Game,
    input: Input,
    previous: Instant,
}

impl App {
    pub fn new(instance: Arc<Instance>, gpu_index: usize) -> Self {
        Self {
            instance,
            window: None,
            surface: None,
            renderer: None,
            gpu_names: Vec::new(),
            active_gpu_index: gpu_index,
            menu: MenuState::new(gpu_index),
            mode: AppMode::Menu,
            game: Game::new(),
            input: Input::default(),
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
        if self.menu.gpu_index > max_index {
            self.menu.gpu_index = max_index;
        }
        let physical = select_physical_device(&devices, self.menu.gpu_index);
        let (device, queue) = create_graphics_context(surface.clone(), &physical);
        let renderer = Renderer::new(device, queue, surface.clone(), window.clone(), &physical);

        self.active_gpu_index = self.menu.gpu_index;
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
        self.game.set_difficulty(self.menu.difficulty);
        if self.menu.take_difficulty_changed() {
            self.game.restart();
            println!("Run restarted (difficulty changed)");
        }
        self.switch_gpu();
        self.mode = AppMode::Playing;
    }

    /// Switches the graphics backend to `menu.gpu_index` when it differs from the
    /// active GPU. The window surface is reused and the run keeps its state.
    fn switch_gpu(&mut self) {
        let chosen = self
            .menu
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
        let (device, queue) = create_graphics_context(surface.clone(), &physical);
        let renderer = Renderer::new(device, queue, surface.clone(), window, &physical);
        self.active_gpu_index = chosen;
        self.renderer = Some(renderer);
    }

    fn handle_menu_key(&mut self, event_loop: &ActiveEventLoop, kb: &KeyEvent, press: bool) {
        if !press {
            return;
        }
        let device_count = self.gpu_names.len();
        match kb.physical_key {
            PhysicalKey::Code(KeyCode::ArrowUp) | PhysicalKey::Code(KeyCode::KeyW) => {
                self.menu.cursor = self.menu.cursor.previous();
            }
            PhysicalKey::Code(KeyCode::ArrowDown) | PhysicalKey::Code(KeyCode::KeyS) => {
                self.menu.cursor = self.menu.cursor.next();
            }
            PhysicalKey::Code(KeyCode::ArrowLeft) | PhysicalKey::Code(KeyCode::KeyA) => {
                match self.menu.cursor {
                    MenuRow::Gpu => self.menu.cycle_gpu(-1, device_count),
                    MenuRow::Difficulty => self.menu.cycle_difficulty(-1),
                    _ => {}
                }
            }
            PhysicalKey::Code(KeyCode::ArrowRight) | PhysicalKey::Code(KeyCode::KeyD) => {
                match self.menu.cursor {
                    MenuRow::Gpu => self.menu.cycle_gpu(1, device_count),
                    MenuRow::Difficulty => self.menu.cycle_difficulty(1),
                    _ => {}
                }
            }
            PhysicalKey::Code(KeyCode::Enter) => match self.menu.cursor {
                MenuRow::Gpu => self.switch_gpu(),
                MenuRow::Start => self.close_menu(),
                MenuRow::Exit => event_loop.exit(),
                MenuRow::Difficulty => {}
            },
            _ => {}
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
                    PhysicalKey::Code(KeyCode::Escape) if press => self.toggle_menu(),
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
                        PhysicalKey::Code(KeyCode::KeyR) => {
                            if press {
                                self.game.restart();
                                println!("Run restarted");
                            }
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

        if matches!(self.mode, AppMode::Playing) {
            self.game.update(dt, &self.input);
            self.input.gear_up = false;
            self.input.gear_down = false;
        }

        let menu = match &self.mode {
            AppMode::Menu => Some(&self.menu),
            AppMode::Playing => None,
        };
        renderer.render(&self.game, dt, menu, &self.gpu_names);
    }
}
