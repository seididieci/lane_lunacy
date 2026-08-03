// SPDX-License-Identifier: MIT

use std::sync::Arc;
use std::time::Instant;

use winit::dpi::LogicalSize;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowAttributes};

use vulkano::instance::Instance;

use crate::game::{DifficultyLevel, Game};
use crate::gpu::create_graphics_context;
use crate::input::Input;
use crate::render::Renderer;

pub struct App {
    instance: Arc<Instance>,
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    game: Game,
    input: Input,
    previous: Instant,
}

impl App {
    pub fn new(instance: Arc<Instance>) -> Self {
        Self {
            instance,
            window: None,
            renderer: None,
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

        let (physical, surface, device, queue) =
            create_graphics_context(self.instance.clone(), window.clone());

        self.renderer = Some(Renderer::new(device, queue, surface, window.clone(), &physical));
        self.window = Some(window);
        self.previous = Instant::now();

        println!(
            "Controls: W / Up = throttle | S / Down = brake | A / Left & D / Right = steer | E = gear up | Q = gear down | 1/2/3 = EASY/NORMAL/HARD"
        );
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
                match kb.physical_key {
                    PhysicalKey::Code(KeyCode::KeyW)
                    | PhysicalKey::Code(KeyCode::ArrowUp) => self.input.throttle = pressed,
                    PhysicalKey::Code(KeyCode::KeyS)
                    | PhysicalKey::Code(KeyCode::ArrowDown) => self.input.brake = pressed,
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
                        if pressed && !kb.repeat {
                            self.input.gear_up = true;
                        }
                    }
                    PhysicalKey::Code(KeyCode::KeyQ) => {
                        if pressed && !kb.repeat {
                            self.input.gear_down = true;
                        }
                    }
                    PhysicalKey::Code(KeyCode::Digit1) => {
                        if pressed && !kb.repeat {
                            self.game.set_difficulty(DifficultyLevel::EasyArcade);
                            println!("Difficulty set to EASY");
                        }
                    }
                    PhysicalKey::Code(KeyCode::Digit2) => {
                        if pressed && !kb.repeat {
                            self.game.set_difficulty(DifficultyLevel::Normal);
                            println!("Difficulty set to NORMAL");
                        }
                    }
                    PhysicalKey::Code(KeyCode::Digit3) => {
                        if pressed && !kb.repeat {
                            self.game.set_difficulty(DifficultyLevel::Hard);
                            println!("Difficulty set to HARD");
                        }
                    }
                    _ => {}
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
        self.game.update(dt, &self.input);
        self.input.gear_up = false;
        self.input.gear_down = false;
        renderer.render(&self.game, dt);
    }
}
