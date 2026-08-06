// SPDX-License-Identifier: MIT

pub mod app;
pub mod font;
pub mod game;
pub mod gpu;
pub mod hud;
pub mod input;
pub mod menu;
pub mod mesh;
pub mod model;
pub mod render;
pub mod road;
pub mod shaders;
pub mod surface;
pub mod ui;
pub mod vertex;

use winit::event_loop::EventLoop;

use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::swapchain::Surface;
use vulkano::VulkanLibrary;

use crate::game::Weather;

pub fn run(gpu_index: usize, weather: Weather, start_hour: Option<f32>) {
    let event_loop = EventLoop::new().expect("failed to create event loop");

    let library = VulkanLibrary::new().expect("failed to load the Vulkan library");
    let required_extensions = Surface::required_extensions(&event_loop)
        .expect("failed to get required extensions");
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            enabled_extensions: required_extensions,
            ..Default::default()
        },
    )
    .expect("failed to create instance");

    let mut app = app::App::new(instance, gpu_index, weather, start_hour);
    event_loop.run_app(&mut app).expect("event loop failed");
}
