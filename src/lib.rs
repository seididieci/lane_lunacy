// SPDX-License-Identifier: MIT

pub mod app;
pub mod cli;
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

use std::sync::Arc;

use winit::event_loop::EventLoop;

use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::swapchain::Surface;
use vulkano::VulkanLibrary;

use crate::cli::SnapshotOptions;
use crate::game::Weather;
use crate::gpu::{create_graphics_context_headless, enumerate_devices, select_physical_device};

/// Creates a Vulkan instance with the extensions needed for a window surface.
pub fn create_surface_instance(event_loop: &EventLoop<()>) -> Arc<Instance> {
    let library = VulkanLibrary::new().expect("failed to load the Vulkan library");
    let required_extensions =
        Surface::required_extensions(event_loop).expect("failed to get required extensions");
    Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            enabled_extensions: required_extensions,
            ..Default::default()
        },
    )
    .expect("failed to create instance")
}

/// Creates a headless Vulkan instance (no surface, no window extensions) for
/// offscreen `--snapshot` rendering.
pub fn create_headless_instance() -> Arc<Instance> {
    let library = VulkanLibrary::new().expect("failed to load the Vulkan library");
    Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            ..Default::default()
        },
    )
    .expect("failed to create instance")
}

pub fn run(gpu_index: usize, weather: Weather, start_hour: Option<f32>, seed: Option<u64>) {
    let event_loop = EventLoop::new().expect("failed to create event loop");
    let instance = create_surface_instance(&event_loop);

    // No `--seed`: keep the historical per-launch randomness by seeding from
    // the clock, so every interactive run still gets a fresh sky and cycle.
    let seed = seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock before epoch")
            .as_nanos() as u64
    });

    let mut app = app::App::new(instance, gpu_index, weather, start_hour, seed);
    event_loop.run_app(&mut app).expect("event loop failed");
}

/// Headless `--snapshot` entry point: boots a surface-less Vulkan context,
/// builds the deterministic `Game` for the scenario, and prints the derived
/// scene state. The offscreen render + PNG write lands in a later step.
pub fn run_snapshot(opts: SnapshotOptions) {
    let instance = create_headless_instance();
    let devices = enumerate_devices(&instance);
    let physical = select_physical_device(&devices, opts.gpu);
    let (device, queue) = create_graphics_context_headless(&physical);

    let mut game = crate::game::Game::new();
    game.set_weather(opts.weather);
    if let Some(hour) = opts.time {
        game.set_start_hour(hour);
    }
    // Last: seed drives the weather phase (and start hour when unpinned).
    game.set_seed(opts.seed);

    println!("snapshot scenario: {:?}", opts);
    println!(
        "derived scene: time_of_day {:.3}h, cloud {:.3}, rain {:.3}",
        game.time_of_day(),
        game.cloud_amount(),
        game.rain_intensity()
    );
    println!(
        "headless context ready: {} / queue {:?}",
        physical.properties().device_name,
        queue.queue_family_index()
    );
    let _ = device;
    let _ = game;
}
