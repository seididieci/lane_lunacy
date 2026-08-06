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

pub fn run(gpu_index: usize, weather: Weather, start_hour: Option<f32>) {
    let event_loop = EventLoop::new().expect("failed to create event loop");
    let instance = create_surface_instance(&event_loop);

    let mut app = app::App::new(instance, gpu_index, weather, start_hour);
    event_loop.run_app(&mut app).expect("event loop failed");
}

/// Headless `--snapshot` entry point: boots a surface-less Vulkan context,
/// prints the requested scenario, and currently verifies the device + queue
/// come up. The offscreen render + PNG write lands in a later step.
pub fn run_snapshot(opts: SnapshotOptions) {
    let instance = create_headless_instance();
    let devices = enumerate_devices(&instance);
    let physical = select_physical_device(&devices, opts.gpu);
    let (device, queue) = create_graphics_context_headless(&physical);

    println!("snapshot scenario: {:?}", opts);
    println!(
        "headless context ready: {} / queue {:?}",
        physical.properties().device_name,
        queue.queue_family_index()
    );
    let _ = device;
}
