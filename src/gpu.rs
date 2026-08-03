// SPDX-License-Identifier: MIT

use std::io::{self, BufRead, IsTerminal, Write};
use std::sync::Arc;

use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags};
use vulkano::instance::Instance;
use vulkano::swapchain::Surface;
use winit::window::Window;

pub fn select_physical_device(instance: &Arc<Instance>) -> Arc<PhysicalDevice> {
    let devices: Vec<Arc<_>> = instance
        .enumerate_physical_devices()
        .expect("failed to enumerate physical devices")
        .collect();

    if devices.is_empty() {
        panic!("no Vulkan-capable GPUs found on this system");
    }

    println!("Available GPUs:");
    for (i, device) in devices.iter().enumerate() {
        let props = device.properties();
        println!("  [{}] {}  ({:?})", i, props.device_name, props.device_type);
    }

    let chosen = if !io::stdin().is_terminal() {
        pick_default(&devices)
    } else {
        print!("Select a GPU to use (index): ");
        io::stdout().flush().unwrap();
        let mut line = String::new();
        io::stdin().lock().read_line(&mut line).unwrap();
        match line.trim().parse::<usize>() {
            Ok(i) if i < devices.len() => i,
            _ => {
                println!("invalid selection, using default");
                pick_default(&devices)
            }
        }
    };

    println!("Using GPU: {}", devices[chosen].properties().device_name);
    devices[chosen].clone()
}

fn pick_default(devices: &[Arc<PhysicalDevice>]) -> usize {
    devices
        .iter()
        .position(|d| d.properties().device_type == PhysicalDeviceType::DiscreteGpu)
        .unwrap_or(0)
}

pub fn create_graphics_context(
    instance: Arc<Instance>,
    window: Arc<Window>,
) -> (Arc<PhysicalDevice>, Arc<Surface>, Arc<Device>, Arc<Queue>) {
    let physical = select_physical_device(&instance);
    let surface = Surface::from_window(instance, window).expect("failed to create surface");

    let queue_family_index = physical
        .queue_family_properties()
        .iter()
        .enumerate()
        .find(|(i, q)| {
            q.queue_flags.intersects(QueueFlags::GRAPHICS)
                && physical
                    .surface_support(*i as u32, &surface)
                    .unwrap_or(false)
        })
        .map(|(i, _)| i as u32)
        .expect("no queue family with graphics + present support");

    let (device, mut queues) = Device::new(
        physical.clone(),
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            enabled_extensions: DeviceExtensions {
                khr_swapchain: true,
                ..DeviceExtensions::empty()
            },
            ..Default::default()
        },
    )
    .expect("failed to create device");
    let queue = queues.next().unwrap();

    (physical, surface, device, queue)
}
