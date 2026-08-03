// SPDX-License-Identifier: MIT

use std::sync::Arc;

use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags};
use vulkano::instance::Instance;
use vulkano::swapchain::Surface;

pub fn enumerate_devices(instance: &Arc<Instance>) -> Vec<Arc<PhysicalDevice>> {
    let all: Vec<Arc<_>> = instance
        .enumerate_physical_devices()
        .expect("failed to enumerate physical devices")
        .collect();

    if all.is_empty() {
        panic!("no Vulkan-capable GPUs found on this system");
    }

    let hardware: Vec<Arc<PhysicalDevice>> = all
        .iter()
        .filter(|d| d.properties().device_type != PhysicalDeviceType::Cpu)
        .cloned()
        .collect();

    if hardware.is_empty() {
        println!("No hardware GPU found; falling back to software rendering (llvmpipe)");
        all
    } else {
        hardware
    }
}

pub fn select_physical_device(
    devices: &[Arc<PhysicalDevice>],
    index: usize,
) -> Arc<PhysicalDevice> {
    let chosen = index.min(devices.len().saturating_sub(1));
    println!(
        "Using GPU [{}]: {}  ({:?})",
        chosen,
        devices[chosen].properties().device_name,
        devices[chosen].properties().device_type
    );
    devices[chosen].clone()
}

pub fn create_graphics_context(
    surface: Arc<Surface>,
    physical: &Arc<PhysicalDevice>,
) -> (Arc<Device>, Arc<Queue>) {
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

    (device, queue)
}
