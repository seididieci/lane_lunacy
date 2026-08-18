// SPDX-License-Identifier: MIT

use std::sync::Arc;

use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo,
    QueueFlags,
};
use vulkano::instance::Instance;
use vulkano::swapchain::Surface;

/// Whether the physical device can run the ray-tracing backend: it needs the
/// acceleration-structure, ray-tracing-pipeline and deferred-host-operations
/// extensions plus buffer device addresses for BLAS/TLAS references.
pub fn ray_tracing_supported(physical: &Arc<PhysicalDevice>) -> bool {
    let ext = physical.supported_extensions();
    ext.khr_acceleration_structure
        && ext.khr_ray_tracing_pipeline
        && ext.khr_deferred_host_operations
        && ext.khr_buffer_device_address
}

/// Extensions to enable on the device when the GPU supports ray tracing.
fn raytrace_extensions() -> DeviceExtensions {
    DeviceExtensions {
        khr_acceleration_structure: true,
        khr_ray_tracing_pipeline: true,
        khr_deferred_host_operations: true,
        khr_buffer_device_address: true,
        ..DeviceExtensions::empty()
    }
}

/// Features to enable on the device when the GPU supports ray tracing.
fn raytrace_features() -> DeviceFeatures {
    DeviceFeatures {
        acceleration_structure: true,
        ray_tracing_pipeline: true,
        buffer_device_address: true,
        ..DeviceFeatures::default()
    }
}

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
    let rt = ray_tracing_supported(&devices[chosen]);
    println!(
        "Using GPU [{}]: {}  ({:?})  ray tracing: {}",
        chosen,
        devices[chosen].properties().device_name,
        devices[chosen].properties().device_type,
        if rt { "supported" } else { "unavailable" }
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

    let rt_supported = ray_tracing_supported(physical);
    let (mut enabled_extensions, mut enabled_features) =
        (DeviceExtensions::empty(), DeviceFeatures::default());
    if rt_supported {
        enabled_extensions = raytrace_extensions();
        enabled_features = raytrace_features();
    }
    enabled_extensions.khr_swapchain = true;

    let (device, mut queues) = Device::new(
        physical.clone(),
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            enabled_extensions,
            enabled_features,
            ..Default::default()
        },
    )
    .expect("failed to create device");
    let queue = queues.next().unwrap();

    (device, queue)
}

/// Creates a device + queue for offscreen rendering without a window/surface.
/// Only needs a `GRAPHICS` queue family (no present support), and no swapchain
/// extension. Used by the headless `--snapshot` path.
pub fn create_graphics_context_headless(
    physical: &Arc<PhysicalDevice>,
) -> (Arc<Device>, Arc<Queue>) {
    let queue_family_index = physical
        .queue_family_properties()
        .iter()
        .enumerate()
        .find(|(_, q)| q.queue_flags.intersects(QueueFlags::GRAPHICS))
        .map(|(i, _)| i as u32)
        .expect("no queue family with graphics support");

    let rt_supported = ray_tracing_supported(physical);
    let (enabled_extensions, enabled_features) =
        if rt_supported {
            (raytrace_extensions(), raytrace_features())
        } else {
            (DeviceExtensions::empty(), DeviceFeatures::default())
        };

    let (device, mut queues) = Device::new(
        physical.clone(),
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            enabled_extensions,
            enabled_features,
            ..Default::default()
        },
    )
    .expect("failed to create device");
    let queue = queues.next().unwrap();

    (device, queue)
}
