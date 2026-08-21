// SPDX-License-Identifier: MIT

//! Headless driving benchmark (`--drive <path.csv> <secs>`).
//!
//! Boots a surface-less Vulkan context, drives the car forward with scripted
//! input at a fixed 60 Hz, and records per-frame [`FrameTimings`] — including
//! world-chunk rebuild crossings — to a profiler CSV (plus a `report.md`).
//!
//! There is no window and no swapchain: the only per-frame work is the CPU
//! scene builder, which is exactly the cost that a chunk crossing can charge
//! the render thread. A crossing that finds its mesh already prefetched by the
//! background pool shows up as `rebuild_ms ≈ 0`; before the pool, every
//! crossing rebuilt in-line at 120–160 ms (windowed sessions).

use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::audio::{AudioCapture, EngineRunner, SharedCapture};
use crate::cli::DriveOptions;
use crate::font::FontAtlas;
use crate::game::{DifficultyLevel, Game};
use crate::hud::build_hud_tree;
use crate::input::Input;
use crate::profiler::{FrameTimings, SessionProfiler};
use crate::render::frame_builder::FrameBuilder;
use crate::render::scene::SceneResources;
use crate::ui::Ui;

/// Drives the car with fixed 60 Hz steps for `opts.seconds` and records the
/// profiled frames. Prints a per-crossing breakdown so a chunk-prefetch
/// regression is visible at a glance.
pub fn run_drive(opts: DriveOptions) {
    let instance = crate::create_headless_instance();
    let devices = crate::gpu::enumerate_devices(&instance);
    let physical = crate::gpu::select_physical_device(&devices, opts.gpu);
    let (device, queue) = crate::gpu::create_graphics_context_headless(&physical);

    println!(
        "drive benchmark: {} s, seed {}, terrain {:?}, gpu {} ({})",
        opts.seconds,
        opts.seed,
        opts.terrain_detail,
        opts.gpu,
        physical.properties().device_name,
    );

    let mut game = Game::new();
    // Easiest mode: the fewest traffic cars and the highest wreck limit, so a
    // scripted straight-line run stays alive for the full benchmark.
    game.set_difficulty(DifficultyLevel::EasyArcade);
    game.set_weather(opts.weather);
    // The seed drives weather phase and cloud tiles; chunk terrain is a pure
    // function of world coordinates, so the drive covers the same chunks
    // regardless of seed.
    game.set_seed(opts.seed);

    let font_atlas = FontAtlas::load();
    let render_pass = vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                format: vulkano::format::Format::R16G16B16A16_SFLOAT,
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },
            depth: {
                format: vulkano::format::Format::D32_SFLOAT,
                samples: 1,
                load_op: Clear,
                store_op: DontCare,
            },
        },
        pass: {
            color: [color],
            depth_stencil: { depth },
        }
    )
    .expect("drive render pass");

    let scene_started = Instant::now();
    let scene = SceneResources::new(
        device.clone(),
        queue.clone(),
        render_pass,
        &font_atlas,
        opts.seed,
        vulkano::image::SampleCount::Sample1,
    );
    println!(
        "scene init: {:.1} ms",
        scene_started.elapsed().as_secs_f32() * 1000.0
    );

    let mut frame_builder = FrameBuilder::with_seed(opts.seed);
    frame_builder.set_terrain_detail(opts.terrain_detail);
    let mut profiler = SessionProfiler::open(&opts.csv).expect("open drive CSV");
    let aspect = 16.0 / 9.0;
    let dt = Duration::from_secs_f64(1.0 / 60.0);
    let frames = (opts.seconds * 60) as usize;
    let ui = Ui::new();

    // Optional audio: run the engine sound in sample-space (no output device)
    // and either capture the per-frame game RPM vs audio pitch pairing and/or
    // render the audible output to a WAV.
    let capture: Option<SharedCapture> = opts.audio_capture.as_deref().and_then(|p| {
        AudioCapture::open(p)
            .inspect_err(|e| eprintln!("audio capture: failed to open {}: {e}", p.display()))
            .ok()
            .map(Arc::new)
    });
    let want_audio = capture.is_some() || opts.audio_wav.is_some();
    let mut runner = want_audio.then(|| EngineRunner::new(capture.clone()));
    let mut wav: Option<Vec<f32>> = opts.audio_wav.as_ref().map(|_| Vec::new());
    if let Some(cap) = &capture {
        println!("audio capture enabled: {}", cap.path().display());
    }
    if let Some(p) = &opts.audio_wav {
        println!("audio wav render: {}", p.display());
    }
    let mut audio_frame = 0u32;

    // Scripted driver: full throttle from the first frame, shifting up only
    // when the current gear revs into the perfect-shift band, straight steering
    // (the road is road-aligned, so a centred heading stays on it).
    let mut input = Input {
        throttle: true,
        ..Default::default()
    };
    let mut crossings: Vec<(u64, f32, usize, usize, usize)> = Vec::new();

    // The sim advances at a fixed 1/60 s per frame; pace the loop to that rate
    // so the benchmark matches a real 60 Hz (vsync) session. An unpaced loop
    // finishes thousands of fps, which makes the first chunk crossing land
    // within microseconds of the background build completing — an artefact that
    // a driven window never sees.
    let frame_dt = Duration::from_secs_f64(1.0 / 60.0);
    let mut next_frame = Instant::now();

    for frame_idx in 0..frames {
        // Shift up when the current gear has revved into the perfect-shift band.
        let gear = game.vehicle.gear;
        input.gear_up =
            gear < 5 && game.vehicle.rpm_frac_for(gear) >= crate::game::vehicle::PERFECT_LO;
        input.gear_down = false;
        input.steer = 0.0;

        let sim_started = Instant::now();
        game.update(dt, &input);
        let sim_ms = sim_started.elapsed().as_secs_f32() * 1000.0;

        // Publish the vehicle state to the engine sound and record the
        // rpm-vs-pitch pairing, mirroring the interactive `update_audio`.
        if let Some(runner) = &mut runner {
            let dt_secs = dt.as_secs_f32();
            audio_frame += 1;
            runner.set_engine(
                game.vehicle.speed,
                game.vehicle.gear,
                dt_secs,
                audio_frame,
                game.engine_blown,
            );
            match &mut wav {
                Some(buf) => runner.advance_frame_into(buf),
                None => runner.advance_frame(),
            }
            if let Some(cap) = &capture {
                let (arpm, ahz) = runner.state();
                cap.record_frame(
                    frame_idx as f32 * dt_secs,
                    audio_frame,
                    dt_secs,
                    game.vehicle.speed,
                    game.vehicle.gear,
                    game.vehicle.rpm_frac(),
                    arpm,
                    ahz,
                    runner.sample_idx(),
                );
            }
        }

        if frame_idx % 300 == 0 {
            let ws = frame_builder.world_stats();
            println!(
                "[{:6}] dist {:.0} m  gear {}  speed {:.0} km/h  heat {:.2}  pending {}  cached {}{}",
                frame_idx,
                game.vehicle.distance,
                game.vehicle.gear,
                game.speed_kmh,
                game.engine_heat,
                ws.chunks_pending,
                ws.chunks_cached,
                if game.game_over { "  GAME_OVER" } else { "" }
            );
        }

        let ui_started = Instant::now();
        let mut hud_root = build_hud_tree(&game, None);
        let hud_verts = ui.build(&mut hud_root, &font_atlas, aspect, 0.0);
        let ui_ms = ui_started.elapsed().as_secs_f32() * 1000.0;

        let build_started = Instant::now();
        let _frame = frame_builder.build(&scene, &game, dt, aspect, hud_verts);
        let build_ms = build_started.elapsed().as_secs_f32() * 1000.0;

        let ws = frame_builder.world_stats();
        if ws.chunks_rebuilt > 0 {
            crossings.push((
                frame_idx as u64,
                ws.last_rebuild_ms,
                ws.chunks_rebuilt,
                ws.chunks_pending,
                ws.chunks_cached,
            ));
        }

        profiler.push(FrameTimings {
            frame_idx: frame_idx as u64,
            elapsed_s: frame_idx as f32 * dt.as_secs_f32(),
            dt_ms: dt.as_secs_f32() * 1000.0,
            sim_ms,
            ui_ms,
            rebuild_ms: ws.last_rebuild_ms,
            chunks_rebuilt: ws.chunks_rebuilt,
            frame_ms: build_ms,
            total_ms: sim_ms + ui_ms + build_ms,
            ..Default::default()
        });

        next_frame += frame_dt;
        if let Some(sleep) = next_frame.checked_duration_since(Instant::now()) {
            std::thread::sleep(sleep);
        }
    }

    let files = profiler.close();
    println!("profiler wrote: {}", files[0].display());
    println!("report wrote:   {}", files[1].display());

    if let Some(cap) = &capture {
        let path = cap.close();
        println!("audio capture closed; wrote {}", path.display());
    }
    if let (Some(buf), Some(path)) = (&wav, &opts.audio_wav) {
        match write_wav_mono(path, buf) {
            Ok(()) => println!("audio wav wrote {} ({} samples)", path.display(), buf.len()),
            Err(e) => eprintln!("audio wav: failed to write {}: {e}", path.display()),
        }
    }

    println!(
        "\nchunk crossings: {} (rebuild_ms = render-thread stall, pending = queued builds)\n    idx  rebuild_ms  chunks  pending  cached",
        crossings.len()
    );
    for (idx, rebuild_ms, chunks, pending, cached) in &crossings {
        println!(
            "  {:5}  {:9.2}  {:6}  {:7}  {:6}",
            idx, rebuild_ms, chunks, pending, cached
        );
    }
    if let Some((_, max, ..)) = crossings.iter().max_by(|a, b| a.1.total_cmp(&b.1)) {
        println!(
            "worst crossing stall: {:.2} ms (pre-pool baseline: 120-160 ms per crossing)",
            max
        );
    }
}

/// Writes mono 16-bit PCM WAV at 48 kHz.
fn write_wav_mono(path: &std::path::Path, samples: &[f32]) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    let data_len = (samples.len() * 2) as u32;
    f.write_all(b"RIFF")?;
    f.write_all(&(36 + data_len).to_le_bytes())?;
    f.write_all(b"WAVEfmt ")?;
    f.write_all(&16u32.to_le_bytes())?;
    f.write_all(&1u16.to_le_bytes())?; // PCM
    f.write_all(&1u16.to_le_bytes())?; // mono
    f.write_all(&48_000u32.to_le_bytes())?;
    f.write_all(&(48_000u32 * 2).to_le_bytes())?; // byte rate
    f.write_all(&2u16.to_le_bytes())?; // block align
    f.write_all(&16u16.to_le_bytes())?; // bits per sample
    f.write_all(b"data")?;
    f.write_all(&data_len.to_le_bytes())?;
    for s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        f.write_all(&v.to_le_bytes())?;
    }
    f.flush()
}
