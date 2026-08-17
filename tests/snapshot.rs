// SPDX-License-Identifier: MIT

//! Snapshot regression tests: pin the deterministic scene to the golden
//! baselines captured by `scripts/snapshot_parity.sh capture`.
//!
//! Two layers:
//! - **CPU probes** are pure scene math (no GPU) and run in every `cargo test`.
//! - **GPU probes** render a real offscreen frame through the Vulkan pipeline,
//!   so they only run when `LANE_SNAPSHOT_TESTS=1` (they need a device and take
//!   a few seconds).

use std::time::Duration;

use lane_lunacy::game::{Game, Weather};
use lane_lunacy::model::CarLightAnchors;
use lane_lunacy::render::frame::{build_frame, Frame, FrameState};
use lane_lunacy::render::probe::{compute_cpu, compute_gpu, CpuProbe, GpuProbe};

/// A deterministic scenario and its golden probes (from the baseline capture).
struct Scenario {
    name: &'static str,
    time: f32,
    weather: Weather,
    cpu: CpuProbe,
    gpu: GpuProbe,
}

fn scenarios() -> [Scenario; 3] {
    [
        Scenario {
            name: "noon_clear",
            time: 12.0,
            weather: Weather::Clear,
            cpu: CpuProbe {
                sun_ndc: Some([0.0, 0.696008]),
                flare_intensity: 0.991975,
                projector_road_coverage: 0.875,
                wet_fac: 0.0,
                night_fac: 0.0,
                mist_fac: 0.0,
            },
            gpu: GpuProbe {
                sky_top_lum: 0.446360,
                road_center_lum: 0.157527,
                sun_disc_max_lum: 4.18458,
                flare_bloom_lum: 4.003115,
            },
        },
        Scenario {
            name: "midnight_rain",
            time: 0.0,
            weather: Weather::Rain,
            cpu: CpuProbe {
                sun_ndc: Some([-0.515831, 0.778845]),
                flare_intensity: 0.008400,
                projector_road_coverage: 0.875,
                wet_fac: 1.0,
                night_fac: 0.6,
                mist_fac: 0.5,
            },
            gpu: GpuProbe {
                sky_top_lum: 0.079253,
                road_center_lum: 0.110233,
                sun_disc_max_lum: 0.331207,
                flare_bloom_lum: 0.332588,
            },
        },
        Scenario {
            name: "dusk",
            time: 18.0,
            weather: Weather::Clear,
            cpu: CpuProbe {
                sun_ndc: Some([-4.094469, 1.626567]),
                flare_intensity: 0.037354,
                projector_road_coverage: 0.875,
                wet_fac: 0.0,
                night_fac: 0.0,
                mist_fac: 0.07413377,
            },
            gpu: GpuProbe {
                sky_top_lum: 0.326593,
                road_center_lum: 0.150075,
                sun_disc_max_lum: 0.0,
                flare_bloom_lum: 0.0,
            },
        },
    ]
}

// Tolerances: the render is deterministic, so a real regression moves the
// values well beyond these bounds. Luminances are left looser (the sun disc is
// a single-pixel peak) so driver changes don't cause false failures.
const TOL_CPU: f32 = 0.005;
const TOL_NDC: f32 = 0.02;
const TOL_LUM: f32 = 0.01;
const TOL_SUN: f32 = 0.1;

fn assert_close(label: &str, actual: f32, expected: f32, tol: f32) {
    let d = (actual - expected).abs();
    assert!(
        d <= tol,
        "{label}: {actual} differs from {expected} by {d} (tol {tol})"
    );
}

fn assert_cpu(label: &str, actual: CpuProbe, expected: CpuProbe) {
    match (actual.sun_ndc, expected.sun_ndc) {
        (Some([ax, ay]), Some([ex, ey])) => {
            assert_close(&format!("{label}.sun_ndc[0]"), ax, ex, TOL_NDC);
            assert_close(&format!("{label}.sun_ndc[1]"), ay, ey, TOL_NDC);
        }
        (a, b) => assert_eq!(a, b, "{label}.sun_ndc presence mismatch"),
    }
    assert_close(
        &format!("{label}.flare_intensity"),
        actual.flare_intensity,
        expected.flare_intensity,
        TOL_CPU,
    );
    assert_close(
        &format!("{label}.projector_road_coverage"),
        actual.projector_road_coverage,
        expected.projector_road_coverage,
        TOL_CPU,
    );
    assert_close(
        &format!("{label}.wet_fac"),
        actual.wet_fac,
        expected.wet_fac,
        TOL_CPU,
    );
    assert_close(
        &format!("{label}.night_fac"),
        actual.night_fac,
        expected.night_fac,
        TOL_CPU,
    );
    assert_close(
        &format!("{label}.mist_fac"),
        actual.mist_fac,
        expected.mist_fac,
        TOL_CPU,
    );
}

fn assert_gpu(label: &str, actual: GpuProbe, expected: GpuProbe) {
    assert_close(
        &format!("{label}.sky_top_lum"),
        actual.sky_top_lum,
        expected.sky_top_lum,
        TOL_LUM,
    );
    assert_close(
        &format!("{label}.road_center_lum"),
        actual.road_center_lum,
        expected.road_center_lum,
        TOL_LUM,
    );
    assert_close(
        &format!("{label}.sun_disc_max_lum"),
        actual.sun_disc_max_lum,
        expected.sun_disc_max_lum,
        TOL_SUN,
    );
    assert_close(
        &format!("{label}.flare_bloom_lum"),
        actual.flare_bloom_lum,
        expected.flare_bloom_lum,
        TOL_SUN,
    );
}

fn scenario_game(scenario: &Scenario) -> Game {
    let mut game = Game::new();
    game.set_start_hour(scenario.time);
    game.set_weather(scenario.weather);
    game.set_seed(42);
    game
}

/// Pure-CPU frame matching the snapshot path: zero dt, 16:9 aspect.
fn frame_for(game: &Game) -> Frame {
    let mut state = FrameState::default();
    let anchors = CarLightAnchors {
        lateral: 0.8,
        long_half: 1.84,
        headlight_y: 0.48,
        taillight_y: 0.48,
    };
    build_frame(
        game,
        Duration::ZERO,
        16.0 / 9.0,
        &mut state,
        &anchors,
        &[anchors],
        Vec::new(),
    )
}

#[test]
fn cpu_probes_match_golden_baselines() {
    for scenario in scenarios() {
        let game = scenario_game(&scenario);
        let frame = frame_for(&game);
        assert_cpu(scenario.name, compute_cpu(&game, &frame), scenario.cpu);
    }
}

/// Renders the scenario offscreen through the real pipeline and checks both
/// CPU and GPU probes against the baselines. Skipped unless explicitly enabled.
#[test]
fn gpu_probes_match_golden_baselines() {
    if std::env::var("LANE_SNAPSHOT_TESTS").as_deref() != Ok("1") {
        eprintln!("skipping GPU snapshot test (set LANE_SNAPSHOT_TESTS=1 to run)");
        return;
    }

    let instance = lane_lunacy::create_headless_instance();
    let devices = lane_lunacy::gpu::enumerate_devices(&instance);
    let physical = lane_lunacy::gpu::select_physical_device(&devices, 0);
    let (device, queue) = lane_lunacy::gpu::create_graphics_context_headless(&physical);
    let font_atlas = lane_lunacy::font::FontAtlas::load();

    for scenario in scenarios() {
        let game = scenario_game(&scenario);
        let output = lane_lunacy::render::snapshot::render_snapshot(
            device.clone(),
            queue.clone(),
            &game,
            &font_atlas,
            42,
            1280,
            720,
            false,
            lane_lunacy::mesh::TerrainDetail::Medium,
        );
        let cpu = compute_cpu(&game, &output.frame);
        let gpu = compute_gpu(
            &output.linear_rgba,
            output.width,
            output.height,
            output.frame.sun_ndc,
        );
        assert_cpu(scenario.name, cpu, scenario.cpu);
        assert_gpu(scenario.name, gpu, scenario.gpu);
    }
}
