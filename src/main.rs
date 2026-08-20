// SPDX-License-Identifier: MIT

use lane_lunacy::cli::{self, RunMode};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match cli::parse(&args) {
        RunMode::Interactive {
            gpu,
            weather,
            start_hour,
            seed,
            windowed,
            debug,
            raytrace,
            profile,
            present_mode,
            fps_limit,
            window_capture,
            capture_dir,
            rockwall_view,
            auto_start,
        } => lane_lunacy::run(
            gpu,
            weather,
            start_hour,
            seed,
            windowed,
            debug,
            raytrace,
            profile,
            present_mode,
            fps_limit,
            window_capture,
            capture_dir,
            rockwall_view,
            auto_start,
        ),
        RunMode::Snapshot(opts) => lane_lunacy::run_snapshot(opts),
        RunMode::Report(csv) => lane_lunacy::run_report(csv),
        RunMode::Drive(opts) => lane_lunacy::run_drive(opts),
    }
}
