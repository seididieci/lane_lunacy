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
        } => lane_lunacy::run(gpu, weather, start_hour, seed, windowed, debug),
        RunMode::Snapshot(opts) => lane_lunacy::run_snapshot(opts),
    }
}
