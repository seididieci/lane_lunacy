// SPDX-License-Identifier: MIT

use lane_lunacy::game::Weather;

fn main() {
    let mut gpu_index = 0;
    let mut weather = Weather::Auto;
    let args: Vec<String> = std::env::args().collect();
    if let Some(pos) = args.iter().position(|a| a == "--gpu") {
        match args.get(pos + 1).and_then(|v| v.parse::<usize>().ok()) {
            Some(n) => gpu_index = n,
            None => eprintln!("invalid value for --gpu, using default index 0"),
        }
    }
    if let Some(pos) = args.iter().position(|a| a == "--weather") {
        match args.get(pos + 1).and_then(|v| Weather::parse(v)) {
            Some(w) => weather = w,
            None => eprintln!(
                "invalid value for --weather (auto|clear|cloudy|rain), using default AUTO"
            ),
        }
    }
    lane_lunacy::run(gpu_index, weather);
}
