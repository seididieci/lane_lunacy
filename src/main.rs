// SPDX-License-Identifier: MIT

fn main() {
    let mut gpu_index = 0;
    let args: Vec<String> = std::env::args().collect();
    if let Some(pos) = args.iter().position(|a| a == "--gpu") {
        match args.get(pos + 1).and_then(|v| v.parse::<usize>().ok()) {
            Some(n) => gpu_index = n,
            None => eprintln!("invalid value for --gpu, using default index 0"),
        }
    }
    lane_lunacy::run(gpu_index);
}
