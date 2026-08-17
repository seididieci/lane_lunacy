// SPDX-License-Identifier: MIT

use std::path::PathBuf;

use crate::game::Weather;
use crate::mesh::TerrainDetail;

/// Options for the headless `--snapshot` mode: render one deterministic frame
/// offscreen (no window) and write it as a PNG the agent can inspect.
#[derive(Debug, Clone)]
pub struct SnapshotOptions {
    /// Output PNG path.
    pub path: PathBuf,
    /// Pinned start hour (0..24) for reproducible lighting.
    pub time: Option<f32>,
    pub weather: Weather,
    /// Framebuffer size in pixels.
    pub width: u32,
    pub height: u32,
    /// Deterministic scene seed (cloud tiles, weather phase, start hour).
    pub seed: u64,
    /// Physical device index to use.
    pub gpu: usize,
    /// Render the F3 debug HUD on top of the scene.
    pub debug: bool,
    /// Terrain ribbon density used to build the world chunks.
    pub terrain_detail: TerrainDetail,
}

/// Which top-level program the CLI selects.
#[derive(Debug, Clone)]
pub enum RunMode {
    Interactive {
        gpu: usize,
        weather: Weather,
        start_hour: Option<f32>,
        /// `--seed`, or `None` for a clock-random scene seed.
        seed: Option<u64>,
        /// `--windowed`: start in a floating 90%-FHD window instead of
        /// borderless fullscreen (the default). F11 toggles at runtime.
        windowed: bool,
        /// `--debug`: start with the F3 debug HUD (FPS etc.) enabled.
        debug: bool,
        /// `--profile <path.csv>`: record per-frame timings to the CSV and write
        /// a Markdown analysis report on close.
        profile: Option<PathBuf>,
    },
    Snapshot(SnapshotOptions),
    /// `--report <path.csv>`: re-read an existing profiling session and
    /// regenerate its `report.md` (useful after fixing the profiler) without
    /// launching the game.
    Report(PathBuf),
}

/// Parses the raw command-line arguments (excluding the program name) into a
/// run mode. Preserves the exact interactive behavior of the previous inline
/// parser while adding the `--snapshot` headless mode.
pub fn parse(args: &[String]) -> RunMode {
    let mut gpu = 0usize;
    let mut weather = Weather::Auto;
    let mut start_hour: Option<f32> = None;
    let mut seed: Option<u64> = None;
    let mut windowed = false;
    let mut profile: Option<PathBuf> = None;

    let mut snapshot: Option<PathBuf> = None;
    let mut report: Option<PathBuf> = None;
    let mut width = 1280u32;
    let mut height = 720u32;
    let mut snapshot_debug = false;
    let mut debug = false;
    let mut terrain_detail = TerrainDetail::Medium;

    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        match arg.as_str() {
            "--gpu" => {
                if let Some(v) = args.get(i + 1).and_then(|v| v.parse::<usize>().ok()) {
                    gpu = v;
                    i += 2;
                } else {
                    eprintln!("invalid value for --gpu, using default index 0");
                    i += 1;
                }
            }
            "--weather" => {
                if let Some(v) = args.get(i + 1).and_then(|v| Weather::parse(v)) {
                    weather = v;
                    i += 2;
                } else {
                    eprintln!(
                        "invalid value for --weather (auto|clear|cloudy|rain), using default AUTO"
                    );
                    i += 1;
                }
            }
            "--time" => match args.get(i + 1).and_then(|v| v.parse::<f32>().ok()) {
                Some(h) if (0.0..24.0).contains(&h) => {
                    start_hour = Some(h);
                    i += 2;
                }
                _ => {
                    eprintln!(
                        "invalid value for --time (a decimal hour 0..24), using a random start"
                    );
                    i += 1;
                }
            },
            "--snapshot" => {
                if let Some(v) = args.get(i + 1).filter(|v| !v.starts_with("--")) {
                    snapshot = Some(PathBuf::from(v));
                    i += 2;
                } else {
                    eprintln!("invalid value for --snapshot (missing output PNG path)");
                    i += 1;
                }
            }
            "--size" => {
                if let Some(v) = args.get(i + 1).and_then(|v| parse_size(v)) {
                    (width, height) = v;
                    i += 2;
                } else {
                    eprintln!(
                        "invalid value for --size (expected WxH, e.g. 1280x720), using 1280x720"
                    );
                    i += 1;
                }
            }
            "--seed" => {
                if let Some(v) = args.get(i + 1).and_then(|v| v.parse::<u64>().ok()) {
                    seed = Some(v);
                    i += 2;
                } else {
                    eprintln!("invalid value for --seed (a u64), using 0");
                    i += 1;
                }
            }
            "--windowed" => {
                windowed = true;
                i += 1;
            }
            "--debug" => {
                debug = true;
                snapshot_debug = true;
                i += 1;
            }
            "--profile" => {
                if let Some(v) = args.get(i + 1).filter(|v| !v.starts_with("--")) {
                    profile = Some(PathBuf::from(v));
                    i += 2;
                } else {
                    eprintln!("invalid value for --profile (missing output CSV path)");
                    i += 1;
                }
            }
            "--report" => {
                if let Some(v) = args.get(i + 1).filter(|v| !v.starts_with("--")) {
                    report = Some(PathBuf::from(v));
                    i += 2;
                } else {
                    eprintln!("invalid value for --report (missing input CSV path)");
                    i += 1;
                }
            }
            "--terrain-detail" => {
                if let Some(v) = args.get(i + 1).and_then(|v| TerrainDetail::parse(v)) {
                    terrain_detail = v;
                    i += 2;
                } else {
                    eprintln!(
                        "invalid value for --terrain-detail (low|med|high), using default MED"
                    );
                    i += 1;
                }
            }
            _ => {
                eprintln!("ignoring unrecognized argument: {arg}");
                i += 1;
            }
        }
    }

    match (snapshot, report) {
        (Some(path), _) => RunMode::Snapshot(SnapshotOptions {
            path,
            time: start_hour,
            weather,
            width,
            height,
            seed: seed.unwrap_or(0),
            gpu,
            debug: snapshot_debug,
            terrain_detail,
        }),
        (None, Some(path)) => RunMode::Report(path),
        (None, None) => RunMode::Interactive {
            gpu,
            weather,
            start_hour,
            seed,
            windowed,
            debug,
            profile,
        },
    }
}

/// Parses a `WxH` string into (width, height), both nonzero.
fn parse_size(s: &str) -> Option<(u32, u32)> {
    let (w, h) = s.split_once('x')?;
    let w = w.parse::<u32>().ok()?;
    let h = h.parse::<u32>().ok()?;
    if w == 0 || h == 0 {
        return None;
    }
    Some((w, h))
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::path::Path;

    fn parse_args(args: &[&str]) -> RunMode {
        let owned: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        parse(&owned)
    }

    #[test]
    fn no_args_is_interactive_with_defaults() {
        match parse_args(&[]) {
            RunMode::Interactive {
                gpu,
                weather,
                start_hour,
                seed,
                windowed,
                debug,
                profile,
            } => {
                assert_eq!(gpu, 0);
                assert_eq!(weather, Weather::Auto);
                assert_eq!(start_hour, None);
                assert_eq!(seed, None);
                assert!(!windowed);
                assert!(!debug);
                assert_eq!(profile, None);
            }
            _ => panic!("expected interactive mode"),
        }
    }

    #[test]
    fn interactive_flags_are_preserved() {
        match parse_args(&["--gpu", "1", "--weather", "RAIN", "--time", "18.5"]) {
            RunMode::Interactive {
                gpu,
                weather,
                start_hour,
                seed,
                windowed,
                ..
            } => {
                assert_eq!(gpu, 1);
                assert_eq!(weather, Weather::Rain);
                assert_eq!(start_hour, Some(18.5));
                assert_eq!(seed, None);
                assert!(!windowed);
            }
            _ => panic!("expected interactive mode"),
        }
    }

    #[test]
    fn interactive_debug_flag_enables_debug_hud() {
        match parse_args(&["--debug"]) {
            RunMode::Interactive { debug, .. } => assert!(debug),
            _ => panic!("expected interactive mode"),
        }
    }

    #[test]
    fn profile_flag_records_output_path() {
        match parse_args(&["--profile", "sess.csv"]) {
            RunMode::Interactive { profile, .. } => {
                assert_eq!(profile.as_deref(), Some(Path::new("sess.csv")))
            }
            _ => panic!("expected interactive mode"),
        }
    }

    #[test]
    fn profile_flag_without_value_falls_back_to_none() {
        match parse_args(&["--profile"]) {
            RunMode::Interactive { profile, .. } => assert_eq!(profile, None),
            _ => panic!("expected interactive mode"),
        }
    }

    #[test]
    fn report_flag_selects_report_mode() {
        match parse_args(&["--report", "sess.csv"]) {
            RunMode::Report(path) => assert_eq!(path, PathBuf::from("sess.csv")),
            _ => panic!("expected report mode"),
        }
    }

    #[test]
    fn snapshot_takes_precedence_over_report() {
        match parse_args(&["--snapshot", "a.png", "--report", "sess.csv"]) {
            RunMode::Snapshot(o) => assert_eq!(o.path, PathBuf::from("a.png")),
            _ => panic!("expected snapshot mode"),
        }
    }

    #[test]
    fn interactive_windowed_flag_is_parsed() {
        match parse_args(&["--windowed", "--seed", "7"]) {
            RunMode::Interactive { seed, windowed, .. } => {
                assert_eq!(seed, Some(7));
                assert!(windowed);
            }
            _ => panic!("expected interactive mode"),
        }
    }

    #[test]
    fn snapshot_mode_captures_all_options() {
        match parse_args(&[
            "--snapshot",
            "/tmp/f.png",
            "--time",
            "12",
            "--weather",
            "clear",
            "--size",
            "640x360",
            "--seed",
            "42",
            "--gpu",
            "2",
            "--debug",
        ]) {
            RunMode::Snapshot(o) => {
                assert_eq!(o.path, PathBuf::from("/tmp/f.png"));
                assert_eq!(o.time, Some(12.0));
                assert_eq!(o.weather, Weather::Clear);
                assert_eq!((o.width, o.height), (640, 360));
                assert_eq!(o.seed, 42);
                assert_eq!(o.gpu, 2);
                assert!(o.debug);
            }
            _ => panic!("expected snapshot mode"),
        }
    }

    #[test]
    fn snapshot_uses_defaults_for_missing_options() {
        match parse_args(&["--snapshot", "/tmp/f.png"]) {
            RunMode::Snapshot(o) => {
                assert_eq!((o.width, o.height), (1280, 720));
                assert_eq!(o.seed, 0);
                assert_eq!(o.time, None);
                assert_eq!(o.weather, Weather::Auto);
                assert_eq!(o.gpu, 0);
                assert!(!o.debug);
                assert_eq!(o.terrain_detail, TerrainDetail::Medium);
            }
            _ => panic!("expected snapshot mode"),
        }
    }

    #[test]
    fn snapshot_terrain_detail_flag_is_parsed() {
        match parse_args(&["--snapshot", "/tmp/f.png", "--terrain-detail", "high"]) {
            RunMode::Snapshot(o) => assert_eq!(o.terrain_detail, TerrainDetail::High),
            _ => panic!("expected snapshot mode"),
        }
        match parse_args(&["--snapshot", "/tmp/f.png", "--terrain-detail", "low"]) {
            RunMode::Snapshot(o) => assert_eq!(o.terrain_detail, TerrainDetail::Low),
            _ => panic!("expected snapshot mode"),
        }
        // Invalid value falls back to MED.
        match parse_args(&["--snapshot", "/tmp/f.png", "--terrain-detail", "bogus"]) {
            RunMode::Snapshot(o) => assert_eq!(o.terrain_detail, TerrainDetail::Medium),
            _ => panic!("expected snapshot mode"),
        }
    }

    #[test]
    fn invalid_size_falls_back_to_default() {
        match parse_args(&["--snapshot", "/tmp/f.png", "--size", "bogus"]) {
            RunMode::Snapshot(o) => assert_eq!((o.width, o.height), (1280, 720)),
            _ => panic!("expected snapshot mode"),
        }
    }

    #[test]
    fn out_of_range_time_is_rejected() {
        match parse_args(&["--time", "30"]) {
            RunMode::Interactive { start_hour, .. } => assert_eq!(start_hour, None),
            _ => panic!("expected interactive mode"),
        }
    }

    #[test]
    fn interactive_accepts_seed() {
        match parse_args(&["--seed", "7"]) {
            RunMode::Interactive { seed, .. } => assert_eq!(seed, Some(7)),
            _ => panic!("expected interactive mode"),
        }
    }

    #[test]
    fn parse_size_accepts_and_rejects() {
        assert_eq!(parse_size("640x360"), Some((640, 360)));
        assert_eq!(parse_size("0x100"), None);
        assert_eq!(parse_size("x100"), None);
        assert_eq!(parse_size("100"), None);
    }
}
