// SPDX-License-Identifier: MIT

//! Per-frame session profiler: records how long each frame takes to generate
//! and where that time goes, so a performance-hunting session can be analyzed
//! after the fact instead of eyeballing the F3 HUD.
//!
//! Enabled with `--profile <path.csv>`: every frame pushes a `FrameTimings`
//! row into the CSV (flushed periodically), and on close the profiler writes a
//! Markdown report next to the CSV (spike frames, dominant phases, percentiles)
//! and prints the list of generated files before the process exits.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// One frame's worth of timings, in milliseconds (except the counters).
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameTimings {
    pub frame_idx: u64,
    /// Seconds since the profiling session started.
    pub elapsed_s: f32,
    /// Wall-clock time between this frame and the previous one (ms).
    pub dt_ms: f32,
    /// `game.update` (simulation) time (ms).
    pub sim_ms: f32,
    /// Menu/HUD widget tree build time (ms).
    pub ui_ms: f32,
    /// World-chunk (re)build time (ms).
    pub rebuild_ms: f32,
    /// Chunks rebuilt in that frame.
    pub chunks_rebuilt: usize,
    /// `FrameBuilder::build` (camera/lights/particles) time (ms).
    pub frame_ms: f32,
    /// Scene render-pass recording time (ms).
    pub scene_ms: f32,
    /// Bloom downsample-chain recording time (ms).
    pub bloom_ms: f32,
    /// Post-composite recording time (ms).
    pub post_ms: f32,
    /// HUD/text pass recording time (ms).
    pub hud_ms: f32,
    /// `record_frame_posted` total time (ms).
    pub record_ms: f32,
    /// Swapchain acquire time (ms).
    pub acquire_ms: f32,
    /// Previous-frame fence wait time (ms).
    pub fence_ms: f32,
    /// Swapchain acquire + previous-frame fence wait (ms). Kept for
    /// compatibility with older CSVs; equals `acquire_ms + fence_ms`.
    pub gpu_wait_ms: f32,
    /// Future chain submit + present + flush time (ms).
    pub submit_ms: f32,
    /// Time spent in the event loop between frames, i.e. outside
    /// `about_to_wait` (ms). 0 on the first frame.
    pub idle_ms: f32,
    /// `Renderer::render` total time (ms).
    pub render_ms: f32,
    /// Whole-frame time (the profiler's own measurement, ms).
    pub total_ms: f32,
}

impl FrameTimings {
    pub const HEADER: &'static str = concat!(
        "frame_idx,elapsed_s,dt_ms,sim_ms,ui_ms,rebuild_ms,chunks_rebuilt,frame_ms,",
        "scene_ms,bloom_ms,post_ms,hud_ms,record_ms,acquire_ms,fence_ms,gpu_wait_ms,",
        "submit_ms,idle_ms,render_ms,total_ms"
    );

    pub fn to_csv_row(&self) -> String {
        format!(
            "{},{:.6},{:.3},{:.3},{:.3},{:.3},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
            self.frame_idx,
            self.elapsed_s,
            self.dt_ms,
            self.sim_ms,
            self.ui_ms,
            self.rebuild_ms,
            self.chunks_rebuilt,
            self.frame_ms,
            self.scene_ms,
            self.bloom_ms,
            self.post_ms,
            self.hud_ms,
            self.record_ms,
            self.acquire_ms,
            self.fence_ms,
            self.gpu_wait_ms,
            self.submit_ms,
            self.idle_ms,
            self.render_ms,
            self.total_ms,
        )
    }

    /// Parses one CSV data row back into `FrameTimings` (for report
    /// regeneration). `None` on malformed input.
    pub fn from_csv_row(line: &str) -> Option<Self> {
        let f: Vec<&str> = line.split(',').collect();
        if f.len() != 20 {
            return None;
        }
        let num = |i: usize| f[i].parse::<f32>().ok();
        Some(FrameTimings {
            frame_idx: f[0].parse().ok()?,
            elapsed_s: num(1)?,
            dt_ms: num(2)?,
            sim_ms: num(3)?,
            ui_ms: num(4)?,
            rebuild_ms: num(5)?,
            chunks_rebuilt: f[6].parse().ok()?,
            frame_ms: num(7)?,
            scene_ms: num(8)?,
            bloom_ms: num(9)?,
            post_ms: num(10)?,
            hud_ms: num(11)?,
            record_ms: num(12)?,
            acquire_ms: num(13)?,
            fence_ms: num(14)?,
            gpu_wait_ms: num(15)?,
            submit_ms: num(16)?,
            idle_ms: num(17)?,
            render_ms: num(18)?,
            total_ms: num(19)?,
        })
    }
}

/// How often the CSV buffer is flushed to disk (frames).
const FLUSH_EVERY_FRAMES: usize = 120;

/// Spike threshold (ms): frames slower than this are flagged in the report.
/// 17 ms ≈ 60 FPS budget, so anything above is a build/stutter candidate.
const SPIKE_MS: f32 = 17.0;

/// Frames captured so far, in memory for the end-of-session report.
struct SessionFrames {
    rows: Vec<FrameTimings>,
    bytes: usize,
}

pub struct SessionProfiler {
    csv_path: PathBuf,
    writer: BufWriter<File>,
    frames: SessionFrames,
}

impl SessionProfiler {
    /// Opens the CSV at `path` (creating parent dirs) and writes the header.
    pub fn open(path: &Path) -> std::io::Result<Self> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let mut file = BufWriter::new(File::create(path)?);
        writeln!(file, "{}", FrameTimings::HEADER)?;
        file.flush()?;
        Ok(SessionProfiler {
            csv_path: path.to_path_buf(),
            writer: file,
            frames: SessionFrames {
                rows: Vec::with_capacity(4096),
                bytes: 0,
            },
        })
    }

    /// Records one frame. The row is buffered and flushed every
    /// `FLUSH_EVERY_FRAMES` frames so per-frame I/O stays off the hot path.
    pub fn push(&mut self, t: FrameTimings) {
        let row = t.to_csv_row();
        // Best-effort write: the session still runs if a flush hiccups.
        if writeln!(self.writer, "{row}").is_err() {
            eprintln!("profiler: failed to write frame row; giving up");
        }
        self.frames.bytes += row.len() + 1;
        self.frames.rows.push(t);
        if self.frames.rows.len().is_multiple_of(FLUSH_EVERY_FRAMES) {
            let _ = self.writer.flush();
        }
    }

    /// Flushes the CSV, writes the Markdown report next to it, and returns the
    /// list of files generated (for the caller to print).
    pub fn close(mut self) -> Vec<PathBuf> {
        let _ = self.writer.flush();

        let mut report_path = self.csv_path.clone();
        report_path.set_extension("report.md");
        let report = build_report(&self.frames.rows);
        let _ = std::fs::write(&report_path, report);

        vec![self.csv_path, report_path]
    }

    /// Re-reads an existing session CSV (header + rows) and regenerates its
    /// `report.md`. Lets you re-analyze a captured session after fixing the
    /// profiler without replaying the game.
    pub fn regenerate_report(csv_path: &Path) -> std::io::Result<PathBuf> {
        let text = std::fs::read_to_string(csv_path)?;
        let mut lines = text.lines();
        let header = lines
            .next()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "empty CSV"))?;
        if header != FrameTimings::HEADER {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "unexpected CSV header ({}), possibly from an older session",
                    truncate_header(header)
                ),
            ));
        }
        let mut rows = Vec::new();
        for (lineno, line) in lines.enumerate() {
            rows.push(FrameTimings::from_csv_row(line).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("malformed row {} in {}", lineno + 2, csv_path.display()),
                )
            })?);
        }
        let mut report_path = csv_path.to_path_buf();
        report_path.set_extension("report.md");
        std::fs::write(&report_path, build_report(&rows))?;
        Ok(report_path)
    }
}

/// Column index used by the percentile/spike analysis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Sim,
    Ui,
    Rebuild,
    Frame,
    Scene,
    Bloom,
    Post,
    Hud,
    Record,
    Acquire,
    Fence,
    Submit,
    Idle,
    Render,
    Total,
}

impl Phase {
    fn label(self) -> &'static str {
        match self {
            Phase::Sim => "sim",
            Phase::Ui => "ui",
            Phase::Rebuild => "rebuild",
            Phase::Frame => "frame",
            Phase::Scene => "scene",
            Phase::Bloom => "bloom",
            Phase::Post => "post",
            Phase::Hud => "hud",
            Phase::Record => "record",
            Phase::Acquire => "acquire",
            Phase::Fence => "fence",
            Phase::Submit => "submit",
            Phase::Idle => "idle",
            Phase::Render => "render",
            Phase::Total => "total",
        }
    }

    fn value(self, t: &FrameTimings) -> f32 {
        match self {
            Phase::Sim => t.sim_ms,
            Phase::Ui => t.ui_ms,
            Phase::Rebuild => t.rebuild_ms,
            Phase::Frame => t.frame_ms,
            Phase::Scene => t.scene_ms,
            Phase::Bloom => t.bloom_ms,
            Phase::Post => t.post_ms,
            Phase::Hud => t.hud_ms,
            Phase::Record => t.record_ms,
            Phase::Acquire => t.acquire_ms,
            Phase::Fence => t.fence_ms,
            Phase::Submit => t.submit_ms,
            Phase::Idle => t.idle_ms,
            Phase::Render => t.render_ms,
            Phase::Total => t.total_ms,
        }
    }
}

/// Phases considered when classifying which function slows down a spike frame.
/// Big totals (render/record/total) are composed of the leaf phases, so spikes
/// are attributed to the first leaf that can explain a large chunk of the cost.
const LEAF_PHASES: [Phase; 11] = [
    Phase::Rebuild,
    Phase::Frame,
    Phase::Scene,
    Phase::Bloom,
    Phase::Post,
    Phase::Hud,
    Phase::Acquire,
    Phase::Fence,
    Phase::Submit,
    Phase::Sim,
    Phase::Ui,
];

fn build_report(rows: &[FrameTimings]) -> String {
    if rows.is_empty() {
        return "# Profile report\n\nNo frames recorded.\n".to_string();
    }

    let n = rows.len();
    let first = rows.first().unwrap();
    let last = rows.last().unwrap();
    let duration_s = last.elapsed_s - first.elapsed_s;
    let fps = if duration_s > 0.0 {
        n as f32 / duration_s
    } else {
        0.0
    };

    let mut out = String::new();
    out.push_str("# Lane Lunacy — frame profile report\n\n");
    out.push_str("## Session overview\n\n");
    out.push_str(&format!("- Frames: **{n}**\n"));
    out.push_str(&format!(
        "- Duration: **{duration_s:.2} s** (avg {fps:.1} FPS)\n"
    ));
    out.push_str(&format!(
        "- Spike threshold: **{SPIKE_MS} ms** (60 FPS budget)\n\n"
    ));

    // Spike frames + the phase that explains the delay.
    //
    // `dt_ms` of frame N is the wall-clock gap between the *start* of frame
    // N-1 and the *start* of frame N: it is paid for by whatever frame N-1
    // did (the actual work, `total_ms[N-1]`) plus the idle time `idle_ms[N]`
    // (time spent waiting in the event loop, not rendering). So a spike must
    // be attributed to the previous frame's dominant phase, not this frame's.
    // The very first frame is bootstrap (GPU context, shader compile): it is
    // shown but flagged as `startup` and excluded from the cause tally.
    let spikes: Vec<(usize, f32, f32, Option<Phase>)> = rows
        .iter()
        .enumerate()
        .filter(|(i, t)| *i == 0 || t.dt_ms > SPIKE_MS)
        .map(|(i, t)| {
            let work = if i > 0 { rows[i - 1].total_ms } else { 0.0 };
            let cause = if i == 0 {
                None // startup
            } else if t.idle_ms > SPIKE_MS && t.idle_ms > rows[i - 1].total_ms {
                Some(Phase::Idle)
            } else {
                Some(dominant_phase(&rows[i - 1]))
            };
            (i, t.dt_ms, work, cause)
        })
        .collect();

    out.push_str("## Spike frames\n\n");
    if spikes.is_empty() {
        out.push_str("_No frames exceeded the threshold._\n\n");
    } else {
        out.push_str("| # | frame | dt (ms) | work (ms) | idle (ms) | cause |\n");
        out.push_str("|---|-------|---------|-----------|-----------|-------|\n");
        for (i, dt, work, cause) in &spikes {
            let label = match cause {
                Some(p) => format!("`{}`", p.label()),
                None => "`startup`".to_string(),
            };
            out.push_str(&format!(
                "| {} | {} | {:.1} | {:.1} | {:.1} | {} |\n",
                i + 1,
                rows[*i].frame_idx,
                dt,
                work,
                rows[*i].idle_ms,
                label
            ));
        }
        out.push('\n');
    }

    // Cause tally (startup excluded).
    out.push_str("## Spike causes\n\n");
    let mut tally: Vec<(Phase, usize)> = Vec::new();
    for (_, _, _, cause) in &spikes {
        let Some(phase) = cause else { continue };
        if let Some(entry) = tally.iter_mut().find(|(p, _)| p == phase) {
            entry.1 += 1;
        } else {
            tally.push((*phase, 1));
        }
    }
    tally.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
    if tally.is_empty() {
        out.push_str("_No spikes._\n\n");
    } else {
        for (phase, count) in &tally {
            out.push_str(&format!("- `{}`: {count} spikes\n", phase.label()));
        }
        out.push('\n');
    }

    // Percentiles per phase.
    out.push_str("## Phase timing (ms)\n\n");
    out.push_str("| phase | mean | p50 | p95 | p99 | max |\n");
    out.push_str("|-------|------|-----|-----|-----|-----|\n");
    let mut all_phases = LEAF_PHASES.to_vec();
    all_phases.push(Phase::Render);
    all_phases.push(Phase::Record);
    all_phases.push(Phase::Idle);
    all_phases.push(Phase::Total);
    for phase in all_phases {
        let vals: Vec<f32> = rows.iter().map(|t| phase.value(t)).collect();
        out.push_str(&format!(
            "| {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} |\n",
            phase.label(),
            mean(&vals),
            percentile(&vals, 0.50),
            percentile(&vals, 0.95),
            percentile(&vals, 0.99),
            vals.iter().copied().fold(0.0f32, f32::max)
        ));
    }
    out.push('\n');

    // Slowest frames, most to least. Like the spike table, the cause is read
    // from the previous frame (the `dt` of N reflects the work of N-1) or from
    // `idle` when the wait happened in the event loop.
    out.push_str("## Slowest frames (top 10)\n\n");
    out.push_str("| frame | dt (ms) | work (ms) | idle (ms) | cause |\n");
    out.push_str("|-------|---------|-----------|-----------|-------|\n");
    let mut slowest: Vec<(usize, f32)> =
        rows.iter().enumerate().map(|(i, t)| (i, t.dt_ms)).collect();
    slowest.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (i, dt) in slowest.into_iter().take(10) {
        let label = if i == 0 {
            "`startup`".to_string()
        } else if rows[i].idle_ms > SPIKE_MS && rows[i].idle_ms > rows[i - 1].total_ms {
            "`idle`".to_string()
        } else {
            format!("`{}`", dominant_phase(&rows[i - 1]).label())
        };
        let work = if i == 0 { 0.0 } else { rows[i - 1].total_ms };
        out.push_str(&format!(
            "| {} | {:.1} | {:.1} | {:.1} | {} |\n",
            rows[i].frame_idx, dt, work, rows[i].idle_ms, label
        ));
    }
    out.push('\n');

    out.push_str("Report generated by Lane Lunacy's session profiler (dt > 17 ms spikes).\n");
    out
}

/// Returns the highest-cost leaf phase, or Total if nothing meaningful was
/// measured. Used to call out which function slowed a frame down.
fn dominant_phase(t: &FrameTimings) -> Phase {
    // `frame_ms` includes the rebuild cost, so without this carve-out a chunk
    // rebuild would be mis-attributed to the generic `frame` phase.
    if t.chunks_rebuilt > 0 && t.rebuild_ms > 0.1 {
        return Phase::Rebuild;
    }
    let mut best = Phase::Total;
    let mut best_v = 0.0f32;
    for phase in LEAF_PHASES {
        let v = phase.value(t);
        // Ignore sub-millisecond noise when attributing spikes.
        if v > best_v && v > 0.1 {
            best = phase;
            best_v = v;
        }
    }
    best
}

/// Truncates a CSV header for error messages.
fn truncate_header(header: &str) -> String {
    const MAX: usize = 60;
    if header.len() <= MAX {
        header.to_string()
    } else {
        format!("{}…", &header[..MAX])
    }
}

fn mean(vals: &[f32]) -> f32 {
    if vals.is_empty() {
        0.0
    } else {
        vals.iter().sum::<f32>() / vals.len() as f32
    }
}

/// Nearest-rank percentile (0..=1) over a copy of `vals` — small enough to sort.
fn percentile(vals: &[f32], p: f32) -> f32 {
    if vals.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f32> = vals.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let rank = ((p * (sorted.len() - 1) as f32).round() as usize).min(sorted.len() - 1);
    sorted[rank]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(idx: u64, dt: f32, rebuild: f32) -> FrameTimings {
        FrameTimings {
            frame_idx: idx,
            elapsed_s: idx as f32 * 0.0166,
            dt_ms: dt,
            sim_ms: 0.5,
            ui_ms: 0.2,
            rebuild_ms: rebuild,
            chunks_rebuilt: 0,
            frame_ms: 0.8,
            scene_ms: 0.6,
            bloom_ms: 0.1,
            post_ms: 0.4,
            hud_ms: 0.1,
            record_ms: 1.2,
            acquire_ms: 0.3,
            fence_ms: 0.6,
            gpu_wait_ms: 0.9,
            submit_ms: 0.4,
            idle_ms: 0.0,
            render_ms: 2.0,
            total_ms: dt,
        }
    }

    #[test]
    fn csv_row_matches_header_arity() {
        let t = row(1, 20.0, 3.0);
        assert_eq!(
            FrameTimings::HEADER.split(',').count(),
            t.to_csv_row().split(',').count()
        );
    }

    #[test]
    fn empty_session_reports_no_frames() {
        let report = build_report(&[]);
        assert!(report.contains("No frames recorded"));
    }

    #[test]
    fn spike_report_flags_dominant_phase() {
        // The `dt` of frame N is paid for by the work of frame N-1, so the
        // rebuild lives in the frame *before* the perceived spike.
        let rows: Vec<FrameTimings> = (0u64..100)
            .map(|i| {
                // Frame 49 does the chunk rebuild (the costly work)…
                if i == 49 {
                    let mut t = row(i, 16.0, 25.0);
                    t.total_ms = 25.0;
                    t
                } else if i == 50 {
                    // …and frame 50 *shows* the 40 ms `dt` caused by it.
                    let mut t = row(i, 40.0, 0.0);
                    t.total_ms = 2.0;
                    t
                } else {
                    row(i, 16.0, 0.0)
                }
            })
            .collect();
        let report = build_report(&rows);
        assert!(report.contains("Spike frames"));
        assert!(report.contains("`rebuild`"));
        assert!(report.contains("40.0"));
    }

    #[test]
    fn spike_attribution_uses_previous_frame_work() {
        let mut idle_frame = row(1, 40.0, 0.0); // big dt, own work tiny
        idle_frame.idle_ms = 38.0;
        idle_frame.total_ms = 2.0;
        // Previous frame did negligible work and idle is < its total: nothing
        // dominates, so the cause falls back to `total` of the previous frame.
        let rows = vec![row(0, 16.0, 0.0), idle_frame];
        let report = build_report(&rows);
        assert!(report.contains("`idle`"));
    }

    #[test]
    fn startup_frame_is_flagged_but_not_tallied() {
        let mut startup = row(0, 3261.0, 0.0);
        startup.total_ms = 3261.0;
        let rows = vec![startup, row(1, 16.0, 0.0)];
        let report = build_report(&rows);
        assert!(report.contains("`startup`"));
        // The startup cause must not appear in the tally.
        let causes = report
            .split("## Spike causes")
            .nth(1)
            .unwrap()
            .split("## Phase timing")
            .next()
            .unwrap();
        assert!(!causes.contains("startup"));
    }

    #[test]
    fn percentile_returns_expected_ranks() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(percentile(&v, 0.0), 1.0);
        assert_eq!(percentile(&v, 1.0), 4.0);
        assert_eq!(percentile(&v, 0.5), 3.0);
    }

    #[test]
    fn dominant_phase_prefers_biggest_leaf() {
        let mut t = row(1, 30.0, 0.0);
        t.rebuild_ms = 20.0;
        assert_eq!(dominant_phase(&t), Phase::Rebuild);
    }

    #[test]
    fn dominant_phase_prefers_rebuild_when_chunks_rebuilt() {
        // frame_ms includes rebuild, so without the carve-out a 428 ms rebuild
        // would be attributed to `frame` (429 ms). chunks_rebuilt forces Rebuild.
        let mut t = row(1, 30.0, 0.0);
        t.chunks_rebuilt = 1;
        t.rebuild_ms = 428.0;
        t.frame_ms = 429.0;
        assert_eq!(dominant_phase(&t), Phase::Rebuild);
    }

    #[test]
    fn csv_row_round_trips_through_from_csv_row() {
        let t = row(7, 23.5, 3.1);
        let parsed = FrameTimings::from_csv_row(&t.to_csv_row()).unwrap();
        assert_eq!(parsed.frame_idx, t.frame_idx);
        assert_eq!(parsed.dt_ms, t.dt_ms);
        assert_eq!(parsed.rebuild_ms, t.rebuild_ms);
        assert_eq!(parsed.chunks_rebuilt, t.chunks_rebuilt);
        assert_eq!(parsed.acquire_ms, t.acquire_ms);
        assert_eq!(parsed.fence_ms, t.fence_ms);
        assert_eq!(parsed.submit_ms, t.submit_ms);
        assert_eq!(parsed.idle_ms, t.idle_ms);
        assert_eq!(parsed.total_ms, t.total_ms);
        assert!(FrameTimings::from_csv_row("garbage").is_none());
    }

    #[test]
    fn regenerate_report_from_csv_matches_in_memory_report() {
        let dir = std::env::temp_dir().join(format!("lane_prof_reg_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let csv = dir.join("session.csv");

        let mut profiler = SessionProfiler::open(&csv).unwrap();
        profiler.push(row(0, 16.0, 0.0));
        profiler.push(row(1, 41.0, 0.0));
        let files = profiler.close();
        assert_eq!(files.len(), 2);

        let regenerated = SessionProfiler::regenerate_report(&csv).unwrap();
        assert_eq!(
            regenerated,
            csv.with_extension("report.md"),
            "regenerated report path"
        );
        let first = std::fs::read_to_string(&regenerated).unwrap();
        let second = std::fs::read_to_string(csv.with_extension("report.md")).unwrap();
        assert_eq!(first, second, "regeneration is deterministic");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn close_writes_csv_and_markdown_report() {
        let dir = std::env::temp_dir().join(format!("lane_prof_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let csv = dir.join("session.csv");

        let mut profiler = SessionProfiler::open(&csv).unwrap();
        // Frame 0 does the rebuild work; frame 1 *shows* the 41 ms `dt` it
        // caused (off-by-one: `dt[N]` reflects the work of frame N-1).
        let mut work = row(0, 16.0, 22.0);
        work.total_ms = 22.0;
        profiler.push(work);
        profiler.push(row(1, 41.0, 0.0));
        let files = profiler.close();

        let csv_text = std::fs::read_to_string(&csv).unwrap();
        assert!(csv_text.starts_with(FrameTimings::HEADER));
        assert!(csv_text.lines().count() == 3, "header + 2 rows");

        let report_path = csv.with_extension("report.md");
        let report = std::fs::read_to_string(&report_path).unwrap();
        assert!(report.contains("## Spike frames"));
        assert!(report.contains("`rebuild`"));

        assert_eq!(files, vec![csv.clone(), report_path.clone()]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
