// SPDX-License-Identifier: MIT

//! Diagnostic audio capture: pairs the game's RPM with the engine sound's
//! actual output pitch so stepping/latency artifacts can be analyzed offline.
//!
//! Enabled with `--audio-capture <path.csv>`. The audio thread pushes high-rate
//! samples into a lock-free-ish queue (it only ever does a `push_back` on a
//! pre-sized `VecDeque`); the game thread drains that queue and writes both the
//! per-frame game state and the drained audio samples to a CSV. No file or
//! allocation work ever happens on the audio callback.

use std::collections::VecDeque;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

/// One high-rate sample recorded by the engine source on the audio thread.
#[derive(Clone, Copy, Debug)]
pub struct TraceSample {
    /// Absolute sample index of the engine source.
    pub sample_idx: u64,
    /// Game frame counter the audio thread last saw.
    pub frame: u32,
    /// Sample position within that game frame (0..=frame_len).
    pub samples_in_frame: u64,
    /// The one-pole-smoothed revs fraction used for pitch.
    pub smooth_rpm: f32,
    /// The fundamental frequency actually used (Hz).
    pub hz: f32,
}

const QUEUE_CAPACITY: usize = 16_384;
/// Flush the CSV every this many game frames so data survives a crash.
const FLUSH_EVERY_FRAMES: u32 = 120;

const HEADER: &str = "kind,elapsed_s,audio_frame,dt_s,speed_mps,gear,game_rpm_frac,audio_smooth_rpm,audio_hz,audio_sample_idx,samples_in_frame";

/// Shared capture sink. `push` is the only method the audio thread calls; the
/// writer is only ever touched by the game thread (behind its own mutex).
pub struct AudioCapture {
    enabled: AtomicBool,
    queue: Mutex<VecDeque<TraceSample>>,
    writer: Mutex<BufWriter<std::fs::File>>,
    csv_path: PathBuf,
    frames_written: AtomicU32,
}

impl AudioCapture {
    /// Opens the CSV (creating parent dirs) and writes the header.
    pub fn open(path: &Path) -> std::io::Result<AudioCapture> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let mut file = BufWriter::new(std::fs::File::create(path)?);
        writeln!(file, "{HEADER}")?;
        file.flush()?;
        Ok(AudioCapture {
            enabled: AtomicBool::new(true),
            queue: Mutex::new(VecDeque::with_capacity(QUEUE_CAPACITY)),
            writer: Mutex::new(file),
            csv_path: path.to_path_buf(),
            frames_written: AtomicU32::new(0),
        })
    }

    pub fn path(&self) -> &Path {
        &self.csv_path
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Audio-thread only: enqueue one high-rate sample. Pre-sized so this never
    /// allocates; the game thread drains each frame so the queue stays small.
    pub fn push(&self, s: TraceSample) {
        if !self.is_enabled() {
            return;
        }
        let _ = self.queue.lock().map(|mut q| q.push_back(s));
    }

    /// Game-thread only: record one game frame, then drain the audio queue into
    /// the CSV.
    #[allow(clippy::too_many_arguments)]
    pub fn record_frame(
        &self,
        elapsed_s: f32,
        frame: u32,
        dt_s: f32,
        speed_mps: f32,
        gear: u32,
        game_rpm_frac: f32,
        audio_smooth_rpm: f32,
        audio_hz: f32,
        audio_sample_idx: u64,
    ) {
        if !self.is_enabled() {
            return;
        }
        let row = format!(
            "frame,{elapsed_s:.6},{frame},{dt_s:.6},{speed_mps:.3},{gear},{game_rpm_frac:.6},{audio_smooth_rpm:.6},{audio_hz:.3},{audio_sample_idx},"
        );
        self.write_row(&row);

        let drained: Vec<TraceSample> = self
            .queue
            .lock()
            .map(|mut q| q.drain(..).collect())
            .unwrap_or_default();
        for s in drained {
            let row = format!(
                "audio,{:.6},{},{},{},{},{},{:.6},{:.3},{},{}",
                elapsed_s,
                s.frame,
                "",
                "",
                "",
                "",
                s.smooth_rpm,
                s.hz,
                s.sample_idx,
                s.samples_in_frame,
            );
            self.write_row(&row);
        }

        let frames = self.frames_written.fetch_add(1, Ordering::Relaxed) + 1;
        if frames.is_multiple_of(FLUSH_EVERY_FRAMES) {
            self.flush();
        }
    }

    /// Flushes and closes the CSV, returning the path.
    pub fn close(&self) -> PathBuf {
        self.enabled.store(false, Ordering::Relaxed);
        self.flush();
        self.csv_path.clone()
    }

    fn write_row(&self, row: &str) {
        if let Ok(mut w) = self.writer.lock() {
            if writeln!(w, "{row}").is_err() {
                eprintln!("audio capture: failed to write row; disabling");
                self.enabled.store(false, Ordering::Relaxed);
            }
        }
    }

    fn flush(&self) {
        if let Ok(mut w) = self.writer.lock() {
            let _ = w.flush();
        }
    }
}

impl std::fmt::Debug for AudioCapture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioCapture")
            .field("csv_path", &self.csv_path)
            .field("enabled", &self.is_enabled())
            .finish()
    }
}

/// Convenience shared handle.
pub type SharedCapture = Arc<AudioCapture>;
