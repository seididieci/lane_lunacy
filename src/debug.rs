// SPDX-License-Identifier: MIT

//! Dev-only frame diagnostics, shown by the debug HUD (toggled with F3).
//!
//! `DebugStats` is a plain data bag filled each frame by `App`: smoothed FPS
//! and frame/CPU times, world-mesh volume and last chunk-rebuild cost, particle
//! counts, and the current distance/terrain state. It deliberately carries no
//! references to render or game types so `hud.rs` can render it without
//! coupling into the renderer.

/// Exponential moving-average blend; `alpha` in 0..1 (higher = more responsive).
pub fn ema(current: f32, sample: f32, alpha: f32) -> f32 {
    current + (sample - current) * alpha
}

/// Frame diagnostics for the debug HUD. Zero-initialized; fields are refreshed
/// from the previous frame so the panel always shows last-known values.
#[derive(Clone, Debug, Default)]
pub struct DebugStats {
    /// Smoothed frames per second (inverse of the real `dt`).
    pub fps: f32,
    /// Smoothed wall-clock time of the whole frame (ms).
    pub frame_ms: f32,
    /// Smoothed CPU time spent inside `Renderer::render` (ms).
    pub cpu_ms: f32,
    /// Number of cached world chunk meshes.
    pub world_chunks: usize,
    /// Total world vertices across all cached chunks.
    pub world_verts: usize,
    /// Total world triangles across all cached chunks.
    pub world_tris: usize,
    /// Duration of the last world-chunk rebuild (ms); 0 until the first one.
    pub chunk_rebuild_ms: f32,
    /// How many chunks were (re)built in that last rebuild.
    pub chunks_rebuilt: usize,
    /// Chunk builds queued but not yet committed by the background pool.
    pub chunks_pending: usize,
    /// Chunks committed and ready in the background cache, including prefetched.
    pub chunks_cached: usize,
    /// Particle vertices emitted this frame (rain + dust + mist + flare).
    pub particles: usize,
    /// HUD/UI vertices this frame.
    pub hud_verts: usize,
    /// Distance travelled this run (m).
    pub distance: f32,
    /// Terrain speed factor at the current distance (0.75..1.0).
    pub terrain_factor: f32,
    /// Index of the world chunk the player currently occupies.
    pub chunk_index: i32,
}

impl DebugStats {
    /// Blends the per-frame FPS and frame time from a real `dt`.
    pub fn sample_frame(&mut self, dt_secs: f32) {
        if dt_secs <= 0.0 {
            return;
        }
        let fps = 1.0 / dt_secs;
        self.fps = ema(self.fps, fps, 0.1);
        self.frame_ms = ema(self.frame_ms, dt_secs * 1000.0, 0.1);
    }

    /// Blends the CPU time measured around `Renderer::render`.
    pub fn sample_cpu(&mut self, ms: f32) {
        self.cpu_ms = ema(self.cpu_ms, ms, 0.1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ema_blends_toward_the_sample() {
        assert_eq!(ema(0.0, 10.0, 0.5), 5.0);
        assert_eq!(ema(5.0, 10.0, 1.0), 10.0);
        assert_eq!(ema(5.0, 10.0, 0.0), 5.0);
    }

    #[test]
    fn sample_frame_ignores_non_positive_dt() {
        let mut d = DebugStats::default();
        d.sample_frame(0.0);
        assert_eq!(d.fps, 0.0);
        d.sample_frame(-1.0);
        assert_eq!(d.fps, 0.0);
    }

    #[test]
    fn sample_frame_smooths_fps_and_frame_ms() {
        let mut d = DebugStats::default();
        // Steady 100 FPS (10 ms frames): the smoothed values converge toward it.
        for _ in 0..100 {
            d.sample_frame(0.01);
        }
        assert!((d.fps - 100.0).abs() < 1.0, "fps {}", d.fps);
        assert!((d.frame_ms - 10.0).abs() < 0.1, "frame_ms {}", d.frame_ms);
        // A single hiccup pulls the smoothed values only partway toward it.
        d.sample_frame(0.1);
        assert!(
            d.fps < 100.0 && d.fps > 80.0,
            "hiccup fps should sag but not slam, got {}",
            d.fps
        );
        assert!(
            d.frame_ms > 15.0 && d.frame_ms < 25.0,
            "hiccup frame_ms should rise partway, got {}",
            d.frame_ms
        );
    }
}
