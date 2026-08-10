// SPDX-License-Identifier: MIT

//! Mutable per-frame state shared by every presenter.
//!
//! `FrameBuilder` owns what changes from frame to frame: the particle systems
//! (rain + drift dust), the smoothed camera heading, the sky clock, and the
//! cached world chunk buffers. `build` advances that state and produces the
//! pure CPU `Frame`; `world_chunks` hands the cached mesh buffers to the
//! command-buffer recorder. Both the windowed `Renderer` and the headless
//! snapshot path use it, so the two targets advance identical state and can
//! never drift apart.

use std::time::Duration;

use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};

use crate::game::Game;
use crate::mesh::build_world_chunk;
use crate::render::frame::{build_frame, Frame, FrameState};
use crate::render::scene::SceneResources;
use crate::render::{WORLD_CHUNKS_AHEAD, WORLD_CHUNKS_BEHIND, WORLD_CHUNK_LEN};
use crate::vertex::{HudVertex, Vertex3d};

/// A single world-chunk mesh: vertex buffer + index buffer.
pub(crate) type WorldChunk = (Subbuffer<[Vertex3d]>, Subbuffer<[u32]>);

/// Volume and timing of the cached world meshes, surfaced for the debug HUD.
#[derive(Clone, Copy, Debug, Default)]
pub struct WorldStats {
    pub chunk_count: usize,
    pub world_verts: usize,
    pub world_tris: usize,
    pub last_rebuild_ms: f32,
    pub chunks_rebuilt: usize,
    pub particles: usize,
}

/// Mutable per-frame scene state, independent of any present target.
pub struct FrameBuilder {
    state: FrameState,
    /// World-chunk mesh buffers, kept in ascending chunk-index order. A parallel
    /// `world_chunk_indices` records which chunk index each entry covers, so a
    /// window crossing only rebuilds the chunks that actually changed.
    world_chunks: Vec<WorldChunk>,
    world_chunk_indices: Vec<i32>,
    world_anchor_chunk: i32,
    stats: WorldStats,
}

impl Default for FrameBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameBuilder {
    /// Live presenter: the particle systems seed from the clock, so every run
    /// gets a fresh rain/dust field (the historical interactive look).
    pub fn new() -> Self {
        FrameBuilder {
            state: FrameState::default(),
            world_chunks: Vec::new(),
            world_chunk_indices: Vec::new(),
            world_anchor_chunk: i32::MIN,
            stats: WorldStats::default(),
        }
    }

    /// Deterministic presenter for the headless snapshot path: the particle
    /// systems seed from the scenario seed, so the render is reproducible.
    pub fn with_seed(seed: u64) -> Self {
        FrameBuilder {
            state: FrameState::with_seed(seed),
            world_chunks: Vec::new(),
            world_chunk_indices: Vec::new(),
            world_anchor_chunk: i32::MIN,
            stats: WorldStats::default(),
        }
    }

    /// Advances the persistent state (world chunks, sky clock, camera
    /// smoothing, particles) and computes the full CPU `Frame` for this `Game`
    /// state. The world chunks are re-anchored to the player's current chunk on
    /// first use and whenever the player crosses into a new one.
    pub fn build(
        &mut self,
        scene: &SceneResources,
        game: &Game,
        dt: Duration,
        aspect: f32,
        hud_verts: Vec<HudVertex>,
    ) -> Frame {
        self.ensure_world_chunks(scene, game.vehicle.distance);
        let frame = build_frame(
            game,
            dt,
            aspect,
            &mut self.state,
            &scene.player_anchors,
            &scene.traffic_anchors,
            hud_verts,
        );
        self.stats.particles = frame.particle_verts.len()
            + frame.dust_verts.len()
            + frame.mist_verts.len()
            + frame.flare_verts.len();
        frame
    }

    /// Cached world mesh buffers for the current anchor chunk, handed to the
    /// command-buffer recorder.
    pub fn world_chunks(&self) -> &[WorldChunk] {
        &self.world_chunks
    }

    /// Volume and timing of the cached world meshes (debug HUD).
    pub fn world_stats(&self) -> WorldStats {
        self.stats
    }

    fn ensure_world_chunks(&mut self, scene: &SceneResources, player_distance: f32) {
        let current_chunk = (player_distance / WORLD_CHUNK_LEN).floor() as i32;
        if current_chunk == self.world_anchor_chunk {
            return;
        }
        let started = std::time::Instant::now();
        // Sliding window of [current-BEHIND, current+AHEAD]. Terrain is a pure
        // function of world coords, so a chunk at a given index is identical
        // every time the car visits it. Only the chunk(s) whose index actually
        // changed need rebuilding: a normal +1/-1 crossing rebuilds exactly one
        // chunk instead of the whole window, keeping the per-crossing stutter
        // ~8x smaller. Teleports (restart) shift many chunks at once.
        let new_first = current_chunk - WORLD_CHUNKS_BEHIND;
        let new_last = current_chunk + WORLD_CHUNKS_AHEAD;

        // Keep the chunks that are still inside the window (both ranges are
        // ascending and contiguous, so a two-pointer merge is exact).
        let (kept_old, to_build) = window_plan(&self.world_chunk_indices, new_first, new_last);
        let mut kept = Vec::with_capacity(self.world_chunks.len());
        let mut kept_idx = Vec::with_capacity(self.world_chunks.len());
        let mut rebuilt = 0usize;
        for &slot in &kept_old {
            kept.push(self.world_chunks[slot].clone());
            kept_idx.push(self.world_chunk_indices[slot]);
        }
        for new_idx in to_build {
            let (wv, wi) = build_world_chunk(new_idx as f32 * WORLD_CHUNK_LEN, WORLD_CHUNK_LEN);
            let world_vertices = Buffer::from_iter(
                scene.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::VERTEX_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                wv,
            )
            .expect("world chunk vertices");
            let world_indices = Buffer::from_iter(
                scene.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::INDEX_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                wi,
            )
            .expect("world chunk indices");
            kept.push((world_vertices, world_indices));
            kept_idx.push(new_idx);
            rebuilt += 1;
        }
        self.world_chunks = kept;
        self.world_chunk_indices = kept_idx;

        let mut verts = 0usize;
        let mut tris = 0usize;
        for (wv, wi) in &self.world_chunks {
            verts += wv.len() as usize;
            tris += (wi.len() / 3) as usize;
        }
        self.stats.chunk_count = self.world_chunks.len();
        self.stats.world_verts = verts;
        self.stats.world_tris = tris;
        self.stats.last_rebuild_ms = started.elapsed().as_secs_f32() * 1000.0;
        self.stats.chunks_rebuilt = rebuilt;
        self.world_anchor_chunk = current_chunk;
    }
}

/// Given the ascending chunk indices currently cached and the ascending
/// `[first, last]` window the car needs, returns the slots to keep from the
/// cache (indices into `cached`) and the window indices that must be built
/// fresh. Both inputs are sorted ascending, so the merge is a linear scan.
fn window_plan(cached: &[i32], first: i32, last: i32) -> (Vec<usize>, Vec<i32>) {
    let mut kept = Vec::with_capacity(cached.len());
    let mut build = Vec::new();
    let mut c = 0usize;
    let mut want = first;
    while want <= last {
        match cached.get(c) {
            Some(&idx) if idx == want => {
                kept.push(c);
                c += 1;
                want += 1;
            }
            Some(&idx) if idx < want => {
                // Cached chunk fell out of the window (behind the car).
                c += 1;
            }
            _ => {
                // This window index isn't cached: needs a rebuild.
                build.push(want);
                want += 1;
            }
        }
    }
    (kept, build)
}

#[cfg(test)]
mod tests {
    use super::window_plan;
    use crate::render::{WORLD_CHUNKS_AHEAD, WORLD_CHUNKS_BEHIND};

    fn window(c: i32) -> (i32, i32) {
        (c - WORLD_CHUNKS_BEHIND, c + WORLD_CHUNKS_AHEAD)
    }

    #[test]
    fn initial_window_builds_every_chunk() {
        let (first, last) = window(0);
        let (kept, build) = window_plan(&[], first, last);
        assert!(kept.is_empty());
        assert_eq!(
            build.len(),
            (WORLD_CHUNKS_AHEAD + WORLD_CHUNKS_BEHIND + 1) as usize
        );
        assert_eq!(build.first().copied(), Some(first));
        assert_eq!(build.last().copied(), Some(last));
    }

    #[test]
    fn forward_crossing_rebuilds_only_the_new_leading_chunk() {
        let (f0, l0) = window(0);
        let cached: Vec<i32> = (f0..=l0).collect();
        let (f1, l1) = window(1);
        let (kept, build) = window_plan(&cached, f1, l1);
        // All but the dropped trailing chunk are reused.
        assert_eq!(kept.len(), cached.len() - 1);
        assert_eq!(build, vec![l1]);
    }

    #[test]
    fn backward_crossing_rebuilds_only_the_new_leading_chunk() {
        let (f0, l0) = window(10);
        let cached: Vec<i32> = (f0..=l0).collect();
        let (f1, l1) = window(9);
        let (kept, build) = window_plan(&cached, f1, l1);
        assert_eq!(kept.len(), cached.len() - 1);
        assert_eq!(build, vec![f1]);
    }

    #[test]
    fn teleport_rebuilds_entirely_non_overlapping_window() {
        let cached: Vec<i32> = (-1..=6).collect();
        let (f, l) = window(100);
        let (kept, build) = window_plan(&cached, f, l);
        assert!(kept.is_empty());
        assert_eq!(build.first().copied(), Some(f));
        assert_eq!(build.last().copied(), Some(l));
    }

    #[test]
    fn overlapping_teleport_keeps_intersection_and_rebuilds_the_rest() {
        let cached: Vec<i32> = (-1..=6).collect();
        // Jump forward 3: overlap is indices 2..=6, rebuild 7..=9.
        let (f, l) = window(3);
        let (kept, build) = window_plan(&cached, f, l);
        assert_eq!(kept.len(), 5);
        assert_eq!(build, vec![7, 8, 9]);
    }
}
