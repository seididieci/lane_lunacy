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

/// Mutable per-frame scene state, independent of any present target.
pub struct FrameBuilder {
    state: FrameState,
    world_chunks: Vec<WorldChunk>,
    world_anchor_chunk: i32,
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
            world_anchor_chunk: i32::MIN,
        }
    }

    /// Deterministic presenter for the headless snapshot path: the particle
    /// systems seed from the scenario seed, so the render is reproducible.
    pub fn with_seed(seed: u64) -> Self {
        FrameBuilder {
            state: FrameState::with_seed(seed),
            world_chunks: Vec::new(),
            world_anchor_chunk: i32::MIN,
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
        build_frame(
            game,
            dt,
            aspect,
            &mut self.state,
            &scene.player_anchors,
            &scene.traffic_anchors,
            hud_verts,
        )
    }

    /// Cached world mesh buffers for the current anchor chunk, handed to the
    /// command-buffer recorder.
    pub fn world_chunks(&self) -> &[WorldChunk] {
        &self.world_chunks
    }

    fn ensure_world_chunks(&mut self, scene: &SceneResources, player_distance: f32) {
        let current_chunk = (player_distance / WORLD_CHUNK_LEN).floor() as i32;
        if current_chunk == self.world_anchor_chunk {
            return;
        }
        self.world_chunks.clear();
        for rel in -WORLD_CHUNKS_BEHIND..=WORLD_CHUNKS_AHEAD {
            let chunk_idx = current_chunk + rel;
            let start_s = chunk_idx as f32 * WORLD_CHUNK_LEN;
            let (wv, wi) = build_world_chunk(start_s, WORLD_CHUNK_LEN);
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
            self.world_chunks.push((world_vertices, world_indices));
        }
        self.world_anchor_chunk = current_chunk;
    }
}
