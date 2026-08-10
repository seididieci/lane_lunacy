// SPDX-License-Identifier: MIT

//! Roadside world objects (marker posts, street lamps, trees — and later
//! hills/cliffs). Each object type lives in its own file and is responsible for
//! its own deterministic placement, geometry, and (for lamps) light-anchor
//! query, so adding a new roadside object is just one more entry in `OBJECTS`.
//!
//! Placement is a pure function of world-`s`, so the chunk mesh and any other
//! consumer (e.g. the frame's street-lamp pools) always agree and chunk
//! rebuilds are stable across snapshot runs.

pub mod marker_post;
pub mod street_lamp;
pub mod tree;

use crate::vertex::Vertex3d;

/// One deterministic instance of a roadside object: `side` is +1/-1, `lateral`
/// the object's offset from the road center (signed by the object), and
/// `variant` any per-instance roll (tree shape). Everything derives from `s`.
#[derive(Debug, PartialEq)]
pub struct Placement {
    pub s: f32,
    pub side: f32,
    pub lateral: f32,
    pub variant: u32,
}

/// A roadside object type. `placements` enumerates the deterministic instances
/// in `[start_s, end_s)`; `push_geometry` appends that instance's triangles to
/// the chunk buffers.
pub trait RoadsideObject: Send + Sync {
    fn placements(&self, start_s: f32, end_s: f32) -> Vec<Placement>;
    fn push_geometry(&self, v: &mut Vec<Vertex3d>, i: &mut Vec<u32>, p: &Placement);
}

/// Every roadside object type, in the order its geometry is appended to a
/// chunk. A new object registers here and is picked up by
/// [`build_world_scenery`].
pub const OBJECTS: &[&dyn RoadsideObject] = &[
    &marker_post::MARKER_POST,
    &street_lamp::STREET_LAMP,
    &tree::TREE,
];

/// Appends every object's geometry in `[start_s, end_s)` to the chunk buffers.
/// Called by `build_world_chunk` after the road base.
pub fn build_world_scenery(start_s: f32, end_s: f32, v: &mut Vec<Vertex3d>, i: &mut Vec<u32>) {
    for obj in OBJECTS {
        for p in obj.placements(start_s, end_s) {
            obj.push_geometry(v, i, &p);
        }
    }
}

/// Deterministic 0..1 hash from a world-space coordinate, so chunk rebuilds and
/// snapshot runs place scenery identically.
pub(crate) fn hash01(s: f32) -> f32 {
    let x = (s * 1000.0).abs() as u64;
    let mut h = x.wrapping_mul(1_103_515_245).wrapping_add(12_345);
    h ^= h >> 16;
    h = h.wrapping_mul(0x85EB_CA6B);
    h ^= h >> 13;
    ((h & 0x00FF_FFFF) as f32) / 16_777_215.0
}

/// Grid cadence: the world-`s` values `ceil(start/interval)*interval + offset`
/// stepped by `interval` while `< end_s`. Faithful to the per-object loops that
/// predate this helper, including their chunk-boundary behavior (the offset is
/// applied after the ceil-anchored grid, so boundary windows follow the grid,
/// not the offset).
pub(crate) fn spaced_placements(start_s: f32, end_s: f32, interval: f32, offset: f32) -> Vec<f32> {
    let mut out = Vec::new();
    let mut s = (start_s / interval).ceil() * interval + offset;
    while s < end_s {
        out.push(s);
        s += interval;
    }
    out
}
