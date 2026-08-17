// SPDX-License-Identifier: MIT

//! Roadside marker posts: a small white post with a colored band (red on the
//! left, blue on the right) every `POST_SPACING` metres on both sides of the
//! road, offset from the grid by `POST_OFFSET`.

use crate::geom::push_box;
use crate::road::{road_curve, ROAD_HALF};
use crate::surface::SurfaceMaterial;
use crate::vertex::Vertex3d;
use crate::world::{spaced_placements, Placement, RoadsideObject};

const POST_SPACING: f32 = 18.0;
const POST_OFFSET: f32 = 12.0;
/// Lateral offset of the posts from the road center (outside the shoulder
/// strips, inside the lamp poles).
const POST_LATERAL: f32 = ROAD_HALF + 1.0;
const ASPHALT_BASE: SurfaceMaterial = SurfaceMaterial::AsphaltBase;

pub static MARKER_POST: MarkerPost = MarkerPost;

pub struct MarkerPost;

impl RoadsideObject for MarkerPost {
    fn placements(&self, start_s: f32, end_s: f32) -> Vec<Placement> {
        spaced_placements(start_s, end_s, POST_SPACING, POST_OFFSET)
            .into_iter()
            .flat_map(|s| {
                [-1.0f32, 1.0].into_iter().map(move |side| Placement {
                    s,
                    side,
                    lateral: POST_LATERAL,
                    variant: 0,
                })
            })
            .collect()
    }

    fn push_geometry(&self, v: &mut Vec<Vertex3d>, i: &mut Vec<u32>, p: &Placement) {
        let x = road_curve(p.s) + p.side * p.lateral;
        let z = -p.s;
        // Posts rise with the terrain height at this position
        let terrain_y = crate::world::terrain::terrain_height(p.s, p.lateral);
        let (slot, scale) = (ASPHALT_BASE.atlas_slot(), ASPHALT_BASE.uv_scale());
        push_box(
            v,
            i,
            [x - 0.07, terrain_y + 0.0, z - 0.07],
            [x + 0.07, terrain_y + 1.05, z + 0.07],
            [0.93, 0.93, 0.9],
            slot,
            scale,
        );
        push_box(
            v,
            i,
            [x - 0.08, terrain_y + 0.62, z - 0.08],
            [x + 0.08, terrain_y + 0.78, z + 0.08],
            if p.side < 0.0 {
                [0.95, 0.3, 0.24]
            } else {
                [0.24, 0.58, 0.95]
            },
            slot,
            scale,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::OBJECTS;

    #[test]
    fn marker_posts_remain_on_both_sides() {
        let (v, _) = crate::mesh::build_world_chunk(0.0, 260.0, crate::mesh::TerrainDetail::Medium);
        let posts = v.iter().filter(|vert| vert.position[1] > 1.0).count();
        assert!(posts > 0, "marker posts should still exist");
    }

    #[test]
    fn marker_posts_tile_cleanly_across_chunk_boundaries() {
        // Windows anchored at multiples of 260 (the world-chunk cadence) must
        // share no post and lose none: the post grid is anchored absolutely, so
        // a boundary can't drop a post that sits inside either window.
        let a = MARKER_POST.placements(0.0, 260.0);
        let b = MARKER_POST.placements(260.0, 520.0);
        assert_eq!(
            a.len() + b.len(),
            MARKER_POST.placements(0.0, 520.0).len(),
            "adjacent windows must tile the post grid exactly"
        );
        let full = MARKER_POST.placements(0.0, 520.0);
        let s_vals: Vec<f32> = full.iter().map(|p| p.s).collect();
        // The absolute grid is 12 + 18k; 264 (between the two windows) must be
        // present now that boundaries are anchored on the grid, not the window.
        assert!(
            s_vals.iter().any(|s| (s - 264.0).abs() < 1e-4),
            "the post at s=264 (formerly dropped at the 260m boundary) must exist"
        );
    }

    #[test]
    fn marker_posts_are_placed_on_both_sides_with_regular_cadence() {
        let posts = MARKER_POST.placements(0.0, 260.0);
        // Grid: 12 + 18k < 260 → 14 posts, one per side = 28.
        assert_eq!(posts.len(), 28, "two posts per 18m grid position");
        assert!(
            posts.iter().any(|p| p.side > 0.0) && posts.iter().any(|p| p.side < 0.0),
            "posts appear on both sides"
        );
        // Left (side < 0) and right (side > 0) come in pairs, left first.
        for pair in posts.chunks(2) {
            assert_eq!(pair[0].side, -1.0);
            assert_eq!(pair[1].side, 1.0);
            assert_eq!(pair[0].s, pair[1].s);
        }
        assert_eq!(
            posts,
            MARKER_POST.placements(0.0, 260.0),
            "placement is deterministic"
        );
        // Cadence: consecutive grid positions are exactly POST_SPACING apart.
        let grid: Vec<f32> = spaced_placements(0.0, 260.0, POST_SPACING, POST_OFFSET);
        for w in grid.windows(2) {
            assert!((w[1] - w[0] - POST_SPACING).abs() < 1e-4);
        }
        assert!(
            (grid[0] - POST_OFFSET).abs() < 1e-4,
            "first post is at the offset"
        );
        // Registered in the world registry.
        assert!(
            OBJECTS.iter().any(|o| o.placements(0.0, 40.0).len() > 0),
            "marker posts are part of the world registry"
        );
    }
}
