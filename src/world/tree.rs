// SPDX-License-Identifier: MIT

//! Roadside trees: a box trunk plus foliage cones (`variant 0` = pine with
//! stacked cones, `variant 1` = broadleaf with a wide crown), one per
//! `TREE_SPACING` metres on alternating sides, deterministically rolled from
//! world-`s` so chunk rebuilds and snapshot runs place trees identically.

use crate::geom::{push_box, push_cone};
use crate::road::{road_curve, ROAD_HALF};
use crate::vertex::Vertex3d;
use crate::world::terrain::{terrain_height, terrain_slope};
use crate::world::{hash01, spaced_placements, Placement, RoadsideObject};

/// Material slot for tree trunks and foliage (atlas slot 4, see mesh.frag.glsl).
const TREE_MATERIAL: f32 = 4.0;
/// World-space UV scale (tiles per metre) for tree foliage.
const TREE_UV_SCALE: f32 = 0.15;
/// Distance (metres) between consecutive trees along the road.
const TREE_SPACING: f32 = 11.0;

pub static TREE: Tree = Tree;

pub struct Tree;

impl RoadsideObject for Tree {
    fn placements(&self, start_s: f32, end_s: f32) -> Vec<Placement> {
        spaced_placements(start_s, end_s, TREE_SPACING, 0.0)
            .into_iter()
            .map(|s| {
                let side = if ((s / TREE_SPACING).floor() as i32) % 2 == 0 {
                    1.0
                } else {
                    -1.0
                };
                let lateral = ROAD_HALF + 2.5 + hash01(s * 1.3 + 5.1) * 5.5;
                let variant = if hash01(s + 7.3) < 0.5 { 0 } else { 1 };
                Placement {
                    s,
                    side,
                    lateral,
                    variant,
                }
            })
            .collect()
    }

    fn push_geometry(&self, v: &mut Vec<Vertex3d>, i: &mut Vec<u32>, p: &Placement) {
        let height = 2.6 + hash01(p.s * 0.37 + 1.7) * 2.0;
        let lateral = p.side * p.lateral;
        let ground_y = terrain_height(p.s, lateral);
        // Trees root on the displaced terrain; a tree whose base would sit on a
        // steep mountain face or high on a ridge gets culled so trunks never
        // poke sideways out of a slope or float over thin air.
        if terrain_slope(p.s, lateral) > 0.7 || ground_y > 8.0 {
            return;
        }
        let x = road_curve(p.s) + lateral;
        push_tree(v, i, x, -p.s, ground_y, height, p.variant, TREE_UV_SCALE);
    }
}

/// Pushes one stylized roadside tree. All parts use `TREE_MATERIAL`; the vertex
/// color distinguishes trunk from foliage. `ground_y` roots the whole tree on
/// the terrain surface.
fn push_tree(
    v: &mut Vec<Vertex3d>,
    i: &mut Vec<u32>,
    x: f32,
    z: f32,
    ground_y: f32,
    height: f32,
    variant: u32,
    scale: f32,
) {
    let trunk_r = height * 0.05;
    let trunk_h = height * if variant == 0 { 0.24 } else { 0.34 };
    let trunk_col = [0.40, 0.29, 0.19];
    let foliage_col = if variant == 0 {
        [0.16, 0.46, 0.22]
    } else {
        [0.26, 0.52, 0.20]
    };
    push_box(
        v,
        i,
        [x - trunk_r, ground_y, z - trunk_r],
        [x + trunk_r, ground_y + trunk_h, z + trunk_r],
        trunk_col,
        TREE_MATERIAL,
        scale,
    );
    if variant == 0 {
        let c_base = trunk_h;
        let c_span = height - trunk_h;
        let mut r = height * 0.30;
        for k in 0..3u32 {
            let cb = ground_y + c_base + c_span * (k as f32 * 0.28);
            let apex =
                (ground_y + c_base + c_span * (0.55 + k as f32 * 0.15)).min(ground_y + height);
            push_cone(
                v,
                i,
                x,
                z,
                r,
                cb,
                apex,
                foliage_col,
                TREE_MATERIAL,
                scale,
                8,
            );
            r *= 0.62;
        }
    } else {
        push_cone(
            v,
            i,
            x,
            z,
            height * 0.52,
            ground_y + trunk_h * 0.9,
            ground_y + height * 0.74,
            foliage_col,
            TREE_MATERIAL,
            scale,
            8,
        );
        push_cone(
            v,
            i,
            x,
            z,
            height * 0.30,
            ground_y + height * 0.64,
            ground_y + height,
            foliage_col,
            TREE_MATERIAL,
            scale,
            8,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trees_are_generated_off_the_road_deterministically() {
        let (v, i) = crate::mesh::build_world_chunk(0.0, 260.0, crate::mesh::TerrainDetail::Medium);
        let is_tree = |vert: &Vertex3d| vert.material >= TREE_MATERIAL && vert.material < 5.0;
        let foliage: Vec<&Vertex3d> = v.iter().filter(|vert| is_tree(vert)).collect();
        assert!(!foliage.is_empty(), "trees should be generated");
        for vert in &foliage {
            let lateral = vert.position[0] - road_curve(-vert.position[2]);
            assert!(
                lateral.abs() > ROAD_HALF + 1.5,
                "tree geometry must stay off the road, got lateral={lateral}"
            );
            assert!(
                vert.position[1] >= 0.0 && vert.position[1] <= 12.0,
                "tree heights out of range: {}",
                vert.position[1]
            );
        }
        let tri_count = i.len() / 3;
        assert!(tri_count > 0, "trees must contribute triangles");

        // Deterministic: a second identical build produces identical tree
        // placement (Vertex3d isn't PartialEq, so compare the tree signatures).
        let (v2, _) = crate::mesh::build_world_chunk(0.0, 260.0, crate::mesh::TerrainDetail::Medium);
        let sig = |verts: &[Vertex3d]| -> Vec<[f32; 3]> {
            let mut out: Vec<[f32; 3]> = verts
                .iter()
                .filter(|vert| is_tree(vert))
                .map(|vert| vert.position)
                .collect();
            out.sort_by(|a, b| a.partial_cmp(b).unwrap());
            out
        };
        assert_eq!(
            sig(&v),
            sig(&v2),
            "tree placement must be deterministic per world-s"
        );
    }

    #[test]
    fn tree_placements_alternate_sides_and_stay_in_range() {
        let trees = TREE.placements(0.0, 260.0);
        assert_eq!(trees.len(), (260.0 / TREE_SPACING).ceil() as usize);
        // Sides alternate; laterals and variants are deterministic.
        for (i, p) in trees.iter().enumerate() {
            let expected_side = if i % 2 == 0 { 1.0 } else { -1.0 };
            assert_eq!(p.side, expected_side, "sides must alternate");
            assert!(
                p.lateral >= ROAD_HALF + 2.5 && p.lateral <= ROAD_HALF + 8.0,
                "lateral out of range: {}",
                p.lateral
            );
            assert!(p.variant <= 1, "variant out of range");
        }
        assert_eq!(
            trees,
            TREE.placements(0.0, 260.0),
            "placement is deterministic"
        );
    }
}
