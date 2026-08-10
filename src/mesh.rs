// SPDX-License-Identifier: MIT

use crate::geom::push_quad;
use crate::road::{road_curve, ROAD_HALF};

/// Half-width of the flat ground ribbon on each side of the road. Wide enough
/// that its outer edge stays behind the fog ramp, so the open terrain reads as
/// intentional instead of ending in a visible cutoff. Widening this adds no
/// vertices (each ground quad already spans the full width).
const GROUND_HALF_W: f32 = 200.0;
use crate::surface::{material_at, SurfaceMaterial};
use crate::vertex::Vertex3d;

/// Shortcuts for the ribbon cross-section's fixed surfaces. The asphalt slots
/// and UV scales live in `surface.rs` so the mesh and gameplay agree.
const ASPHALT_BASE: SurfaceMaterial = SurfaceMaterial::AsphaltBase;
const GRASS: SurfaceMaterial = SurfaceMaterial::Grass;

/// Builds one chunk of the world mesh: the flat ground ribbon and the road
/// (asphalt, shoulders, verges, edge + center lines), then the roadside objects
/// (marker posts, street lamps, trees) via `world::build_world_scenery`. Each
/// object type owns its deterministic placement and geometry in `src/world/`.
pub fn build_world_chunk(start_s: f32, chunk_len: f32) -> (Vec<Vertex3d>, Vec<u32>) {
    let mut v = Vec::new();
    let mut i = Vec::new();
    let half_w = ROAD_HALF;
    let end_s = start_s + chunk_len;
    let step = 2.0;

    // local ground ribbon around the road (per-chunk)
    let ground = [0.7, 0.85, 0.6];
    let mut s_ground = start_s;
    while s_ground < end_s {
        let s0 = s_ground;
        let s1 = (s_ground + step).min(end_s);
        let x0 = road_curve(s0);
        let x1 = road_curve(s1);
        let z0 = -s0;
        let z1 = -s1;
        push_quad(
            &mut v,
            &mut i,
            [x0 - GROUND_HALF_W, 0.0, z0],
            [x0 + GROUND_HALF_W, 0.0, z0],
            [x1 + GROUND_HALF_W, 0.0, z1],
            [x1 - GROUND_HALF_W, 0.0, z1],
            [0.0, 1.0, 0.0],
            ground,
            GRASS.atlas_slot(),
            GRASS.uv_scale(),
        );
        s_ground += step;
    }

    // road ribbon along -Z
    let road = [0.55, 0.55, 0.58];
    let edge_line = [0.95, 0.95, 0.92];
    let center_line = [0.95, 0.84, 0.36];
    let shoulder_a = [0.62, 0.12, 0.12];
    let shoulder_b = [0.86, 0.86, 0.8];
    let verge = [0.6, 0.75, 0.55];
    let mut s = start_s;
    while s < end_s {
        let s0 = s;
        let s1 = (s + step).min(end_s);
        let x0 = road_curve(s0);
        let x1 = road_curve(s1);
        let z0 = -s0;
        let z1 = -s1;
        let asphalt = material_at(s0, 0.0);
        let asphalt_slot = asphalt.atlas_slot();
        let asphalt_scale = asphalt.uv_scale();

        // asphalt
        push_quad(
            &mut v,
            &mut i,
            [x0 - half_w, 0.02, z0],
            [x0 + half_w, 0.02, z0],
            [x1 + half_w, 0.02, z1],
            [x1 - half_w, 0.02, z1],
            [0.0, 1.0, 0.0],
            road,
            asphalt_slot,
            asphalt_scale,
        );

        let shoulder_col = if ((s0 / 4.0) as i32) % 2 == 0 {
            shoulder_a
        } else {
            shoulder_b
        };

        // left shoulder strip
        push_quad(
            &mut v,
            &mut i,
            [x0 - half_w - 0.55, 0.021, z0],
            [x0 - half_w, 0.021, z0],
            [x1 - half_w, 0.021, z1],
            [x1 - half_w - 0.55, 0.021, z1],
            [0.0, 1.0, 0.0],
            shoulder_col,
            ASPHALT_BASE.atlas_slot(),
            ASPHALT_BASE.uv_scale(),
        );

        // right shoulder strip
        push_quad(
            &mut v,
            &mut i,
            [x0 + half_w, 0.021, z0],
            [x0 + half_w + 0.55, 0.021, z0],
            [x1 + half_w + 0.55, 0.021, z1],
            [x1 + half_w, 0.021, z1],
            [0.0, 1.0, 0.0],
            shoulder_col,
            ASPHALT_BASE.atlas_slot(),
            ASPHALT_BASE.uv_scale(),
        );

        // grass verge strips to soften shoulder->terrain transition
        push_quad(
            &mut v,
            &mut i,
            [x0 - half_w - 1.1, 0.016, z0],
            [x0 - half_w - 0.55, 0.016, z0],
            [x1 - half_w - 0.55, 0.016, z1],
            [x1 - half_w - 1.1, 0.016, z1],
            [0.0, 1.0, 0.0],
            verge,
            GRASS.atlas_slot(),
            GRASS.uv_scale(),
        );
        push_quad(
            &mut v,
            &mut i,
            [x0 + half_w + 0.55, 0.016, z0],
            [x0 + half_w + 1.1, 0.016, z0],
            [x1 + half_w + 1.1, 0.016, z1],
            [x1 + half_w + 0.55, 0.016, z1],
            [0.0, 1.0, 0.0],
            verge,
            GRASS.atlas_slot(),
            GRASS.uv_scale(),
        );

        // edge lines
        push_quad(
            &mut v,
            &mut i,
            [x0 - half_w + 0.10, 0.025, z0],
            [x0 - half_w + 0.18, 0.025, z0],
            [x1 - half_w + 0.18, 0.025, z1],
            [x1 - half_w + 0.10, 0.025, z1],
            [0.0, 1.0, 0.0],
            edge_line,
            ASPHALT_BASE.atlas_slot(),
            ASPHALT_BASE.uv_scale(),
        );
        push_quad(
            &mut v,
            &mut i,
            [x0 + half_w - 0.18, 0.025, z0],
            [x0 + half_w - 0.10, 0.025, z0],
            [x1 + half_w - 0.10, 0.025, z1],
            [x1 + half_w - 0.18, 0.025, z1],
            [0.0, 1.0, 0.0],
            edge_line,
            ASPHALT_BASE.atlas_slot(),
            ASPHALT_BASE.uv_scale(),
        );

        // dashed center line
        if ((s0 / 9.0) as i32) % 2 == 0 {
            push_quad(
                &mut v,
                &mut i,
                [x0 - 0.09, 0.026, z0],
                [x0 + 0.09, 0.026, z0],
                [x1 + 0.09, 0.026, z1],
                [x1 - 0.09, 0.026, z1],
                [0.0, 1.0, 0.0],
                center_line,
                ASPHALT_BASE.atlas_slot(),
                ASPHALT_BASE.uv_scale(),
            );
        }
        s += step;
    }

    // Roadside objects (marker posts, street lamps, trees). Each type owns its
    // deterministic placement and geometry in `src/world/`; registering a new
    // object there is all that's needed to draw it.
    crate::world::build_world_scenery(start_s, end_s, &mut v, &mut i);

    (v, i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn world_chunk_has_open_ground_and_no_banks() {
        let (v, _) = build_world_chunk(0.0, 260.0);
        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut tall = 0;
        for vert in &v {
            min_x = min_x.min(vert.position[0]);
            max_x = max_x.max(vert.position[0]);
            if vert.position[1] > 1.5 {
                tall += 1;
            }
        }
        // The flat ground ribbon spans ±GROUND_HALF_W around the road.
        assert!(
            min_x <= -GROUND_HALF_W + 0.01 && max_x >= GROUND_HALF_W - 0.01,
            "ground must span ±GROUND_HALF_W, got [{min_x}, {max_x}]"
        );
        // No wall-like geometry: nothing tall may sit inside the road lanes
        // (|lateral| < half_w - 0.5), and the total tall-vertex budget stays
        // small (trees + lamp poles + arms ≈ ~2k verts). The old banks were a
        // continuous ~3.8m wall hugging the road edge (~12.5k verts/chunk).
        for vert in v.iter().filter(|vert| vert.position[1] > 1.5) {
            let lateral = vert.position[0] - road_curve(-vert.position[2]);
            assert!(
                lateral.abs() > ROAD_HALF - 0.5,
                "tall geometry must clear the road lanes, got lateral={lateral}"
            );
        }
        assert!(
            tall < 5000,
            "no continuous wall may remain, got {tall} tall vertices"
        );
    }

    #[test]
    fn world_chunk_builds_roadside_scenery() {
        let (v, _) = build_world_chunk(0.0, 260.0);
        let trees = v.iter().filter(|vert| vert.material >= 4.0).count();
        let elevated = v.iter().filter(|vert| vert.position[1] > 1.0).count();
        assert!(trees > 0, "trees should be part of the chunk");
        assert!(
            elevated > 0,
            "posts and lamp poles should be part of the chunk"
        );
    }
}

/// Unit hemisphere (radius 1, y >= 0) centered at the origin, used for the sky
/// dome. Drawn with `model = translate(eye) * scale(radius)` so it follows the
/// camera; the normalized position is the sky direction sampled by the fragment
/// shader.
pub fn build_sky_dome(rings: u32, sectors: u32) -> (Vec<Vertex3d>, Vec<u32>) {
    let mut v = Vec::new();
    let mut i = Vec::new();

    for r in 0..=rings {
        let theta = (r as f32 / rings as f32) * std::f32::consts::FRAC_PI_2;
        let y = theta.cos();
        let ring_r = theta.sin();
        for s in 0..=sectors {
            let phi = (s as f32 / sectors as f32) * std::f32::consts::TAU;
            let pos = [ring_r * phi.cos(), y, ring_r * phi.sin()];
            v.push(Vertex3d {
                position: pos,
                normal: pos,
                color: [1.0, 1.0, 1.0],
                tex_coord: [0.0, 0.0],
                material: 0.0,
            });
        }
    }

    let cols = sectors + 1;
    for r in 0..rings {
        for s in 0..sectors {
            let a = r * cols + s;
            let b = a + 1;
            let c = (r + 1) * cols + s;
            let d = c + 1;
            i.extend_from_slice(&[a, b, d, a, d, c]);
        }
    }

    (v, i)
}
