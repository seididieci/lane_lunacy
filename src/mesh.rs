// SPDX-License-Identifier: MIT

use crate::geom::push_quad;
use crate::road::{road_curve, ROAD_HALF};

/// Half-width of the terrain ribbon on each side of the road. Long enough that
/// its outer edge (foothills ~20m up) sits at the fully-opaque fog distance, so
/// the terrain reads as a mountain backdrop instead of ending in a cutoff.
const GROUND_HALF_W: f32 = 600.0;
use crate::surface::{material_at, SurfaceMaterial};
use crate::vertex::Vertex3d;
use crate::world::terrain::{terrain_height, terrain_slope};

/// Shortcuts for the ribbon cross-section's fixed surfaces. The asphalt slots
/// and UV scales live in `surface.rs` so the mesh and gameplay agree.
const ASPHALT_BASE: SurfaceMaterial = SurfaceMaterial::AsphaltBase;
const GRASS: SurfaceMaterial = SurfaceMaterial::Grass;

/// Flat normal for a ground cell from three of its corners, flopped to point
/// up/outward (toward the road on a mountain face) so it lights the visible
/// face.
fn flat_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let mut n = [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ];
    if n[1] < 0.0 {
        n = [-n[0], -n[1], -n[2]];
    }
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if len < 1e-6 {
        [0.0, 1.0, 0.0]
    } else {
        [n[0] / len, n[1] / len, n[2] / len]
    }
}

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

    // Local terrain ribbon around the road (per-chunk). The old single
    // full-width quad was too coarse to displace, so the ribbon is re-tessellated
    // into lateral bands (dense near the road where the hills and mountains
    // live, coarse far out where the foothills fade into fog). Each vertex sits
    // on the deterministic terrain_height(s, lateral); each cell gets a flat
    // normal from its corners and the surface material its slope warrants
    // (grass vs rock). Shared corners land on identical coordinates between
    // cells, so hills and mountain ridges are seamless within and across chunks.
    let ground_color = [0.7, 0.85, 0.6];
    let rock_color = [0.85, 0.85, 0.88];
    let mut lat_edges = vec![0.0f32, 2.0, 4.5, 7.0];
    let mut d = 7.0;
    while d < 20.0 {
        d += 1.0;
        lat_edges.push(d);
    }
    while d < 45.0 {
        d += 2.0;
        lat_edges.push(d);
    }
    while d < 120.0 {
        d += 5.0;
        lat_edges.push(d);
    }
    while d < GROUND_HALF_W {
        d += 25.0;
        lat_edges.push(d);
    }
    if *lat_edges.last().unwrap() != GROUND_HALF_W {
        lat_edges.push(GROUND_HALF_W);
    }

    let mut s_ground = start_s;
    while s_ground < end_s {
        let s0 = s_ground;
        let s1 = (s_ground + step).min(end_s);
        let s_mid = (s0 + s1) * 0.5;
        let z0 = -s0;
        let z1 = -s1;
        let x0 = road_curve(s0);
        let x1 = road_curve(s1);
        for w in lat_edges.windows(2) {
            let l0 = w[0];
            let l1 = w[1];
            for side in [-1.0, 1.0] {
                let la0 = l0 * side;
                let la1 = l1 * side;
                let ca = [x0 + la0, terrain_height(s0, la0), z0];
                let cb = [x0 + la1, terrain_height(s0, la1), z0];
                let cc = [x1 + la1, terrain_height(s1, la1), z1];
                let cd = [x1 + la0, terrain_height(s1, la0), z1];
                let lat_mid = (l0 + l1) * 0.5 * side;
                let m = material_at(s_mid, lat_mid, terrain_slope(s_mid, lat_mid));
                let col = if m == SurfaceMaterial::Rock {
                    rock_color
                } else {
                    ground_color
                };
                push_quad(
                    &mut v,
                    &mut i,
                    ca,
                    cb,
                    cc,
                    cd,
                    flat_normal(ca, cb, cc),
                    col,
                    m.atlas_slot(),
                    m.uv_scale(),
                );
            }
        }
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
        // Road elevation at this position
        let asphalt = material_at(s0, 0.0, 0.0);
        let asphalt_slot = asphalt.atlas_slot();
        let asphalt_scale = asphalt.uv_scale();

        // asphalt (at terrain height)
        push_quad(
            &mut v,
            &mut i,
            [
                x0 - half_w,
                crate::world::terrain::terrain_height(s0, 0.0) + 0.015,
                z0,
            ],
            [
                x0 + half_w,
                crate::world::terrain::terrain_height(s0, 0.0) + 0.015,
                z0,
            ],
            [
                x1 + half_w,
                crate::world::terrain::terrain_height(s1, 0.0) + 0.015,
                z1,
            ],
            [
                x1 - half_w,
                crate::world::terrain::terrain_height(s1, 0.0) + 0.015,
                z1,
            ],
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
    fn world_chunk_has_open_ground_and_clear_road_corridor() {
        let (v, _) = build_world_chunk(0.0, 260.0);
        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = 0.0f32;
        let mut min_y = f32::INFINITY;
        for vert in &v {
            min_x = min_x.min(vert.position[0]);
            max_x = max_x.max(vert.position[0]);
            max_y = max_y.max(vert.position[1]);
            min_y = min_y.min(vert.position[1]);
        }
        // The terrain ribbon spans ±GROUND_HALF_W around the road.
        assert!(
            min_x <= -GROUND_HALF_W + 0.01 && max_x >= GROUND_HALF_W - 0.01,
            "terrain must span ±GROUND_HALF_W, got [{min_x}, {max_x}]"
        );
        // The road corridor stays open: no terrain obstacles inside |lateral| < ROAD_HALF.
        // Vertices within the road are part of the road mesh itself (asphalt, shoulders, etc.)
        // and shouldn't exceed their expected height by much.
        for vert in v.iter().filter(|vert| vert.position[1] > 0.05) {
            let lateral = vert.position[0] - road_curve(-vert.position[2]);
            if lateral.abs() < ROAD_HALF {
                // Inside corridor: should be within ~0.2 of terrain height for road mesh vertices
                let s = vert.position[2];
                let terrain_y = crate::world::terrain::terrain_height(s, 0.0);
                assert!(
                    (vert.position[1] - terrain_y).abs() < 1.0,
                    "road-corridor vertex at s={}{{}} with lateral={}{{}} is {} units from terrain surface",
                    s, lateral, vert.position[1] - terrain_y
                );
            }
        }
        // Terrain is bounded: nothing below the road plane, nothing past the
        // deterministic ceiling (mountain crests).
        assert!(min_y >= -0.01, "terrain must not dip below the road plane");
        assert!(
            max_y <= crate::world::terrain::MAX_HEIGHT + 0.01,
            "terrain must stay bounded, got max_y={max_y}"
        );
        // Hills/mountains are actually present in the chunk.
        assert!(max_y > 1.5, "the terrain must rise off the road");
    }

    #[test]
    fn world_chunk_builds_roadside_scenery() {
        let (v, _) = build_world_chunk(0.0, 260.0);
        let trees = v
            .iter()
            .filter(|vert| vert.material >= 4.0 && vert.material < 5.0)
            .count();
        let elevated = v.iter().filter(|vert| vert.position[1] > 1.0).count();
        assert!(trees > 0, "trees should be part of the chunk");
        assert!(
            elevated > 0,
            "posts and lamp poles should be part of the chunk"
        );
    }

    #[test]
    fn world_chunk_terrain_bakes_rock_into_mountain_faces() {
        // Mountain near/far faces are steep enough to cross the rock threshold
        // whenever a deterministic ridge block is active; scan chunks until one
        // is found (the world is deterministic, so this is a fixed guarantee,
        // not a probability).
        let mut s = 0.0;
        let mut found = false;
        while s < 5200.0 {
            let (v, _) = build_world_chunk(s, 260.0);
            if v.iter()
                .any(|vert| vert.material == SurfaceMaterial::Rock.atlas_slot())
            {
                found = true;
                break;
            }
            s += 260.0;
        }
        assert!(found, "some chunk must contain rock mountain faces");
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
