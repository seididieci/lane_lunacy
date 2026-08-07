// SPDX-License-Identifier: MIT

use crate::road::{road_curve, ROAD_HALF};
use crate::surface::{material_at, SurfaceMaterial};
use crate::vertex::Vertex3d;

/// Shortcuts for the ribbon cross-section's fixed surfaces. The asphalt slots
/// and UV scales live in `surface.rs` so the mesh and gameplay agree.
const ASPHALT_BASE: SurfaceMaterial = SurfaceMaterial::AsphaltBase;
const GRASS: SurfaceMaterial = SurfaceMaterial::Grass;

#[allow(clippy::too_many_arguments)]
fn push_quad(
    v: &mut Vec<Vertex3d>,
    i: &mut Vec<u32>,
    a: [f32; 3],
    b: [f32; 3],
    c: [f32; 3],
    d: [f32; 3],
    n: [f32; 3],
    col: [f32; 3],
    material: f32,
    scale: f32,
) {
    let base = v.len() as u32;
    let uv = |p: [f32; 3]| [p[0] * scale, p[2] * scale];
    v.push(Vertex3d {
        position: a,
        normal: n,
        color: col,
        tex_coord: uv(a),
        material,
    });
    v.push(Vertex3d {
        position: b,
        normal: n,
        color: col,
        tex_coord: uv(b),
        material,
    });
    v.push(Vertex3d {
        position: c,
        normal: n,
        color: col,
        tex_coord: uv(c),
        material,
    });
    v.push(Vertex3d {
        position: d,
        normal: n,
        color: col,
        tex_coord: uv(d),
        material,
    });
    i.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

fn push_box(
    v: &mut Vec<Vertex3d>,
    i: &mut Vec<u32>,
    min: [f32; 3],
    max: [f32; 3],
    col: [f32; 3],
    material: f32,
    scale: f32,
) {
    let (x0, y0, z0) = (min[0], min[1], min[2]);
    let (x1, y1, z1) = (max[0], max[1], max[2]);
    push_quad(
        v,
        i,
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y0, z1],
        [x0, y0, z1],
        [0.0, -1.0, 0.0],
        col,
        material,
        scale,
    );
    push_quad(
        v,
        i,
        [x0, y1, z1],
        [x1, y1, z1],
        [x1, y1, z0],
        [x0, y1, z0],
        [0.0, 1.0, 0.0],
        col,
        material,
        scale,
    );
    push_quad(
        v,
        i,
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
        [0.0, 0.0, 1.0],
        col,
        material,
        scale,
    );
    push_quad(
        v,
        i,
        [x1, y0, z0],
        [x0, y0, z0],
        [x0, y1, z0],
        [x1, y1, z0],
        [0.0, 0.0, -1.0],
        col,
        material,
        scale,
    );
    push_quad(
        v,
        i,
        [x1, y0, z0],
        [x1, y0, z1],
        [x1, y1, z1],
        [x1, y1, z0],
        [1.0, 0.0, 0.0],
        col,
        material,
        scale,
    );
    push_quad(
        v,
        i,
        [x0, y0, z1],
        [x0, y0, z0],
        [x0, y1, z0],
        [x0, y1, z1],
        [-1.0, 0.0, 0.0],
        col,
        material,
        scale,
    );
}

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
            [x0 - 44.0, 0.0, z0],
            [x0 + 44.0, 0.0, z0],
            [x1 + 44.0, 0.0, z1],
            [x1 - 44.0, 0.0, z1],
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

    // roadside marker posts
    let mut post_s = (start_s / 18.0).ceil() * 18.0 + 12.0;
    while post_s < end_s {
        let x = road_curve(post_s);
        let z = -post_s;
        for side in [-1.0, 1.0] {
            let px = x + side * (half_w + 1.0);
            push_box(
                &mut v,
                &mut i,
                [px - 0.07, 0.0, z - 0.07],
                [px + 0.07, 1.05, z + 0.07],
                [0.93, 0.93, 0.9],
                ASPHALT_BASE.atlas_slot(),
                ASPHALT_BASE.uv_scale(),
            );
            push_box(
                &mut v,
                &mut i,
                [px - 0.08, 0.62, z - 0.08],
                [px + 0.08, 0.78, z + 0.08],
                if side < 0.0 {
                    [0.95, 0.3, 0.24]
                } else {
                    [0.24, 0.58, 0.95]
                },
                ASPHALT_BASE.atlas_slot(),
                ASPHALT_BASE.uv_scale(),
            );
        }
        post_s += 18.0;
    }

    // stepped banks on both sides (denser segments to reduce blocky silhouettes)
    let bank_cols = [
        [0.3, 0.25, 0.19],
        [0.34, 0.28, 0.21],
        [0.38, 0.31, 0.24],
        [0.42, 0.35, 0.27],
    ];
    let mut s_bank = (start_s / 4.0).floor() * 4.0;
    let bank_step = 4.0;
    while s_bank < end_s {
        let s0 = s_bank;
        let s1 = (s_bank + bank_step).min(end_s);
        let z0 = -s1 - 0.45;
        let z1 = -s0 + 0.45;
        let cx = road_curve((s0 + s1) * 0.5);
        let und = (((s0 * 0.11).sin()) + 1.0) * 0.5;

        for side in [-1.0, 1.0] {
            for (lvl, col) in bank_cols.iter().enumerate() {
                let inner = half_w + 1.3 + lvl as f32 * 1.3;
                let outer = inner + 1.25;
                let base_y = lvl as f32 * 0.85;
                let top_y = base_y + 0.95 + und * 0.32;
                let x_inner = cx + side * inner;
                let x_outer = cx + side * outer;
                let min_x = x_inner.min(x_outer);
                let max_x = x_inner.max(x_outer);
                push_box(
                    &mut v,
                    &mut i,
                    [min_x, base_y, z0],
                    [max_x, top_y, z1],
                    *col,
                    GRASS.atlas_slot(),
                    GRASS.uv_scale(),
                );
            }
        }

        s_bank += bank_step;
    }

    (v, i)
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
