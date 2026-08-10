// SPDX-License-Identifier: MIT

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

/// Material slot for tree trunks and foliage (atlas slot 4, see mesh.frag.glsl).
const TREE_MATERIAL: f32 = 4.0;
/// World-space UV scale (tiles per metre) for tree foliage.
const TREE_UV_SCALE: f32 = 0.15;

/// World-space spacing (metres) between street-lamp pairs; one lamp is placed
/// on each side of the road per spacing interval.
pub const LAMP_SPACING: f32 = 40.0;
/// Lateral offset of the lamp pole base from the road center (well past the
/// verge strips and marker posts, inside the tree line).
pub const LAMP_LATERAL: f32 = ROAD_HALF + 1.7;
/// Pole height to the arm (luminaire head sits just above it).
const LAMP_HEIGHT: f32 = 5.6;

/// Deterministic list of street-lamp positions in `[start_s, end_s)`: every
/// `LAMP_SPACING` metres, one lamp on each side of the road. Pure function of
/// world-`s`, so the chunk mesh (pole geometry) and the frame lights (pools +
/// glow sprites) always agree and chunk rebuilds are stable.
pub fn roadside_lamps(start_s: f32, end_s: f32) -> Vec<(f32, f32)> {
    let mut out = Vec::new();
    let mut s = (start_s / LAMP_SPACING).ceil() * LAMP_SPACING;
    while s < end_s {
        out.push((s, 1.0));
        out.push((s, -1.0));
        s += LAMP_SPACING;
    }
    out
}

/// World-space luminaire head position for a lamp-pair entry `(s, side)`,
/// matching the geometry `push_lamp` builds so the light pools and glow
/// sprites sit exactly on the head.
pub fn lamp_head_pos(s: f32, side: f32) -> [f32; 3] {
    let x = road_curve(s) + side * LAMP_LATERAL;
    let head_lateral = ROAD_HALF + 0.35;
    let head_x = x - side * (LAMP_LATERAL - head_lateral);
    [head_x, LAMP_HEIGHT - 0.21, -s]
}

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

fn push_tri(
    v: &mut Vec<Vertex3d>,
    i: &mut Vec<u32>,
    a: [f32; 3],
    b: [f32; 3],
    c: [f32; 3],
    n: [f32; 3],
    col: [f32; 3],
    material: f32,
    scale: f32,
) {
    let base = v.len() as u32;
    let uv = |p: [f32; 3]| [p[0] * scale, p[2] * scale];
    for p in [a, b, c] {
        v.push(Vertex3d {
            position: p,
            normal: n,
            color: col,
            tex_coord: uv(p),
            material,
        });
    }
    i.extend_from_slice(&[base, base + 1, base + 2]);
}

/// Flat-shaded cone with its base ring centered on (cx, cz) at height `y0` and
/// apex at (cx, y1, cz). Triangles are wound so the outward normal is front
/// facing under the mesh pipeline's back-face culling.
#[allow(clippy::too_many_arguments)]
fn push_cone(
    v: &mut Vec<Vertex3d>,
    i: &mut Vec<u32>,
    cx: f32,
    cz: f32,
    base_r: f32,
    y0: f32,
    y1: f32,
    col: [f32; 3],
    material: f32,
    scale: f32,
    segments: usize,
) {
    if segments < 3 || base_r <= 0.0 || y1 <= y0 {
        return;
    }
    let apex = [cx, y1, cz];
    let ring: Vec<[f32; 3]> = (0..segments)
        .map(|k| {
            let a = k as f32 / segments as f32 * std::f32::consts::TAU;
            [cx + base_r * a.cos(), y0, cz + base_r * a.sin()]
        })
        .collect();
    for k in 0..segments {
        let p0 = ring[k];
        let p1 = ring[(k + 1) % segments];
        let e1 = [p1[0] - apex[0], p1[1] - apex[1], p1[2] - apex[2]];
        let e2 = [p0[0] - apex[0], p0[1] - apex[1], p0[2] - apex[2]];
        let mut n = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        // Keep the normal on the outside of the cone (toward the horizontal
        // axis direction of the face centroid).
        let mid = [
            (apex[0] + p0[0] + p1[0]) / 3.0 - cx,
            (apex[2] + p0[2] + p1[2]) / 3.0 - cz,
        ];
        if n[0] * mid[0] + n[2] * mid[1] < 0.0 {
            n = [-n[0], -n[1], -n[2]];
        }
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-6);
        push_tri(
            v,
            i,
            apex,
            p1,
            p0,
            [n[0] / len, n[1] / len, n[2] / len],
            col,
            material,
            scale,
        );
    }
}

/// Pushes one stylized roadside tree: a box trunk plus foliage cones
/// (`variant 0` = pine with stacked cones, `variant 1` = broadleaf with a wide
/// crown). All parts use `TREE_MATERIAL`; the vertex color distinguishes trunk
/// from foliage.
fn push_tree(
    v: &mut Vec<Vertex3d>,
    i: &mut Vec<u32>,
    x: f32,
    z: f32,
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
        [x - trunk_r, 0.0, z - trunk_r],
        [x + trunk_r, trunk_h, z + trunk_r],
        trunk_col,
        TREE_MATERIAL,
        scale,
    );
    if variant == 0 {
        let c_base = trunk_h;
        let c_span = height - trunk_h;
        let mut r = height * 0.30;
        for k in 0..3u32 {
            let cb = c_base + c_span * (k as f32 * 0.28);
            let apex = (c_base + c_span * (0.55 + k as f32 * 0.15)).min(height);
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
            trunk_h * 0.9,
            height * 0.74,
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
            height * 0.64,
            height,
            foliage_col,
            TREE_MATERIAL,
            scale,
            8,
        );
    }
}

/// Pushes one street lamp: a dark pole at the roadside edge with an arm
/// reaching toward the road and a small warm-off-white luminaire head. All
/// parts use the asphalt slot so they stay free of the terrain tint and grass
/// flattening; the head's light comes from the projector pool + glow sprite,
/// not the mesh color.
fn push_lamp(v: &mut Vec<Vertex3d>, i: &mut Vec<u32>, x: f32, z: f32, side: f32) {
    let pole_col = [0.20, 0.20, 0.22];
    let head_col = [0.95, 0.88, 0.72];
    let pole_r = 0.09;
    // Pole at the roadside edge.
    push_box(
        v,
        i,
        [x - pole_r, 0.0, z - pole_r],
        [x + pole_r, LAMP_HEIGHT, z + pole_r],
        pole_col,
        ASPHALT_BASE.atlas_slot(),
        ASPHALT_BASE.uv_scale(),
    );
    // Arm reaching inward from the pole top to the head over the road edge.
    let head_lateral = ROAD_HALF + 0.35;
    let head_x = x - side * (LAMP_LATERAL - head_lateral);
    let arm_h = LAMP_HEIGHT - 0.14;
    push_box(
        v,
        i,
        [x.min(head_x) - 0.06, arm_h, z - 0.06],
        [x.max(head_x) + 0.06, arm_h + 0.14, z + 0.06],
        pole_col,
        ASPHALT_BASE.atlas_slot(),
        ASPHALT_BASE.uv_scale(),
    );
    // Luminaire head hanging from the arm end.
    push_box(
        v,
        i,
        [head_x - 0.16, arm_h - 0.16, z - 0.10],
        [head_x + 0.16, arm_h + 0.02, z + 0.10],
        head_col,
        ASPHALT_BASE.atlas_slot(),
        ASPHALT_BASE.uv_scale(),
    );
}

/// Deterministic 0..1 hash from a world-space coordinate, so chunk rebuilds and
/// snapshot runs place scenery identically.
fn hash01(s: f32) -> f32 {
    let x = (s * 1000.0).abs() as u64;
    let mut h = x.wrapping_mul(1_103_515_245).wrapping_add(12_345);
    h ^= h >> 16;
    h = h.wrapping_mul(0x85EB_CA6B);
    h ^= h >> 13;
    ((h & 0x00FF_FFFF) as f32) / 16_777_215.0
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

    // roadside trees: deterministic per world-s (hash01) so chunk rebuilds and
    // snapshot runs place trees identically. Every ~11m, alternating sides,
    // lateral offset well past the verge strips (±7.3..12.8m), heights 2.6-4.6m.
    let mut tree_s = (start_s / 11.0).ceil() * 11.0;
    while tree_s < end_s {
        let side = if ((tree_s / 11.0).floor() as i32) % 2 == 0 {
            1.0
        } else {
            -1.0
        };
        let lateral = half_w + 2.5 + hash01(tree_s * 1.3 + 5.1) * 5.5;
        let height = 2.6 + hash01(tree_s * 0.37 + 1.7) * 2.0;
        let variant = if hash01(tree_s + 7.3) < 0.5 { 0 } else { 1 };
        let x = road_curve(tree_s);
        let z = -tree_s;
        push_tree(
            &mut v,
            &mut i,
            x + side * lateral,
            z,
            height,
            variant,
            TREE_UV_SCALE,
        );
        tree_s += 11.0;
    }

    // street lamps: a pole on each side every LAMP_SPACING, placed via the same
    // deterministic list the frame lights use, so poles and their light pools
    // always line up.
    for (lamp_s, side) in roadside_lamps(start_s, end_s) {
        let x = road_curve(lamp_s);
        let z = -lamp_s;
        push_lamp(&mut v, &mut i, x + side * LAMP_LATERAL, z, side);
    }

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
    fn marker_posts_remain_on_both_sides() {
        let (v, _) = build_world_chunk(0.0, 260.0);
        let posts = v.iter().filter(|vert| vert.position[1] > 1.0).count();
        assert!(posts > 0, "marker posts should still exist");
    }

    #[test]
    fn world_chunk_builds_lamps_both_sides_deterministically() {
        let (v, _) = build_world_chunk(0.0, 260.0);
        // Head boxes hang just under the arm (y ≈ 5.30..5.48) and overhang the
        // road edge (lateral ≈ ±(half_w + 0.35)), while poles stand at
        // ±LAMP_LATERAL. Check the luminaires exist and sit inward of the poles.
        let head_lateral = ROAD_HALF + 0.35;
        let heads: Vec<[f32; 3]> = v
            .iter()
            .filter(|vert| vert.position[1] > 5.28 && vert.position[1] < 5.50)
            .map(|vert| vert.position)
            .collect();
        let heads_on_road: Vec<[f32; 3]> = heads
            .iter()
            .copied()
            .filter(|pos| {
                let lateral = (pos[0] - road_curve(-pos[2])).abs();
                lateral < head_lateral + 0.3 && lateral > head_lateral - 0.3
            })
            .collect();
        assert!(
            !heads_on_road.is_empty(),
            "lamp luminaires must hang over the road edge"
        );
        for pos in &heads_on_road {
            let lateral = pos[0] - road_curve(-pos[2]);
            assert!(
                lateral.abs() < LAMP_LATERAL,
                "lamp heads must sit inward of the pole, got lateral={lateral}"
            );
        }
        // Both sides per LAMP_SPACING: one lamp on each side every 40m.
        let lamps = roadside_lamps(0.0, 260.0);
        let expected = (260.0 / LAMP_SPACING).ceil() as usize * 2;
        assert_eq!(lamps.len(), expected, "one lamp on each side per spacing");
        assert!(
            lamps.iter().any(|(_, side)| *side > 0.0) && lamps.iter().any(|(_, side)| *side < 0.0),
            "lamps appear on both sides"
        );
        assert_eq!(
            lamps,
            roadside_lamps(0.0, 260.0),
            "placement is deterministic"
        );
        // Boundary consistency: adjacent windows share no lamp and lose none.
        let a = roadside_lamps(0.0, 260.0);
        let b = roadside_lamps(260.0, 520.0);
        assert_eq!(a.len() + b.len(), roadside_lamps(0.0, 520.0).len());
        // The head-position helper matches the mesh: its lateral sits at the
        // head over the road edge, not at the pole.
        for (s, side) in &lamps {
            let head = lamp_head_pos(*s, *side);
            let lateral = head[0] - road_curve(-head[2]);
            assert!(
                (lateral.abs() - head_lateral).abs() < 1e-3,
                "lamp_head_pos must sit at the luminaire over the road edge"
            );
            assert!(head[1] > 5.2, "lamp head is elevated");
        }
    }

    #[test]
    fn world_chunk_builds_trees_off_the_road_deterministically() {
        let (v, i) = build_world_chunk(0.0, 260.0);
        let foliage: Vec<&crate::vertex::Vertex3d> = v
            .iter()
            .filter(|vert| vert.material >= TREE_MATERIAL)
            .collect();
        assert!(!foliage.is_empty(), "trees should be generated");
        for vert in &foliage {
            let lateral = vert.position[0] - road_curve(-vert.position[2]);
            assert!(
                lateral.abs() > ROAD_HALF + 1.5,
                "tree geometry must stay off the road, got lateral={lateral}"
            );
            assert!(
                vert.position[1] >= 0.0 && vert.position[1] <= 5.0,
                "tree heights out of range: {}",
                vert.position[1]
            );
        }
        let tri_count = i.len() / 3;
        assert!(tri_count > 0, "trees must contribute triangles");

        // Deterministic: a second identical build produces identical tree
        // placement (Vertex3d isn't PartialEq, so compare the tree signatures).
        let (v2, _) = build_world_chunk(0.0, 260.0);
        let sig = |verts: &[crate::vertex::Vertex3d]| -> Vec<[f32; 3]> {
            let mut out: Vec<[f32; 3]> = verts
                .iter()
                .filter(|vert| vert.material >= TREE_MATERIAL)
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
