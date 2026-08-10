// SPDX-License-Identifier: MIT

//! Shared mesh geometry primitives used by both the road base (`mesh.rs`) and
//! the roadside world objects (`world/`). Each primitive takes its material and
//! world-space UV scale explicitly, so the callers decide which atlas slot and
//! tiling a surface uses.

use crate::vertex::Vertex3d;

#[allow(clippy::too_many_arguments)]
pub(crate) fn push_quad(
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

pub(crate) fn push_box(
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

pub(crate) fn push_tri(
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
pub(crate) fn push_cone(
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
