// SPDX-License-Identifier: MIT

//! Low-level NDC vertex primitives used by the UI draw pass.
//!
//! These are the "canvas" the widgets draw on. They operate directly in NDC
//! coordinates (y up); widgets work in layout units and convert via `DrawCtx`.

use crate::font::FontAtlas;
use crate::vertex::HudVertex;

pub(crate) const SOLID_UV: [f32; 2] = [-1.0, -1.0];

pub(crate) fn push_rect(
    out: &mut Vec<HudVertex>,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    uv: [f32; 2],
    color: [f32; 4],
) {
    let c = color;
    out.push(HudVertex {
        position: [x, y],
        color: c,
        uv,
    });
    out.push(HudVertex {
        position: [x + w, y],
        color: c,
        uv,
    });
    out.push(HudVertex {
        position: [x + w, y - h],
        color: c,
        uv,
    });
    out.push(HudVertex {
        position: [x, y],
        color: c,
        uv,
    });
    out.push(HudVertex {
        position: [x + w, y - h],
        color: c,
        uv,
    });
    out.push(HudVertex {
        position: [x, y - h],
        color: c,
        uv,
    });
}

pub(crate) fn push_glyph_quad(
    out: &mut Vec<HudVertex>,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    uv: (f32, f32, f32, f32),
    color: [f32; 4],
) {
    let c = color;
    let (u0, v0, u1, v1) = uv;
    // The vertex shader flips Y (see hud.vert.glsl), so the quad top maps to the
    // screen top and samples the top of the glyph cell (v0).
    out.push(HudVertex {
        position: [x, y],
        color: c,
        uv: [u0, v0],
    });
    out.push(HudVertex {
        position: [x + w, y],
        color: c,
        uv: [u1, v0],
    });
    out.push(HudVertex {
        position: [x + w, y - h],
        color: c,
        uv: [u1, v1],
    });
    out.push(HudVertex {
        position: [x, y],
        color: c,
        uv: [u0, v0],
    });
    out.push(HudVertex {
        position: [x + w, y - h],
        color: c,
        uv: [u1, v1],
    });
    out.push(HudVertex {
        position: [x, y - h],
        color: c,
        uv: [u0, v1],
    });
}

/// Draws text with the top-left of the em box at (x, y) in NDC.
#[allow(clippy::too_many_arguments)]
pub(crate) fn draw_text(
    out: &mut Vec<HudVertex>,
    atlas: &FontAtlas,
    text: &str,
    x: f32,
    y: f32,
    em_ndc: f32,
    aspect: f32,
    color: [f32; 4],
) {
    let scale_y = em_ndc / atlas.raster_px;
    let scale_x = scale_y / aspect;
    // Baseline from text-top y using the max top extent above baseline.
    let baseline = y - atlas.ascender * scale_y;
    let mut cx = x;
    for c in text.chars() {
        if let Some(g) = atlas.glyph(c) {
            if g.width > 0.0 && g.height > 0.0 {
                let gx = cx + g.bearing_x * scale_x;
                let gy = baseline + (g.bearing_y + g.height) * scale_y;
                let gw = g.width * scale_x;
                let gh = g.height * scale_y;
                push_glyph_quad(out, gx, gy, gw, gh, (g.u0, g.v0, g.u1, g.v1), color);
            }
            cx += g.advance * scale_x;
        }
    }
}

/// A point on a ring in NDC. Angles are degrees from +x, counter-clockwise
/// (NDC y-up). `aspect` stretches the x axis so rings render physically
/// circular on any window aspect.
pub(crate) fn ring_point(
    cx: f32,
    cy: f32,
    r: f32,
    angle_deg: f32,
    aspect: f32,
) -> [f32; 2] {
    let rad = angle_deg.to_radians();
    [cx + r * rad.cos() * aspect, cy + r * rad.sin()]
}

/// Tessellated ring band from `a0_deg` to `a1_deg` (counter-clockwise) between
/// radii `r0` and `r1`. Emitted as solid-colored triangles (SOLID_UV).
#[allow(clippy::too_many_arguments)]
pub(crate) fn push_ring_segment(
    out: &mut Vec<HudVertex>,
    cx: f32,
    cy: f32,
    r0: f32,
    r1: f32,
    a0_deg: f32,
    a1_deg: f32,
    aspect: f32,
    color: [f32; 4],
) {
    if (a1_deg - a0_deg).abs() < 1e-3 {
        return;
    }
    // One quad every ~5 degrees, bounded for tiny sweeps.
    let steps = (((a1_deg - a0_deg).abs() / 5.0).ceil() as usize).clamp(1, 96);
    let c = color;
    for i in 0..steps {
        let t0 = i as f32 / steps as f32;
        let t1 = (i + 1) as f32 / steps as f32;
        let a = a0_deg + (a1_deg - a0_deg) * t0;
        let b = a0_deg + (a1_deg - a0_deg) * t1;
        let p0 = ring_point(cx, cy, r1, a, aspect);
        let p1 = ring_point(cx, cy, r0, a, aspect);
        let p2 = ring_point(cx, cy, r0, b, aspect);
        let p3 = ring_point(cx, cy, r1, b, aspect);
        out.push(HudVertex { position: p0, color: c, uv: SOLID_UV });
        out.push(HudVertex { position: p1, color: c, uv: SOLID_UV });
        out.push(HudVertex { position: p2, color: c, uv: SOLID_UV });
        out.push(HudVertex { position: p0, color: c, uv: SOLID_UV });
        out.push(HudVertex { position: p2, color: c, uv: SOLID_UV });
        out.push(HudVertex { position: p3, color: c, uv: SOLID_UV });
    }
}

/// A thin needle from radius `r0` to `r1` at `angle_deg`, `thick` wide,
/// centered on `(cx, cy)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn push_needle(
    out: &mut Vec<HudVertex>,
    cx: f32,
    cy: f32,
    r0: f32,
    r1: f32,
    angle_deg: f32,
    thick: f32,
    aspect: f32,
    color: [f32; 4],
) {
    let rad = angle_deg.to_radians();
    // Direction with aspect stretch; perpendicular is the unstretched flip.
    let (sin, cos) = rad.sin_cos();
    let dx = cos * aspect;
    let dy = sin;
    let len = (dx * dx + dy * dy).sqrt();
    let (ux, uy) = (-dy / len, dx / len);
    let h = thick / 2.0;
    let a = [cx + r0 * dx, cy + r0 * dy];
    let b = [cx + r1 * dx, cy + r1 * dy];
    let c = color;
    out.push(HudVertex { position: [a[0] + ux * h, a[1] + uy * h], color: c, uv: SOLID_UV });
    out.push(HudVertex { position: [b[0] + ux * h, b[1] + uy * h], color: c, uv: SOLID_UV });
    out.push(HudVertex { position: [b[0] - ux * h, b[1] - uy * h], color: c, uv: SOLID_UV });
    out.push(HudVertex { position: [a[0] + ux * h, a[1] + uy * h], color: c, uv: SOLID_UV });
    out.push(HudVertex { position: [b[0] - ux * h, b[1] - uy * h], color: c, uv: SOLID_UV });
    out.push(HudVertex { position: [a[0] - ux * h, a[1] - uy * h], color: c, uv: SOLID_UV });
}

