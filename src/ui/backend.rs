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

