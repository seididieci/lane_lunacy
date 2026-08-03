// SPDX-License-Identifier: MIT

use crate::font::FontAtlas;
use crate::game::Game;
use crate::vertex::HudVertex;

const SOLID_UV: [f32; 2] = [-1.0, -1.0];

fn push_solid_rect(out: &mut Vec<HudVertex>, x: f32, y: f32, w: f32, h: f32, color: [f32; 3]) {
    let c = color;
    out.push(HudVertex { position: [x, y], color: c, uv: SOLID_UV });
    out.push(HudVertex { position: [x + w, y], color: c, uv: SOLID_UV });
    out.push(HudVertex { position: [x + w, y - h], color: c, uv: SOLID_UV });
    out.push(HudVertex { position: [x, y], color: c, uv: SOLID_UV });
    out.push(HudVertex { position: [x + w, y - h], color: c, uv: SOLID_UV });
    out.push(HudVertex { position: [x, y - h], color: c, uv: SOLID_UV });
}

fn push_glyph_quad(
    out: &mut Vec<HudVertex>,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    uv: (f32, f32, f32, f32),
    color: [f32; 3],
) {
    let c = color;
    let (u0, v0, u1, v1) = uv;
    // swap v0/v1 to correct vertical mirror (texture origin vs screen origin)
    out.push(HudVertex { position: [x, y], color: c, uv: [u0, v1] });
    out.push(HudVertex { position: [x + w, y], color: c, uv: [u1, v1] });
    out.push(HudVertex { position: [x + w, y - h], color: c, uv: [u1, v0] });
    out.push(HudVertex { position: [x, y], color: c, uv: [u0, v1] });
    out.push(HudVertex { position: [x + w, y - h], color: c, uv: [u1, v0] });
    out.push(HudVertex { position: [x, y - h], color: c, uv: [u0, v0] });
}

fn text_width(atlas: &FontAtlas, text: &str, px_to_ndc_x: f32) -> f32 {
    let mut w = 0.0;
    for c in text.chars() {
        if let Some(g) = atlas.glyph(c) {
            w += g.advance * px_to_ndc_x;
        }
    }
    w
}

/// Draws text with the top-left of the em box at (x, y) in NDC.
fn draw_text(
    out: &mut Vec<HudVertex>,
    atlas: &FontAtlas,
    text: &str,
    x: f32,
    y: f32,
    em_ndc: f32,
    aspect: f32,
    color: [f32; 3],
) {
    let scale_y = em_ndc / atlas.raster_px;
    let scale_x = scale_y / aspect;
    // Baseline from text-top y using the actual ascender (most negative ymin).
    let baseline = y - atlas.ascender * scale_y;
    let mut cx = x;
    for c in text.chars() {
        if let Some(g) = atlas.glyph(c) {
            if g.width > 0.0 && g.height > 0.0 {
                let gx = cx + g.bearing_x * scale_x;
                let gy = baseline - g.bearing_y * scale_y;
                let gw = g.width * scale_x;
                let gh = g.height * scale_y;
                push_glyph_quad(out, gx, gy, gw, gh, (g.u0, g.v0, g.u1, g.v1), color);
            }
            cx += g.advance * scale_x;
        }
    }
}

pub fn build_hud_vertices(game: &Game, atlas: &FontAtlas, aspect: f32) -> Vec<HudVertex> {
    let mut out: Vec<HudVertex> = Vec::new();
    let white = [1.0, 1.0, 1.0];
    let green = [0.2, 1.0, 0.3];
    let yellow = [1.0, 0.9, 0.3];
    let red = [1.0, 0.2, 0.2];

    let px_to_ndc_x = |em: f32| em / atlas.raster_px / aspect;

    // Speed value (large, centered lower)
    let speed_em = 0.14;
    let speed_str = format!("{:.0}", game.speed_kmh);
    let sw = text_width(atlas, &speed_str, px_to_ndc_x(speed_em));
    draw_text(&mut out, atlas, &speed_str, -sw / 2.0, -0.6, speed_em, aspect, white);

    // KM/H label under it
    let label_em = 0.06;
    let lw = text_width(atlas, "KM/H", px_to_ndc_x(label_em));
    draw_text(&mut out, atlas, "KM/H", -lw / 2.0, -0.78, label_em, aspect, green);

    // Gear (top-left)
    draw_text(
        &mut out,
        atlas,
        &format!("GEAR {}", game.vehicle.gear),
        -0.95,
        0.92,
        0.07,
        aspect,
        yellow,
    );

    draw_text(
        &mut out,
        atlas,
        &format!("MODE {}", game.difficulty.label()),
        -0.95,
        0.82,
        0.05,
        aspect,
        [0.78, 0.9, 1.0],
    );

    // Speed bar
    let ratio = (game.speed_kmh / 288.0).clamp(0.0, 1.0);
    push_solid_rect(&mut out, -0.9, -0.85, 1.8 * ratio, 0.04, green);

    // Alerts
    let alert_em = 0.16;
    if game.game_over {
        let t = "GAME OVER";
        let w = text_width(atlas, t, px_to_ndc_x(alert_em));
        draw_text(&mut out, atlas, t, -w / 2.0, 0.15, alert_em, aspect, red);
    } else if game.wreck_timer > 0.0 {
        let t = "WRECK";
        let w = text_width(atlas, t, px_to_ndc_x(alert_em));
        draw_text(&mut out, atlas, t, -w / 2.0, 0.15, alert_em, aspect, [1.0, 0.4, 0.1]);
    }

    out
}
