// SPDX-License-Identifier: MIT

use crate::font::FontAtlas;
use crate::game::Game;
use crate::vertex::HudVertex;

const SOLID_UV: [f32; 2] = [-1.0, -1.0];

const PANEL_BG: [f32; 4] = [0.03, 0.04, 0.06, 0.6];
const PANEL_PAD: f32 = 0.012;

fn push_rect(out: &mut Vec<HudVertex>, x: f32, y: f32, w: f32, h: f32, uv: [f32; 2], color: [f32; 4]) {
    let c = color;
    out.push(HudVertex { position: [x, y], color: c, uv });
    out.push(HudVertex { position: [x + w, y], color: c, uv });
    out.push(HudVertex { position: [x + w, y - h], color: c, uv });
    out.push(HudVertex { position: [x, y], color: c, uv });
    out.push(HudVertex { position: [x + w, y - h], color: c, uv });
    out.push(HudVertex { position: [x, y - h], color: c, uv });
}

fn push_solid_rect(out: &mut Vec<HudVertex>, x: f32, y: f32, w: f32, h: f32, color: [f32; 4]) {
    push_rect(out, x, y, w, h, SOLID_UV, color);
}

fn push_panel(out: &mut Vec<HudVertex>, x: f32, y: f32, w: f32, h: f32) {
    push_rect(out, x, y, w, h, SOLID_UV, PANEL_BG);
}

fn push_glyph_quad(
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

/// Tight NDC bounding box of a string's glyph quads: (min_x, top, max_x, bottom).
/// Uses the exact same layout math as `draw_text`, so it covers ascenders and
/// descenders (e.g. `/` and the tail of `Q`) that fall outside the nominal em box.
fn text_bounds(
    atlas: &FontAtlas,
    text: &str,
    x: f32,
    y: f32,
    em_ndc: f32,
    aspect: f32,
) -> Option<(f32, f32, f32, f32)> {
    let scale_y = em_ndc / atlas.raster_px;
    let scale_x = scale_y / aspect;
    let baseline = y - atlas.ascender * scale_y;
    let mut min_x = f32::INFINITY;
    let mut top = f32::NEG_INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut bottom = f32::INFINITY;
    let mut cx = x;
    for c in text.chars() {
        if let Some(g) = atlas.glyph(c) {
            if g.width > 0.0 && g.height > 0.0 {
                let gx = cx + g.bearing_x * scale_x;
                let gy = baseline - g.bearing_y * scale_y;
                let gw = g.width * scale_x;
                let gh = g.height * scale_y;
                min_x = min_x.min(gx);
                top = top.max(gy);
                max_x = max_x.max(gx + gw);
                bottom = bottom.min(gy - gh);
            }
            cx += g.advance * scale_x;
        }
    }
    if top.is_finite() {
        Some((min_x, top, max_x, bottom))
    } else {
        None
    }
}

fn push_line_panel(
    out: &mut Vec<HudVertex>,
    atlas: &FontAtlas,
    text: &str,
    x: f32,
    y: f32,
    em_ndc: f32,
    aspect: f32,
) {
    if let Some((min_x, top, max_x, bottom)) = text_bounds(atlas, text, x, y, em_ndc, aspect) {
        push_panel(
            out,
            min_x - PANEL_PAD,
            top + PANEL_PAD,
            (max_x - min_x) + 2.0 * PANEL_PAD,
            (top - bottom) + 2.0 * PANEL_PAD,
        );
    }
}

fn push_block_panel(
    out: &mut Vec<HudVertex>,
    lines: &[(&str, f32, f32, f32)],
    atlas: &FontAtlas,
    aspect: f32,
) {
    let mut min_x = f32::INFINITY;
    let mut top = f32::NEG_INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut bottom = f32::INFINITY;
    for &(text, x, y, em) in lines {
        if let Some((b_min_x, b_top, b_max_x, b_bottom)) =
            text_bounds(atlas, text, x, y, em, aspect)
        {
            min_x = min_x.min(b_min_x);
            top = top.max(b_top);
            max_x = max_x.max(b_max_x);
            bottom = bottom.min(b_bottom);
        }
    }
    if top.is_finite() {
        push_panel(
            out,
            min_x - PANEL_PAD,
            top + PANEL_PAD,
            (max_x - min_x) + 2.0 * PANEL_PAD,
            (top - bottom) + 2.0 * PANEL_PAD,
        );
    }
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
    color: [f32; 4],
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
    let white = [1.0, 1.0, 1.0, 1.0];
    let green = [0.2, 1.0, 0.3, 1.0];
    let yellow = [1.0, 0.9, 0.3, 1.0];
    let red = [1.0, 0.2, 0.2, 1.0];
    let dim = [0.72, 0.78, 0.82, 1.0];
    let mode_col = [0.78, 0.9, 1.0, 1.0];
    let wreck_orange = [1.0, 0.4, 0.1, 1.0];
    let right_x = 0.95;

    let px_to_ndc_x = |em: f32| em / atlas.raster_px / aspect;

    // Top-right: score / best / average speed
    let score_t = format!("SCORE {}", game.score);
    let best_t = format!("BEST {}", game.best_score);
    let avg_t = format!("AVG {:.0} KM/H", game.avg_speed * 3.6);
    let score_x = right_x - text_width(atlas, &score_t, px_to_ndc_x(0.07));
    let best_x = right_x - text_width(atlas, &best_t, px_to_ndc_x(0.05));
    let avg_x = right_x - text_width(atlas, &avg_t, px_to_ndc_x(0.045));
    let lines = [
        (score_t.as_str(), score_x, 0.92, 0.07),
        (best_t.as_str(), best_x, 0.82, 0.05),
        (avg_t.as_str(), avg_x, 0.72, 0.045),
    ];
    push_block_panel(&mut out, &lines, atlas, aspect);
    draw_text(&mut out, atlas, &score_t, score_x, 0.92, 0.07, aspect, white);
    draw_text(&mut out, atlas, &best_t, best_x, 0.82, 0.05, aspect, green);
    draw_text(&mut out, atlas, &avg_t, avg_x, 0.72, 0.045, aspect, dim);

    // Top-left: gear / mode / wrecks remaining
    let gear_t = format!("GEAR {}", game.vehicle.gear);
    let mode_t = format!("MODE {}", game.difficulty.label());
    let tuning = game.difficulty.tuning();
    let wreck_col = if tuning.wreck_limit - game.wrecks <= 1 {
        red
    } else {
        white
    };
    let wrecks_t = format!("WRECKS {}/{}", game.wrecks, tuning.wreck_limit);
    let lines = [
        (gear_t.as_str(), -0.95, 0.92, 0.07),
        (mode_t.as_str(), -0.95, 0.82, 0.05),
        (wrecks_t.as_str(), -0.95, 0.72, 0.05),
    ];
    push_block_panel(&mut out, &lines, atlas, aspect);
    draw_text(&mut out, atlas, &gear_t, -0.95, 0.92, 0.07, aspect, yellow);
    draw_text(&mut out, atlas, &mode_t, -0.95, 0.82, 0.05, aspect, mode_col);
    draw_text(&mut out, atlas, &wrecks_t, -0.95, 0.72, 0.05, aspect, wreck_col);

    // Speed value (large, centered lower) + KM/H label
    let speed_em = 0.14;
    let speed_str = format!("{:.0}", game.speed_kmh);
    let sw = text_width(atlas, &speed_str, px_to_ndc_x(speed_em));
    let label_em = 0.06;
    let lw = text_width(atlas, "KM/H", px_to_ndc_x(label_em));
    let lines = [
        (speed_str.as_str(), -sw / 2.0, -0.6, speed_em),
        ("KM/H", -lw / 2.0, -0.78, label_em),
    ];
    push_block_panel(&mut out, &lines, atlas, aspect);
    draw_text(&mut out, atlas, &speed_str, -sw / 2.0, -0.6, speed_em, aspect, white);
    draw_text(&mut out, atlas, "KM/H", -lw / 2.0, -0.78, label_em, aspect, green);

    // Speed bar (scaled to true top speed ~342 km/h)
    let ratio = (game.speed_kmh / 342.0).clamp(0.0, 1.0);
    push_solid_rect(&mut out, -0.9, -0.85, 1.8 * ratio, 0.04, green);

    // Alerts
    let alert_em = 0.16;
    if game.game_over {
        let t = "GAME OVER";
        let score_t = format!("SCORE {}", game.score);
        let best_t = format!("BEST {}", game.best_score);
        let hint = "PRESS R TO RESTART";
        let t_w = text_width(atlas, t, px_to_ndc_x(alert_em));
        let s_em = 0.08;
        let s_w = text_width(atlas, &score_t, px_to_ndc_x(s_em));
        let b_em = 0.06;
        let b_w = text_width(atlas, &best_t, px_to_ndc_x(b_em));
        let hint_em = 0.05;
        let hint_w = text_width(atlas, hint, px_to_ndc_x(hint_em));
        let lines = [
            (t, -t_w / 2.0, 0.15, alert_em),
            (score_t.as_str(), -s_w / 2.0, -0.05, s_em),
            (best_t.as_str(), -b_w / 2.0, -0.18, b_em),
            (hint, -hint_w / 2.0, -0.32, hint_em),
        ];
        push_block_panel(&mut out, &lines, atlas, aspect);

        draw_text(&mut out, atlas, t, -t_w / 2.0, 0.15, alert_em, aspect, red);
        draw_text(&mut out, atlas, &score_t, -s_w / 2.0, -0.05, s_em, aspect, white);
        draw_text(&mut out, atlas, &best_t, -b_w / 2.0, -0.18, b_em, aspect, green);
        if ((game.ui_time * 2.0) as i32) % 2 == 0 {
            draw_text(&mut out, atlas, hint, -hint_w / 2.0, -0.32, hint_em, aspect, yellow);
        }
    } else if game.wreck_timer > 0.0 {
        let t = "WRECK";
        let w = text_width(atlas, t, px_to_ndc_x(alert_em));
        push_line_panel(&mut out, atlas, t, -w / 2.0, 0.15, alert_em, aspect);
        draw_text(&mut out, atlas, t, -w / 2.0, 0.15, alert_em, aspect, wreck_orange);
    }

    out
}
