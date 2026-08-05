// SPDX-License-Identifier: MIT

//! Screen-space sun lens flare: procedural sprites plus the quad geometry that
//! projects the sun into NDC and fans ghosts + an anamorphic streak along the
//! line from the sun to the screen center.

use crate::vertex::FlareVertex;

/// Quad kind selector for the flare fragment shader.
pub const FLARE_CORE: f32 = 0.0;
pub const FLARE_GLOW: f32 = 1.0;
pub const FLARE_STREAK: f32 = 2.0;

/// Tight hot disc with a soft falloff, used for the sun's own bright core.
pub fn generate_sun_core(size: u32) -> Vec<u8> {
    let size = size as f32;
    let r0 = size * 0.20;
    let mut pixels = Vec::with_capacity((size * size * 4.0) as usize);
    for y in 0..size as i32 {
        for x in 0..size as i32 {
            let dx = (x as f32 + 0.5) - size * 0.5;
            let dy = (y as f32 + 0.5) - size * 0.5;
            let d = (dx * dx + dy * dy).sqrt();
            let a = (-(d / r0) * (d / r0)).exp();
            let a = a * a;
            pixels.push(255);
            pixels.push(248);
            pixels.push(235);
            pixels.push((a * 255.0) as u8);
        }
    }
    pixels
}

/// Horizontally elongated streak used for the anamorphic ghost line.
pub fn generate_flare_streak(w: u32, h: u32) -> Vec<u8> {
    let (w, h) = (w as f32, h as f32);
    let sx = w * 0.45;
    let sy = h * 0.30;
    let mut pixels = Vec::with_capacity((w * h * 4.0) as usize);
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let dx = (x as f32 + 0.5) - w * 0.5;
            let dy = (y as f32 + 0.5) - h * 0.5;
            let a = (-(dx / sx) * (dx / sx)).exp() * (-(dy / sy) * (dy / sy)).exp();
            pixels.push(255);
            pixels.push(243);
            pixels.push(220);
            pixels.push((a * 255.0) as u8);
        }
    }
    pixels
}

/// Builds the flare quads for a frame.
///
/// `sun_ndc` is the projected NDC position of the sun, `aspect` the framebuffer
/// aspect (so quads stay circular), and `intensity` the combined fade factor
/// (sun brightness, cloud cover, off-screen falloff). Returns an empty list
/// when the flare should be invisible.
pub fn build_flare_verts(sun_ndc: [f32; 2], aspect: f32, intensity: f32) -> Vec<FlareVertex> {
    let mut out = Vec::new();
    if intensity <= 0.001 {
        return out;
    }

    // Axis from the sun toward the screen center, plus its perpendicular.
    let dx = -sun_ndc[0];
    let dy = -sun_ndc[1];
    let dist = (dx * dx + dy * dy).sqrt();
    let (ax, ay) = if dist > 1e-4 {
        (dx / dist, dy / dist)
    } else {
        (0.0, 1.0)
    };
    let (px, py) = (-ay, ax);
    let inv_aspect = 1.0 / aspect;

    // (distance along axis as a fraction of sun->center, half-height in NDC,
    //  horizontal half-extent scale, color, kind, alpha)
    let elements: [(f32, f32, f32, [f32; 3], f32, f32); 5] = [
        (0.00, 0.045, 0.60, [1.0, 0.96, 0.88], FLARE_CORE, 1.0),
        (0.00, 0.120, 0.60, [1.0, 0.80, 0.55], FLARE_GLOW, 0.45),
        (0.22, 0.030, 1.0, [1.0, 0.70, 0.40], FLARE_GLOW, 0.35),
        (0.42, 0.022, 1.0, [0.60, 0.80, 1.0], FLARE_GLOW, 0.28),
        (0.62, 0.016, 1.0, [1.0, 0.60, 0.35], FLARE_GLOW, 0.20),
    ];
    for (t, hh, hw_scale, rgb, kind, alpha) in elements {
        let cx = sun_ndc[0] + ax * dist * t;
        let cy = sun_ndc[1] + ay * dist * t;
        let hw = hh * inv_aspect * hw_scale;
        push_quad(&mut out, cx, cy, ax, ay, px, py, hw, hh, rgb, kind, alpha * intensity);
    }

    // Anamorphic streak bridging the sun and the screen center.
    let sx = sun_ndc[0] + ax * dist * 0.5;
    let sy = sun_ndc[1] + ay * dist * 0.5;
    let streak_len = (dist * 0.55).max(0.06) * inv_aspect;
    push_quad(
        &mut out,
        sx,
        sy,
        ax,
        ay,
        px,
        py,
        streak_len,
        0.012,
        [1.0, 0.95, 0.80],
        FLARE_STREAK,
        0.22 * intensity,
    );
    out
}

/// Appends a quad (two triangles, 6 vertices) centered at `(cx, cy)` with
/// half-extents `hw` along the axis and `hh` along the perpendicular.
fn push_quad(
    out: &mut Vec<FlareVertex>,
    cx: f32,
    cy: f32,
    ax: f32,
    ay: f32,
    px: f32,
    py: f32,
    hw: f32,
    hh: f32,
    rgb: [f32; 3],
    kind: f32,
    alpha: f32,
) {
    let c = [cx, cy, 1.0];
    let a = [ax * hw, ay * hw];
    let p = [px * hh, py * hh];
    let corners = [
        [c[0] + a[0] + p[0], c[1] + a[1] + p[1]],
        [c[0] + a[0] - p[0], c[1] + a[1] - p[1]],
        [c[0] - a[0] - p[0], c[1] - a[1] - p[1]],
        [c[0] - a[0] + p[0], c[1] - a[1] + p[1]],
    ];
    let uvs = [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]];
    let color = [rgb[0], rgb[1], rgb[2], alpha];
    for idx in [0usize, 1, 2, 0, 2, 3] {
        out.push(FlareVertex {
            position: corners[idx],
            color,
            uv: uvs[idx],
            kind,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_intensity_emits_nothing() {
        assert!(build_flare_verts([0.3, 0.2], 16.0 / 9.0, 0.0).is_empty());
    }

    #[test]
    fn flare_emits_two_triangles_per_quad() {
        let verts = build_flare_verts([0.3, 0.2], 16.0 / 9.0, 1.0);
        assert_eq!(verts.len() % 6, 0);
        assert!(verts.len() >= 6 * 6, "core+glows+streak quads");
        assert!(verts.iter().all(|v| v.color[3] > 0.0));
        // The core quad sits at the sun's position.
        assert!((verts[0].position[0] - 0.3).abs() < 0.1);
    }

    #[test]
    fn sun_at_screen_center_cannot_divide_by_zero() {
        let verts = build_flare_verts([0.0, 0.0], 1.0, 1.0);
        assert!(verts.iter().all(|v| v.position.iter().all(|p| p.is_finite())));
    }

    #[test]
    fn sprite_textures_are_opaque_in_the_middle() {
        let core = generate_sun_core(64);
        let mid = (64 * 32 + 32) * 4;
        assert!(core[mid + 3] > 250, "core is near-opaque at its center");
        let streak = generate_flare_streak(256, 32);
        let smid = (16 * 256 + 128) * 4;
        assert!(streak[smid + 3] > 250, "streak is near-opaque at its center");
    }
}
