// SPDX-License-Identifier: MIT

//! Procedural cloud layer tiles.
//!
//! `generate_cloud_tile` bakes a seamless RGBA cloud tile (alpha = cloud
//! coverage). The noise lattice is periodic, so the tile wraps in both axes,
//! letting the sky shader scroll it endlessly without visible seams.

/// Bakes a `size`×`size` RGBA tile (white clouds, alpha = coverage).
///
/// Low-frequency fBm drives the main cloud masses, a second band adds gentle
/// breakup, and modest domain warping prevents obvious bands/repetition. The
/// per-run `seed` gives every launch a different cloud layout.
pub fn generate_cloud_tile(size: u32, seed: u64) -> Vec<u8> {
    let n = size as usize;
    let mut out = Vec::with_capacity(n * n * 4);

    let warp = size as f32 * 0.05;
    let warp_seed = seed
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add(0xBF58476D1CE4E5B9);

    for py in 0..n {
        let y = py as f32;
        for px in 0..n {
            let x = px as f32;
            // Two decorrelated coarse fields warp X and Y independently.
            let wx = fbm(warp_seed, size, x, y, 2, 3);
            let wy = fbm(warp_seed ^ 0xDEADBEEF, size, x + 37.3, y + 91.7, 2, 3);
            let sx = x + (wx - 0.5) * warp;
            let sy = y + (wy - 0.5) * warp;

            // Distinct puffs from a mid-frequency base + a little breakup.
            // A generous threshold keeps the tile mostly covered with real
            // cloud mass (instead of a near-empty field of specks), and enough
            // base cells break it into scattered clusters around the sky so it
            // never collapses into one continuous wrapping bank.
            let base = fbm(seed, size, sx, sy, 8, 4);
            let detail = fbm(
                seed ^ 0xA5A5_A5A5_F0F0_F0F0,
                size,
                sx * 1.73 + 19.0,
                sy * 1.53 + 7.0,
                12,
                3,
            );
            let shape = (base * 0.8 + detail * 0.2).clamp(0.0, 1.0);
            let cov = smoothstep((shape - 0.45) / (0.75 - 0.45));
            let lum = (220.0 + cov * 35.0) as u8;

            out.push(lum);
            out.push(lum);
            out.push(lum);
            out.push((cov * 255.0) as u8);
        }
    }

    out
}

/// Bakes a `size`×`size` opaque RGBA foliage tile for the world atlas (slot 4).
///
/// A soft low-contrast green noise gives tree surfaces organic variation while
/// keeping per-tree `v_color` in control of the hue (the mesh shader mixes the
/// tile with mid-grey for material 4). The lattice is periodic, so the tile
/// tiles seamlessly over tree faces.
pub fn generate_foliage_tile(size: u32, seed: u64) -> Vec<u8> {
    let n = size as usize;
    let mut out = Vec::with_capacity(n * n * 4);
    for py in 0..n {
        let y = py as f32;
        for px in 0..n {
            let x = px as f32;
            let n1 = fbm(seed, size, x + 11.0, y + 5.0, 6, 4);
            let n2 = fbm(
                seed ^ 0x5DEECE66D,
                size,
                x * 1.7 + 3.0,
                y * 1.7 + 9.0,
                10,
                3,
            );
            let n3 = fbm(
                seed ^ 0xDEADBEEF,
                size,
                x * 3.1 + 31.0,
                y * 3.1 + 17.0,
                18,
                2,
            );
            // Base leaf green with subtle hue/luma jitter (light patchy canopy).
            let r = (0.30 + 0.18 * (n1 - 0.5) + 0.06 * (n3 - 0.5)) * 255.0;
            let g = (0.55 + 0.22 * (n1 - 0.5) + 0.10 * (n2 - 0.5) + 0.05 * (n3 - 0.5)) * 255.0;
            let b = (0.22 + 0.14 * (n1 - 0.5) + 0.05 * (n3 - 0.5)) * 255.0;
            out.push(r.clamp(0.0, 255.0) as u8);
            out.push(g.clamp(0.0, 255.0) as u8);
            out.push(b.clamp(0.0, 255.0) as u8);
            out.push(255);
        }
    }
    out
}

/// Bakes a `size`×`size` opaque RGBA rock tile for the world atlas (slot 5).
///
/// Neutral grey rock with coarse horizontal strata bands plus a little mid and
/// fine speckle, so cliff faces get organic banding without strong color. The
/// lattice is periodic, so the tile tiles seamlessly over large cliff faces.
pub fn generate_rock_tile(size: u32, seed: u64) -> Vec<u8> {
    let n = size as usize;
    let mut out = Vec::with_capacity(n * n * 4);
    for py in 0..n {
        let y = py as f32;
        for px in 0..n {
            let x = px as f32;
            let strata = fbm(seed, size, x * 0.4 + 3.0, y * 1.3 + 11.0, 4, 3);
            let n1 = fbm(seed ^ 0x9E3779B9, size, x + 17.0, y + 3.0, 8, 3);
            let n2 = fbm(
                seed ^ 0x5DEECE66D,
                size,
                x * 2.3 + 5.0,
                y * 2.3 + 7.0,
                14,
                2,
            );
            // Neutral grey with slight warm/cool variation from the bands.
            let g = 0.44 + 0.10 * (strata - 0.5) + 0.05 * (n1 - 0.5) + 0.04 * (n2 - 0.5);
            let r = g + 0.02 * (n1 - 0.5);
            let b = g - 0.02 * (n2 - 0.5);
            out.push((r * 255.0).clamp(0.0, 255.0) as u8);
            out.push((g * 255.0).clamp(0.0, 255.0) as u8);
            out.push((b * 255.0).clamp(0.0, 255.0) as u8);
            out.push(255);
        }
    }
    out
}

/// Fractal value noise, periodic across the tile.
fn fbm(seed: u64, size: u32, x: f32, y: f32, base_cells: i32, octaves: usize) -> f32 {
    let mut sum = 0.0;
    let mut amp = 1.0;
    let mut norm = 0.0;
    let mut s = seed;
    for k in 0..octaves {
        let cells = base_cells << k;
        if cells > size as i32 {
            break;
        }
        let fx = (x / size as f32) * cells as f32;
        let fy = (y / size as f32) * cells as f32;
        sum += amp * value_noise(s, cells, fx, fy);
        norm += amp;
        amp *= 0.58;
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
    }
    sum / norm
}

/// Smooth, periodic value noise over a `cells`×`cells` lattice.
fn value_noise(seed: u64, cells: i32, fx: f32, fy: f32) -> f32 {
    let xi = fx.floor() as i32;
    let yi = fy.floor() as i32;
    let tx = smoothstep5(fx - xi as f32);
    let ty = smoothstep5(fy - yi as f32);

    let a = cell(seed, cells, xi, yi);
    let b = cell(seed, cells, xi + 1, yi);
    let c = cell(seed, cells, xi, yi + 1);
    let d = cell(seed, cells, xi + 1, yi + 1);

    let ab = a + (b - a) * tx;
    let cd = c + (d - c) * tx;
    ab + (cd - ab) * ty
}

/// Random lattice value, wrapping coordinates so the tile is seamless.
fn cell(seed: u64, cells: i32, x: i32, y: i32) -> f32 {
    let xi = x.rem_euclid(cells);
    let yi = y.rem_euclid(cells);

    let mut h = seed
        .wrapping_add((xi as u64).wrapping_mul(0x9E3779B185EBCA87))
        .wrapping_add((yi as u64).wrapping_mul(0xC2B2AE3D27D4EB4F));
    h ^= h >> 30;
    h = h.wrapping_mul(0xBF58476D1CE4E5B9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94D049BB133111EB);
    h ^= h >> 31;

    (h & 0x00FF_FFFF) as f32 / 16_777_215.0
}

fn smoothstep5(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

fn smoothstep(t: f32) -> f32 {
    let x = t.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn foliage_tile_is_opaque_green_noise() {
        let size = 64u32;
        let tile = generate_foliage_tile(size, 7);
        assert_eq!(tile.len(), (size * size * 4) as usize);
        let mut green_biased = true;
        for px in tile.chunks_exact(4).step_by(64) {
            assert_eq!(px[3], 255, "foliage is opaque");
            green_biased &= px[1] >= px[0] && px[1] >= px[2];
        }
        assert!(green_biased, "foliage tile reads green");
    }

    #[test]
    fn rock_tile_is_opaque_neutral_grey_noise() {
        let size = 64u32;
        let tile = generate_rock_tile(size, 7);
        assert_eq!(tile.len(), (size * size * 4) as usize);
        for px in tile.chunks_exact(4).step_by(64) {
            assert_eq!(px[3], 255, "rock is opaque");
            // Neutral grey: channels close together, mid luminance.
            let (r, g, b) = (px[0] as i16, px[1] as i16, px[2] as i16);
            assert!(
                (r - g).abs() <= 12 && (b - g).abs() <= 12,
                "rock must read neutral grey, got ({r},{g},{b})"
            );
            assert!((80..=170).contains(&g), "rock luminance mid-range");
        }
    }

    #[test]
    fn rock_tile_is_deterministic_per_seed() {
        let a = generate_rock_tile(32, 9);
        let b = generate_rock_tile(32, 9);
        let c = generate_rock_tile(32, 10);
        assert_eq!(a, b, "same seed -> same tile");
        assert_ne!(a, c, "different seed -> different tile");
    }
}
