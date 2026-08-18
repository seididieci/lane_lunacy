// SPDX-License-Identifier: MIT
// Temporary diagnostic for LANE_DEBUG_POST captures.
//   mask:   counts "puddle" pixels (luma above a threshold) in the road band.
//   planar: counts valid (greenish) vs invalid (reddish) planar samples.
// Usage: cargo run --release --example pngdebug -- <mask|planar> <png>...

use image::GenericImageView;

fn srgb_to_linear(c: u8) -> f32 {
    let v = c as f32 / 255.0;
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mode = args.next().expect("mode: mask|planar");
    for path in args {
        let img = image::open(&path)
            .unwrap_or_else(|e| panic!("{}: {e}", path))
            .to_rgba8();
        let (w, h) = img.dimensions();
        let mut total = 0u64;
        let mut active = 0u64;
        let mut min_x = u32::MAX;
        let mut max_x = 0u32;
        let mut min_y = u32::MAX;
        let mut max_y = 0u32;
        let mut sum = 0f64;
        match mode.as_str() {
            "mask" => {
                for y in 0..h {
                    for x in 0..w {
                        let px = *img.get_pixel(x, y);
                        let l = 0.299 * srgb_to_linear(px[0]) + 0.587 * srgb_to_linear(px[1])
                            + 0.114 * srgb_to_linear(px[2]);
                        total += 1;
                        sum += l as f64;
                        if l > 0.2 {
                            active += 1;
                            min_x = min_x.min(x);
                            max_x = max_x.max(x);
                            min_y = min_y.min(y);
                            max_y = max_y.max(y);
                        }
                    }
                }
                println!(
                    "{}\n  mask active(luma>0.2): {:.4} mean_luma {:.4} bbox x[{}..{}] y[{}..{}]",
                    std::path::Path::new(&path).file_name().unwrap().to_string_lossy(),
                    active as f32 / total as f32,
                    sum / total as f64,
                    if active > 0 { min_x } else { 0 },
                    if active > 0 { max_x } else { 0 },
                    if active > 0 { min_y } else { 0 },
                    if active > 0 { max_y } else { 0 }
                );
            }
            "planar" => {
                // Only look at the bottom 45% (road region).
                let band_top = (h as f32 * 0.55) as u32;
                let mut valid = 0u64;
                let mut invalid = 0u64;
                let mut vsum_r = 0f64;
                let mut vsum_g = 0f64;
                let mut vsum_b = 0f64;
                for y in band_top..h {
                    for x in 0..w {
                        let px = *img.get_pixel(x, y);
                        let r = srgb_to_linear(px[0]);
                        let g = srgb_to_linear(px[1]);
                        let b = srgb_to_linear(px[2]);
                        if r + g + b < 0.02 {
                            continue;
                        }
                        if g > r {
                            valid += 1;
                            vsum_r += r as f64;
                            vsum_g += g as f64;
                            vsum_b += b as f64;
                        } else {
                            invalid += 1;
                        }
                    }
                }
                let v = valid as f32 / (valid + invalid) as f32;
                let (mr, mg, mb) = if valid > 0 {
                    (
                        vsum_r / valid as f64,
                        vsum_g / valid as f64,
                        vsum_b / valid as f64,
                    )
                } else {
                    (0.0, 0.0, 0.0)
                };
                println!(
                    "{}\n  planar samples: valid(g) {:.4} invalid(r) {:.4} total_sampled {} | valid mean rgb ({:.3}, {:.3}, {:.3})",
                    std::path::Path::new(&path).file_name().unwrap().to_string_lossy(),
                    v,
                    1.0 - v,
                    valid + invalid,
                    mr,
                    mg,
                    mb
                );
            }
            _ => panic!("unknown mode {mode}"),
        }
    }
}
