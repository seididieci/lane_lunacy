// SPDX-License-Identifier: MIT
// Temporary diagnostic: prints luma statistics for captured PNGs so visual
// properties (puddles, reflections) can be measured without eyeballing images.
// Usage: cargo run --release --example pngstats -- <png> [<png> ...]

use image::GenericImageView;

fn srgb_to_linear(c: u8) -> f32 {
    let v = c as f32 / 255.0;
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

fn luma(px: image::Rgba<u8>) -> f32 {
    0.299 * srgb_to_linear(px[0]) + 0.587 * srgb_to_linear(px[1]) + 0.114 * srgb_to_linear(px[2])
}

fn main() {
    for path in std::env::args().skip(1) {
        let img = image::open(&path)
            .unwrap_or_else(|e| panic!("{}: {e}", path))
            .to_rgba8();
        let (w, h) = img.dimensions();

        // Road band: bottom-center strip, excludes sky and most HUD.
        let band_top = (h as f32 * 0.55) as u32;
        let band_w0 = (w as f32 * 0.15) as u32;
        let band_w1 = (w as f32 * 0.85) as u32;
        let mut vals: Vec<f32> = Vec::new();
        for y in band_top..h {
            for x in band_w0..band_w1 {
                vals.push(luma(*img.get_pixel(x, y)));
            }
        }
        let n = vals.len() as f32;
        let mean = vals.iter().sum::<f32>() / n;
        let mut sorted = vals.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[sorted.len() / 2];
        let p10 = sorted[sorted.len() / 10];
        let p90 = sorted[(sorted.len() * 9) / 10];
        let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let bright_thresh = median * 1.4 + 0.02;
        let bright = vals.iter().filter(|&&v| v > bright_thresh).count() as f32;
        println!(
            "{}\n  size {}x{} band(y{}..{}) mean {:.4} median {:.4} std {:.4} p10 {:.4} p90 {:.4} bright>40% {:.4}",
            std::path::Path::new(&path).file_name().unwrap().to_string_lossy(),
            w,
            h,
            band_top,
            h,
            mean,
            median,
            var.sqrt(),
            p10,
            p90,
            bright / n
        );

        // Thirds breakdown.
        for (label, y0, y1) in [
            ("top", 0, h / 3),
            ("mid", h / 3, 2 * h / 3),
            ("bot", 2 * h / 3, h),
        ] {
            let mut sum = 0f32;
            let mut cnt = 0u32;
            for y in y0..y1 {
                for x in 0..w {
                    sum += luma(*img.get_pixel(x, y));
                    cnt += 1;
                }
            }
            println!("  {label} mean {:.4}", sum / cnt as f32);
        }

        // 8x6 grid of mean luma to locate bright/dark regions.
        let (gx, gy) = (8usize, 6usize);
        let mut grid = vec![0f64; gx * gy];
        let mut counts = vec![0u64; gx * gy];
        for y in 0..h {
            for x in 0..w {
                let row = ((y as usize) * gy / h as usize).min(gy - 1);
                let col = ((x as usize) * gx / w as usize).min(gx - 1);
                let idx = row * gx + col;
                grid[idx] += luma(*img.get_pixel(x, y)) as f64;
                counts[idx] += 1;
            }
        }
        print!("  grid(rows=top->bottom):");
        for row in 0..gy {
            if row > 0 {
                print!(" |");
            }
            for col in 0..gx {
                print!(" {:5.3}", grid[row * gx + col] / counts[row * gx + col] as f64);
            }
        }
        println!();
    }
}
