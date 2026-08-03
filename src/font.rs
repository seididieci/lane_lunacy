// SPDX-License-Identifier: MIT

use std::collections::HashMap;

use fontdue::Font;

const FONT_PATH: &str = "/usr/share/fonts/maple-mono/MapleMono-NF-Regular.ttf";
const RASTER_PX: f32 = 48.0;
const PADDING: usize = 2;
const CHARSET: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,:!?/-+%()";

#[derive(Clone, Copy, Debug)]
pub struct Glyph {
    pub u0: f32,
    pub v0: f32,
    pub u1: f32,
    pub v1: f32,
    pub width: f32,
    pub height: f32,
    pub bearing_x: f32,
    pub bearing_y: f32,
    pub advance: f32,
}

pub struct FontAtlas {
    pub pixels: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub raster_px: f32,
    /// Most negative ymin (above baseline) — used to compute baseline from text-top.
    pub ascender: f32,
    glyphs: HashMap<char, Glyph>,
}

impl FontAtlas {
    pub fn load() -> Self {
        let data = std::fs::read(FONT_PATH).expect("failed to read Maple Mono font file");
        let font =
            Font::from_bytes(data, fontdue::FontSettings::default()).expect("failed to parse font");

        let mut rasters: Vec<(char, fontdue::Metrics, Vec<u8>)> = Vec::new();
        let mut row_height = 0usize;
        for c in CHARSET.chars() {
            let (metrics, bitmap) = font.rasterize(c, RASTER_PX);
            row_height = row_height.max(metrics.height);
            rasters.push((c, metrics, bitmap));
        }

        let cell_w = RASTER_PX.ceil() as usize + PADDING * 2;
        let cell_h = row_height + PADDING * 2;
        let cols = 12usize;
        let rows = (rasters.len() + cols - 1) / cols;
        let width = cell_w * cols;
        let height = cell_h * rows;

        let mut pixels = vec![0u8; width * height];
        let mut glyphs = HashMap::new();
        let mut ascender = 0.0f32;

        for (i, (c, metrics, bitmap)) in rasters.iter().enumerate() {
            let cx = (i % cols) * cell_w + PADDING;
            let cy = (i / cols) * cell_h + PADDING;

            for gy in 0..metrics.height {
                for gx in 0..metrics.width {
                    let src = gy * metrics.width + gx;
                    let dst = (cy + gy) * width + (cx + gx);
                    pixels[dst] = bitmap[src];
                }
            }

            let u0 = cx as f32 / width as f32;
            let v0 = cy as f32 / height as f32;
            let u1 = (cx + metrics.width) as f32 / width as f32;
            let v1 = (cy + metrics.height) as f32 / height as f32;

            let by = metrics.ymin as f32;
            ascender = ascender.min(by);

            glyphs.insert(
                *c,
                Glyph {
                    u0,
                    v0,
                    u1,
                    v1,
                    width: metrics.width as f32,
                    height: metrics.height as f32,
                    bearing_x: metrics.xmin as f32,
                    bearing_y: by,
                    advance: metrics.advance_width,
                },
            );
        }

        FontAtlas {
            pixels,
            width: width as u32,
            height: height as u32,
            raster_px: RASTER_PX,
            ascender,
            glyphs,
        }
    }

    pub fn glyph(&self, c: char) -> Option<&Glyph> {
        self.glyphs.get(&c)
    }
}
