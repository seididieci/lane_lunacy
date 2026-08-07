// SPDX-License-Identifier: MIT

use crate::font::FontAtlas;
use crate::ui::layout::{Constraints, Point, Rect, Size};
use crate::vertex::HudVertex;

/// A positioned widget: the widget box plus its laid-out geometry.
///
/// The rect is filled in by the widget's parent during the layout pass and is
/// relative to the parent's top-left corner (absolute for the root node).
pub struct Node {
    pub rect: Rect,
    pub widget: Box<dyn Widget>,
}

impl Node {
    pub fn new(widget: impl Widget + 'static) -> Node {
        Node {
            rect: Rect::new(0.0, 0.0, 0.0, 0.0),
            widget: Box::new(widget),
        }
    }

    /// This node's rect expressed inside its parent's absolute `rect`.
    pub(crate) fn placed(&self, parent: Rect) -> Rect {
        self.rect.at_origin(parent.pos)
    }
}

/// A composable piece of UI.
///
/// The trait follows a two-phase flow driven by `Ui`:
///  1. `layout` measures the widget (and lays out its children, assigning their
///     rects relative to this widget's origin);
///  2. `draw` emits the vertices for the final absolute `rect`.
///
/// `hit_test` / `handle_pointer` back the pointer API (wired to input later).
pub trait Widget {
    fn layout(&mut self, ctx: &mut LayoutCtx, constraints: Constraints) -> Size;

    fn draw(&self, ctx: &mut DrawCtx, rect: Rect);

    /// Return the topmost interactive hit under `p` (absolute layout units).
    fn hit_test(&self, _p: Point, _rect: Rect) -> Option<Hit> {
        None
    }

    /// Route a pointer event to this widget (and children). Returns true when
    /// the event was consumed (e.g. a button press/release).
    fn handle_pointer(&mut self, _ev: PointerEvent, _rect: Rect) -> bool {
        false
    }
}

/// An interactive target reported by `hit_test`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hit {
    pub id: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PointerEvent {
    Press { pos: Point },
    Release { pos: Point },
}

/// Measurement context for the layout pass.
pub struct LayoutCtx<'a> {
    pub atlas: &'a FontAtlas,
}

impl<'a> LayoutCtx<'a> {
    /// Extent of `text` at `em` layout units (the nominal em box).
    pub fn measure(&self, text: &str, em: f32) -> Size {
        let scale = em / self.atlas.raster_px;
        let mut w = 0.0f32;
        for c in text.chars() {
            if let Some(g) = self.atlas.glyph(c) {
                w += g.advance * scale;
            }
        }
        Size::new(w, em)
    }

    /// Tight bounding box of the glyphs (covers ascenders and descenders).
    pub fn measure_tight(&self, text: &str, em: f32) -> Size {
        let scale = em / self.atlas.raster_px;
        let baseline = self.atlas.ascender * scale;
        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        let mut cx = 0.0f32;
        for c in text.chars() {
            if let Some(g) = self.atlas.glyph(c) {
                if g.width > 0.0 && g.height > 0.0 {
                    let gx = cx + g.bearing_x * scale;
                    let top = baseline - (g.bearing_y + g.height) * scale;
                    let bottom = baseline - g.bearing_y * scale;
                    let gw = g.width * scale;
                    min_x = min_x.min(gx);
                    max_x = max_x.max(gx + gw);
                    min_y = min_y.min(top);
                    max_y = max_y.max(bottom);
                }
                cx += g.advance * scale;
            }
        }
        if min_x.is_finite() {
            Size::new(max_x - min_x, max_y - min_y)
        } else {
            Size::ZERO
        }
    }

    /// Greedy word-wrap of `text` to at most `max_w` layout units.
    pub fn wrap(&self, text: &str, em: f32, max_w: f32) -> Vec<String> {
        let scale = em / self.atlas.raster_px;
        let word_w = |word: &str| -> f32 {
            word.chars()
                .filter_map(|c| self.atlas.glyph(c).map(|g| g.advance))
                .sum::<f32>()
                * scale
        };
        let space_w = word_w(" ");
        let mut lines: Vec<String> = Vec::new();
        let mut current = String::new();
        let mut current_w = 0.0f32;
        for word in text.split_whitespace() {
            let w = word_w(word);
            let would_be = if current.is_empty() {
                w
            } else {
                current_w + space_w + w
            };
            if would_be <= max_w || current.is_empty() {
                if !current.is_empty() {
                    current.push(' ');
                    current_w += space_w;
                }
                current.push_str(word);
                current_w += w;
            } else {
                lines.push(std::mem::take(&mut current));
                current.push_str(word);
                current_w = w;
            }
        }
        if !current.is_empty() {
            lines.push(current);
        }
        lines
    }
}

/// Vertex-emission context for the draw pass. All coordinates are absolute
/// layout units; conversion to NDC happens here.
pub struct DrawCtx<'a> {
    pub out: &'a mut Vec<HudVertex>,
    pub atlas: &'a FontAtlas,
    pub virtual_size: Size,
    /// Monotonic UI clock in seconds, for time-based effects (e.g. blinking).
    pub time: f32,
}

impl<'a> DrawCtx<'a> {
    /// Width of `text` at `em` layout units (nominal em box).
    pub fn measure_text(&self, text: &str, em: f32) -> f32 {
        let scale = em / self.atlas.raster_px;
        text.chars()
            .filter_map(|c| self.atlas.glyph(c).map(|g| g.advance))
            .sum::<f32>()
            * scale
    }

    /// Draw `text` centered on `center` (layout units), with its em box
    /// vertically centered too.
    pub fn draw_text_centered(&mut self, text: &str, em: f32, color: [f32; 4], center: Point) {
        let w = self.measure_text(text, em);
        self.draw_text(
            text,
            em,
            color,
            Point::new(center.x - w / 2.0, center.y - em / 2.0),
        );
    }

    /// Draw `text` with the top-left of its em box at `top_left`.
    pub fn draw_text(&mut self, text: &str, em: f32, color: [f32; 4], top_left: Point) {
        let em_ndc = 2.0 * em / self.virtual_size.h;
        let aspect = self.virtual_size.w / self.virtual_size.h;
        crate::ui::backend::draw_text(
            self.out,
            self.atlas,
            text,
            to_ndc_x(self.virtual_size, top_left.x),
            to_ndc_y(self.virtual_size, top_left.y),
            em_ndc,
            aspect,
            color,
        );
    }

    /// Fill `rect` with a solid color panel.
    pub fn push_panel(&mut self, rect: Rect, color: [f32; 4]) {
        crate::ui::backend::push_rect(
            self.out,
            to_ndc_x(self.virtual_size, rect.pos.x),
            to_ndc_y(self.virtual_size, rect.pos.y),
            rect.size.w / self.virtual_size.w * 2.0,
            rect.size.h / self.virtual_size.h * 2.0,
            crate::ui::backend::SOLID_UV,
            color,
        );
    }

    /// Ring band from `frac0` to `frac1` (fractions of the gauge sweep) between
    /// `r_in` and `r_out` layout units, centered on `center`.
    pub fn draw_ring_segment(
        &mut self,
        center: Point,
        r_in: f32,
        r_out: f32,
        frac0: f32,
        frac1: f32,
        color: [f32; 4],
    ) {
        let (a0, a1) = gauge_angles(frac0, frac1);
        let scale = 2.0 / self.virtual_size.h;
        crate::ui::backend::push_ring_segment(
            self.out,
            to_ndc_x(self.virtual_size, center.x),
            to_ndc_y(self.virtual_size, center.y),
            r_in * scale,
            r_out * scale,
            a0,
            a1,
            self.virtual_size.h / self.virtual_size.w,
            color,
        );
    }

    /// A thin needle from radius `r0` to `r1` at gauge fraction `frac`.
    pub fn draw_needle(
        &mut self,
        center: Point,
        r0: f32,
        r1: f32,
        frac: f32,
        thick: f32,
        color: [f32; 4],
    ) {
        let angle = gauge_angle(frac);
        let scale = 2.0 / self.virtual_size.h;
        crate::ui::backend::push_needle(
            self.out,
            to_ndc_x(self.virtual_size, center.x),
            to_ndc_y(self.virtual_size, center.y),
            r0 * scale,
            r1 * scale,
            angle,
            thick * scale,
            self.virtual_size.h / self.virtual_size.w,
            color,
        );
    }
}

/// Map a fraction of the 270-degree gauge sweep (0 = lower-left, 0.5 = top,
/// 1 = lower-right) to an angle in degrees, y-up.
pub(crate) fn gauge_angle(frac: f32) -> f32 {
    let t = frac.clamp(0.0, 1.0);
    // Sweep clockwise from 225 deg (7:30) down to -45 deg (4:30).
    225.0 - 270.0 * t
}

fn gauge_angles(frac0: f32, frac1: f32) -> (f32, f32) {
    (gauge_angle(frac0), gauge_angle(frac1))
}

fn to_ndc_x(canvas: Size, x: f32) -> f32 {
    x / canvas.w * 2.0 - 1.0
}

fn to_ndc_y(canvas: Size, y: f32) -> f32 {
    1.0 - y / canvas.h * 2.0
}
