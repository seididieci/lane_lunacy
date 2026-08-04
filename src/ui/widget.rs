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
}

fn to_ndc_x(canvas: Size, x: f32) -> f32 {
    x / canvas.w * 2.0 - 1.0
}

fn to_ndc_y(canvas: Size, y: f32) -> f32 {
    1.0 - y / canvas.h * 2.0
}
