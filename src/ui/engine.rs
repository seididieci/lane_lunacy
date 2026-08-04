// SPDX-License-Identifier: MIT

use crate::font::FontAtlas;
use crate::ui::layout::{Constraints, Point, Rect, Size};
use crate::ui::widget::{DrawCtx, Hit, LayoutCtx, Node, PointerEvent};
use crate::vertex::HudVertex;

/// Height of the virtual canvas in layout units. Width is derived from the
/// window aspect ratio, so text and touch targets keep a constant physical feel.
pub const VIRTUAL_HEIGHT: f32 = 1080.0;

/// The UI engine: owns the virtual coordinate space and runs the layout, draw
/// and hit-test passes over a screen's widget tree.
#[derive(Default)]
pub struct Ui;

impl Ui {
    pub fn new() -> Ui {
        Ui
    }

    /// Size of the virtual canvas for the given window aspect ratio.
    pub fn virtual_size(&self, aspect: f32) -> Size {
        Size::new(VIRTUAL_HEIGHT * aspect, VIRTUAL_HEIGHT)
    }

    /// Layout `root`, draw it and return the HUD vertices for this frame.
    ///
    /// `time` is the UI clock in seconds, used for time-based widget effects
    /// such as blinking text.
    pub fn build(
        &self,
        root: &mut Node,
        atlas: &FontAtlas,
        aspect: f32,
        time: f32,
    ) -> Vec<HudVertex> {
        let virtual_size = self.virtual_size(aspect);

        let mut layout_ctx = LayoutCtx { atlas };
        let size = root
            .widget
            .layout(&mut layout_ctx, Constraints::tight(virtual_size));
        root.rect = Rect::new(0.0, 0.0, size.w, size.h);

        let mut out: Vec<HudVertex> = Vec::new();
        let mut draw_ctx = DrawCtx {
            out: &mut out,
            atlas,
            virtual_size,
            time,
        };
        root.widget.draw(&mut draw_ctx, root.rect);
        out
    }

    /// Return the interactive hit under `pos` (absolute layout units).
    pub fn hit_test(&self, root: &Node, pos: Point) -> Option<Hit> {
        if root.rect.contains(pos) {
            root.widget.hit_test(pos, root.rect)
        } else {
            None
        }
    }

    /// Route a pointer event (absolute layout units) to the widget tree.
    pub fn handle_pointer(&self, root: &mut Node, ev: PointerEvent) -> bool {
        let pos = match ev {
            PointerEvent::Press { pos } | PointerEvent::Release { pos } => pos,
        };
        if root.rect.contains(pos) {
            root.widget.handle_pointer(ev, root.rect)
        } else {
            false
        }
    }
}
