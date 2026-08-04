// SPDX-License-Identifier: MIT

use crate::ui::layout::{Align, Constraints, HAlign, Insets, Point, Rect, Size, VAlign};
use crate::ui::widget::{DrawCtx, Hit, LayoutCtx, Node, PointerEvent, Widget};

/// Highlight color used by focused `Button`s.
const HIGHLIGHT: [f32; 4] = [0.35, 0.5, 0.7, 0.35];

/// A block of text, optionally wrapped to its allocated width.
pub struct Text {
    pub text: String,
    pub em: f32,
    pub color: [f32; 4],
    pub wrap: bool,
    pub align: HAlign,
    lines: Vec<(String, f32, f32)>,
}

impl Text {
    pub fn new(text: impl Into<String>, em: f32, color: [f32; 4]) -> Text {
        Text {
            text: text.into(),
            em,
            color,
            wrap: false,
            align: HAlign::Left,
            lines: Vec::new(),
        }
    }

    /// Wrap to the allocated width when the text is wider than the constraint.
    pub fn wrapped(mut self) -> Text {
        self.wrap = true;
        self
    }

    /// Align each line within its allocated box.
    pub fn aligned(mut self, align: HAlign) -> Text {
        self.align = align;
        self
    }
}

impl Widget for Text {
    fn layout(&mut self, ctx: &mut LayoutCtx, constraints: Constraints) -> Size {
        let intrinsic = ctx.measure(&self.text, self.em);
        let max_w = constraints.max.w;
        let wrapped = if self.wrap && max_w.is_finite() && intrinsic.w > max_w {
            ctx.wrap(&self.text, self.em, max_w)
        } else {
            Vec::new()
        };
        // Reserve each line's tight glyph height (glyphs can exceed the em box,
        // e.g. tall ascenders/descenders) so stacked lines never overlap.
        self.lines = wrapped
            .into_iter()
            .chain(std::iter::once(self.text.clone()))
            .map(|line| {
                let w = ctx.measure(&line, self.em).w;
                let h = ctx.measure_tight(&line, self.em).h.max(self.em);
                (line, w, h)
            })
            .collect();
        let line_w = self.lines.iter().map(|(_, w, _)| *w).fold(0.0f32, f32::max);
        let total_h = self.lines.iter().map(|(_, _, h)| *h).sum::<f32>();
        let w = line_w.min(max_w);
        constraints.clamp_size(Size::new(w, total_h))
    }

    fn draw(&self, ctx: &mut DrawCtx, rect: Rect) {
        let mut y = rect.pos.y;
        for (line, line_w, line_h) in &self.lines {
            let x = rect.pos.x
                + match self.align {
                    HAlign::Left => 0.0,
                    HAlign::Center => ((rect.size.w - line_w) / 2.0).max(0.0),
                    HAlign::Right => (rect.size.w - line_w).max(0.0),
                };
            ctx.draw_text(line, self.em, self.color, Point::new(x, y));
            y += line_h;
        }
    }
}

/// A tappable menu entry with a focus highlight and a stable `id` for input.
pub struct Button {
    pub label: String,
    pub em: f32,
    pub color: [f32; 4],
    pub focused: bool,
    pub id: u64,
    /// Minimum hit-target extent in layout units (use >= ~44 px for touch).
    pub touch_min: f32,
    text_size: Size,
    pressed: bool,
}

impl Button {
    pub fn new(label: impl Into<String>, em: f32, color: [f32; 4], id: u64) -> Button {
        Button {
            label: label.into(),
            em,
            color,
            focused: false,
            id,
            touch_min: 0.0,
            text_size: Size::ZERO,
            pressed: false,
        }
    }

    pub fn touch_target(mut self, min: f32) -> Button {
        self.touch_min = min;
        self
    }

    pub fn focused(mut self, yes: bool) -> Button {
        self.focused = yes;
        self
    }
}

impl Widget for Button {
    fn layout(&mut self, ctx: &mut LayoutCtx, constraints: Constraints) -> Size {
        let text = ctx.measure_tight(&self.label, self.em);
        self.text_size = text;
        let w = (text.w + 2.0 * self.em * 0.4).max(self.touch_min);
        let h = (text.h + 2.0 * self.em * 0.3).max(self.touch_min);
        constraints.clamp_size(Size::new(w, h))
    }

    fn draw(&self, ctx: &mut DrawCtx, rect: Rect) {
        let cx = rect.pos.x + (rect.size.w - self.text_size.w) / 2.0;
        let cy = rect.pos.y + (rect.size.h - self.text_size.h) / 2.0;
        if self.focused {
            let pad = self.em * 0.3;
            ctx.push_panel(
                Rect::new(
                    cx - pad,
                    cy - pad,
                    self.text_size.w + 2.0 * pad,
                    self.text_size.h + 2.0 * pad,
                ),
                HIGHLIGHT,
            );
        }
        ctx.draw_text(&self.label, self.em, self.color, Point::new(cx, cy));
    }

    fn hit_test(&self, p: Point, rect: Rect) -> Option<Hit> {
        rect.contains(p).then_some(Hit { id: self.id })
    }

    fn handle_pointer(&mut self, ev: PointerEvent, rect: Rect) -> bool {
        match ev {
            PointerEvent::Press { pos } => {
                if rect.contains(pos) {
                    self.pressed = true;
                    true
                } else {
                    false
                }
            }
            PointerEvent::Release { pos } => {
                let activated = self.pressed && rect.contains(pos);
                self.pressed = false;
                activated
            }
        }
    }
}

/// A colored box that optionally wraps a single child. Without a child it
/// renders as a plain rectangle (e.g. a speed bar).
pub struct Panel {
    pub color: [f32; 4],
    pub padding: Insets,
    pub size: Option<Size>,
    pub child: Option<Node>,
}

impl Panel {
    pub fn colored(color: [f32; 4]) -> Panel {
        Panel {
            color,
            padding: Insets::ZERO,
            size: None,
            child: None,
        }
    }

    pub fn sized(color: [f32; 4], size: Size) -> Panel {
        Panel {
            color,
            padding: Insets::ZERO,
            size: Some(size),
            child: None,
        }
    }

    pub fn wrap(color: [f32; 4], padding: Insets, child: Node) -> Panel {
        Panel {
            color,
            padding,
            size: None,
            child: Some(child),
        }
    }

    pub fn with_child(mut self, child: Node) -> Panel {
        self.child = Some(child);
        self
    }

    pub fn padded(mut self, padding: Insets) -> Panel {
        self.padding = padding;
        self
    }
}

impl Widget for Panel {
    fn layout(&mut self, ctx: &mut LayoutCtx, constraints: Constraints) -> Size {
        let size = match (self.size, &mut self.child) {
            (Some(size), Some(child)) => {
                let inner = Size::new(
                    (size.w - self.padding.l - self.padding.r).max(0.0),
                    (size.h - self.padding.t - self.padding.b).max(0.0),
                );
                child.widget.layout(ctx, Constraints::tight(inner));
                child.rect = Rect::new(self.padding.l, self.padding.t, inner.w, inner.h);
                size
            }
            (None, Some(child)) => {
                let inner_max = Size::new(
                    (constraints.max.w - self.padding.l - self.padding.r).max(0.0),
                    (constraints.max.h - self.padding.t - self.padding.b).max(0.0),
                );
                let child_size = child.widget.layout(ctx, Constraints::loose(inner_max));
                child.rect = Rect::new(
                    self.padding.l,
                    self.padding.t,
                    child_size.w,
                    child_size.h,
                );
                Size::new(
                    child_size.w + self.padding.l + self.padding.r,
                    child_size.h + self.padding.t + self.padding.b,
                )
            }
            (size, None) => size.unwrap_or(constraints.max),
        };
        constraints.clamp_size(size)
    }

    fn draw(&self, ctx: &mut DrawCtx, rect: Rect) {
        ctx.push_panel(rect, self.color);
        if let Some(child) = &self.child {
            child.widget.draw(ctx, child.placed(rect));
        }
    }

    fn hit_test(&self, p: Point, rect: Rect) -> Option<Hit> {
        self.child
            .as_ref()
            .and_then(|child| child.widget.hit_test(p, child.placed(rect)))
    }

    fn handle_pointer(&mut self, ev: PointerEvent, rect: Rect) -> bool {
        self.child
            .as_mut()
            .map_or(false, |child| child.widget.handle_pointer(ev, child.placed(rect)))
    }
}

/// A vertical stack of widgets with a cross-axis alignment.
pub struct Column {
    pub gap: f32,
    pub align: HAlign,
    pub children: Vec<Node>,
}

impl Column {
    pub fn new(children: Vec<Node>, gap: f32, align: HAlign) -> Column {
        Column {
            gap,
            align,
            children,
        }
    }

    pub fn push(&mut self, node: Node) {
        self.children.push(node);
    }
}

impl Widget for Column {
    fn layout(&mut self, ctx: &mut LayoutCtx, constraints: Constraints) -> Size {
        let max_w = constraints.max.w;
        let mut y = 0.0f32;
        let mut max_child_w = 0.0f32;
        let count = self.children.len();
        for (i, node) in self.children.iter_mut().enumerate() {
            let size =
                node.widget
                    .layout(ctx, Constraints::loose(Size::new(max_w, f32::INFINITY)));
            node.rect = Rect::new(0.0, y, size.w, size.h);
            y += size.h;
            if i + 1 < count {
                y += self.gap;
            }
            max_child_w = max_child_w.max(size.w);
        }
        let col_w = max_child_w.min(max_w);
        for node in &mut self.children {
            node.rect.pos.x = h_align_offset(self.align, col_w, node.rect.size.w);
        }
        constraints.clamp_size(Size::new(col_w, y))
    }

    fn draw(&self, ctx: &mut DrawCtx, rect: Rect) {
        for node in &self.children {
            node.widget.draw(ctx, node.placed(rect));
        }
    }

    fn hit_test(&self, p: Point, rect: Rect) -> Option<Hit> {
        self.children
            .iter()
            .find_map(|node| node.widget.hit_test(p, node.placed(rect)))
    }

    fn handle_pointer(&mut self, ev: PointerEvent, rect: Rect) -> bool {
        self.children
            .iter_mut()
            .any(|node| node.widget.handle_pointer(ev, node.placed(rect)))
    }
}

/// A horizontal stack of widgets with a cross-axis alignment.
pub struct Row {
    pub gap: f32,
    pub align: VAlign,
    pub children: Vec<Node>,
}

impl Row {
    pub fn new(children: Vec<Node>, gap: f32, align: VAlign) -> Row {
        Row {
            gap,
            align,
            children,
        }
    }

    pub fn push(&mut self, node: Node) {
        self.children.push(node);
    }
}

impl Widget for Row {
    fn layout(&mut self, ctx: &mut LayoutCtx, constraints: Constraints) -> Size {
        let max_h = constraints.max.h;
        let mut x = 0.0f32;
        let mut max_child_h = 0.0f32;
        let count = self.children.len();
        for (i, node) in self.children.iter_mut().enumerate() {
            let size =
                node.widget
                    .layout(ctx, Constraints::loose(Size::new(f32::INFINITY, max_h)));
            node.rect = Rect::new(x, 0.0, size.w, size.h);
            x += size.w;
            if i + 1 < count {
                x += self.gap;
            }
            max_child_h = max_child_h.max(size.h);
        }
        let row_h = max_child_h.min(max_h);
        for node in &mut self.children {
            node.rect.pos.y = v_align_offset(self.align, row_h, node.rect.size.h);
        }
        constraints.clamp_size(Size::new(x, row_h))
    }

    fn draw(&self, ctx: &mut DrawCtx, rect: Rect) {
        for node in &self.children {
            node.widget.draw(ctx, node.placed(rect));
        }
    }

    fn hit_test(&self, p: Point, rect: Rect) -> Option<Hit> {
        self.children
            .iter()
            .find_map(|node| node.widget.hit_test(p, node.placed(rect)))
    }

    fn handle_pointer(&mut self, ev: PointerEvent, rect: Rect) -> bool {
        self.children
            .iter_mut()
            .any(|node| node.widget.handle_pointer(ev, node.placed(rect)))
    }
}

/// A child anchored within an `Overlay`.
pub struct AlignChild {
    pub align: Align,
    pub node: Node,
}

/// Fills its container and positions each child at an `Align` anchor. This is
/// the root of every screen and the way HUD corner blocks are placed.
pub struct Overlay {
    pub children: Vec<AlignChild>,
}

impl Overlay {
    pub fn new() -> Overlay {
        Overlay {
            children: Vec::new(),
        }
    }

    pub fn push(&mut self, align: Align, node: Node) {
        self.children.push(AlignChild { align, node });
    }

    pub fn child(mut self, align: Align, node: Node) -> Overlay {
        self.push(align, node);
        self
    }
}

impl Widget for Overlay {
    fn layout(&mut self, ctx: &mut LayoutCtx, constraints: Constraints) -> Size {
        let canvas = constraints.max;
        for child in &mut self.children {
            let size = child
                .node
                .widget
                .layout(ctx, Constraints::loose(Size::INFINITY));
            let offset = child.align.offset_in(canvas, size);
            child.node.rect = Rect::new(offset.x, offset.y, size.w, size.h);
        }
        constraints.clamp_size(canvas)
    }

    fn draw(&self, ctx: &mut DrawCtx, rect: Rect) {
        for child in &self.children {
            child.node.widget.draw(ctx, child.node.placed(rect));
        }
    }

    fn hit_test(&self, p: Point, rect: Rect) -> Option<Hit> {
        // Reverse order: the last pushed child is the topmost.
        self.children
            .iter()
            .rev()
            .find_map(|child| child.node.widget.hit_test(p, child.node.placed(rect)))
    }

    fn handle_pointer(&mut self, ev: PointerEvent, rect: Rect) -> bool {
        self.children
            .iter_mut()
            .rev()
            .any(|child| child.node.widget.handle_pointer(ev, child.node.placed(rect)))
    }
}

/// Empty widget that reserves a fixed amount of space.
pub struct Spacer {
    pub size: Size,
}

impl Spacer {
    pub fn new(w: f32, h: f32) -> Spacer {
        Spacer {
            size: Size::new(w, h),
        }
    }
}

impl Widget for Spacer {
    fn layout(&mut self, _ctx: &mut LayoutCtx, constraints: Constraints) -> Size {
        constraints.clamp_size(self.size)
    }

    fn draw(&self, _ctx: &mut DrawCtx, _rect: Rect) {}
}

fn h_align_offset(align: HAlign, parent_w: f32, child_w: f32) -> f32 {
    match align {
        HAlign::Left => 0.0,
        HAlign::Center => ((parent_w - child_w) / 2.0).max(0.0),
        HAlign::Right => (parent_w - child_w).max(0.0),
    }
}

fn v_align_offset(align: VAlign, parent_h: f32, child_h: f32) -> f32 {
    match align {
        VAlign::Top => 0.0,
        VAlign::Center => ((parent_h - child_h) / 2.0).max(0.0),
        VAlign::Bottom => (parent_h - child_h).max(0.0),
    }
}
