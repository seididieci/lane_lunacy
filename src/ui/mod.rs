// SPDX-License-Identifier: MIT

//! Retained widget-tree UI system.
//!
//! Screens are composed from `Widget`s laid out in a virtual canvas (see
//! `engine::VIRTUAL_HEIGHT`). Widgets measure themselves in a layout pass and
//! emit vertices in a draw pass; `Overlay` + `Align` place HUD corner blocks
//! and menus. Low-level NDC primitives live in `backend`.
//!
//! TODO: `Row`/`VAlign` and the pointer API (`Hit`, `PointerEvent`,
//! `handle_pointer`) are exercised only by tests until the pointer input is
//! wired in, hence the temporary dead-code allow. Remove `dead_code` with that
//! follow-up.

#![allow(dead_code)]

pub(crate) mod backend;
mod engine;
mod layout;
mod widget;
mod widgets;

#[allow(unused_imports)] // pointer API consumed by tests until input wiring lands
pub(crate) use engine::{Ui, VIRTUAL_HEIGHT};
#[allow(unused_imports)] // Row/VAlign reserved for future screens
pub(crate) use layout::{Align, Constraints, HAlign, Insets, Point, Rect, Size, VAlign};
#[allow(unused_imports)] // pointer API consumed by tests until input wiring lands
pub(crate) use widget::{DrawCtx, Hit, LayoutCtx, Node, PointerEvent, Widget};
#[allow(unused_imports)] // Row reserved for future screens
pub(crate) use widgets::{Button, Column, Overlay, Panel, Row, Spacer, Text};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::font::FontAtlas;

    fn atlas() -> FontAtlas {
        FontAtlas::load()
    }

    fn aspect_ratio() -> f32 {
        16.0 / 9.0
    }

    #[test]
    fn builds_and_renders_a_screen() {
        let ui = Ui::new();
        let mut root = Node::new(
            Overlay::new().child(
                Align::Center,
                Node::new(Panel::wrap(
                    [0.1, 0.1, 0.1, 0.5],
                    Insets::uniform(20.0),
                    Node::new(Column::new(
                        vec![
                            Node::new(Button::new(
                                "START",
                                40.0,
                                [1.0, 1.0, 1.0, 1.0],
                                7,
                            )),
                            Node::new(
                                Text::new(
                                    "a very long line that should wrap when constrained",
                                    24.0,
                                    [1.0, 1.0, 1.0, 1.0],
                                )
                                .wrapped()
                                .aligned(HAlign::Center),
                            ),
                        ],
                        12.0,
                        HAlign::Center,
                    )),
                )),
            ),
        );
        let verts = ui.build(&mut root, &atlas(), aspect_ratio());
        assert!(!verts.is_empty());
    }

    #[test]
    fn a_centered_button_hits_at_canvas_center() {
        let ui = Ui::new();
        let mut root = Node::new(Overlay::new().child(
            Align::Center,
            Node::new(Button::new("START", 40.0, [1.0, 1.0, 1.0, 1.0], 42)),
        ));
        let atlas = atlas();
        ui.build(&mut root, &atlas, aspect_ratio());

        let canvas = ui.virtual_size(aspect_ratio());
        let center = Point::new(canvas.w / 2.0, canvas.h / 2.0);
        assert_eq!(ui.hit_test(&root, center), Some(Hit { id: 42 }));

        assert!(ui.handle_pointer(
            &mut root,
            PointerEvent::Press { pos: center },
        ));
        assert!(ui.handle_pointer(
            &mut root,
            PointerEvent::Release { pos: center },
        ));
    }

    #[test]
    fn a_button_ignores_pointer_events_outside_its_box() {
        let ui = Ui::new();
        let mut root = Node::new(Overlay::new().child(
            Align::TopLeft,
            Node::new(Button::new("X", 20.0, [1.0, 1.0, 1.0, 1.0], 1)),
        ));
        let atlas = atlas();
        ui.build(&mut root, &atlas, aspect_ratio());

        let canvas = ui.virtual_size(aspect_ratio());
        let far = Point::new(canvas.w - 3.0, canvas.h - 3.0);
        assert!(!ui.handle_pointer(
            &mut root,
            PointerEvent::Press { pos: far },
        ));
        assert_eq!(ui.hit_test(&root, far), None);
    }

    #[test]
    fn text_measure_wrap_and_tight_bounds_agree() {
        let ctx = LayoutCtx { atlas: &atlas() };
        let em = 24.0;

        let plain = ctx.measure("START", em);
        assert!(plain.w > 0.0 && (plain.h - em).abs() < f32::EPSILON);

        let tight = ctx.measure_tight("START", em);
        assert!(tight.w > 0.0 && tight.h > 0.0);
        assert!(tight.h <= em);

        let lines = ctx.wrap("one two three four five six seven", em, 90.0);
        assert!(lines.len() > 1);
        assert!(lines.iter().all(|l| ctx.measure(l, em).w <= 90.0));
    }
}
