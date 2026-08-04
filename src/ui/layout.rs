// SPDX-License-Identifier: MIT

/// A two-dimensional extent in layout units.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Size {
    pub w: f32,
    pub h: f32,
}

impl Size {
    pub const ZERO: Size = Size { w: 0.0, h: 0.0 };
    pub const INFINITY: Size = Size { w: f32::INFINITY, h: f32::INFINITY };

    pub const fn new(w: f32, h: f32) -> Size {
        Size { w, h }
    }
}

/// A point in layout units. Y grows downward.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl Point {
    pub const ZERO: Point = Point { x: 0.0, y: 0.0 };

    pub const fn new(x: f32, y: f32) -> Point {
        Point { x, y }
    }
}

/// An axis-aligned box in layout units.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Rect {
    pub pos: Point,
    pub size: Size,
}

impl Rect {
    pub const fn new(x: f32, y: f32, w: f32, h: f32) -> Rect {
        Rect {
            pos: Point { x, y },
            size: Size { w, h },
        }
    }

    pub fn contains(&self, p: Point) -> bool {
        p.x >= self.pos.x
            && p.x < self.pos.x + self.size.w
            && p.y >= self.pos.y
            && p.y < self.pos.y + self.size.h
    }

    /// This rect shifted so its top-left corner lands on `origin`.
    pub fn at_origin(&self, origin: Point) -> Rect {
        Rect {
            pos: Point::new(origin.x + self.pos.x, origin.y + self.pos.y),
            size: self.size,
        }
    }
}

/// Uniform / asymmetric padding around a box's content.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Insets {
    pub l: f32,
    pub t: f32,
    pub r: f32,
    pub b: f32,
}

impl Insets {
    pub const ZERO: Insets = Insets {
        l: 0.0,
        t: 0.0,
        r: 0.0,
        b: 0.0,
    };

    pub const fn uniform(v: f32) -> Insets {
        Insets {
            l: v,
            t: v,
            r: v,
            b: v,
        }
    }

    pub const fn new(l: f32, t: f32, r: f32, b: f32) -> Insets {
        Insets { l, t, r, b }
    }
}

/// Min/max extents a widget is allowed to occupy during layout.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Constraints {
    pub min: Size,
    pub max: Size,
}

impl Constraints {
    pub const fn new(min: Size, max: Size) -> Constraints {
        Constraints { min, max }
    }

    pub const fn tight(size: Size) -> Constraints {
        Constraints { min: size, max: size }
    }

    pub const fn loose(max: Size) -> Constraints {
        Constraints {
            min: Size::ZERO,
            max,
        }
    }

    /// Clamp `size` into `[min, max]` (width and height independently).
    pub fn clamp_size(&self, size: Size) -> Size {
        Size::new(
            size.w.clamp(self.min.w, self.max.w),
            size.h.clamp(self.min.h, self.max.h),
        )
    }
}

/// Horizontal alignment of content within its allocated box.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HAlign {
    #[default]
    Left,
    Center,
    Right,
}

/// Vertical alignment of content within its allocated box.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VAlign {
    #[default]
    Top,
    Center,
    Bottom,
}

/// Full 3x3 anchoring of a child within a container (used by `Overlay`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Align {
    #[default]
    TopLeft,
    TopCenter,
    TopRight,
    CenterLeft,
    Center,
    CenterRight,
    BottomLeft,
    BottomCenter,
    BottomRight,
}

impl Align {
    /// Offset to place a `child`-sized box inside a `container`-sized one.
    pub fn offset_in(&self, container: Size, child: Size) -> Point {
        let x = match self {
            Align::TopLeft | Align::CenterLeft | Align::BottomLeft => 0.0,
            Align::TopCenter | Align::Center | Align::BottomCenter => {
                (container.w - child.w) / 2.0
            }
            _ => container.w - child.w,
        };
        let y = match self {
            Align::TopLeft | Align::TopCenter | Align::TopRight => 0.0,
            Align::CenterLeft | Align::Center | Align::CenterRight => {
                (container.h - child.h) / 2.0
            }
            _ => container.h - child.h,
        };
        Point::new(x.max(0.0), y.max(0.0))
    }
}
