// SPDX-License-Identifier: MIT

use crate::font::ICON_TROPHY;
use crate::game::Game;
use crate::ui::{
    Align, Column, HAlign, Insets, Node, Overlay, Panel, Size, Text, VIRTUAL_HEIGHT,
};

const WHITE: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
const GREEN: [f32; 4] = [0.2, 1.0, 0.3, 1.0];
const YELLOW: [f32; 4] = [1.0, 0.9, 0.3, 1.0];
const RED: [f32; 4] = [1.0, 0.2, 0.2, 1.0];
const DIM: [f32; 4] = [0.72, 0.78, 0.82, 1.0];
const MODE_COL: [f32; 4] = [0.78, 0.9, 1.0, 1.0];
const WRECK_ORANGE: [f32; 4] = [1.0, 0.4, 0.1, 1.0];
const PANEL_BG: [f32; 4] = [0.03, 0.04, 0.06, 0.6];
const TRANSPARENT: [f32; 4] = [0.0, 0.0, 0.0, 0.0];

const PANEL_INSETS: Insets = Insets::uniform(16.0);
const EDGE: f32 = 24.0;
const ROW_GAP: f32 = 14.0;

const EM_ALERT: f32 = 86.0;
const EM_SPEED: f32 = 76.0;
const EM_LG: f32 = 38.0;
const EM_MD: f32 = 27.0;
const EM_SM: f32 = 24.0;
const EM_LABEL: f32 = 32.0;

/// True top speed (~342 km/h) used to scale the speed bar.
const TOP_SPEED: f32 = 342.0;

/// Builds the in-game HUD widget tree for the current game state.
pub(crate) fn build_hud_tree(game: &Game, aspect: f32) -> Node {
    let mut overlay = Overlay::new();
    overlay.push(Align::TopLeft, top_left(game));
    overlay.push(Align::TopRight, top_right(game));
    overlay.push(Align::BottomCenter, speed_block(game));
    overlay.push(Align::BottomLeft, speed_bar(game, aspect));
    if let Some(alert) = alert(game) {
        overlay.push(Align::Center, alert);
    }
    Node::new(overlay)
}

/// A transparent spacer that pushes a child in from the screen edges.
fn margin(child: Node, insets: Insets) -> Node {
    Node::new(
        Panel::colored(TRANSPARENT)
            .padded(insets)
            .with_child(child),
    )
}

fn top_left(game: &Game) -> Node {
    let tuning = game.difficulty.tuning();
    let wreck_col = if tuning.wreck_limit - game.wrecks <= 1 {
        RED
    } else {
        WHITE
    };
    let col = Column::new(
        vec![
            Node::new(Text::new(
                format!("GEAR {}", game.vehicle.gear),
                EM_LG,
                YELLOW,
            )),
            Node::new(Text::new(
                format!("MODE {}", game.difficulty.label()),
                EM_MD,
                MODE_COL,
            )),
            Node::new(Text::new(
                format!("WRECKS {}/{}", game.wrecks, tuning.wreck_limit),
                EM_MD,
                wreck_col,
            )),
        ],
        ROW_GAP,
        HAlign::Left,
    );
    let panel = Panel::wrap(PANEL_BG, PANEL_INSETS, Node::new(col));
    margin(Node::new(panel), Insets::new(EDGE, EDGE, 0.0, 0.0))
}

fn top_right(game: &Game) -> Node {
    let col = Column::new(
        vec![
            Node::new(Text::new(format!("SCORE {}", game.score), EM_LG, WHITE)),
            Node::new(Text::new(format!("BEST {}", game.best_score), EM_MD, GREEN)),
            Node::new(Text::new(
                format!("AVG {:.0} KM/H", game.avg_speed * 3.6),
                EM_SM,
                DIM,
            )),
        ],
        ROW_GAP,
        HAlign::Right,
    );
    let panel = Panel::wrap(PANEL_BG, PANEL_INSETS, Node::new(col));
    margin(Node::new(panel), Insets::new(0.0, EDGE, EDGE, 0.0))
}

fn speed_block(game: &Game) -> Node {
    let col = Column::new(
        vec![
            Node::new(Text::new(format!("{:.0}", game.speed_kmh), EM_SPEED, WHITE)),
            Node::new(Text::new("KM/H", EM_LABEL, GREEN)),
        ],
        8.0,
        HAlign::Center,
    );
    let panel = Panel::wrap(PANEL_BG, PANEL_INSETS, Node::new(col));
    margin(Node::new(panel), Insets::new(0.0, 0.0, 0.0, EDGE * 10.0))
}

fn speed_bar(game: &Game, aspect: f32) -> Node {
    let ratio = (game.speed_kmh / TOP_SPEED).clamp(0.0, 1.0);
    let vw = VIRTUAL_HEIGHT * aspect;
    let side = 0.05 * vw;
    let bar_w = (vw - 2.0 * side) * ratio;
    let bar = Panel::sized(GREEN, Size::new(bar_w, 24.0));
    margin(Node::new(bar), Insets::new(side, 0.0, 0.0, EDGE * 4.0))
}

fn alert(game: &Game) -> Option<Node> {
    if game.game_over {
        let lines = vec![
            Node::new(Text::new("GAME OVER", EM_ALERT, RED)),
            Node::new(Text::new(format!("SCORE {}", game.score), EM_LG, WHITE)),
            Node::new(Text::new(
                format!("{} BEST {}", ICON_TROPHY, game.best_score),
                EM_MD,
                GREEN,
            )),
            // Always laid out (its space stays reserved) but blinks on/off.
            Node::new(Text::new("PRESS R TO RESTART", EM_MD, YELLOW).blinking(1.0)),
        ];
        let col = Column::new(lines, 12.0, HAlign::Center);
        Some(Node::new(Panel::wrap(
            PANEL_BG,
            Insets::uniform(28.0),
            Node::new(col),
        )))
    } else if game.wreck_timer > 0.0 {
        Some(Node::new(Panel::wrap(
            PANEL_BG,
            Insets::uniform(20.0),
            Node::new(Text::new("WRECK", EM_ALERT, WRECK_ORANGE)),
        )))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::font::FontAtlas;
    use crate::game::Game;
    use crate::ui::Ui;

    #[test]
    fn hud_builds_vertices_for_playing_and_game_over() {
        let ui = Ui::new();
        let atlas = FontAtlas::load();

        let mut playing = build_hud_tree(&Game::new(), 16.0 / 9.0);
        assert!(!ui.build(&mut playing, &atlas, 16.0 / 9.0, 0.0).is_empty());

        let mut over = Game::new();
        over.game_over = true;
        let mut over_root = build_hud_tree(&over, 16.0 / 9.0);
        assert!(!ui.build(&mut over_root, &atlas, 16.0 / 9.0, 0.0).is_empty());
    }
}
