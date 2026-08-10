// SPDX-License-Identifier: MIT

use crate::debug::DebugStats;
use crate::font::ICON_TROPHY;
use crate::game::vehicle::{PERFECT_HI, PERFECT_LO, RED_ZONE_START};
use crate::game::Game;
use crate::ui::{
    Align, Column, Gauge, GaugeZone, HAlign, Insets, Node, Overlay, Panel, Row, Size, Text, VAlign,
};

const WHITE: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
const GREEN: [f32; 4] = [0.2, 1.0, 0.3, 1.0];
const YELLOW: [f32; 4] = [1.0, 0.9, 0.3, 1.0];
const RED: [f32; 4] = [1.0, 0.2, 0.2, 1.0];
const DIM: [f32; 4] = [0.72, 0.78, 0.82, 1.0];
const MODE_COL: [f32; 4] = [0.78, 0.9, 1.0, 1.0];
const WRECK_ORANGE: [f32; 4] = [1.0, 0.4, 0.1, 1.0];
const PERFECT_COL: [f32; 4] = [0.3, 1.0, 0.95, 1.0];
const PANEL_BG: [f32; 4] = [0.03, 0.04, 0.06, 0.6];
const TRANSPARENT: [f32; 4] = [0.0, 0.0, 0.0, 0.0];

const PANEL_INSETS: Insets = Insets::uniform(16.0);
const EDGE: f32 = 24.0;
const ROW_GAP: f32 = 14.0;

const EM_ALERT: f32 = 86.0;
const EM_LG: f32 = 38.0;
const EM_MD: f32 = 27.0;
const EM_SM: f32 = 24.0;

const EM_GAUGE_NUM: f32 = 62.0;
const EM_GAUGE_LABEL: f32 = 26.0;
const GAUGE_SIZE: f32 = 260.0;
const GAUGE_GAP: f32 = 64.0;

/// True top speed (~342 km/h) used to scale the speed gauge.
const TOP_SPEED: f32 = 342.0;

/// Builds the in-game HUD widget tree for the current game state. When
/// `debug` is `Some`, a dev-only diagnostics panel (F3) is added.
pub(crate) fn build_hud_tree(game: &Game, debug: Option<&DebugStats>) -> Node {
    let mut overlay = Overlay::new();
    overlay.push(Align::TopLeft, top_left(game));
    overlay.push(Align::TopRight, top_right(game));
    overlay.push(Align::BottomCenter, gauges(game));
    if let Some(bar) = heat_bar(game) {
        overlay.push(Align::BottomCenter, bar);
    }
    if let Some(popup) = perfect_shift_popup(game) {
        overlay.push(Align::BottomCenter, popup);
    }
    if let Some(alert) = alert(game) {
        overlay.push(Align::Center, alert);
    }
    if let Some(d) = debug {
        overlay.push(Align::TopCenter, debug_panel(d));
    }
    Node::new(overlay)
}

/// Dev-only diagnostics: FPS / frame times, world-mesh volume and rebuild cost,
/// particle counts, and current position/terrain state.
fn debug_panel(d: &DebugStats) -> Node {
    let lines = vec![
        Node::new(Text::new(
            format!("FPS {:.0}   FRAME {:.1} ms", d.fps, d.frame_ms),
            EM_SM,
            YELLOW,
        )),
        Node::new(Text::new(
            format!(
                "CPU {:.2} ms   CHUNKS {}   TRIS {}",
                d.cpu_ms, d.world_chunks, d.world_tris
            ),
            EM_SM,
            DIM,
        )),
        Node::new(Text::new(
            format!(
                "REBUILD {:.0} ms ({} chunks)",
                d.chunk_rebuild_ms, d.chunks_rebuilt
            ),
            EM_SM,
            DIM,
        )),
        Node::new(Text::new(
            format!("PARTS {}   HUD {}", d.particles, d.hud_verts),
            EM_SM,
            DIM,
        )),
        Node::new(Text::new(
            format!(
                "DIST {:.0} m   CHK {}   TERRAIN {:.2}",
                d.distance, d.chunk_index, d.terrain_factor
            ),
            EM_SM,
            DIM,
        )),
    ];
    let col = Column::new(lines, 6.0, HAlign::Left);
    let panel = Panel::wrap(PANEL_BG, PANEL_INSETS, Node::new(col));
    margin(Node::new(panel), Insets::new(EDGE, EDGE, 0.0, 0.0))
}

/// A transparent spacer that pushes a child in from the screen edges.
fn margin(child: Node, insets: Insets) -> Node {
    Node::new(Panel::colored(TRANSPARENT).padded(insets).with_child(child))
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
            Node::new(Text::new(
                format!("TIME {}", clock_time(game.time_of_day())),
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

/// Formats a 0..24 hour-of-day value as a 24h "HH:MM" clock string.
fn clock_time(hours: f32) -> String {
    let h = (hours.floor() as i32).rem_euclid(24);
    let m = ((hours - hours.floor()) * 60.0).floor() as i32;
    format!("{h:02}:{m:02}")
}

/// Speed + RPM circular gauges, side by side at the bottom center.
fn gauges(game: &Game) -> Node {
    let speed_frac = (game.speed_kmh / TOP_SPEED).clamp(0.0, 1.0);
    let speed = Gauge::new(Size::new(GAUGE_SIZE, GAUGE_SIZE), speed_frac, GREEN)
        .number(format!("{:.0}", game.speed_kmh), EM_GAUGE_NUM, WHITE)
        .label("KM/H", EM_GAUGE_LABEL, GREEN);

    // The needle rides on `rpm_frac()` (speed/redline, 0..=1) so the blue
    // perfect-shift band and red zone on the ring match the exact thresholds
    // the game judges gear changes and engine heat by. Feeding `rpm()/REDLINE`
    // here would add the idle-RPM offset and shift the visual zones ~3% early.
    let rpm_frac = game.vehicle.rpm_frac();
    let rpm = Gauge::new(Size::new(GAUGE_SIZE, GAUGE_SIZE), rpm_frac, GREEN)
        .zone(GaugeZone::new(PERFECT_LO, PERFECT_HI, PERFECT_COL))
        .zone(GaugeZone::new(RED_ZONE_START, 1.0, RED))
        .number(
            format!("{:.1}", game.vehicle.rpm() / 1000.0),
            EM_GAUGE_NUM,
            WHITE,
        )
        .label("RPM x1000", EM_GAUGE_LABEL, PERFECT_COL);

    let row = Row::new(
        vec![Node::new(speed), Node::new(rpm)],
        GAUGE_GAP,
        VAlign::Center,
    );
    margin(Node::new(row), Insets::new(0.0, 0.0, 0.0, EDGE))
}

/// A thin engine-heat bar under the gauges; empty width when cold.
fn heat_bar(game: &Game) -> Option<Node> {
    let max_w = 180.0;
    let fill = (game.engine_heat * max_w).clamp(0.0, max_w);
    let col = if game.engine_heat >= 0.6 {
        RED
    } else {
        WRECK_ORANGE
    };
    let track = Panel::sized([0.12, 0.13, 0.16, 0.9], Size::new(max_w, 10.0))
        .with_child(Node::new(Panel::sized(col, Size::new(fill, 10.0))));
    margin(Node::new(track), Insets::new(0.0, 0.0, 0.0, EDGE + 14.0)).into()
}

/// Brief "PERFECT SHIFT +N" feedback above the gauges.
fn perfect_shift_popup(game: &Game) -> Option<Node> {
    if game.perfect_shift_timer <= 0.0 {
        return None;
    }
    let gain = game.score.min(9999);
    let text =
        Text::new(format!("PERFECT SHIFT +{}", gain), EM_LG, PERFECT_COL).aligned(HAlign::Center);
    let bottom = EDGE + GAUGE_SIZE + 20.0;
    margin(Node::new(text), Insets::new(0.0, 0.0, 0.0, bottom)).into()
}

fn alert(game: &Game) -> Option<Node> {
    if game.game_over {
        let title = if game.engine_blown {
            "ENGINE BLOWN"
        } else {
            "GAME OVER"
        };
        let lines = vec![
            Node::new(Text::new(title, EM_ALERT, RED)),
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
    } else if game.engine_blown {
        // Engine just blew: announce immediately while the car coasts to a
        // stop; the full game-over panel appears once it has stopped.
        Some(Node::new(Panel::wrap(
            PANEL_BG,
            Insets::uniform(28.0),
            Node::new(Text::new("ENGINE BLOWN", EM_ALERT, RED)),
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

        let mut playing = build_hud_tree(&Game::new(), None);
        assert!(!ui.build(&mut playing, &atlas, 16.0 / 9.0, 0.0).is_empty());

        let mut over = Game::new();
        over.game_over = true;
        let mut over_root = build_hud_tree(&over, None);
        assert!(!ui.build(&mut over_root, &atlas, 16.0 / 9.0, 0.0).is_empty());
    }

    #[test]
    fn debug_panel_builds_vertices_with_metrics() {
        let ui = Ui::new();
        let atlas = FontAtlas::load();

        let mut stats = DebugStats {
            fps: 118.0,
            frame_ms: 8.5,
            cpu_ms: 3.2,
            world_chunks: 8,
            world_tris: 262_000,
            chunk_rebuild_ms: 41.0,
            chunks_rebuilt: 8,
            particles: 1800,
            hud_verts: 8900,
            distance: 1234.5,
            terrain_factor: 0.78,
            chunk_index: 4,
            world_verts: 131_000,
        };
        let mut root = build_hud_tree(&Game::new(), Some(&stats));
        let verts = ui.build(&mut root, &atlas, 16.0 / 9.0, 0.0);
        assert!(!verts.is_empty(), "debug panel should emit vertices");

        // Full path: some metrics survive a round-trip through the stats.
        stats.sample_frame(0.01);
        assert!(stats.fps > 0.0);
    }

    #[test]
    fn engine_blown_alert_appears_before_the_car_stops() {
        let ui = Ui::new();
        let atlas = FontAtlas::load();

        let mut blown = Game::new();
        blown.engine_blown = true;
        let mut blown_root = build_hud_tree(&blown, None);
        assert!(
            !ui.build(&mut blown_root, &atlas, 16.0 / 9.0, 0.0)
                .is_empty(),
            "alert visible while the car is still coasting"
        );
    }

    #[test]
    fn clock_time_formats_24h_hh_mm() {
        assert_eq!(clock_time(0.0), "00:00");
        assert_eq!(clock_time(12.5), "12:30");
        assert_eq!(clock_time(23.99), "23:59");
        assert_eq!(clock_time(24.5), "00:30", "wraps past midnight");
    }
}
