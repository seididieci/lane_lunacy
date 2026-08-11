// SPDX-License-Identifier: MIT

//! Roadside street lamps: a dark pole at the verge edge with an arm reaching
//! toward the road and a warm-off-white luminaire head, one per side every
//! `LAMP_SPACING` metres. The mesh owns only the geometry; the light itself
//! (projector pool + glow sprites) is packed by the frame from the same
//! deterministic placement list via [`head_pos`].

use crate::geom::push_box;
use crate::road::{road_curve, ROAD_HALF};
use crate::surface::SurfaceMaterial;
use crate::vertex::Vertex3d;
use crate::world::{spaced_placements, Placement, RoadsideObject};

/// World-space spacing (metres) between street-lamp pairs; one lamp is placed
/// on each side of the road per spacing interval.
const LAMP_SPACING: f32 = 40.0;
/// Lateral offset of the lamp pole base from the road center (well past the
/// verge strips and marker posts, inside the tree line).
const LAMP_LATERAL: f32 = ROAD_HALF + 1.7;
/// Pole height to the arm (luminaire head sits just above it).
const LAMP_HEIGHT: f32 = 5.6;
const ASPHALT_BASE: SurfaceMaterial = SurfaceMaterial::AsphaltBase;

pub static STREET_LAMP: StreetLamp = StreetLamp;

pub struct StreetLamp;

impl RoadsideObject for StreetLamp {
    fn placements(&self, start_s: f32, end_s: f32) -> Vec<Placement> {
        spaced_placements(start_s, end_s, LAMP_SPACING, 0.0)
            .into_iter()
            .flat_map(|s| {
                [1.0f32, -1.0].into_iter().map(move |side| Placement {
                    s,
                    side,
                    lateral: LAMP_LATERAL,
                    variant: 0,
                })
            })
            .collect()
    }

    fn push_geometry(&self, v: &mut Vec<Vertex3d>, i: &mut Vec<u32>, p: &Placement) {
        let x = road_curve(p.s) + p.side * p.lateral;
        push_lamp(v, i, x, -p.s, p.side);
    }
}

/// World-space luminaire head position for a lamp placement, matching the
/// geometry [`push_lamp`] builds so the light pools and glow sprites sit
/// exactly on the head. Used by the frame to fill the lamp projector pool.
pub fn head_pos(p: &Placement) -> [f32; 3] {
    let x = road_curve(p.s) + p.side * p.lateral;
    let head_lateral = ROAD_HALF + 0.35;
    let head_x = x - p.side * (LAMP_LATERAL - head_lateral);
    [head_x, LAMP_HEIGHT - 0.21, -p.s]
}

/// Pushes one street lamp: a dark pole at the roadside edge with an arm
/// reaching toward the road and a small warm-off-white luminaire head. All
/// parts use the asphalt slot so they stay free of the terrain tint and grass
/// flattening; the head's light comes from the projector pool + glow sprite,
/// not the mesh color.
fn push_lamp(v: &mut Vec<Vertex3d>, i: &mut Vec<u32>, x: f32, z: f32, side: f32) {
    let (slot, scale) = (ASPHALT_BASE.atlas_slot(), ASPHALT_BASE.uv_scale());
    let pole_col = [0.20, 0.20, 0.22];
    let head_col = [0.95, 0.88, 0.72];
    let pole_r = 0.09;
    // Pole at the roadside edge.
    push_box(
        v,
        i,
        [x - pole_r, 0.0, z - pole_r],
        [x + pole_r, LAMP_HEIGHT, z + pole_r],
        pole_col,
        slot,
        scale,
    );
    // Arm reaching inward from the pole top to the head over the road edge.
    let head_lateral = ROAD_HALF + 0.35;
    let head_x = x - side * (LAMP_LATERAL - head_lateral);
    let arm_h = LAMP_HEIGHT - 0.14;
    push_box(
        v,
        i,
        [x.min(head_x) - 0.06, arm_h, z - 0.06],
        [x.max(head_x) + 0.06, arm_h + 0.14, z + 0.06],
        pole_col,
        slot,
        scale,
    );
    // Luminaire head hanging from the arm end.
    push_box(
        v,
        i,
        [head_x - 0.16, arm_h - 0.16, z - 0.10],
        [head_x + 0.16, arm_h + 0.02, z + 0.10],
        head_col,
        slot,
        scale,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn street_lamps_placed_on_both_sides_every_spacing() {
        // One lamp on each side per LAMP_SPACING.
        let lamps = STREET_LAMP.placements(0.0, 260.0);
        let expected = (260.0 / LAMP_SPACING).ceil() as usize * 2;
        assert_eq!(lamps.len(), expected, "one lamp on each side per spacing");
        assert!(
            lamps.iter().any(|p| p.side > 0.0) && lamps.iter().any(|p| p.side < 0.0),
            "lamps appear on both sides"
        );
        assert!(
            lamps
                .iter()
                .all(|p| (p.lateral - LAMP_LATERAL).abs() < 1e-4),
            "lamp poles stand at LAMP_LATERAL"
        );
        assert_eq!(
            lamps,
            STREET_LAMP.placements(0.0, 260.0),
            "placement is deterministic"
        );
        // Boundary consistency: adjacent windows share no lamp and lose none.
        let a = STREET_LAMP.placements(0.0, 260.0);
        let b = STREET_LAMP.placements(260.0, 520.0);
        assert_eq!(a.len() + b.len(), STREET_LAMP.placements(0.0, 520.0).len());
    }

    #[test]
    fn head_pos_sits_at_the_luminaire_over_the_road_edge() {
        for p in STREET_LAMP.placements(0.0, 260.0) {
            let head = head_pos(&p);
            let lateral = head[0] - road_curve(-head[2]);
            assert!(
                (lateral.abs() - (ROAD_HALF + 0.35)).abs() < 1e-3,
                "head_pos must sit at the luminaire over the road edge"
            );
            assert!(head[1] > 5.2, "lamp head is elevated");
        }
    }

    #[test]
    fn chunk_mesh_builds_luminaire_heads_inward_of_the_poles() {
        let (v, _) =
            crate::mesh::build_world_chunk(0.0, 260.0, crate::mesh::TerrainDetail::Medium);
        // Head boxes hang just under the arm (y ≈ 5.30..5.48) and overhang the
        // road edge (lateral ≈ ±(ROAD_HALF + 0.35)), while poles stand at
        // ±LAMP_LATERAL. Check the luminaires exist and sit inward of the poles.
        let head_lateral = ROAD_HALF + 0.35;
        let heads_on_road: Vec<[f32; 3]> = v
            .iter()
            .filter(|vert| vert.position[1] > 5.28 && vert.position[1] < 5.50)
            .map(|vert| vert.position)
            .filter(|pos| {
                let lateral = (pos[0] - road_curve(-pos[2])).abs();
                lateral < head_lateral + 0.3 && lateral > head_lateral - 0.3
            })
            .collect();
        assert!(
            !heads_on_road.is_empty(),
            "lamp luminaires must hang over the road edge"
        );
        for pos in &heads_on_road {
            let lateral = pos[0] - road_curve(-pos[2]);
            assert!(
                lateral.abs() < LAMP_LATERAL,
                "lamp heads must sit inward of the pole, got lateral={lateral}"
            );
        }
    }
}
