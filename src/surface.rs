// SPDX-License-Identifier: MIT

//! Road surface material model shared between the world mesh and gameplay
//! systems (e.g. per-surface drift dust).
//!
//! A `SurfaceMaterial` maps 1:1 to a slot in the world texture atlas and
//! carries a dust profile. Adding a new surface (e.g. gravel) is one enum
//! variant plus an atlas slot, a texture, and a `DustProfile`.

use crate::road::ROAD_HALF;

/// Atlas sentinel for the car colormap path (not a world surface). Kept far
/// above the surface slots so new surfaces can be appended without conflicts.
pub const MAT_CAR: f32 = 99.0;

/// Shoulder strip width (metres), mirroring `mesh.rs`'s cross-section.
const SHOULDER_W: f32 = 0.55;
/// Off-road terrain steeper than this (rise/m) renders as rock instead of grass.
pub const ROCK_SLOPE: f32 = 0.8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceMaterial {
    AsphaltBase,
    AsphaltWorn,
    AsphaltCracked,
    Grass,
    Rock,
}

impl SurfaceMaterial {
    /// Atlas slot (0..n) used by `mesh.frag.glsl` for the world texture.
    pub fn atlas_slot(self) -> f32 {
        match self {
            SurfaceMaterial::AsphaltBase => 0.0,
            SurfaceMaterial::AsphaltWorn => 1.0,
            SurfaceMaterial::AsphaltCracked => 2.0,
            SurfaceMaterial::Grass => 3.0,
            SurfaceMaterial::Rock => 5.0,
        }
    }

    /// World-space UV scale (tiles per metre) for this surface.
    pub fn uv_scale(self) -> f32 {
        match self {
            SurfaceMaterial::AsphaltBase
            | SurfaceMaterial::AsphaltWorn
            | SurfaceMaterial::AsphaltCracked => 0.32,
            SurfaceMaterial::Grass => 0.10,
            // Larger tiles on big cliff faces so the rock doesn't swim.
            SurfaceMaterial::Rock => 0.08,
        }
    }

    /// Dust kicked up by drifting over this surface. `emission` is a 0..1
    /// multiplier on the drift-driven spawn rate; the color tint dresses the
    /// puffs so each surface reads differently.
    pub fn dust_profile(self) -> DustProfile {
        match self {
            SurfaceMaterial::AsphaltBase => DustProfile {
                emission: 0.35,
                color: [0.45, 0.42, 0.40],
                puff_scale: 1.0,
                alpha: 0.75,
            },
            SurfaceMaterial::AsphaltWorn => DustProfile {
                emission: 0.6,
                color: [0.48, 0.44, 0.41],
                puff_scale: 1.05,
                alpha: 0.8,
            },
            SurfaceMaterial::AsphaltCracked => DustProfile {
                emission: 0.8,
                color: [0.5, 0.45, 0.42],
                puff_scale: 1.1,
                alpha: 0.85,
            },
            SurfaceMaterial::Grass => DustProfile {
                emission: 0.45,
                color: [0.4, 0.45, 0.34],
                puff_scale: 1.2,
                alpha: 0.75,
            },
            SurfaceMaterial::Rock => DustProfile {
                emission: 0.2,
                color: [0.42, 0.4, 0.38],
                puff_scale: 1.1,
                alpha: 0.6,
            },
        }
    }
}

/// Visual/behavioral dust characteristics of a surface.
#[derive(Clone, Copy, Debug)]
pub struct DustProfile {
    pub emission: f32,
    pub color: [f32; 3],
    pub puff_scale: f32,
    pub alpha: f32,
}

/// The surface under a point of the road cross-section, expressed as a
/// distance along the ribbon plus a lateral offset from the center line.
/// `slope` (rise per metre) only matters beyond the shoulder, where the
/// surrounding terrain is grass unless it's steep enough to read as rock.
/// Mirrors the geometry built in `mesh.rs` so gameplay queries (drift dust)
/// always agree with what is actually drawn.
pub fn material_at(distance: f32, offset: f32, slope: f32) -> SurfaceMaterial {
    let abs_x = offset.abs();
    if abs_x <= ROAD_HALF {
        // Asphalt ribbon: pick the per-block variant deterministically.
        asphalt_variant(distance)
    } else if abs_x <= ROAD_HALF + SHOULDER_W {
        // Shoulder strips are always asphalt base.
        SurfaceMaterial::AsphaltBase
    } else if slope > ROCK_SLOPE {
        // Steep off-road terrain (cliff faces, escarpments) reads as rock.
        SurfaceMaterial::Rock
    } else {
        // Verge and the surrounding ground are grass.
        SurfaceMaterial::Grass
    }
}

/// Picks an asphalt variant per long block so worn/cracked stretches appear
/// occasionally and subtly instead of alternating every few metres.
fn asphalt_variant(s: f32) -> SurfaceMaterial {
    let block = (s / 96.0).floor() as i32;
    let h = block.wrapping_mul(1_103_515_245).wrapping_add(12_345);
    let r = ((h >> 16) & 0x7fff) as f32 / 32767.0;
    if r < 0.03 {
        SurfaceMaterial::AsphaltCracked
    } else if r < 0.15 {
        SurfaceMaterial::AsphaltWorn
    } else {
        SurfaceMaterial::AsphaltBase
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inside_the_ribbon_is_asphalt() {
        for offset in [-3.0, 0.0, 4.7] {
            let m = material_at(0.0, offset, 0.0);
            assert!(
                matches!(
                    m,
                    SurfaceMaterial::AsphaltBase
                        | SurfaceMaterial::AsphaltWorn
                        | SurfaceMaterial::AsphaltCracked
                ),
                "offset {offset} must be asphalt"
            );
        }
    }

    #[test]
    fn shoulder_is_asphalt_and_beyond_is_grass() {
        assert_eq!(
            material_at(0.0, ROAD_HALF + 0.2, 0.0),
            SurfaceMaterial::AsphaltBase
        );
        assert_eq!(
            material_at(0.0, ROAD_HALF + 1.0, 0.0),
            SurfaceMaterial::Grass
        );
        assert_eq!(material_at(0.0, 50.0, 0.0), SurfaceMaterial::Grass);
    }

    #[test]
    fn steep_off_road_terrain_is_rock() {
        assert_eq!(
            material_at(0.0, ROAD_HALF + 1.0, ROCK_SLOPE + 0.1),
            SurfaceMaterial::Rock
        );
        // The slope only matters beyond the shoulder: the ribbon is asphalt
        // regardless of how steep the terrain beyond it is.
        assert_eq!(material_at(0.0, 0.0, 10.0), material_at(0.0, 0.0, 0.0));
    }

    #[test]
    fn variant_pick_is_deterministic() {
        assert_eq!(material_at(0.0, 0.0, 0.0), material_at(0.0, 0.0, 0.0));
        assert_eq!(material_at(1234.0, 1.0, 0.0), material_at(1234.0, 1.0, 0.0));
    }

    #[test]
    fn atlas_slots_are_stable_and_uv_scale_matches_surface() {
        assert_eq!(SurfaceMaterial::AsphaltBase.atlas_slot(), 0.0);
        assert_eq!(SurfaceMaterial::AsphaltWorn.atlas_slot(), 1.0);
        assert_eq!(SurfaceMaterial::AsphaltCracked.atlas_slot(), 2.0);
        assert_eq!(SurfaceMaterial::Grass.atlas_slot(), 3.0);
        assert_eq!(SurfaceMaterial::Rock.atlas_slot(), 5.0);
        assert!(SurfaceMaterial::Grass.uv_scale() < SurfaceMaterial::AsphaltBase.uv_scale());
    }

    #[test]
    fn dust_profile_emission_stays_within_unit_range() {
        for m in [
            SurfaceMaterial::AsphaltBase,
            SurfaceMaterial::AsphaltWorn,
            SurfaceMaterial::AsphaltCracked,
            SurfaceMaterial::Grass,
            SurfaceMaterial::Rock,
        ] {
            let p = m.dust_profile();
            assert!((0.0..=1.0).contains(&p.emission));
            assert!(p.alpha > 0.0);
        }
        assert!(
            SurfaceMaterial::AsphaltCracked.dust_profile().emission
                > SurfaceMaterial::AsphaltBase.dust_profile().emission
        );
    }
}
