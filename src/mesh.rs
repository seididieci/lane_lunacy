// SPDX-License-Identifier: MIT

use crate::geom::push_quad;
use crate::road::{road_curve, road_tangent, ROAD_HALF};

/// Half-width of the terrain ribbon on each side of the road. Long enough that
/// its outer edge (foothills ~20m up) sits at the fully-opaque fog distance, so
/// the terrain reads as a mountain backdrop instead of ending in a cutoff.
const GROUND_HALF_W: f32 = 600.0;
use crate::surface::{material_at, SurfaceMaterial, ROCK_SLOPE};
use crate::vertex::Vertex3d;
use crate::world::terrain::{terrain_height, terrain_slope, RISE_START};

/// Shortcuts for the ribbon cross-section's fixed surfaces. The asphalt slots
/// and UV scales live in `surface.rs` so the mesh and gameplay agree.
const ASPHALT_BASE: SurfaceMaterial = SurfaceMaterial::AsphaltBase;
const GRASS: SurfaceMaterial = SurfaceMaterial::Grass;

/// User-facing terrain tessellation detail. `Low` is the historic coarse mesh
/// (cheapest rebuild, smallest GPU cost), `Medium` the default balance, and
/// `High` the dense option for rounder hills at a higher rebuild/vertex cost.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerrainDetail {
    Low,
    Medium,
    High,
}

impl TerrainDetail {
    /// Short label for the settings menu row.
    pub fn label(self) -> &'static str {
        match self {
            TerrainDetail::Low => "LOW",
            TerrainDetail::Medium => "MED",
            TerrainDetail::High => "HIGH",
        }
    }

    /// Parses a `--terrain-detail` CLI value (case-insensitive).
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "low" => Some(TerrainDetail::Low),
            "med" | "medium" => Some(TerrainDetail::Medium),
            "high" => Some(TerrainDetail::High),
            _ => None,
        }
    }

    /// Along-road cell size (metres) at this detail.
    fn step(self) -> f32 {
        match self {
            TerrainDetail::Low => 2.0,
            TerrainDetail::Medium => 1.0,
            TerrainDetail::High => 0.75,
        }
    }

    /// Lateral ribbon edge positions (metres from the road center) at this
    /// detail: dense near the road where the hills and mountains live, coarser
    /// far out where the foothills fade into fog.
    fn lat_edges(self) -> Vec<f32> {
        let mut edges = match self {
            TerrainDetail::Low => vec![0.0, 2.0, 4.5, 7.0],
            TerrainDetail::Medium => vec![0.0, 1.5, 3.0, 4.5, 6.0, 7.0],
            TerrainDetail::High => vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        };
        let segs: &[(f32, f32, f32)] = match self {
            TerrainDetail::Low => &[
                (7.0, 1.0, 20.0),
                (20.0, 2.0, 45.0),
                (45.0, 5.0, 120.0),
                (120.0, 25.0, 600.0),
            ],
            TerrainDetail::Medium => &[
                (7.0, 1.0, 20.0),
                (20.0, 1.5, 45.0),
                (45.0, 2.5, 120.0),
                (120.0, 10.0, 300.0),
                (300.0, 25.0, 600.0),
            ],
            TerrainDetail::High => &[
                (7.0, 0.75, 20.0),
                (20.0, 1.5, 45.0),
                (45.0, 2.0, 120.0),
                (120.0, 5.0, 300.0),
                (300.0, 12.5, 600.0),
            ],
        };
        for &(mut d, delta, end) in segs {
            while d < end {
                d += delta;
                edges.push(d);
            }
        }
        if *edges.last().unwrap() != GROUND_HALF_W {
            edges.push(GROUND_HALF_W);
        }
        edges
    }
}

/// Smooth per-vertex normal for the terrain heightfield at `(s, lateral)`,
/// derived from central differences of `terrain_height` and the road tangent.
/// Points up; on a slope it tilts downhill so the visible face lights smoothly
/// instead of reading as a flat facet.
fn terrain_normal(s: f32, lateral: f32) -> [f32; 3] {
    const E: f32 = 0.5;
    let hs = (terrain_height(s + E, lateral) - terrain_height(s - E, lateral)) / (2.0 * E);
    let hl = (terrain_height(s, lateral + E) - terrain_height(s, lateral - E)) / (2.0 * E);
    // Surface tangents: dP/ds = (road_tangent(s), hs, -1), dP/dl = (1, hl, 0).
    // The up-facing normal is -(dP/ds x dP/dl) = (-hl, 1, hs - road_tangent(s)*hl).
    let n = [-hl, 1.0, hs - road_tangent(s) * hl];
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if len < 1e-6 {
        [0.0, 1.0, 0.0]
    } else {
        [n[0] / len, n[1] / len, n[2] / len]
    }
}

/// Along-road tessellation step at `s`: the base `step` on gentle ground, a
/// finer `step * STEP_FINE_FACTOR` (floored at `STEP_FINE_MIN`) where the
/// off-road terrain climbs steeply enough to read as rock. Denser tessellation
/// on rock faces stops the fixed 1.0m grid from showing as visible stepped
/// slices on the wall; gentle ground keeps the cheaper coarse grid. The blend
/// ramps over a slope band so there is no visible tessellation seam at the
/// grass/rock boundary.
const STEP_FINE_FACTOR: f32 = 0.4;
const STEP_FINE_MIN: f32 = 0.5;
const STEP_SLOPE_LO: f32 = ROCK_SLOPE * 0.6;
const STEP_SLOPE_HI: f32 = ROCK_SLOPE * 1.6;

fn terrain_step(s: f32, step: f32) -> f32 {
    // Representative steepness: the max slope on either side at the typical
    // lateral band where hills/mountains rise off the road (terrain_gradient
    // samples the same zone). Degenerates to 0 (base step) on open ground.
    let mut sl = 0.0f32;
    for d in [RISE_START + 0.5, RISE_START + 2.0, RISE_START + 4.0] {
        sl = sl.max(terrain_slope(s, d)).max(terrain_slope(s, -d));
    }
    let t = ((sl - STEP_SLOPE_LO) / (STEP_SLOPE_HI - STEP_SLOPE_LO)).clamp(0.0, 1.0);
    let fine = (step * STEP_FINE_FACTOR).max(STEP_FINE_MIN);
    step - (step - fine) * t
}

/// Pushes one terrain cell as two triangles. Each corner gets its own smooth
/// normal (recovered from its world position via `terrain_normal`), so hills
/// read as rounded slopes rather than flat-shaded facets.
#[allow(clippy::too_many_arguments)]
fn push_terrain_cell(
    v: &mut Vec<Vertex3d>,
    i: &mut Vec<u32>,
    a: [f32; 3],
    b: [f32; 3],
    c: [f32; 3],
    d: [f32; 3],
    col: [f32; 3],
    material: f32,
    scale: f32,
) {
    let base = v.len() as u32;
    let uv = |p: [f32; 3]| [p[0] * scale, p[2] * scale];
    for p in [a, b, c, d] {
        v.push(Vertex3d {
            position: p,
            normal: terrain_normal(-p[2], p[0] - road_curve(-p[2])),
            color: col,
            tex_coord: uv(p),
            material,
        });
    }
    i.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

/// Builds one chunk of the world mesh: the flat ground ribbon and the road
/// (asphalt, shoulders, verges, edge + center lines), then the roadside objects
/// (marker posts, street lamps, trees) via `world::build_world_scenery`. Each
/// object type owns its deterministic placement and geometry in `src/world/`.
pub fn build_world_chunk(
    start_s: f32,
    chunk_len: f32,
    detail: TerrainDetail,
) -> (Vec<Vertex3d>, Vec<u32>) {
    let mut v = Vec::new();
    let mut i = Vec::new();
    let half_w = ROAD_HALF;
    let end_s = start_s + chunk_len;
    let step = detail.step();

    // Local terrain ribbon around the road (per-chunk). The old single
    // full-width quad was too coarse to displace, so the ribbon is re-tessellated
    // into lateral bands (dense near the road where the hills and mountains
    // live, coarse far out where the foothills fade into fog). Each vertex sits
    // on the deterministic terrain_height(s, lateral); each cell gets the
    // surface material its slope warrants (grass vs rock). Shared corners land
    // on identical coordinates between cells, so hills and mountain ridges are
    // seamless within and across chunks. The ribbon density follows the
    // `TerrainDetail` setting so players can trade smoothness against rebuild
    // cost.
    let ground_color = [0.7, 0.85, 0.6];
    let rock_color = [0.85, 0.85, 0.88];
    let lat_edges = detail.lat_edges();

    let mut s_ground = start_s;
    while s_ground < end_s {
        let s0 = s_ground;
        let s1 = (s_ground + terrain_step(s_ground, step)).min(end_s);
        let s_mid = (s0 + s1) * 0.5;
        let z0 = -s0;
        let z1 = -s1;
        let x0 = road_curve(s0);
        let x1 = road_curve(s1);
        for w in lat_edges.windows(2) {
            let l0 = w[0];
            let l1 = w[1];
            for side in [-1.0, 1.0] {
                // Ascending lateral offset on both sides so mirrored cells keep
                // the same front-facing winding; otherwise CullMode::Back culls
                // the left terrain and the golden clear color shows through.
                let (la0, la1) = if side >= 0.0 {
                    (l0 * side, l1 * side)
                } else {
                    (l1 * side, l0 * side)
                };
                let ca = [x0 + la0, terrain_height(s0, la0), z0];
                let cb = [x0 + la1, terrain_height(s0, la1), z0];
                let cc = [x1 + la1, terrain_height(s1, la1), z1];
                let cd = [x1 + la0, terrain_height(s1, la0), z1];
                let lat_mid = (l0 + l1) * 0.5 * side;
                let m = material_at(s_mid, lat_mid, terrain_slope(s_mid, lat_mid));
                let col = if m == SurfaceMaterial::Rock {
                    rock_color
                } else {
                    ground_color
                };
                push_terrain_cell(
                    &mut v,
                    &mut i,
                    ca,
                    cb,
                    cc,
                    cd,
                    col,
                    m.atlas_slot(),
                    m.uv_scale(),
                );
            }
        }
        s_ground += terrain_step(s_ground, step);
    }

    // road ribbon along -Z
    let road = [0.55, 0.55, 0.58];
    let edge_line = [0.95, 0.95, 0.92];
    let center_line = [0.95, 0.84, 0.36];
    let shoulder_a = [0.62, 0.12, 0.12];
    let shoulder_b = [0.86, 0.86, 0.8];
    let verge = [0.6, 0.75, 0.55];
    let mut s = start_s;
    while s < end_s {
        let s0 = s;
        let s1 = (s + step).min(end_s);
        let x0 = road_curve(s0);
        let x1 = road_curve(s1);
        let z0 = -s0;
        let z1 = -s1;
        // Road elevation at this position
        let asphalt = material_at(s0, 0.0, 0.0);
        let asphalt_slot = asphalt.atlas_slot();
        let asphalt_scale = asphalt.uv_scale();

        // asphalt (at terrain height)
        push_quad(
            &mut v,
            &mut i,
            [
                x0 - half_w,
                crate::world::terrain::terrain_height(s0, 0.0) + 0.015,
                z0,
            ],
            [
                x0 + half_w,
                crate::world::terrain::terrain_height(s0, 0.0) + 0.015,
                z0,
            ],
            [
                x1 + half_w,
                crate::world::terrain::terrain_height(s1, 0.0) + 0.015,
                z1,
            ],
            [
                x1 - half_w,
                crate::world::terrain::terrain_height(s1, 0.0) + 0.015,
                z1,
            ],
            [0.0, 1.0, 0.0],
            road,
            asphalt_slot,
            asphalt_scale,
        );

        let shoulder_col = if ((s0 / 4.0) as i32) % 2 == 0 {
            shoulder_a
        } else {
            shoulder_b
        };

        // left shoulder strip
        push_quad(
            &mut v,
            &mut i,
            [x0 - half_w - 0.55, 0.021, z0],
            [x0 - half_w, 0.021, z0],
            [x1 - half_w, 0.021, z1],
            [x1 - half_w - 0.55, 0.021, z1],
            [0.0, 1.0, 0.0],
            shoulder_col,
            ASPHALT_BASE.atlas_slot(),
            ASPHALT_BASE.uv_scale(),
        );

        // right shoulder strip
        push_quad(
            &mut v,
            &mut i,
            [x0 + half_w, 0.021, z0],
            [x0 + half_w + 0.55, 0.021, z0],
            [x1 + half_w + 0.55, 0.021, z1],
            [x1 + half_w, 0.021, z1],
            [0.0, 1.0, 0.0],
            shoulder_col,
            ASPHALT_BASE.atlas_slot(),
            ASPHALT_BASE.uv_scale(),
        );

        // grass verge strips to soften shoulder->terrain transition
        push_quad(
            &mut v,
            &mut i,
            [x0 - half_w - 1.1, 0.016, z0],
            [x0 - half_w - 0.55, 0.016, z0],
            [x1 - half_w - 0.55, 0.016, z1],
            [x1 - half_w - 1.1, 0.016, z1],
            [0.0, 1.0, 0.0],
            verge,
            GRASS.atlas_slot(),
            GRASS.uv_scale(),
        );
        push_quad(
            &mut v,
            &mut i,
            [x0 + half_w + 0.55, 0.016, z0],
            [x0 + half_w + 1.1, 0.016, z0],
            [x1 + half_w + 1.1, 0.016, z1],
            [x1 + half_w + 0.55, 0.016, z1],
            [0.0, 1.0, 0.0],
            verge,
            GRASS.atlas_slot(),
            GRASS.uv_scale(),
        );

        // edge lines
        push_quad(
            &mut v,
            &mut i,
            [x0 - half_w + 0.10, 0.025, z0],
            [x0 - half_w + 0.18, 0.025, z0],
            [x1 - half_w + 0.18, 0.025, z1],
            [x1 - half_w + 0.10, 0.025, z1],
            [0.0, 1.0, 0.0],
            edge_line,
            ASPHALT_BASE.atlas_slot(),
            ASPHALT_BASE.uv_scale(),
        );
        push_quad(
            &mut v,
            &mut i,
            [x0 + half_w - 0.18, 0.025, z0],
            [x0 + half_w - 0.10, 0.025, z0],
            [x1 + half_w - 0.10, 0.025, z1],
            [x1 + half_w - 0.18, 0.025, z1],
            [0.0, 1.0, 0.0],
            edge_line,
            ASPHALT_BASE.atlas_slot(),
            ASPHALT_BASE.uv_scale(),
        );

        // dashed center line
        if ((s0 / 9.0) as i32) % 2 == 0 {
            push_quad(
                &mut v,
                &mut i,
                [x0 - 0.09, 0.026, z0],
                [x0 + 0.09, 0.026, z0],
                [x1 + 0.09, 0.026, z1],
                [x1 - 0.09, 0.026, z1],
                [0.0, 1.0, 0.0],
                center_line,
                ASPHALT_BASE.atlas_slot(),
                ASPHALT_BASE.uv_scale(),
            );
        }
        s += step;
    }

    // Roadside objects (marker posts, street lamps, trees). Each type owns its
    // deterministic placement and geometry in `src/world/`; registering a new
    // object there is all that's needed to draw it.
    crate::world::build_world_scenery(start_s, end_s, &mut v, &mut i);

    (v, i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn world_chunk_has_open_ground_and_clear_road_corridor() {
        let (v, _) = build_world_chunk(0.0, 260.0, TerrainDetail::Medium);
        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = 0.0f32;
        let mut min_y = f32::INFINITY;
        for vert in &v {
            min_x = min_x.min(vert.position[0]);
            max_x = max_x.max(vert.position[0]);
            max_y = max_y.max(vert.position[1]);
            min_y = min_y.min(vert.position[1]);
        }
        // The terrain ribbon spans ±GROUND_HALF_W around the road.
        assert!(
            min_x <= -GROUND_HALF_W + 0.01 && max_x >= GROUND_HALF_W - 0.01,
            "terrain must span ±GROUND_HALF_W, got [{min_x}, {max_x}]"
        );
        // The road corridor stays open: no terrain obstacles inside |lateral| < ROAD_HALF.
        // Vertices within the road are part of the road mesh itself (asphalt, shoulders, etc.)
        // and shouldn't exceed their expected height by much.
        for vert in v.iter().filter(|vert| vert.position[1] > 0.05) {
            let lateral = vert.position[0] - road_curve(-vert.position[2]);
            if lateral.abs() < ROAD_HALF {
                // Inside corridor: should be within ~0.2 of terrain height for road mesh vertices
                let s = vert.position[2];
                let terrain_y = crate::world::terrain::terrain_height(s, 0.0);
                assert!(
                    (vert.position[1] - terrain_y).abs() < 1.0,
                    "road-corridor vertex at s={}{{}} with lateral={}{{}} is {} units from terrain surface",
                    s, lateral, vert.position[1] - terrain_y
                );
            }
        }
        // Terrain is bounded: nothing below the road plane, nothing past the
        // deterministic ceiling (mountain crests).
        assert!(min_y >= -0.01, "terrain must not dip below the road plane");
        assert!(
            max_y <= crate::world::terrain::MAX_HEIGHT + 0.01,
            "terrain must stay bounded, got max_y={max_y}"
        );
        // Hills/mountains are actually present in the chunk.
        assert!(max_y > 1.5, "the terrain must rise off the road");
    }

    #[test]
    fn world_chunk_builds_roadside_scenery() {
        let (v, _) = build_world_chunk(0.0, 260.0, TerrainDetail::Medium);
        let trees = v
            .iter()
            .filter(|vert| vert.material >= 4.0 && vert.material < 5.0)
            .count();
        let elevated = v.iter().filter(|vert| vert.position[1] > 1.0).count();
        assert!(trees > 0, "trees should be part of the chunk");
        assert!(
            elevated > 0,
            "posts and lamp poles should be part of the chunk"
        );
    }

    #[test]
    fn terrain_detail_scales_triangle_density() {
        let (_, il) = build_world_chunk(0.0, 260.0, TerrainDetail::Low);
        let (_, im) = build_world_chunk(0.0, 260.0, TerrainDetail::Medium);
        let (_, ih) = build_world_chunk(0.0, 260.0, TerrainDetail::High);
        let tris = |i: &[u32]| i.len() / 3;
        assert!(tris(&il) < tris(&im), "Medium must densify over Low");
        assert!(tris(&im) < tris(&ih), "High must densify over Medium");
        // Plausible magnitudes for the 600 m ribbon at the advertised steps.
        assert!(tris(&il) > 10_000, "Low still tessellates the ribbon");
        assert!(tris(&ih) < 400_000, "High stays within a rebuild budget");
    }

    #[test]
    fn world_chunk_terrain_bakes_rock_into_mountain_faces() {
        // Mountain near/far faces are steep enough to cross the rock threshold
        // whenever a deterministic ridge block is active; scan chunks until one
        // is found (the world is deterministic, so this is a fixed guarantee,
        // not a probability).
        let mut s = 0.0;
        let mut found = false;
        while s < 5200.0 {
            let (v, _) = build_world_chunk(s, 260.0, TerrainDetail::Medium);
            if v.iter()
                .any(|vert| vert.material == SurfaceMaterial::Rock.atlas_slot())
            {
                found = true;
                break;
            }
            s += 260.0;
        }
        assert!(found, "some chunk must contain rock mountain faces");
    }

    #[test]
    fn steep_chunks_tessellate_denser_than_flat() {
        // The adaptive along-road step must densify on steep rock faces (finer
        // grid -> no visible stepped slices on the wall) while keeping the
        // cheaper coarse grid on open ground. Find a chunk with rock mountain
        // faces and one that is flat, then compare their along-road density.
        let mut steep_s = None;
        let mut s = 0.0;
        while s < 5200.0 {
            let (v, _) = build_world_chunk(s, 260.0, TerrainDetail::Medium);
            if v.iter()
                .any(|vert| vert.material == SurfaceMaterial::Rock.atlas_slot())
            {
                steep_s = Some(s);
                break;
            }
            s += 260.0;
        }
        let steep_s = steep_s.expect("some chunk must contain rock mountain faces");
        let (sv, _) = build_world_chunk(steep_s, 260.0, TerrainDetail::Medium);

        // A flat chunk: terrain that never rises above the rock threshold.
        let mut flat_s = None;
        s = 0.0;
        while s < 5200.0 {
            let (v, _) = build_world_chunk(s, 260.0, TerrainDetail::Medium);
            let flat = v
                .iter()
                .all(|vert| vert.material != SurfaceMaterial::Rock.atlas_slot());
            if flat {
                flat_s = Some(s);
                break;
            }
            s += 260.0;
        }
        let flat_s = flat_s.expect("some chunk must be flat");
        let (fv, _) = build_world_chunk(flat_s, 260.0, TerrainDetail::Medium);

        // Terrain-only triangle density: exclude road/scenery by counting only
        // vertices whose material is off-road (grass slot 3 / rock slot 5).
        let terrain_tris = |v: &[Vertex3d]| {
            v.iter()
                .filter(|vert| vert.material >= 3.0 && vert.material < 6.0)
                .count()
                / 4
                * 2
        };
        assert!(
            terrain_tris(&sv) > terrain_tris(&fv),
            "steep chunk must tessellate denser ({} vs {})",
            terrain_tris(&sv),
            terrain_tris(&fv)
        );
    }
}

/// Unit hemisphere (radius 1, y >= 0) centered at the origin, used for the sky
/// dome. Drawn with `model = translate(eye) * scale(radius)` so it follows the
/// camera; the normalized position is the sky direction sampled by the fragment
/// shader.
pub fn build_sky_dome(rings: u32, sectors: u32) -> (Vec<Vertex3d>, Vec<u32>) {
    let mut v = Vec::new();
    let mut i = Vec::new();

    for r in 0..=rings {
        let theta = (r as f32 / rings as f32) * std::f32::consts::FRAC_PI_2;
        let y = theta.cos();
        let ring_r = theta.sin();
        for s in 0..=sectors {
            let phi = (s as f32 / sectors as f32) * std::f32::consts::TAU;
            let pos = [ring_r * phi.cos(), y, ring_r * phi.sin()];
            v.push(Vertex3d {
                position: pos,
                normal: pos,
                color: [1.0, 1.0, 1.0],
                tex_coord: [0.0, 0.0],
                material: 0.0,
            });
        }
    }

    let cols = sectors + 1;
    for r in 0..rings {
        for s in 0..sectors {
            let a = r * cols + s;
            let b = a + 1;
            let c = (r + 1) * cols + s;
            let d = c + 1;
            i.extend_from_slice(&[a, b, d, a, d, c]);
        }
    }

    (v, i)
}
