// SPDX-License-Identifier: MIT

use glam::{Mat3, Mat4, Vec3};

use crate::math::mix;
use crate::vertex::Vertex3d;

const TARGET_CAR_LENGTH: f32 = 4.0;

/// Light anchor offsets (in world units, model space) derived from the car's
/// real geometry, used to place headlights/taillights on each model's actual
/// front/rear corners. Offsets are relative to the car's origin (the point the
/// renderer places on the road), with front facing -Z.
#[derive(Clone, Copy, Debug)]
pub struct CarLightAnchors {
    /// Lateral half-offset of each light from the centerline.
    pub lateral: f32,
    /// Distance from the center to the front/rear face (half the length).
    pub long_half: f32,
    /// Headlight height above the base (front corner fascia).
    pub headlight_y: f32,
    /// Taillight height above the base (rear corner fascia).
    pub taillight_y: f32,
}

pub fn load_gltf_mesh_from_bytes(
    data: &[u8],
    source_label: &str,
) -> Result<(Vec<Vertex3d>, Vec<u32>, CarLightAnchors), String> {
    let gltf = gltf::Gltf::from_slice(data)
        .map_err(|e| format!("failed to parse glTF {source_label}: {e}"))?;
    let mut buffers = Vec::new();
    for buffer in gltf.document.buffers() {
        match buffer.source() {
            gltf::buffer::Source::Bin => {
                let blob = gltf
                    .blob
                    .as_deref()
                    .ok_or_else(|| format!("missing GLB binary blob in {source_label}"))?;
                if blob.len() < buffer.length() {
                    return Err(format!(
                        "binary blob too small in {source_label}: {} < {}",
                        blob.len(),
                        buffer.length()
                    ));
                }
                buffers.push(gltf::buffer::Data(blob[..buffer.length()].to_vec()));
            }
            gltf::buffer::Source::Uri(uri) => {
                return Err(format!(
                    "external buffer URI '{uri}' in {source_label} is not supported for embedded loading"
                ));
            }
        }
    }

    load_from_document(&gltf.document, &buffers, source_label)
}

fn load_from_document(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    source_label: &str,
) -> Result<(Vec<Vertex3d>, Vec<u32>, CarLightAnchors), String> {

    let mut vertices = Vec::<Vertex3d>::new();
    let mut indices = Vec::<u32>::new();

    let scene = document
        .default_scene()
        .or_else(|| document.scenes().next())
        .ok_or_else(|| format!("no scene found in {source_label}"))?;

    for node in scene.nodes() {
        append_node_meshes(
            node,
            Mat4::IDENTITY,
            buffers,
            &mut vertices,
            &mut indices,
            source_label,
        )?;
    }

    if vertices.is_empty() || indices.is_empty() {
        return Err(format!("no renderable geometry in {source_label}"));
    }

    normalize_car_scale(&mut vertices);
    let anchors = compute_anchors(&vertices);

    Ok((vertices, indices, anchors))
}

fn append_node_meshes(
    node: gltf::Node<'_>,
    parent_transform: Mat4,
    buffers: &[gltf::buffer::Data],
    vertices: &mut Vec<Vertex3d>,
    indices: &mut Vec<u32>,
    source_label: &str,
) -> Result<(), String> {
    let local_transform = Mat4::from_cols_array_2d(&node.transform().matrix());
    let world_transform = parent_transform * local_transform;
    let normal_transform = Mat3::from_mat4(world_transform).inverse().transpose();

    if let Some(mesh) = node.mesh() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()].0));

            let positions: Vec<[f32; 3]> = reader
                .read_positions()
                .ok_or_else(|| format!("mesh without positions: {source_label}"))?
                .collect();

            let normals: Vec<[f32; 3]> = reader
                .read_normals()
                .map(|iter| iter.collect())
                .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]);

            let tex_coords: Vec<[f32; 2]> = reader
                .read_tex_coords(0)
                .map(|iter| iter.into_f32().collect())
                .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);

            let vertex_colors: Vec<[f32; 3]> = reader
                .read_colors(0)
                .map(|iter| iter.into_rgb_f32().collect())
                .unwrap_or_else(|| {
                    let base = primitive.material().pbr_metallic_roughness().base_color_factor();
                    vec![[base[0], base[1], base[2]]; positions.len()]
                });

            let base_index = vertices.len() as u32;
            for (idx, position) in positions.iter().enumerate() {
                let pos = world_transform.transform_point3(Vec3::from(*position));
                let nrm = (normal_transform
                    * Vec3::from(*normals.get(idx).unwrap_or(&[0.0, 1.0, 0.0])))
                .normalize_or_zero();
                vertices.push(Vertex3d {
                    position: pos.to_array(),
                    normal: nrm.to_array(),
                    color: *vertex_colors.get(idx).unwrap_or(&[1.0, 1.0, 1.0]),
                    tex_coord: *tex_coords.get(idx).unwrap_or(&[0.0, 0.0]),
                    // Car meshes use their own colormap texture, not the world atlas.
                    material: crate::surface::MAT_CAR,
                });
            }

            if let Some(read_indices) = reader.read_indices() {
                indices.extend(read_indices.into_u32().map(|idx| base_index + idx));
            } else {
                indices.extend((0..positions.len() as u32).map(|idx| base_index + idx));
            }
        }
    }

    for child in node.children() {
        append_node_meshes(
            child,
            world_transform,
            buffers,
            vertices,
            indices,
            source_label,
        )?;
    }

    Ok(())
}

fn normalize_car_scale(vertices: &mut [Vertex3d]) {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];

    for v in vertices.iter() {
        for k in 0..3 {
            min[k] = min[k].min(v.position[k]);
            max[k] = max[k].max(v.position[k]);
        }
    }

    let center_x = (min[0] + max[0]) * 0.5;
    let base_y = min[1];
    let center_z = (min[2] + max[2]) * 0.5;
    let model_len = (max[2] - min[2]).max(0.001);
    let scale = TARGET_CAR_LENGTH / model_len;

    for v in vertices.iter_mut() {
        v.position[0] = (v.position[0] - center_x) * scale;
        v.position[1] = (v.position[1] - base_y) * scale;
        v.position[2] = (v.position[2] - center_z) * scale;
        // Rotate 180° around Y so model front faces -Z (game forward direction)
        v.position[0] = -v.position[0];
        v.position[2] = -v.position[2];
        v.normal[0] = -v.normal[0];
        v.normal[2] = -v.normal[2];
    }
}

/// Derives the car light anchors from the normalized mesh so headlights and
/// taillights sit on the model's real front/rear corners. Expects vertices in
/// `normalize_car_scale` output space: centered, base at y=0, length 4.0, front
/// facing -Z. Headlight/taillight heights come from the vertical extent of the
/// front/rear corner fascias (the uppermost ~80% of the corner band).
fn compute_anchors(vertices: &[Vertex3d]) -> CarLightAnchors {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for v in vertices {
        for k in 0..3 {
            min[k] = min[k].min(v.position[k]);
            max[k] = max[k].max(v.position[k]);
        }
    }

    let half_width = (max[0] - min[0]) * 0.5;
    let half_len = (max[2] - min[2]) * 0.5;
    let band = (max[2] - min[2]) * 0.15;
    let corner_x = half_width * 0.7;

    // Vertical extent of the corner fascia between two z-bounds.
    let corner_y = |z_lo: f32, z_hi: f32| -> (f32, f32) {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for v in vertices {
            if v.position[0].abs() < corner_x || v.position[2] < z_lo || v.position[2] > z_hi {
                continue;
            }
            lo = lo.min(v.position[1]);
            hi = hi.max(v.position[1]);
        }
        (lo, hi)
    };

    let (f_lo, f_hi) = corner_y(min[2], min[2] + band);
    let (r_lo, r_hi) = corner_y(max[2] - band, max[2]);
    let body_height = (max[1] - min[1]).max(0.001);

    let fallback = body_height * 0.5;
    CarLightAnchors {
        lateral: half_width * 0.8,
        long_half: half_len * 0.92,
        headlight_y: if f_lo.is_finite() {
            mix(f_lo, f_hi, 0.8)
        } else {
            fallback
        },
        taillight_y: if r_lo.is_finite() {
            mix(r_lo, r_hi, 0.8)
        } else {
            fallback
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vertex(x: f32, y: f32, z: f32) -> Vertex3d {
        Vertex3d {
            position: [x, y, z],
            normal: [0.0, 1.0, 0.0],
            color: [1.0, 1.0, 1.0],
            tex_coord: [0.0, 0.0],
            material: 4.0,
        }
    }

    #[test]
    fn anchors_from_synthetic_car_match_expected() {
        // A box 2.0 wide, 4.0 long, base at y=0, front at -Z. The corner
        // fascias rise to y=0.6 on both ends.
        let mut verts = Vec::new();
        for z in [-2.0f32, 2.0] {
            for x in [-1.0f32, 1.0] {
                for y in [0.0f32, 0.4, 0.5, 0.6] {
                    verts.push(vertex(x, y, z));
                }
            }
        }
        let a = compute_anchors(&verts);
        assert_eq!(a.lateral, 0.8);
        assert_eq!(a.long_half, 1.84);
        assert!((a.headlight_y - 0.48).abs() < 1e-4, "front corner band 0..0.6");
        assert!((a.taillight_y - 0.48).abs() < 1e-4, "rear corner band 0..0.6");
    }

    #[test]
    fn anchors_fall_back_to_mid_height_without_corners() {
        // Wide rear body but a narrow nose: no verts in the front corner band
        // (|x| > 0.7 * half-width), so the headlight height falls back.
        let mut verts = Vec::new();
        for x in [-1.0f32, 1.0] {
            for y in [0.0f32, 1.0] {
                for z in [0.0f32, 2.0] {
                    verts.push(vertex(x, y, z));
                }
            }
        }
        // Narrow nose at the front, only near the centerline.
        for x in [-0.1f32, 0.1] {
            verts.push(vertex(x, 0.5, -2.0));
        }
        let a = compute_anchors(&verts);
        assert!((a.lateral - 0.8).abs() < 1e-4, "half-width 1.0 * 0.8");
        assert_eq!(a.headlight_y, 0.5, "falls back to half body height");
        assert!((a.taillight_y - 0.8).abs() < 1e-4, "rear corners 0..1 -> mix 0.8");
    }
}
