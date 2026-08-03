// SPDX-License-Identifier: MIT

use glam::{Mat3, Mat4, Vec3};

use crate::vertex::Vertex3d;

const TARGET_CAR_LENGTH: f32 = 4.0;

pub fn load_gltf_mesh_from_bytes(
    data: &[u8],
    source_label: &str,
) -> Result<(Vec<Vertex3d>, Vec<u32>), String> {
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
) -> Result<(Vec<Vertex3d>, Vec<u32>), String> {

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
            &buffers,
            &mut vertices,
            &mut indices,
            source_label,
        )?;
    }

    if vertices.is_empty() || indices.is_empty() {
        return Err(format!("no renderable geometry in {source_label}"));
    }

    normalize_car_scale(&mut vertices);

    Ok((vertices, indices))
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
            for idx in 0..positions.len() {
                let pos = world_transform.transform_point3(Vec3::from(positions[idx]));
                let nrm = (normal_transform * Vec3::from(*normals.get(idx).unwrap_or(&[0.0, 1.0, 0.0])))
                    .normalize_or_zero();
                vertices.push(Vertex3d {
                    position: pos.to_array(),
                    normal: nrm.to_array(),
                    color: *vertex_colors.get(idx).unwrap_or(&[1.0, 1.0, 1.0]),
                    tex_coord: *tex_coords.get(idx).unwrap_or(&[0.0, 0.0]),
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
