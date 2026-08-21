// SPDX-License-Identifier: MIT
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use shaderc::{CompileOptions, Compiler, EnvVersion, ShaderKind, TargetEnv};

/// The glslang bundled with shaderc 0.10.x omits the `Location` decoration on
/// `RayPayloadKHR`/`IncomingRayPayloadKHR` variables (an old SPIR-V behavior),
/// but Vulkan ray tracing requires it to link the payload interface across the
/// raygen / closest-hit / miss stages. Without it the driver cannot match the
/// payloads and the raygen reads back garbage. Patch the missing decorations
/// into the binary after compilation.
///
/// SPIR-V constants (from `spirv.h`): the module is little-endian 32-bit words,
/// each instruction is `(word_count << 16) | opcode` followed by its operands.
const SPV_OP_VARIABLE: u16 = 59;
const SPV_OP_FUNCTION: u16 = 54;
const SPV_OP_DECORATE: u16 = 71;
const SPV_STORAGE_CLASS_RAY_PAYLOAD_KHR: u32 = 5338;
const SPV_STORAGE_CLASS_INCOMING_RAY_PAYLOAD_KHR: u32 = 5342;
const SPV_DECORATION_LOCATION: u32 = 30;

fn patch_ray_payload_locations(spv: &[u8]) -> Vec<u8> {
    let words: Vec<u32> = spv
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let mut payload_ids: Vec<u32> = Vec::new();
    let mut i = 5; // after the 5-word module header
    // The annotations section (where OpDecorate lives) ends where the type
    // declarations begin. Module-scope variables (including the payload) come
    // after the types and before the first function.
    let mut first_type = words.len();
    while i < words.len() {
        let word = words[i];
        let word_count = (word >> 16) as usize;
        let opcode = (word & 0xffff) as u16;
        if opcode == SPV_OP_FUNCTION {
            break;
        }
        if first_type == words.len() && (19..=33).contains(&opcode) {
            first_type = i;
        }
        if opcode == SPV_OP_VARIABLE && word_count >= 4 {
            let storage_class = words[i + 3];
            if storage_class == SPV_STORAGE_CLASS_RAY_PAYLOAD_KHR
                || storage_class == SPV_STORAGE_CLASS_INCOMING_RAY_PAYLOAD_KHR
            {
                payload_ids.push(words[i + 2]);
            }
        }
        i += word_count;
    }

    if payload_ids.is_empty() || first_type == words.len() {
        return spv.to_vec();
    }

    // `OpDecorate <id> Location N` per payload variable, all inserted at the
    // start of the types section (the end of the annotations section) BEFORE
    // the type declarations (inserting before the type section header would
    // shift the type IDs every decoration references). Every RT stage declares
    // exactly one `RTShade` payload variable, so each module gets a single
    // `Location 0` decoration and the driver links them all to the raygen's
    // `layout(location = 0)`. The shadow any-hit/miss stages compile with the
    // same struct, so glslang links the shared payload without explicit
    // locations (glslang rejects `layout(location=...)` on `rayPayloadEXT` in
    // non-raygen stages). Discovery order in the module matches declaration
    // order in the source; numbered per ID so a future multi-payload shader
    // still gets distinct, deterministic locations.
    let mut decorations: Vec<u32> = Vec::new();
    for (idx, id) in payload_ids.iter().enumerate() {
        decorations.push((4u32 << 16) | SPV_OP_DECORATE as u32);
        decorations.push(*id);
        decorations.push(SPV_DECORATION_LOCATION);
        decorations.push(idx as u32);
    }
    let mut out = Vec::with_capacity(words.len() + decorations.len());
    out.extend_from_slice(&words[..first_type]);
    out.extend_from_slice(&decorations);
    out.extend_from_slice(&words[first_type..]);
    out.iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<u8>>()
}

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let spv_dir = out_dir.join("spv");
    fs::create_dir_all(&spv_dir).expect("create spv output dir");

    let shader_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders");
    println!("cargo:rerun-if-changed={}", shader_dir.display());
    let mut entries: Vec<_> = fs::read_dir(&shader_dir)
        .expect("read shaders dir")
        .map(|e| e.expect("shader dir entry"))
        .filter(|e| e.path().extension().is_some_and(|x| x == "glsl"))
        .collect();
    entries.sort_by_key(|e| e.file_name());

    let compiler = Compiler::new().expect("failed to create shaderc compiler");

    for entry in &entries {
        let src_path = entry.path();
        let src = fs::read_to_string(&src_path).expect("read shader source");
        let stem = src_path
            .file_stem()
            .and_then(|n| n.to_str())
            .expect("shader file stem");
        let (kind, stage) = match stem.rsplit_once('.').map(|(_, s)| s) {
            Some("vert") => (ShaderKind::Vertex, "vertex"),
            Some("frag") => (ShaderKind::Fragment, "fragment"),
            Some("rgen") => (ShaderKind::RayGeneration, "ray generation"),
            Some("rchit") => (ShaderKind::ClosestHit, "closest hit"),
            Some("rmiss") => (ShaderKind::Miss, "miss"),
            Some("rsmiss") => (ShaderKind::Miss, "shadow miss"),
            Some("rshad") => (ShaderKind::AnyHit, "shadow any-hit"),
            Some("rahit") => (ShaderKind::AnyHit, "any hit"),
            other => panic!(
                "unrecognized shader stage in {}: {:?}",
                src_path.display(),
                other
            ),
        };

        let mut options = CompileOptions::new().expect("failed to create compile options");
        options.set_target_env(TargetEnv::Vulkan, EnvVersion::Vulkan1_0 as u32);
        options.set_warnings_as_errors();
        // Ray-tracing shader stages need SPIR-V 1.6 (GL_EXT_ray_tracing); the
        // graphics stages stay on the 1.0-compatible target shaderc picks by
        // default so the existing binary snapshot doesn't shift.
        let is_rt = matches!(
            stem.rsplit_once('.').map(|(_, s)| s),
            Some("rgen")
                | Some("rchit")
                | Some("rmiss")
                | Some("rsmiss")
                | Some("rshad")
                | Some("rahit")
        );
        if is_rt {
            options.set_target_spirv(shaderc::SpirvVersion::V1_6);
        }

        let artifact = compiler
            .compile_into_spirv(&src, kind, stem, "main", Some(&options))
            .unwrap_or_else(|e| {
                panic!(
                    "failed to compile {stage} shader {}: {e}",
                    src_path.display()
                )
            });

        let spv = if is_rt {
            patch_ray_payload_locations(artifact.as_binary_u8())
        } else {
            artifact.as_binary_u8().to_vec()
        };

        fs::write(spv_dir.join(format!("{stem}.spv")), spv)
            .unwrap_or_else(|e| panic!("failed to write {stem}.spv: {e}"));

        println!("cargo:rerun-if-changed={}", src_path.display());
    }
}
