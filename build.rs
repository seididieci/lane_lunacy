use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use shaderc::{CompileOptions, Compiler, EnvVersion, ShaderKind, TargetEnv};

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let spv_dir = out_dir.join("spv");
    fs::create_dir_all(&spv_dir).expect("create spv output dir");

    let shader_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders");
    let mut entries: Vec<_> = fs::read_dir(&shader_dir)
        .expect("read shaders dir")
        .map(|e| e.expect("shader dir entry"))
        .filter(|e| e.path().extension().map_or(false, |x| x == "glsl"))
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
            other => panic!("unrecognized shader stage in {}: {:?}", src_path.display(), other),
        };

        let mut options = CompileOptions::new().expect("failed to create compile options");
        options.set_target_env(TargetEnv::Vulkan, EnvVersion::Vulkan1_0 as u32);
        options.set_warnings_as_errors();

        let artifact = compiler
            .compile_into_spirv(&src, kind, stem, "main", Some(&options))
            .unwrap_or_else(|e| panic!("failed to compile {stage} shader {}: {e}", src_path.display()));

        fs::write(spv_dir.join(format!("{stem}.spv")), artifact.as_binary_u8())
            .unwrap_or_else(|e| panic!("failed to write {stem}.spv: {e}"));

        println!("cargo:rerun-if-changed={}", src_path.display());
    }
}
