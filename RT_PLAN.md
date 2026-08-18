# RT-Ready Seams + Minimal Ray-Tracing Scaffold

> Written before implementation so the plan survives context compaction.
> Repo: `/home/work/work/lane_lunacy`. Branch: `feature/plane_reflections`.

## Goal
Restructure so full RT lighting + reflections can later be added as **one new
function** (`record_raytrace_scene`) without touching the raster pipeline,
bloom, post composite, or HUD. Deliver the seams now plus a minimal working RT
render to prove them.

## Architecture: the insertion seam
`record_frame_posted` (src/render/record.rs) draws scene->offscreen via
`record_scene_contents` (raster). Add `raytrace: Option<&RayTraceResources>`
param; when RT is enabled:
- The "scene into offscreen" step becomes `raytrace.record(...)` — RT writes
  the fully lit HDR image into the **same offscreen image** via `imageStore`
  (storage image).
- Raster scene pass, planar reflection pass, puddle-mask pass are skipped
  (`POST_REFLECT` off).
- Bloom -> post composite -> HUD run **unchanged** (they only read offscreen).
- Everything RT-specific lives in `src/render/raytrace.rs` + `shaders/raytrace.*.glsl`.

## Phase 1 — Device + shader plumbing
- `src/gpu.rs`:
  - `pub fn ray_tracing_supported(physical) -> bool` (checks
    `khr_acceleration_structure` + `khr_ray_tracing_pipeline` +
    `khr_deferred_host_operations` + `khr_buffer_device_address`).
  - Enable those extensions + features (`acceleration_structure`,
    `ray_tracing_pipeline`, `buffer_device_address`) in both
    `create_graphics_context` and `create_graphics_context_headless`.
- `build.rs`: map `.rgen/.rchit/.rmiss/.rahit` -> `ShaderKind::{RayGeneration,
  ClosestHit, Miss, AnyHit}` (shaderc 0.10.1 supports them).
- `src/shaders.rs`: add `RAYTRACE_RGEN_SPV`, `RAYTRACE_RCHIT_SPV`,
  `RAYTRACE_RMISS_SPV`.

## Phase 2 — Formalize reflection backend seam
- `src/render/reflection.rs`: add `ReflectionMethod::RayTraced`; add `REFLECT_RT`
  const in `src/shaders.rs` + `shaders/post.frag.glsl`.
- Small `ReflectionBackend` trait (`color_view()`, `sampler()`) so
  `record_frame_posted` stops depending on the concrete planar type.

## Phase 3 — RT scaffold (`src/render/raytrace.rs`, the "new function")
- `RayTraceResources`:
  - `RayTracingPipeline` (rgen+rchit+rmiss) + `ShaderBindingTable`.
  - Descriptors: `b0` TLAS, `b1` camera/light UBO (reuse `MVP` block), `b2`
    output storage image (offscreen), `b3` world atlas + car colormap (albedo),
    `b4` sky/cloud (rmiss).
  - **BLAS**: built once per mesh source reusing `SceneResources` buffers
    (world chunk mesh, player car, each traffic model); rebuilt when
    `world_stats().chunks_rebuilt > 0`.
  - **TLAS**: rebuilt/updated per frame using the same `Mat4` transforms the
    raster draws use.
  - `record(...)` dispatches `trace_rays` into the offscreen storage image.
- Shaders:
  - `raytrace.rgen.glsl`: camera ray per pixel, `traceRayEXT`, write HDR.
  - `raytrace.rchit.glsl`: reconstruct world pos/normal/uv/material from
    compact vertex SSBO (`gl_PrimitiveID` + `hitAttributeT`), simplified
    lighting mirroring `mesh.frag` (ambient + sun diffuse + fog).
  - `raytrace.rmiss.glsl`: return sky horizon color.
  - Reflection/GI rays are later refinements inside this module only.

## Phase 4 — UI (separate RAYTRACING row)
- `src/menu.rs`: `SettingsState.raytrace: bool` (default false); new
  `SettingsRow::Raytracing` button "RAYTRACING ON/OFF", gated by
  `ray_tracing_supported` (forced OFF when unsupported). Update cursor
  next/prev and row ids (Apply/Back shift by one; fix
  `settings_screen_builds_all_rows` which asserts ids 10..=22 -> 10..=23).
- `src/app.rs`: `FxSettings.raytrace` from `self.applied.raytrace`; pass
  capability into `build_menu_tree`.

## Phase 5 — Offscreen storage image
- `create_offscreen_view` (src/render/mod.rs): add `ImageUsage::STORAGE`
  (R16G16B16A16_SFLOAT storage is universally supported).

## Phase 6 — Verification
- `cargo build` + `cargo test` (snapshot parity + menu tests stay green).
- RT OFF -> visually identical to today.
- RT ON (menu) -> scene rendered by RT, flows through bloom/post/HUD.
- `LANE_DEBUG_POST` diagnostics still work in raster mode.

## Notes
- Headless snapshot path (`record_frame`) stays raster-only this pass; RT for
  snapshots can reuse the module later.
- llvmpipe advertises RT extensions (software RT slow) — gating on the
  capability query handles it.

## Key file map
- `src/gpu.rs` — device/extensions/features + `ray_tracing_supported`.
- `src/render/raytrace.rs` — new RT resources + `record()`.
- `src/render/record.rs` — `record_frame_posted` dispatch seam.
- `src/render/mod.rs` — `FxSettings.raytrace`, offscreen STORAGE, wiring.
- `src/render/reflection.rs` — `ReflectionMethod::RayTraced` + trait.
- `src/render/post.rs` / `src/render/puddle_mask.rs` — untouched.
- `src/menu.rs` / `src/app.rs` — RAYTRACING row wiring.
- `build.rs` / `src/shaders.rs` — RT stage plumbing.
- `shaders/raytrace.rgen.glsl` / `.rchit.glsl` / `.rmiss.glsl` — new.
