# Ray Tracing Debug Status — resume here

Last updated: 2026-08-18 ~23:50. STOPPED MID-DEBUG at the payload-linking problem.

## Where we are

The RT backend is fully wired and RUNNING (ray generation traces rays, hit/miss
shaders execute), but the **ray payload does not propagate from the hit/miss
shaders back to the raygen**, so the output is black. This is a cross-stage
interface bug, NOT a Rust/vulkano bug. Everything compiles, all 170 tests pass,
and the pipeline/descriptors/ASes are valid.

## IMPORTANT: shader files are in a BROKEN TEST STATE right now

`shaders/raytrace.rgen.glsl`, `.rchit.glsl`, `.rmiss.glsl` currently contain
diagnostic test code (flat `vec4 rtp` payload, hardcoded magenta/red/blue colors,
the full real body deleted from rgen after `return`). **They must be restored to
the real implementation before resuming.** See "Canonical shader content" below.

## Root cause being chased: payload does not propagate

Evidence (all captured with `--raytrace --seed 11`, daytime, RADV Mesa 26.1.6):
1. Gradient test: raygen + imageStore + copy to offscreen all work.
2. Hit/miss shaders DO run: when they write `rt_out` directly (imageStore green in
   rchit / red in rmiss), green+red pixels appear in the capture.
3. But when the raygen reads the payload after `traceRayEXT`, it gets the
   value the raygen itself wrote BEFORE the trace (initialized magenta stays
   magenta; hit/miss never overwrite it). Tested with:
   - 6-vec4 struct payload `RTShade` (identical across all 3 stages).
   - flat `vec4 rtp` payload (identical across all 3 stages).
   Both fail identically.

### What was already tried / verified
- **SPIR-V `Location` decoration missing**: the glslang bundled with
  `shaderc-sys 0.10.1` (the latest shaderc crate) deliberately omits
  `OpDecorate <payload> Location 0` on `RayPayloadKHR` variables. Verified in
  `GlslangToSpv.cpp` ~line 10224: it skips Location for RayPayloadKHR when
  `GL_EXT_ray_tracing` is requested (old SPIR-V behavior comment).
- **Fix applied**: `build.rs` now post-processes the 3 RT shaders' SPIR-V,
  inserting `OpDecorate %payload Location 0` before the type section
  (`patch_ray_payload_locations`). `spirv-val` passes, decorations confirmed
  present in all 3 `.spv` files AND embedded in the `target/debug/lane_lunacy`
  binary. **It did NOT fix the propagation.**
- Verified the raygen's SPIR-V is textbook-correct: init store -> OpTraceRayKHR
  (last operand = `%payload` var) -> read payload member -> imageStore.
- Hit/miss SPIR-V: payload var is RayPayloadKHR, Location 0, correct member
  stores. All 3 payload structs are 6x v4float.
- Driver: Mesa 26.1.6, RADV, VK_KHR_ray_tracing_pipeline available, api 1.4.354.

### Next debugging leads (not yet tried)
1. **Question the `OpDecorate %RTShade Block`** decoration that glslang adds to
   the payload struct type. Payload types arguably should NOT be `Block`-decorated.
   Try removing the struct entirely (flat vec4 payload already tried -> still
   fails, so Block on struct is probably not it).
2. **Check if RADV needs `maxPipelineRayPayloadSize` / maxPipelineRayHitAttributeSize
   in `RayTracingPipelineCreateInfo`** — vulkano may default to 0 which could make
   RADV drop payloads. Inspect vulkano's `RayTracingPipelineCreateInfo` defaults.
3. **Try a completely different driver path**: run with `VK_ICD_FILENAMES` pointing
   at a software ray tracer (e.g. llvmpipe/lvp doesn't do RT; maybe `radeonsi`? no).
   Or check if RADV has a `RADV_DEBUG`/`MESA_*` option affecting RT payloads.
4. **Check the SBT more**: maybe `ShaderBindingTable` records are fine (hit/miss
   run) but the payload needs the SBT hit record to reference the right group.
   Not obviously related.
5. **Look at how other people call this exact toolchain (shaderc 0.10.1 +
   vulkano 0.35 + RADV) and get payloads working** — search the web.
6. **Fallback architecture if payload cannot be made to work**: move ALL shading
   into the raygen using `rayQueryEXT` (OpRayQueryKHR) — read hit attributes
   inline, no cross-stage payload needed. Bigger rewrite but self-contained.
   Alternatively render everything in the rchit into `rt_out` directly
   (imageStore per-hit) which is proven to work — the rchit already has all the
   data (world pos, normal, uv, material from the vertex pool) and the MVP/RtUniforms
   UBOs; only the "is this a primary or reflected ray" distinction and the sky
   (miss) need handling. Sky could be handled by the raygen writing a sky pass
   first, then hit shaders overwrite the exact pixels they hit (but rchit runs
   for the reflected ray too — would overwrite primary). The cleanest proven
   fallback: keep imageStore from rchit/rmiss (works) and have the raygen NOT
   write, but then the "mix primary/reflection" and fog-on-sky logic must move
   into rchit/rmiss.

## Verified working (keep these)
- `build.rs`: `.rgen/.rchit/.rmiss` -> RayGeneration/ClosestHit/Miss at SPIR-V
  1.6 + `patch_ray_payload_locations` (DO NOT remove the patch until the payload
  issue is understood).
- `src/render/raytrace.rs`: full backend. Vertex pools sized for HIGH detail
  (`VERT_CAP_VEC4 = 14_000_000`, `INDEX_CAP = 7_000_000` — measured worst case
  3.24M verts / 4.86M indices at High). Pool layout is 3 vec4/vertex:
  `[pos.xyz, uv.x] [normal.xyz, material] [color.xyz, uv.y]` (matches rchit).
  BLAS/TLAS build, instance writes (player + traffic + chunks), `trace_rays`,
  copy rt_out -> offscreen. `ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY`
  added to vert/index/instance buffers (VUID 03673). Build-size queries use the
  real `max_vertex`/`vertex_stride` (03673/undersize fix).
- Wiring: `FxSettings.raytrace`, `SettingsState.raytrace` + `SettingsRow::Raytrace`
  menu row (id 21, APPLY=22, BACK=23), `--raytrace` CLI flag, `record_frame_posted`
  RT branch (skips raster scene + puddle + planar, uses REFLECT_RT), lazy
  `RayTraceResources::new` in `Renderer::render`, `world_chunk_indices()` accessor.
- `cargo check`, `cargo test` (170 passed), `cargo clippy` (new code clean),
  `cargo build` all green. Runtime capture works (gradient + green/red tests).

## Current diffs / files touched this session
- `build.rs` — SPIR-V patch fn + SPIR-V 1.6 for RT stages.
- `src/shaders.rs` — RAYTRACE_*_SPV consts, `RtUniforms`, `REFLECT_RT`.
- `src/render/raytrace.rs` — new module (the backend).
- `src/render/mod.rs` — `pub mod raytrace;`, `FxSettings.raytrace`, Renderer
  `raytrace`/`ray_tracing_supported` fields + lazy init + RT flags/method.
- `src/render/record.rs` — `record_frame_posted` RT branch (params: raytrace,
  chunk_indices, image_i).
- `src/render/frame_builder.rs` — `world_chunk_indices()`.
- `src/menu.rs` — `SettingsState.raytrace`, `SettingsRow::Raytrace`, row ids.
- `src/app.rs` — `--raytrace` plumbing into SettingsState (menu + applied).
- `src/cli.rs`, `src/lib.rs`, `src/main.rs` — `--raytrace` flag.
- `Cargo.toml` — added `smallvec = "1.8"`.
- `shaders/raytrace.rgen.glsl` / `.rchit.glsl` / `.rmiss.glsl` — **NEED RESTORE**.
- `RT_PLAN.md` — original plan (still valid).

## To resume
1. `git diff` / `git status` to see the test-state shaders; restore them.
2. Solve the payload propagation (see "Next debugging leads").
3. Restore rgen (full body: primary trace, save, optional reflected ray, mix),
   rchit (full shading -> `payload.color = vec4(final_col, 1.0)` + extra/world
   fields), rmiss (full sky -> `payload.color = vec4(col, 1.0); w = 0.0`).
4. `cargo build` + capture: expect a bright daytime scene (raster diff ~129).
5. Final: `cargo test`, `cargo clippy`.

## Canonical shader content (restore targets)
- **rgen**: `#version 460` + `#extension GL_EXT_ray_tracing : require`; payload
  `RTShade { vec4 color; vec4 normal; vec4 world_pos; vec4 albedo; vec4 uv;
  vec4 extra; }` at location 0; RtUniforms b0; rt_out b2 (rgba16f); tlas b0 set1.
  Body: NDC (y-up) -> `inv_view_proj` unproject -> `traceRayEXT(tlas, Opaque,
  0xFF, 0,0,0, eye.xyz, 0.001, dir, 2000.0, 0)`; save primary color/normal/
  world_pos/refl; if hit && wet, reset payload, fire reflected ray, mix
  `payload.color.rgb = mix(primary.rgb, payload.rgb, refl)`; final
  `imageStore(rt_out, ivec2(launch.xy), vec4(payload.color.rgb, 1.0))`.
- **rchit**: payload; MVP b1; VertexBuf/IndexBuf/SlotBuf b3/4/5 (caps
  14000000/7000000/256); world_tex b6; car_tex b7. Reads slot by
  `gl_InstanceCustomIndexEXT`, indices by `gl_PrimitiveID`, verts as
  `[pos.xyz,uv.x][normal.xyz,material][color.xyz,uv.y]`, barycentrics computed
  geometrically from `gl_ObjectRayOriginEXT + gl_HitTEXT * gl_ObjectRayDirectionEXT`
  (glslang has no bare `hitAttributeEXT` expression), full mesh.frag lighting,
  `payload.color = vec4(final_col, 1.0)` etc.
- **rmiss**: payload; RtUniforms; clouds_a/b b8/9; full sky.frag reproduction;
  `payload.color = vec4(col, 1.0); payload.color.w = 0.0;` + zeros.

## Commands
- Test capture: `cargo run -- --windowed --raytrace --seed 11 --window-capture /tmp/opencode/rt_x.png`
- Compare vs raster: `cargo run -- --windowed --seed 11 --window-capture /tmp/opencode/raster.png`
- Tests: `cargo test`; Lints: `cargo clippy`; Check: `cargo check`.
