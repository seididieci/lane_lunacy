# Ray Tracing Debug Status — resolved

Last updated: 2026-08-19. The payload-propagation bug is FIXED, the per-image
BLAS desync (fast-flashing geometry) is FIXED, the rain puddle reflections are
FIXED (road is no longer a flat mirror), and the ray-traced render now produces
a correct, stable image.

## Post-flash bug: flat-mirror road in rain (FIXED)

Symptom: in rain weather the entire road reflected like a sheet of glass. Root
cause: `raytrace.rchit.glsl` set `rtp.extra.x = wet_look` (≈1.0) for **every**
asphalt pixel, so the raygen fired a full-strength reflected ray and mixed it in
everywhere. The raster path gates reflections with a deterministic world-space
`puddle_mask` (`post.frag.glsl`) and only blends where `pm > 0.001` at
`blend = pm * 0.72` + `1 - 0.22*blend` darkening.

Fix (shader-only):
- Ported the `puddle_mask` function (hash12 / value_noise / puddle_noise /
  road_* helpers, exact constants `ROAD_HALF=4.8`, `SHOULDER_W=0.55`,
  `12*sin(0.02*s)`) into `raytrace.rchit.glsl`.
- `rtp.extra.x = wet_look * puddle_mask(v_world_pos, wet_fac) * 0.72` — only
  puddle patches mirror; the rest of the asphalt keeps the matte wet sheen.
- `raytrace.rgen.glsl` now applies the raster composite's darkening
  `color *= 1 - 0.22 * blend` after mixing the reflected ray.

Verified: RT road sky-reflecting pixels dropped from ~72k (full mirror) to
~24k, close to the raster's ~21k; vision review confirms discrete patchy puddles
on matte asphalt matching the raster look. `cargo test` (170 pass) and
`cargo clippy` clean.

## Post-payload bug: fast-flashing geometry (FIXED)

Symptom: while driving, geometry appeared and disappeared every few frames (a
"flash"). Multi-frame captures showed 5 of 6 frames as pure sky (no geometry).
Root cause: `build_blas_objects` recreated fresh (unbuilt) BLAS storage +
`AccelerationStructure` objects for **all** image-in-flight copies whenever the
chunk window changed, but `record_blas_builds` only issued build commands for the
*current* `image_i`. Every swapchain image except the one active at rebuild time
traced against garbage acceleration structures → nothing hit → pure sky. Each
subsequent chunk-window change re-triggered it.

Fix (`src/render/raytrace.rs`): lazy per-image geometry application.
- `sync_geometry` now only repacks into a cached `packed` field and sets a
  per-image `blas_dirty` flag (was: wrote every image's pools immediately).
- `record` applies the packed pools + rebuilds BLAS **only for the current
  `image_i`** on that image's own frame, after the renderer has waited its fence
  (matching the module's existing per-image safety model).
- `build_blas_objects` now rebuilds a single image's BLAS instead of all images.

Verified: a 6-frame F10 capture burst after the fix shows every frame with the
full scene (sky ~36-40%, matching the raster layout); the raster diff is ~2.5/255.
`cargo test` (170 pass) and `cargo clippy` clean.

## Root cause (two compounding bugs)

1. **Wrong payload qualifier in hit/miss shaders.** The closest-hit and miss
   shaders declared the payload as `rayPayloadEXT` (SPIR-V `RayPayloadKHR`). In a
   hit/miss shader a `rayPayloadEXT` declares a *new* outgoing payload for rays
   that shader itself launches — **not** the payload coming back from the raygen.
   The incoming payload must be declared `rayPayloadInEXT` (`IncomingRayPayloadKHR`).
   With `rayPayloadEXT` in rchit/rmiss the driver never links the interface, so the
   raygen read back its own pre-trace value (magenta). Fixed by switching rchit and
   rmiss to `rayPayloadInEXT` (`shaders/raytrace.rchit.glsl`,
   `shaders/raytrace.rmiss.glsl`). This was the actual bug the whole time — the
   `Location` patch was a red herring.

2. **vulkano 0.35.2 drops `maxPipelineRayPayloadSize` / `maxPipelineRayHitAttributeSize`.**
   `RayTracingPipelineCreateInfo::to_vk` never sets them, so both stay `0`. Per
   VUID-VkRayTracingPipelineCreateInfoKHR-maxPipelineRayPayloadSize-03447 the pipeline
   value must be ≥ the largest payload any shader declares; with `0` RADV silently
   discards payload writes. Fixed in `src/render/raytrace.rs` by creating the pipeline
   through the raw `create_ray_tracing_pipelines_khr` with
   `RayTracingPipelineInterfaceCreateInfoKHR { max_pipeline_ray_payload_size: 96,
   max_pipeline_ray_hit_attribute_size: 8 }`, then wrapping the handle with
   `RayTracingPipeline::from_handle`. `ash = "=0.38.0"` added as a direct dependency.

## Verified

- `cargo build`, `cargo test` (170 pass), `cargo clippy` (no new warnings) all green.
- Daytime capture (`--raytrace --seed 11`) now renders a bright scene that matches
  the raster reference almost exactly (mean abs diff ~2.5/255; 78% of pixels within
  5/255). Sky gradient, terrain, road and cars all line up. The sun disc is slightly
  tighter in RT than raster because the raster sky dome is a coarse 32-ring mesh that
  interpolates `v_dir`, while RT computes exact per-pixel rays.
- Wet/night rain capture renders: RT reflects the sky on wet asphalt via the
  reflected-ray path (`extra.x` reflectivity → raygen secondary trace → mix). RT sky
  is brighter than raster under rain because the raster branch overlays rain/mist
  particle layers that the RT branch skips (expected, not a payload issue).

## Current diffs vs the debug session
- `build.rs` — SPIR-V 1.6 for RT stages + `patch_ray_payload_locations`. The Location
  patch is now redundant but harmless; keep until the shaderc glslang gap is upstreamed.
- `src/render/raytrace.rs` — raw pipeline creation with payload/hit-attribute sizes
  (the real fix on the Rust side) + the existing backend.
- `shaders/raytrace.rgen.glsl` — `RTShade` payload struct; primary ray; save color /
  normal / world_pos / reflectivity; if hit && wet fire a reflected ray and mix.
- `shaders/raytrace.rchit.glsl` — `rayPayloadInEXT`; full mesh.frag lighting into the
  payload fields; `extra.x` = wet reflectivity.
- `shaders/raytrace.rmiss.glsl` — `rayPayloadInEXT`; full sky.frag reproduction;
  `color.w = 0` marks a miss.
- `Cargo.toml` / `Cargo.lock` — `ash = "=0.38.0"`.

## Chunk-window stutter (FIXED with incremental repack)

Symptom: every ~260m chunk-window crossing hitched the render thread for ~0.9 s
(1× ~500 ms frame + 3× ~130 ms follow-ups). Root cause: `sync_geometry`
re-packed **all 8 window chunks and rebuilt all 13 BLAS** on every slide,
regardless of how many chunks actually changed.

Fix (`src/render/raytrace.rs`): fixed per-slot pool regions + per-chunk dirty
tracking.
- `SlotLayout` = fixed regions inside the vertex/index/instance pools. Statics
  (player + traffic) own slots `0..static_slots`; chunks own
  `chunk_slot(idx) = static_slots + idx.rem_euclid(CHUNK_SLOT_COUNT)` (8 slots,
  window is 8 consecutive indices so a +1 slide maps the entering chunk onto the
  leaving chunk's slot). Caps `CHUNK_VERT_CAP 430_000` / `CHUNK_INDEX_CAP 630_000`
  (measured HIGH max 403,576 verts / 604,596 idx; MED max 209,928 / 314,124).
- `slot_data` caches the packed data per slot; `chunk_owner` remembers which
  chunk index occupies each slot. `sync_geometry` diffs the incoming window
  against `last_chunk_indices`, re-packs **only the entering chunk(s)** into
  their (now-free) slots, and marks `blas_dirty[image_i][slot]` for every image.
- `apply_dirty_slots` + `record_blas_builds` take a `&dirty` list and write /
  rebuild only those slots for the current `image_i` after its fence wait.
- `write_instances` maps each window chunk via `chunk_slot(idx)` so instance
  references stay valid across repacks.

Verified (`--profile`, driven session): crossings at 13 window changes now cost
**single frames of 32-47 ms `scene_ms`** (was 470-550 ms + follow-ups); log shows
`RT pack: 1 chunk(s) repacked` per crossing (was 8). Average FPS rose 85 → 98;
extra time lost to spikes dropped from 5,061 ms/34 s (15%) to 1,198 ms/68 s
(1.8%). Remaining ~41 ms/crossing is the entering chunk's buffer read + repack +
one BLAS build (not yet backgrounded). `cargo test` (170 pass) and
`cargo clippy` clean.

## Missing rain particles under RT (FIXED, depth-correct overlay)

Symptom: with `--raytrace` the camera wet-lens effect rendered but the 3D rain
droplets were invisible. Root cause: the RT branch of `record_frame_posted`
replaces `record_scene_contents` entirely, and the CPU particle pass (rain /
mist / drift dust / night light glows) lived only in that raster function
(`raytrace.rs` had no particle handling), so the per-frame `Frame.particle_verts`
were never drawn.

Fix: a dedicated color-only **RT particle overlay pass** that composites the
existing CPU quads over the offscreen after the RT trace, with per-pixel
occlusion against the exact RT primary-ray depth:
- `raytrace.rgen.glsl` writes a new R32 depth image (binding 10) per pixel with
  `length(primary_hit - eye)` — linear eye distance, chosen because NDC depth
  collapses near `far=2000` (a half-float or NDC comparison cannot resolve
  metre-scale gaps at 30 m). Sky (miss) stores `1e30`. Only `rtp.world_pos` +
  `eye` are used, so no new uniform matrix was needed.
- `shaders/rt_particle.frag.glsl` (new) = `particle.frag` + occlusion discard:
  `if (occl_d < length(v_view_pos) - 0.08) discard;`. The particle's distance
  comes from a new `v_view_pos` varying in `particle.vert.glsl` (unused by the
  raster path). Depth sampled with `post.depth_sampler` (NEAREST).
- `scene.rs` gains `rt_particle_render_pass` (color-only, `Load` so the RT image
  is kept — macro defaults `initial_layout` to `ColorAttachmentOptimal`) +
  `rt_particle_pipeline` (additive) + `rt_dust_pipeline` (alpha) built against
  it at Sample1 (offscreen is 1x even under MSAA), plus `draw_rt_particles`
  (binds MVP / sprite / rt-depth). `mod.rs` builds/rebuilds the overlay
  framebuffer (offscreen color attachment only) on init/resize/AA; `record.rs`
  records mist → dust → rain+glows in the same order as raster.

Verified: `--raytrace --weather rain` capture now shows rain streaks over sky and
road comparable to the raster reference (vision-confirmed on crops), no
validation errors, and the scene still composites correctly. Occlusion
(rain not overdrawing cars) is verified by code inspection (same linear-distance
space, exact RT hit depth, metre bias) and needs a quick in-traffic drive to
confirm visually. `cargo test` (170 pass) and `cargo clippy` clean (only
pre-existing warnings). Bonus: night headlight/taillight/lamp glows are restored
under RT too.

## Commands
- Test capture: `cargo run -- --windowed --raytrace --seed 11 --window-capture /tmp/opencode/rt_x.png`
- Rain capture: `cargo run -- --windowed --raytrace --seed 11 --weather rain --window-capture /tmp/opencode/rt_rain.png`
- Compare vs raster: `cargo run -- --windowed --seed 11 --window-capture /tmp/opencode/raster.png`
- Drive profile: `cargo run -- --windowed --raytrace --seed 11 --profile /tmp/opencode/rt_inc_profile.csv`
- Tests: `cargo test`; Lints: `cargo clippy`.
