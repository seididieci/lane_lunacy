# Lane Lunacy TODO

> Art assets: add PNGs under `assets/textures/`, embed with `include_bytes!`, and
> upload via `upload_rgba8_texture`. Update `LICENSE-ASSETS` for any new art.

> Feature tasks are ordered by implementation difficulty (easiest first).

## 0. Refactor: SOLID + reuse + layering + a programmatic "eye"

Goals:
- SOLID principles, maximum code reuse, and clean layering.
- A headless "programmatic eye": render deterministic frames offscreen to CPU
  buffers, write PNGs the agent can view, and emit numeric probes that are
  assertable in tests — instead of human-taken screenshots.

Target layering (top → bottom):
- `presenters` — windowed `Renderer` (swapchain) | `render/snapshot.rs` (offscreen + readback + PNG + probes)
- `record.rs` — `record_frame(builder, &SceneResources, &Frame, framebuffer)`, one command-buffer recorder for any target
- `scene.rs` — GPU resources built once: pipelines, textures, models, mesh buffers, samplers
- `frame.rs` — pure CPU per-frame math: view/proj, palette, lights, sky uniform, headlight arrays, particle/flare/hud verts (no vulkano types, unit-testable)
- domain — `game/`, `road`, `surface`, `vertex`

Order: build the eye first (baseline), refactor, then re-run snapshots to prove
visual parity.

- [x] `--snapshot` CLI (path, `--time`, `--weather`, `--size`, `--seed`); headless branch in `main.rs` before winit
- [x] Headless GPU context: Instance/Device/queue without a surface (no present queue); reuse `gpu.rs`
- [x] Extract pure CPU `Frame` + builder from `render()` (no vulkano types)
- [x] Deterministic scene seeding: injectable seed for cloud tiles, weather phase, start hour (traffic already deterministic)
- [x] Extract `SceneResources` (pipelines, textures, models, buffers, samplers) shared by windowed + offscreen
- [x] Extract `record.rs` command-buffer recorder reused by both presenters
- [x] `snapshot.rs`: offscreen color+depth images, framebuffer, render, readback via `copy_image_to_buffer`, PNG via `image`
- [x] Probes: CPU (sun NDC, flare intensity, projector road coverage, wet/night fac) + GPU from pixels (sky-top lum, road-center lum, sun-disc max lum, flare bloom); print/JSON
- [x] Capture baseline snapshots + probe JSON (noon clear, midnight rain, dusk) as golden reference
- [x] Pipeline factory `graphics_pipeline()` (stages, vertex input, blend, depth, cull, samples) killing the 6 duplicated blocks in `render/mod.rs`
- [x] Deduplicate math: shared `smoothstep`/`mix`; remove copies in `game/mod.rs`, `daynight.rs`, `flare.rs`
- [x] Bundle headlight/projector arrays into structs; shrink `mvp_buffer`/`draw_particles` signatures (ISP)
- [x] Decouple windowed `Renderer` to only own swapchain/acquire/present; delegate math + recording (via shared `FrameBuilder`)
- [x] Snapshot regression tests: CPU probes always run; GPU probe tests gated behind `LANE_SNAPSHOT_TESTS=1`
- [x] Re-run baselines after refactor; diff probe JSON + PNG to prove visual parity (`scripts/snapshot_parity.sh`; pre-refactor HEAD vs post-refactor = 0 differing pixels)
- [x] `cargo test`, `cargo build`, clippy/fmt clean; document snapshot usage in README

## 1. Realistic road textures
- [x] Create/replace asphalt + grass tile textures under `assets/textures/`
- [x] Embed and upload them; bind as the world texture (replacing the 1×1 white)
- [x] Update `src/mesh.rs` UVs so textures tile in world space (not 1 tile per quad)
- [x] Per-surface texture selection (asphalt / grass / shoulder) via tint channel or second binding
- [x] Add subtle variation/noise so it doesn't look flat

## 2. Sky clouds
- [x] New `sky.vert/frag.glsl` + sky pipeline (or fullscreen pass), drawn before the scene, depth disabled
- [x] Cloud layer image with wrap-around scrolling; tie to time-of-day palette (task 5)
- [x] Register new shaders in `src/shaders.rs`
- [x] Procedural seamless cloud tiles (two cross-faded layers, per-run seed) + golden-hour palette
- [x] Weather state: `cloud_amount` uniform (clear / partly / dramatic), menu-selectable AUTO/CLEAR/CLOUDY/RAIN; RAIN is the placeholder hook for task 4 rain particles — coverage range spread across the full 0..1 scale with a threshold curve (CLEAR stays genuinely clear), cloud tiles densified into scattered clusters (no single wrapping bank), sunlit clouds brightened to pop against the azure sky, and cloud presence emphasised in the low horizon band the camera frames; RAIN darkens the whole sky

## 3. Sun + lens flare
- [x] Sun disc + halo in `sky.frag.glsl` at the day/night sun direction (task 5), gated by elevation and cloud cover
- [x] Lens flare: project sun → NDC, additive sprites along the sun-to-center line (core + ghosts + anamorphic streak); fades by sun brightness, cloud cover, and off-screen falloff
- [x] Procedural flare sprites (`src/render/flare.rs`), no art assets needed

## 4. Particles: rain + drift dust (+ optional local clouds/mist)
- [x] Particle pipeline with additive blending + soft sprite texture (procedural, runtime-baked)
- [x] Reusable CPU billboard particle system: Rust-side update, vertex buffer per frame, capped count (also serves local cloud puffs/mist)
- [x] Rain: fast-falling streaks in a volume around the camera, tied to weather/night (RAIN = full downpour; AUTO rains as its cover cycle peaks)
- [x] (Night) Red taillight billboards on traffic via the same particle pipeline, scaled by night darkness
- [x] Drift dust: puffs on hard steering/sideslip (lateral velocity while speed high), quantity bound to the road material under the car (per-surface `DustProfile`, ready for future surfaces like gravel)
- [x] (Hybrid) Optional local cloud puffs / low-hanging mist near the camera via the same billboard system; the ambient sky layer stays tile-based on the dome (task 2)

## 5. Night / Day cycle
- [x] Per-difficulty cycle: `day_fraction`, `cycle_speed`, `night_darkness` (EASY mostly day / NORMAL full / HARD long dark nights)
- [x] Sun elevation drives `light_dir`; sky, fog, ambient, cloud-tint palettes interpolate day↔night with a dawn/dusk warm tint; night gets a faint moon, moonlit `light_dir`, and procedural stars
- [x] Night-aware overcast colors (cloudy nights stay dark), weather-dimmed fog matching the horizon
- [x] Headlight cone + taillights at night (scaled by `night_darkness`), HUD clock (HH:MM) top-right, lamps placed at real per-model corner geometry (`CarLightAnchors`, incl. the player car's own rear taillights)

## 6. Menu polish + Settings: antialiasing, post-processing, visual filters

> Detailed implementation plan: `PLAN.md` (section 6).

Goals:
- Main menu selects START first; difficulty + weather are value rows on it;
  Settings is a submenu off the main menu.
- Settings exposes GPU, AA (MSAA) and a post-FX stack, gated to what the
  selected GPU supports, all applied live; every effect defaults to ON.
- Post-processing is built on a dedicated offscreen target + fullscreen pass,
  leaving the headless snapshot/probe path untouched.

- [x] Main menu opens with START as the first selected value (title + pause menu)
- [x] Two-screen menu model (Main: START/MODE/WEATHER/SETTINGS/EXIT with MODE/WEATHER cycled inline; Settings: GPU/AA/post-FX + BACK); per-screen keyboard routing
- [x] Post-processing foundation: offscreen color target + fullscreen post pass + `PostSettings` UBO + passthrough shader (windowed only)
- [x] MSAA 2x/4x: `samples` in the pipeline factory + render pass, resolve to offscreen target, backend rebuild on toggle, gated by device sample-count support
- [x] FXAA post effect + toggle
- [x] Bloom: ½→¼→⅛ downsample chain with per-level viewport + soft-knee luminance threshold (only bright sources glow) + toggle
- [x] Cheap FX set: vignette + film grain + saturation + chromatic aberration + toggles
- [x] Live apply wiring: renderer rebuild on AA change, `PostSettings` UBO update on filter toggles, staged APPLY model
- [x] Settings layout polish (10 rows fit 720p), menu tests, README controls, final test/build/clippy/fmt, snapshot parity re-check
- [x] Difficulty + weather moved to the main menu (committed as you cycle); all effects (best supported MSAA + post stack) default to ON at launch, capability-gated

## 7. World building

- [x] Remove the brown stepped banks on the road sides so the camera has a more
      open view (the bank boxes in `src/mesh.rs` box the road in; the car stays
      on the tarmac — the lateral `offset` clamp in `src/game/vehicle.rs` stays;
      the ground ribbon was widened to ±200m so the open field reads clean)
- [ ] Add a gravel road material: new `SurfaceMaterial` variant + atlas slot +
      texture under `assets/textures/` + a `DustProfile` (higher emission, gray
      puffs) and a `material_at` rule (e.g. shoulders/verges become gravel) so
      it reads and behaves differently from asphalt
- [x] Make the terrain respond to night/day: currently the grass/verge stays one
      color — tie the terrain palette to `night_fac` (and dusk tint) like the sky
      (grass/verge albedo tint in `mesh.frag.glsl`, driven by a new
      `terrain_tint` in `daynight::compute`; asphalt stays pure)
- [x] Add trees around the street (procedural, like the cloud/flare sprites —
      no art assets, culled by the road-mesh chunks; cone/pine + broadleaf
      built in `src/mesh.rs` with a deterministic per-world-s placement and a
      generated foliage tile as atlas slot 4 in `src/render/cloud.rs`)
- [x] Add street lamps that switch on at night, reusing the projector/headlight
      path in `src/render/frame.rs` (pole/arm/head geometry in `src/mesh.rs`,
      a dedicated warm `lamp_*` UBO pool projected down from each luminaire in
      `mesh.frag.glsl`, and warm lantern glow sprites in the particle pass; all
      gated by `night_fac`, both sides every 40m)
- [ ] Add road cliffs that affect the car's speed (local terrain gradient feeds
      into `vehicle.update`, e.g. uphill drag / downhill assist)
