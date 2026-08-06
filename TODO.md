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
^- [x] Deduplicate math: shared `smoothstep`/`mix`; remove copies in `game/mod.rs`, `daynight.rs`, `flare.rs`
- [ ] Bundle headlight/projector arrays into structs; shrink `mvp_buffer`/`draw_particles` signatures (ISP)
- [ ] Decouple windowed `Renderer` to only own swapchain/acquire/present; delegate math + recording
- [ ] Snapshot regression tests: CPU probes always run; GPU probe tests gated behind `LANE_SNAPSHOT_TESTS=1`
- [ ] Re-run baselines after refactor; diff probe JSON + PNG to prove visual parity
- [ ] `cargo test`, `cargo build`, clippy/fmt clean; document snapshot usage in README

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
- [ ] (Hybrid) Optional local cloud puffs / low-hanging mist near the camera via the same billboard system; the ambient sky layer stays tile-based on the dome (task 2)

## 5. Night / Day cycle
- [x] Per-difficulty cycle: `day_fraction`, `cycle_speed`, `night_darkness` (EASY mostly day / NORMAL full / HARD long dark nights)
- [x] Sun elevation drives `light_dir`; sky, fog, ambient, cloud-tint palettes interpolate day↔night with a dawn/dusk warm tint; night gets a faint moon, moonlit `light_dir`, and procedural stars
- [x] Night-aware overcast colors (cloudy nights stay dark), weather-dimmed fog matching the horizon
- [x] Headlight cone + taillights at night (scaled by `night_darkness`), HUD clock (HH:MM) top-right, lamps placed at real per-model corner geometry (`CarLightAnchors`, incl. the player car's own rear taillights)
