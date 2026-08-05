# Graphics Upgrade TODO

> Art assets: add PNGs under `assets/textures/`, embed with `include_bytes!`, and
> upload via `upload_rgba8_texture`. Update `LICENSE-ASSETS` for any new art.

> Ordered by implementation difficulty (easiest first).

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
- [ ] Sun billboard at the time-of-day sun direction (from task 5)
- [ ] Lens flare: project sun → screen, sprites along sun-to-center line; fade by horizon occlusion
- [ ] Sun/flare sprite assets in `assets/textures/`

## 4. Particles: rain + drift dust (+ optional local clouds/mist)
- [ ] Particle pipeline with additive blending + soft sprite texture (`assets/textures/`)
- [ ] Reusable CPU billboard particle system: Rust-side update, vertex buffer per frame, capped count (also serves local cloud puffs/mist)
- [ ] Rain: fast-falling streaks in a volume around the camera, tied to weather/night
- [ ] Drift dust: puffs on hard steering/sideslip (lateral velocity while speed high)
- [ ] (Hybrid) Optional local cloud puffs / low-hanging mist near the camera via the same billboard system; the ambient sky layer stays tile-based on the dome (task 2)

## 5. Night / Day cycle
- [ ] Add `time_of_day` to the `MVP` uniform so shaders animate
- [ ] Sun elevation drives `light_dir`; interpolate sky, fog, ambient, and light palettes
- [ ] Difficulty-driven cycle: EASY = mostly day, NORMAL = full cycle, HARD = longer/darker nights
- [ ] (HARD-focused) Car headlights/taillights + darker road at night
