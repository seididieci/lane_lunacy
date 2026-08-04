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
- [ ] New `sky.vert/frag.glsl` + sky pipeline (or fullscreen pass), drawn before the scene, depth disabled
- [ ] Cloud layer image with wrap-around scrolling; tie to time-of-day palette (task 5)
- [ ] Register new shaders in `src/shaders.rs`

## 3. Sun + lens flare
- [ ] Sun billboard at the time-of-day sun direction (from task 5)
- [ ] Lens flare: project sun → screen, sprites along sun-to-center line; fade by horizon occlusion
- [ ] Sun/flare sprite assets in `assets/textures/`

## 4. Particles: rain + drift dust
- [ ] Particle pipeline with additive blending + soft sprite texture (`assets/textures/`)
- [ ] CPU particle system: Rust-side update, vertex buffer per frame, capped count
- [ ] Rain: fast-falling streaks in a volume around the camera, tied to weather/night
- [ ] Drift dust: puffs on hard steering/sideslip (lateral velocity while speed high)

## 5. Night / Day cycle
- [ ] Add `time_of_day` to the `MVP` uniform so shaders animate
- [ ] Sun elevation drives `light_dir`; interpolate sky, fog, ambient, and light palettes
- [ ] Difficulty-driven cycle: EASY = mostly day, NORMAL = full cycle, HARD = longer/darker nights
- [ ] (HARD-focused) Car headlights/taillights + darker road at night
