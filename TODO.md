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
- [x] Sun disc + halo in `sky.frag.glsl` at the day/night sun direction (task 5), gated by elevation and cloud cover
- [x] Lens flare: project sun → NDC, additive sprites along the sun-to-center line (core + ghosts + anamorphic streak); fades by sun brightness, cloud cover, and off-screen falloff
- [x] Procedural flare sprites (`src/render/flare.rs`), no art assets needed

## 4. Particles: rain + drift dust (+ optional local clouds/mist)
- [x] Particle pipeline with additive blending + soft sprite texture (procedural, runtime-baked)
- [x] Reusable CPU billboard particle system: Rust-side update, vertex buffer per frame, capped count (also serves local cloud puffs/mist)
- [x] Rain: fast-falling streaks in a volume around the camera, tied to weather/night (RAIN = full downpour; AUTO rains as its cover cycle peaks)
- [x] (Night) Red taillight billboards on traffic via the same particle pipeline, scaled by night darkness
- [ ] Drift dust: puffs on hard steering/sideslip (lateral velocity while speed high)
- [ ] (Hybrid) Optional local cloud puffs / low-hanging mist near the camera via the same billboard system; the ambient sky layer stays tile-based on the dome (task 2)

## 5. Night / Day cycle
- [x] Per-difficulty cycle: `day_fraction`, `cycle_speed`, `night_darkness` (EASY mostly day / NORMAL full / HARD long dark nights)
- [x] Sun elevation drives `light_dir`; sky, fog, ambient, cloud-tint palettes interpolate day↔night with a dawn/dusk warm tint; night gets a faint moon, moonlit `light_dir`, and procedural stars
- [x] Night-aware overcast colors (cloudy nights stay dark), weather-dimmed fog matching the horizon
- [x] Headlight cone + taillights at night (scaled by `night_darkness`), HUD clock (HH:MM) top-right, lamps placed at real per-model corner geometry (`CarLightAnchors`, incl. the player car's own rear taillights)
