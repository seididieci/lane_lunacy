# Lane Lunacy — implementation plans

> Session-continuity companion to `TODO.md`. Each section records the goal,
> design decisions, the exact code touch-points (file:line, current at last
> update), and how to verify. Follow it section by section; update the
> checkboxes in `TODO.md` as tasks land.

---

## Section 6 — Menu polish + Settings: AA, post-processing, visual filters

Status: planned (TODO.md section 6 all unchecked). Corresponds to
`PLAN.md` section 6.

### Goals (agreed with the user)

1. The main menu selects **START first** (both the title menu and the pause menu).
2. **Settings** is a submenu off the main menu (main = START / SETTINGS / EXIT).
3. Settings exposes: GPU, MODE, WEATHER, **ANTIALIASING** (OFF / MSAA 2× / MSAA 4×),
   **FXAA** (OFF/ON), **BLOOM**, **VIGNETTE**, **GRAIN**, **SATURATION**,
   **CHROMATIC ABERRATION**, and **BACK**.
4. AA + filters are **gated to what the selected GPU supports** and applied
   **live** on toggle (like the weather row already applies live today).
5. Post-processing runs on a dedicated **offscreen color target + fullscreen
   pass** so the headless snapshot/probe path stays untouched and deterministic.

### Design decisions locked in

- AA is modeled as **two rows**: `ANTIALIASING` (MSAA modes, capability-gated) and
  `FXAA` (always available, composable with MSAA).
- Post stack is the **full set**, built incrementally on the foundation task.
- MSAA toggle = full render-backend rebuild (reuses the `switch_gpu` teardown
  pattern); filter toggles = update a `PostSettings` UBO, no rebuild.
- Snapshot/probe path (`src/render/snapshot.rs`) keeps its own samples:1 direct
  render — it does **not** run the post pass, so `scripts/snapshot_parity.sh`
  baselines stay green.

### Current code touch-points (at planning time)

| Concern | Where |
|---|---|
| Menu state/rows/cursor | `src/menu.rs` — `MenuRow` (21), `MenuState::new` cursor `MenuRow::Gpu` (69), `open_for_pause` (74), `build_menu_tree` (132), unit test (185) |
| Menu keyboard routing + live apply | `src/app.rs` — `handle_menu_key` (208), `cycle_weather` (248), `resume_game` (157), `switch_gpu` (170), `build_menu_tree` call (353) |
| Renderer / render pass `samples: 1` | `src/render/mod.rs` — render pass (104–115), `Renderer::new` (72), `create_framebuffers` (169), `recreate_swapchain` (186), `render` (224) |
| Pipeline factory, `MultisampleState::default()` = Sample1 | `src/render/pipeline.rs` — `graphics_pipeline` (121), multisample_state (173) |
| 6 pipelines from the render-pass subpass | `src/render/scene.rs` — `SceneResources::new` (112), pipelines (129–252) |
| Headless path (keep untouched) | `src/render/snapshot.rs` — `render_snapshot` (50), its own render pass (59–75) |
| Shader registration | `src/shaders.rs` (SPIR-V consts) |
| HUD/menu font icons | `src/font.rs` — `ICON_*` (13–19); only 7 glyphs defined so far |

### Task breakdown (implement in this order)

**T1 — START-first default.**
`MenuState::new` → `cursor: MenuRow::Start`. Title + pause both open on START.
No behaviour change elsewhere. Existing menu test only hit-tests the START id, so
it stays valid.

**T2 — Two-screen menu model.**
- Add `MenuScreen { Main, Settings }`. Split rows: `MainRow { Start, Settings,
  Exit }` and `SettingsRow { Gpu, Mode, Weather, Antialias, Fxaa, Bloom, Vignette,
  Grain, Saturation, ChromaticAberration, Back }`, each with per-screen
  `previous`/`next` (clamp, matching current style).
- `MenuState` keeps `screen`, `main_cursor`, `settings_cursor`.
- `build_menu_tree` renders the active screen; Main = 3 rows, Settings = 11 rows.
- `handle_menu_key` routes by screen: Main Up/Down, Enter→START (close menu) /
  SETTINGS (screen=Settings, cursor=Gpu) / EXIT (quit). Settings Up/Down, Left/Right
  cycle row values, Enter **or Esc** on BACK → back to Main.
- Keep stable button ids per row so tests can address them.
- Note: with START as first row, pressing Esc on the title screen still starts the
  game (pre-existing `toggle_menu` behaviour) — accept or revisit later.

**T3 — Post-processing foundation (predisposition).**
- Windowed flow becomes: scene → offscreen `R16G16B16A16_SFLOAT` image → fullscreen
  post pass → swapchain. (Cannot sample + present the same image, hence the target.)
- New `src/render/post.rs`: POST render pass (samples:1, swapchain format) +
  fullscreen-triangle pipeline (clip-space vertices from the vertex shader, no
  vertex buffer). `shaders/post.vert.glsl` + `shaders/post.frag.glsl` (passthrough
  by default) + a `PostSettings` UBO:
  `flags: u32` (bit per FX) + intensities/time/saturation/vignette/bloom/chroma/
  grain factors.
- `record.rs` records scene→offscreen then post→swapchain. Renderer owns the
  offscreen image (recreated on resize) and rebuilds its viewport set.
- Register new shaders in `src/shaders.rs` (build.rs compiles `shaders/*.glsl`
  automatically).
- Leave `snapshot.rs` untouched.

**T4 — MSAA 2×/4×.**
- `src/render/pipeline.rs`: add `samples: SampleCount` to `PipelineSpec` (or
  factory param); set `MultisampleState { rasterization_samples: samples,
  ..Default::default() }`. `SceneResources::new` plumbs it to all 6 pipelines.
- Windowed render pass when MSAA on: color `samples=N` resolved into the 1×
  offscreen target, depth `samples=N`. When off: current 1× direct path.
- Renderer owns an MSAA color image + MSAA depth image (recreated on resize);
  framebuffers attach `[msaa_color, msaa_depth, offscreen_resolve]`.
- **Capability gating**: query `physical.properties().limits`
  `framebuffer_color_sample_counts` **and** `framebuffer_depth_sample_counts`;
  ANTIALIASING row offers only counts both include (skip unsupported on cycle).
- vkteck: vulkano 0.35 — `SampleCount::Sample2/Sample4`, `single_pass_renderpass!`
  supports a `resolve:` attachment; `MultisampleState::default()` is Sample1.

**T5 — FXAA.**
Edge-blend in `post.frag` behind the FXAA flag. No device gating (pure GLSL).

**T6 — Bloom.**
Downsample chain (½, ¼, ⅛) + blur + composite in the post stage, toggled by flag.
Heaviest piece; keep its passes owned by `post.rs`.

**T7 — Cheap FX set.**
Vignette, animated film grain (needs `time` in UBO), saturation, chromatic
aberration — all in `post.frag` behind individual flags.

**T8 — Live apply wiring.**
- Generalize `switch_gpu` (app.rs:170) into `recreate_renderer()` that reads both
  `gpu_index` and the current AA mode and rebuilds once (dedupe when both change).
- AA row cycle → rebuild immediately. Filter/weather toggles → update the live
  renderer (PostSettings UBO; weather already live via `cycle_weather`).
- Store `supported_aa: Vec<SampleCount>` in `App`; refresh it on GPU switch.

**T9 — Polish + verify.**
- Settings card is 11 rows tall — tighten ROW_EM/ROW_GAP for the Settings screen
  (or a second column) so it fits 1280×720.
- New icons if desired (e.g. BACK, palette): verify the glyph exists in the bundled
  Maple Mono Nerd Font before using (add a codepoint in `src/font.rs`; render a
  snapshot to confirm). Otherwise reuse existing `ICON_*`.
- Tests: update menu tests for the two-screen tree; add `MenuState` navigation
  tests (screen transitions, per-screen cursor clamp, AA cycle over the supported
  list only). No new GPU tests needed.
- README: update Controls + menu descriptions.
- Final gates:
  ```bash
  cargo test
  LANE_SNAPSHOT_TESTS=1 cargo test
  scripts/snapshot_parity.sh check     # expect PARITY OK, 0 differing pixels
  cargo clippy --all-targets
  cargo fmt --check                     # only touched files (repo-wide drift pre-exists)
  cargo build
  ```

### Verification baseline (unchanged by this work)

- Baselines: `snapshots/baseline/*` (gitignored), probes noon flare 0.991975 /
  sun_disc 4.1846, midnight wet 1.0 / night 0.6 / sky 0.08607, dusk
  sun_ndc [-4.094, 1.627].
- `tests/snapshot.rs`: CPU always run, GPU behind `LANE_SNAPSHOT_TESTS=1`.
