# Lane Lunacy

![Rust](https://img.shields.io/badge/Rust-1.87%2B-orange) ![License](https://img.shields.io/badge/license-MIT-orange)

> **A howling-good lane-racing game. One car, one endless ribbon of asphalt, and a
> whole lot of traffic that really, really wants a piece of you.**

You are the midnight lane-crawler. The road is a snake, the traffic is a gauntlet,
and your only defence is a five-speed stick shift, a steady thumb, and nerves of
spilled espresso. Dodge, weave, punch the gears, and pray the next blind corner
isn't bumper-to-bumper with your name on it.

Buckle up. It gets spicy.

---

## 🏎️ The Game

- **Drive forever.** A procedurally generated ribbon of road unwinds ahead of you
  — sweeping curves, roadside posts, grassy verges, and the occasional bank you'd
  rather not meet at speed.
- **Dodge the traffic.** Sedans, SUVs, taxis, vans. They come in waves, they don't
  blink, and they are *not* moving over.
- **A real gearbox.** Five gears, manual shifting. Revs idle and scream toward the
  redline on a live RPM dial. Shift up inside the *perfect band* — just before the
  red — for a score bonus and a short burst of acceleration. Push into the **red
  zone** and the engine starts to cook; cook it too long and **the engine blows**,
  ending the run. Top speed: a hair over **340 km/h**.
- **Choose your pain.** Three difficulty modes. Pick one in the start menu — or
  change it anytime via the pause menu (this restarts the run):
  | Mode   | The deal                                              |
  |--------|-------------------------------------------------------|
  | EASY   | Sparse traffic, gentle walls, forgiving crashes       |
  | NORMAL | The way it's meant to be played                       |
  | HARD   | Wall-to-wall chaos. Good luck.                        |
- **Wreck or be wrecked.** Hit a car and it's a *WRECK* — you slow to a crawl and
  the HUD screams it at you. Too many wrecks and it's **GAME OVER** — and if you
  let the engine cook, it **blows** and the car coasts to a stop.

---

## 🕹️ Controls

### Start / pause menu

| Input          | Action                              |
|----------------|-------------------------------------|
| `↑` / `↓` (`W`/`S`) | Move between rows              |
| `←` / `→` (`A`/`D`) | Cycle the selected row's value |
| `Enter`        | Select (START / SETTINGS / EXIT)   |
| `Esc`          | Open pause menu during a run / back |

The main menu offers **START**, **MODE**, **WEATHER**, **SETTINGS** and **EXIT**.
**MODE** and **WEATHER** are value rows: use `←`/`→` to cycle them and they take
effect immediately (changing MODE restarts the run). **START** begins the run
with everything in effect; **SETTINGS** opens a category screen with **GRAPHICS**
and **AUDIO** submenus.

#### Settings

Settings are *staged*: tweak as much as you like, then press **APPLY** to commit
them in one shot (it is dimmed until something changes). **BACK** returns to the
category screen / main menu without committing.

##### Graphics

| Row               | Values                                                |
|-------------------|-------------------------------------------------------|
| GPU               | Every Vulkan device (device 0 is the default)         |
| ANTIALIASING      | OFF / MSAA 2x / MSAA 4x (only modes the GPU supports) |
| FXAA              | ON / OFF — edge-aware smoothing on the post pass      |
| BLOOM             | ON / OFF — multi-level glow on bright lights          |
| VIGNETTE          | ON / OFF — darkened corners                           |
| GRAIN             | ON / OFF — animated film grain                        |
| SATURATION        | ON / OFF — boosted color saturation                   |
| CHROMATIC         | ON / OFF — radial red/blue shift                      |
| APPLY / BACK      | Commit graphics / return to settings                  |

Switching GPU re-uses the window and keeps your run going. Antialiasing and
post-processing apply live. Every effect defaults to ON at launch (the best
MSAA mode the GPU supports, plus all post effects, which every Vulkan device
can run) — dial any of them down in SETTINGS. MSAA modes are gated by what the
chosen GPU supports.

##### Audio

| Row              | Values                                              |
|------------------|-----------------------------------------------------|
| DEVICE           | Every output device (the default is marked "default") |
| MASTER           | 0–100 overall volume slider                        |
| MUSIC            | 0–100 music-channel volume slider                  |
| SFX              | 0–100 effect-channel volume slider                 |
| FX ON            | ON / OFF — engine loop + one-shot effects          |
| MUSIC ON         | ON / OFF — music channel (silent until a track ships) |
| APPLY / BACK     | Commit audio / return to settings                  |

The engine is a **real recorded loop** (a 4-cylinder car at ~7500 RPM, CC-BY
qubodup) played back at a variable rate that follows the RPM needle — it idles
deep, spools up smoothly with the throttle, and sings at the redline — with
speed-scaled wind noise layered on top. The one-shot effects (wrecks, perfect
shifts, the blown engine, gear changes) are **recorded sounds** (CC0) embedded in
the binary. **APPLY** reopens the stream on a new DEVICE and plays a short test
tone so you hear the change immediately. The ALSA host may occasionally report a
benign `get_htstamp ... earlier than get_trigger_htstamp` timestamp race;
playback is unaffected and these messages are suppressed.

### Driving

| Input          | Action                              |
|----------------|-------------------------------------|
| `W` / `↑`      | Throttle                            |
| `S` / `↓`      | Brake                               |
| `A` / `←` `D` / `→` | Steer left / right              |
| `E`            | Gear up                             |
| `Q`            | Gear down                           |
| `R`            | Restart run                         |
| `F10`          | Save a runtime screenshot (if enabled) |
| `F11`          | Toggle windowed / fullscreen        |
| `Esc`          | Pause menu                          |

### Window mode

The game starts **borderless fullscreen** on the current monitor. Start it in a
floating 90%-FHD window (1728×972) with `--windowed`, and toggle between the two
at any time with `F11`.

On **sway** a normal Wayland window can't float itself, so windowed mode asks
sway to float it via IPC (`swaymsg`); for the most reliable behavior add this to
`~/.config/sway/config`:

```
for_window [app_id="lane_lunacy"] floating enable
```

---

## 🧱 Tech Stack

- **Rust** + **Vulkan** via [vulkano](https://github.com/vulkano-rs/vulkano) 0.35
- **winit** 0.30 for windowing and input
- **GLSL shaders** compiled to SPIR-V automatically at build time with
  [shaderc](https://github.com/google/shaderc-rs) (glslang backend)
- **rodio** + **cpal** for audio output (device enumeration and playback)
- **Kenney** car models and textures embedded straight into the binary — no asset
  pipeline, no disk access at runtime
- **fontdue** for a from-scratch HUD font atlas

---

## 📦 Requirements

- **Linux / macOS / Windows** with a **Vulkan-capable GPU** and driver
- **Rust toolchain** (stable, edition 2021)
- **First build only** — shaderc compiles glslang from source, so you need:
  - `cmake`
  - a C/C++ toolchain (`gcc`/`clang`, or MSVC on Windows)
  - `git` (glslang source is vendored, but the build fetches pins)
  - `python3`

> 💡 Everything else is embedded in the binary — car models, textures, sound
> effects, and the **Maple Mono** HUD font (bundled under the SIL Open Font
> License). No runtime asset files or system fonts are required.

**Audio** playback uses cpal's ALSA host on Linux, so the ALSA development
headers are required to build (already present on most desktops):
`alsa-lib-devel` on Fedora, `libasound2-dev` on Debian/Ubuntu.

---

## 🔨 Build & Run

```bash
# debug build
cargo build

# release build (faster, recommended for playing)
cargo build --release

# run
cargo run --release

# force a specific GPU (see the [index] shown in the start menu)
cargo run --release -- --gpu 1

# start with a fixed sky state (auto | clear | cloudy | rain)
cargo run --release -- --weather rain

# start in a floating 90%-FHD window instead of fullscreen
cargo run --release -- --windowed

# enable runtime captures while driving (press F10 to save each frame)
cargo run --release -- --windowed --weather rain --capture-dir snapshots/current

# one-shot capture from the full windowed post path, then auto-exit
cargo run --release -- --windowed --weather rain --window-capture snapshots/current/shot_once.png
```

Rain renders in the **RAIN** sky state (full downpour) and periodically in
**AUTO**, where it fades in as the cloud-cover cycle peaks — additive rain
streaks that lean into your forward motion and melt into the horizon fog.

A low band of **ground mist** banks along the road near the camera whenever the
sky is heavily overcast or the sun sits low at dawn/dusk — dim, drifting puffs
that haze the asphalt and melt into the fog, rendered by the same particle
system as the rain and dust.

That's it. The first build takes a few minutes while shaderc compiles glslang;
after that, incremental builds are fast and shaders are recompiled automatically
whenever you edit anything in `shaders/`.

The game launches straight into the start menu — pick a GPU (default device 0,
or set one up front with `--gpu <N>`) and a difficulty, then press **START**.

---

## 📸 Headless snapshots & the programmatic eye

The renderer is fully deterministic given a seed, time of day, and weather, so a
frame can be rendered **offscreen, without a window or display**:

```bash
# one deterministic frame → PNG (1280x720 unless --size is given)
cargo run --release -- --snapshot shot.png --size 1280x720 --seed 42 --time 12 --weather clear
cargo run --release -- --snapshot night.png --size 1280x720 --seed 42 --time 0 --weather rain
# terrain ribbon density: low|med|high (default MED)
cargo run --release -- --snapshot rough.png --size 1280x720 --seed 42 --terrain-detail high
```

This is the backbone of visual regression checking. Golden baselines (PNG +
probe JSON) live in `snapshots/baseline/` (gitignored) and are captured/compared
with the parity harness:

```bash
# capture baselines from the current code
scripts/snapshot_parity.sh capture

# compare current output against the captured baselines (probes + pixels)
scripts/snapshot_parity.sh check
```

Each scenario is probed both ways: **CPU probes** (`CpuProbe` — sun position,
flare intensity, wet/night factors) are pure scene math, while **GPU probes**
(`GpuProbe` — sky, road, and sun luminance) read the actual rendered pixels.
The regression tests in `tests/snapshot.rs` pin both against the baselines:

```bash
cargo test                              # CPU probes always run
LANE_SNAPSHOT_TESTS=1 cargo test        # + GPU probes (needs a Vulkan device)
```

---

## 🧰 Project Layout

```
assets/models/       Embedded 3D models + textures (GLB, PNG)
assets/fonts/        Embedded HUD font (TTF + OFL license)
shaders/*.glsl       GLSL shader sources (compiled to SPIR-V at build time)
src/
  app.rs             Window, event loop, input handling, audio event wiring
  audio/             Audio output: device enumeration, procedural engine + SFX synthesis
  game/              Game state: vehicle physics, traffic AI, difficulty
  render/            Renderer: Vulkan pipelines, camera, texture uploads
  mesh.rs            Procedural road/world geometry
  hud.rs, font.rs    HUD + font atlas
  menu.rs            Menu tree: main + settings with graphics/audio submenus
  model.rs           GLB mesh loading
  gpu.rs             Device/surface/queue selection
  build.rs*          Shader compilation pipeline
```

Post-processing runs only in the windowed renderer: the scene is rendered into an
offscreen HDR target, then the FX composite (and the optional bloom downsample
chain) produces the final swapchain image. The headless snapshot path bypasses it
so the golden baselines stay deterministic.

---

## 📜 Credits

- Car models & textures: [Kenney](https://kenney.nl/) (see `assets/models/KENNEY_LICENSE.txt`)
- Built on [vulkano](https://github.com/vulkano-rs/vulkano), [winit](https://github.com/rust-windowing/winit),
  [shaderc](https://github.com/google/shaderc-rs), [fontdue](https://github.com/mooman219/fontdue),
  [rodio](https://github.com/RustAudio/rodio), [cpal](https://github.com/RustAudio/cpal)

---

## ⚖️ License

- **Code** (Rust source + GLSL shaders): [MIT](LICENSE) — Copyright (c) 2026 Lane Lunacy contributors.
- **Original art/audio** (added by the project): [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
- **Kenney car models & textures**: [CC0](https://creativecommons.org/publicdomain/zero/1.0/) public domain.
- **HUD font** (Maple Mono Nerd Font): bundled under the [SIL OFL 1.1](assets/fonts/OFL-MapleMono-NF.txt).

See [LICENSE-ASSETS](LICENSE-ASSETS) for the full asset breakdown.

---

*Drive fast, take corners lucky.*
