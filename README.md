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
- **A real gearbox.** Five gears, manual shifting. Keep the needle in the zone or
  bog down like a lawnmower. Top speed: a hair over **340 km/h**.
- **Choose your pain.** Three difficulty modes, mid-run, with a keystroke:
  | Key | Mode   | The deal                                              |
  |-----|--------|-------------------------------------------------------|
  | `1` | EASY   | Sparse traffic, gentle walls, forgiving crashes       |
  | `2` | NORMAL | The way it's meant to be played                       |
  | `3` | HARD   | Wall-to-wall chaos. Good luck.                        |
- **Wreck or be wrecked.** Hit a car and it's a *WRECK* — you slow to a crawl and
  the HUD screams it at you. Too many wrecks and it's **GAME OVER**.

---

## 🕹️ Controls

| Input          | Action                              |
|----------------|-------------------------------------|
| `W` / `↑`      | Throttle                            |
| `S` / `↓`      | Brake                               |
| `A` / `←` `D` / `→` | Steer left / right              |
| `E`            | Gear up                             |
| `Q`            | Gear down                           |
| `1` / `2` / `3`| Difficulty: Easy / Normal / Hard    |

---

## 🧱 Tech Stack

- **Rust** + **Vulkan** via [vulkano](https://github.com/vulkano-rs/vulkano) 0.35
- **winit** 0.30 for windowing and input
- **GLSL shaders** compiled to SPIR-V automatically at build time with
  [shaderc](https://github.com/google/shaderc-rs) (glslang backend)
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

> 💡 **HUD font:** the game loads **Maple Mono Nerd Font** from
> `/usr/share/fonts/maple-mono/MapleMono-NF-Regular.ttf` for the on-screen HUD.
> Install the font (or place a copy at that path) or the game will refuse to start.

---

## 🔨 Build & Run

```bash
# debug build
cargo build

# release build (faster, recommended for playing)
cargo build --release

# run
cargo run --release
```

That's it. The first build takes a few minutes while shaderc compiles glslang;
after that, incremental builds are fast and shaders are recompiled automatically
whenever you edit anything in `shaders/`.

On startup you'll be asked to pick a GPU if more than one is available (or you can
pipe your selection / run non-interactively to auto-pick a discrete GPU).

---

## 🧰 Project Layout

```
assets/models/       Embedded 3D models + textures (GLB, PNG)
shaders/*.glsl       GLSL shader sources (compiled to SPIR-V at build time)
src/
  app.rs             Window, event loop, input handling
  game/              Game state: vehicle physics, traffic AI, difficulty
  render/            Renderer: Vulkan pipelines, camera, texture uploads
  mesh.rs            Procedural road/world geometry
  hud.rs, font.rs    HUD + font atlas
  model.rs           GLB mesh loading
  gpu.rs             Device/surface/queue selection
  build.rs*          Shader compilation pipeline
```

---

## 📜 Credits

- Car models & textures: [Kenney](https://kenney.nl/) (see `assets/models/KENNEY_LICENSE.txt`)
- Built on [vulkano](https://github.com/vulkano-rs/vulkano), [winit](https://github.com/rust-windowing/winit),
  [shaderc](https://github.com/google/shaderc-rs), [fontdue](https://github.com/mooman219/fontdue)

---

## ⚖️ License

- **Code** (Rust source + GLSL shaders): [MIT](LICENSE) — Copyright (c) 2026 Lane Lunacy contributors.
- **Original art/audio** (added by the project): [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
- **Kenney car models & textures**: [CC0](https://creativecommons.org/publicdomain/zero/1.0/) public domain.
- **HUD font** (Maple Mono Nerd Font): [SIL OFL](https://openfontlicense.org/), loaded from the system, not redistributed.

See [LICENSE-ASSETS](LICENSE-ASSETS) for the full asset breakdown.

---

*Drive fast, take corners lucky.*
