# Sound-effect sources

All files in this directory are embedded into the game binary via
`include_bytes!`. They were converted to mono 16-bit PCM @ 48 kHz (from the
original rates) with `ffmpeg -ac 1 -ar 48000`; nothing else was edited.

| File | Source | Author | License | Original URL |
| --- | --- | --- | --- | --- |
| engine_loop.wav | "Car Engine Loop 96kHz, 4s" (`engine-loop-1.wav`, stereo 96 kHz 24-bit) | qubodup | CC-BY 3.0 | https://opengameart.org/content/car-engine-loop-96khz-4s |
| wreck.wav | "Metal Impact Sounds" (`bong1.wav`) | BMacZero / Brian MacIntosh | CC0 | https://opengameart.org/content/metal-impact-sounds |
| gear.wav | "Metal Impact Sounds" (`clink1.wav`) | BMacZero / Brian MacIntosh | CC0 | https://opengameart.org/content/metal-impact-sounds |
| perfect_shift.wav | "Metal Impact Sounds" (`bing1.wav`) | BMacZero / Brian MacIntosh | CC0 | https://opengameart.org/content/metal-impact-sounds |
| blow.wav | "Dynamite sound effect" (`Dynamite with sensor.wav`) | Listener | CC0 (public domain) | https://opengameart.org/content/dynamite-sound-effect |

## License texts

- **CC-BY 3.0** (engine_loop.wav): https://creativecommons.org/licenses/by/3.0/
  qubodup (OGA user qubodup) must be credited; see LICENSE-ASSETS.
- **CC0** (wreck.wav, gear.wav, perfect_shift.wav, blow.wav):
  https://creativecommons.org/publicdomain/zero/1.0/ — public domain.
  Attribution optional.

The game's synthesized `Test` beep (device/volume confirmation) is original
project audio and is not derived from these files.