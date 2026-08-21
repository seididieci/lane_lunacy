// SPDX-License-Identifier: MIT

mod capture;
pub use capture::{AudioCapture, SharedCapture, TraceSample};

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::time::Duration;

use rodio::buffer::SamplesBuffer;
use rodio::cpal::{self, traits::{DeviceTrait, HostTrait}, Device};
use rodio::source::Source;
use rodio::{
    ChannelCount, DeviceSinkBuilder, MixerDeviceSink, Player, SampleRate,
};

/// Audio channel volumes and toggles, staged in the menu and committed on APPLY.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AudioSettings {
    /// Index into the enumerated output-device list.
    pub device_index: usize,
    /// Master volume in 0..=100.
    pub master: u8,
    /// Music channel volume in 0..=100.
    pub music: u8,
    /// Sound-effect channel volume in 0..=100.
    pub sfx: u8,
    /// Master switch for sound effects (engine + one-shots).
    pub fx_enabled: bool,
    /// Master switch for the music channel.
    pub music_enabled: bool,
}

impl Default for AudioSettings {
    fn default() -> Self {
        AudioSettings {
            device_index: 0,
            master: 80,
            music: 70,
            sfx: 80,
            fx_enabled: true,
            music_enabled: true,
        }
    }
}

impl AudioSettings {
    /// Linear master gain in 0..=1.
    pub fn master_gain(&self) -> f32 {
        self.master.clamp(0, 100) as f32 / 100.0
    }

    /// Perceptual gain for the sound-effect channel (squared volume curve).
    pub fn sfx_gain(&self) -> f32 {
        self.master_gain() * (self.sfx.clamp(0, 100) as f32 / 100.0).powi(2)
    }

    /// Perceptual gain for the music channel.
    pub fn music_gain(&self) -> f32 {
        self.master_gain() * (self.music.clamp(0, 100) as f32 / 100.0).powi(2)
    }
}

/// Stream-error callback for the output sink. cpal's ALSA host reports a
/// benign race when the hardware playback timestamp (`get_htstamp`) trails the
/// trigger timestamp (`get_trigger_htstamp`) by a hair; playback is unaffected
/// but the default rodio handler would print one line per occurrence. That case
/// is ignored; every other stream error still reaches stderr.
fn ignore_benign_alsa_errors(err: cpal::StreamError) {
    match err {
        cpal::StreamError::BackendSpecific { err }
            if err.description.contains("earlier than get_trigger_htstamp") => {}
        other => eprintln!("audio stream error: {other}"),
    }
}

/// Lists every output device the host exposes, plus the index of the default
/// output device (when the host has one). Used to populate the AUDIO > DEVICE row.
pub fn enumerate_output_devices() -> (Vec<String>, Option<usize>) {
    let host = cpal::default_host();
    let devices: Vec<Device> = host
        .output_devices()
        .map(|iter| iter.collect())
        .unwrap_or_default();
    let names = devices
        .iter()
        .map(|d| {
            d.description()
                .map(|desc| desc.name().to_string())
                .unwrap_or_else(|_| "Unknown device".to_string())
        })
        .collect::<Vec<_>>();
    let default = host
        .default_output_device()
        .and_then(|d| d.id().ok())
        .and_then(|id| devices.iter().position(|d| d.id().ok().as_ref() == Some(&id)));
    (names, default)
}

/// One-shot sound effects. All are synthesized in code so they stay trivial to
/// tune and never depend on external asset files.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sfx {
    Wreck,
    PerfectShift,
    Blow,
    Gear,
    /// Short beep used to confirm a device/volume change.
    Test,
}

/// Parameters shared between the game thread and the engine-sound source.
///
/// The game publishes the raw vehicle state (`speed_mps`, `gear`, `dt`,
/// `frame`) once per frame. The engine source itself reconstructs the
/// continuous rev curve with a sample-accurate first-order hold (clocked by the
/// audio sample counter, not a wall-clock thread), so the pitch never sees the
/// render loop's quantization and cannot be starved into stepping.
struct EngineParams {
    /// Published by the game: road speed in m/s.
    speed_mps: AtomicU32,
    /// Published by the game: current gear.
    gear: AtomicU32,
    /// Published by the game: frame duration in seconds.
    dt: AtomicU32,
    /// Published by the game: monotonic frame counter (bumped every frame).
    frame: AtomicU32,
    /// Written by the engine when capturing: the fundamental actually used (Hz).
    audio_hz: AtomicU32,
    /// Written by the engine when capturing: the smoothed revs used for pitch.
    audio_smooth_rpm: AtomicU32,
    /// Written by the engine when capturing: absolute sample index.
    audio_sample_idx: AtomicU64,
    /// Published by the game: engine blown flag.
    blown: AtomicBool,
}

impl Default for EngineParams {
    fn default() -> Self {
        EngineParams {
            speed_mps: AtomicU32::new(f32::to_bits(0.0)),
            gear: AtomicU32::new(1),
            dt: AtomicU32::new(f32::to_bits(1.0 / 60.0)),
            frame: AtomicU32::new(0),
            audio_hz: AtomicU32::new(f32::to_bits(0.0)),
            audio_smooth_rpm: AtomicU32::new(f32::to_bits(0.0)),
            audio_sample_idx: AtomicU64::new(0),
            blown: AtomicBool::new(false),
        }
    }
}

/// Push a high-rate capture sample every this many engine samples (48 = 1 kHz
/// at 48 kHz).
const TRACE_EVERY_SAMPLES: u64 = 48;

const SAMPLE_RATE: u32 = 48_000;

/// Output stream period size in frames (double-buffered by cpal).
/// Debug builds run the unoptimized audio callback too slowly for a tight
/// buffer, so keep a generous period there to avoid underruns/stepping;
/// release keeps a low-latency buffer so the pitch stays glued to the needle.
#[cfg(debug_assertions)]
const OUTPUT_PERIOD_FRAMES: u32 = 2400; // ~50 ms
#[cfg(not(debug_assertions))]
const OUTPUT_PERIOD_FRAMES: u32 = 1024; // ~21 ms

/// Reference fundamental of the pre-rendered engine loop (Hz). Midpoint of the
/// 55..165 base range, so runtime resampling stays within 0.55..=1.65x.
const LOOP_F0: f32 = 100.0;
/// Duration of the pre-rendered engine loop (seconds). 1.5 s at f0 = 100 Hz is
/// 150 integer cycles, so the loop wraps seamlessly.
const LOOP_DURATION_S: f32 = 1.5;
/// How many harmonics are baked into the loop. At redline the top harmonic
/// (128 * 165 Hz) stays just under Nyquist.
const LOOP_HARMONICS: usize = 128;
/// High-harmonic rolloff: harmonics above this frequency get an extra -6 dB/oct
/// so the tone stays warm instead of buzzing like a plain sawtooth.
const LOOP_ROLLOFF_HZ: f32 = 2000.0;
/// Exhaust formant: a gentle mid-band boost so the tone reads as an engine
/// rather than an oscillator.
const FORMANT_CENTER_HZ: f32 = 1200.0;
const FORMANT_WIDTH_HZ: f32 = 600.0;
const FORMANT_GAIN: f32 = 1.5;

/// Gain of the sub-harmonic (fundamental / 2) for the deep V8-style rumble.
const SUB_HARMONIC_GAIN: f32 = 0.6;

/// Fractional amplitude boost applied at each combustion pulse.
const PULSE_DEPTH: f32 = 0.2;

/// Combustion-pulse envelope decay time constant, in seconds.
const PULSE_DECAY_S: f32 = 0.008;

/// Exhaust noise level (broadband "air"/hiss) baked into the loop.
const EXHAUST_LEVEL: f32 = 0.09;

/// Low-pass cutoff of the baked exhaust noise, in Hz.
const EXHAUST_CUTOFF_HZ: f32 = 1600.0;

/// Level of the brief noise burst fired alongside each combustion pulse.
const FIRE_NOISE_LEVEL: f32 = 0.10;

/// Idle "lope": amplitude-wobble depth and rate that fade out as revs rise.
const LOPE_DEPTH: f32 = 0.05;
const LOPE_RATE_HZ: f32 = 7.0;

/// First-order-hold reconstruction of the rev curve: interpolate the road speed
/// between the previous and current frame's published samples, then map it
/// through the current gear to a revs fraction. `frac` is the fraction of the
/// frame that has elapsed (0..=1).
fn interpolated_rpm_frac(speed_prev: f32, speed_now: f32, frac: f32, gear: u32) -> f32 {
    let speed = speed_prev + (speed_now - speed_prev) * frac.clamp(0.0, 1.0);
    let redline = crate::game::vehicle::redline_speed((gear as usize).min(5));
    if redline <= 0.0 {
        0.0
    } else {
        (speed / redline).clamp(0.0, 1.0)
    }
}

/// Renders the reference engine loop: a dense `LOOP_HARMONICS` harmonic body at
/// `f0`, a sub-harmonic rumble at f0/2, a per-cycle combustion pulse, and a
/// baked low-passed exhaust noise bed with a burst per firing. `duration_s` must
/// contain an integer number of cycles at `f0` so the loop wraps seamlessly.
/// This is the "loop + resample" engine pattern used by TORCS/SuperTuxKart:
/// runtime pitch is a phase ramp over this dense spectrum, so it glides
/// smoothly with no sparse-line "steps".
fn render_engine_loop(f0: f32, duration_s: f32) -> Vec<f32> {
    let len = (SAMPLE_RATE as f32 * duration_s) as usize;
    let mut out = Vec::with_capacity(len);
    let mut seed = 0x1234_5678_9ABC_DEF0u64;
    let lp_alpha = std::f32::consts::TAU * EXHAUST_CUTOFF_HZ / SAMPLE_RATE as f32;
    let pulse_decay = (-1.0 / (SAMPLE_RATE as f32 * PULSE_DECAY_S)).exp();
    let mut exhaust_lp = 0.0f32;
    let mut pulse = 0.0f32;
    let mut prev_phase = 0.0f32;
    for i in 0..len {
        let t = i as f32 / SAMPLE_RATE as f32;
        let phase = (t * f0) % 1.0;
        // Dense harmonic body with a high-end rolloff and a mid-band formant.
        let mut v = 0.0f32;
        for n in 1..=LOOP_HARMONICS {
            let f = f0 * n as f32;
            let amp = (1.0 / n as f32)
                * (LOOP_ROLLOFF_HZ / f).min(1.0)
                * (1.0
                    + FORMANT_GAIN
                        * (-((f - FORMANT_CENTER_HZ) / FORMANT_WIDTH_HZ).powi(2)).exp());
            v += (std::f32::consts::TAU * phase * n as f32).sin() * amp;
        }
        // Normalize to ~a unit sawtooth peak, then add the sub-harmonic rumble.
        v /= std::f32::consts::FRAC_PI_2;
        v += (std::f32::consts::TAU * t * f0 * 0.5).sin() * SUB_HARMONIC_GAIN;

        // Combustion pulse at each cycle start.
        if phase < prev_phase {
            pulse = 1.0;
        }
        let pulse_env = 1.0 + PULSE_DEPTH * pulse;
        pulse *= pulse_decay;

        // Exhaust: low-passed noise bed plus a noise burst on firing.
        exhaust_lp += lp_alpha * (noise_rng(&mut seed) - exhaust_lp);
        let exhaust = exhaust_lp * EXHAUST_LEVEL
            + pulse * noise_rng(&mut seed) * FIRE_NOISE_LEVEL;

        out.push(v * pulse_env + exhaust);
        prev_phase = phase;
    }
    out
}

/// Procedural engine source built from a pre-rendered dense loop. The loop is
/// resampled with an `f64` phase accumulator so the pitch is a sample-accurate
/// ramp (no stepping), and the baked-in dense spectrum glides smoothly instead
/// of reading as discrete organ-like notes. An idle lope and speed-proportional
/// wind are layered at runtime, then saturated with a soft tanh.
struct EngineSound {
    params: Arc<EngineParams>,
    /// Pre-rendered dense reference engine loop.
    loop_buf: Arc<Vec<f32>>,
    /// Reference fundamental the loop was rendered at.
    f0: f32,
    /// Fractional read position (in samples) into the loop.
    read_pos: f64,
    /// Idle-lope phase, in seconds.
    lope_time: f32,
    noise_state: u64,
    smooth_rpm: f32,
    smooth_speed: f32,
    /// Precomputed one-pole coefficients (no per-sample `exp`).
    rpm_alpha: f32,
    speed_alpha: f32,
    /// Sample-accurate first-order hold state: last seen frame counter, the
    /// previous frame's speed, and how many samples have elapsed in the frame.
    last_frame: u32,
    prev_speed: f32,
    last_speed: f32,
    samples_in_frame: u64,
    /// Measured length (in samples) of the previous frame, used to size the
    /// current ramp so it completes when the next frame arrives — regardless of
    /// how fast the audio clock runs relative to the game.
    prev_frame_len: u64,
    /// Absolute sample counter (for the capture trace).
    sample_count: u64,
    /// Optional diagnostic capture sink (audio thread pushes only).
    capture: Option<SharedCapture>,
}

impl EngineSound {
    fn new(
        params: Arc<EngineParams>,
        loop_buf: Arc<Vec<f32>>,
        f0: f32,
        capture: Option<SharedCapture>,
    ) -> Self {
        EngineSound {
            params,
            loop_buf,
            f0,
            read_pos: 0.0,
            lope_time: 0.0,
            noise_state: 0x9E37_79B9_7F4A_7C15,
            smooth_rpm: 0.0,
            smooth_speed: 0.0,
            rpm_alpha: 1.0 - (-1.0 / (SAMPLE_RATE as f32 * 0.012)).exp(),
            speed_alpha: 1.0 - (-1.0 / (SAMPLE_RATE as f32 * 0.1)).exp(),
            last_frame: 0,
            prev_speed: 0.0,
            last_speed: 0.0,
            samples_in_frame: 0,
            prev_frame_len: SAMPLE_RATE as u64 / 60,
            sample_count: 0,
            capture,
        }
    }

    /// Fundamental frequency (Hz): 55 at idle, 165 at the redline.
    fn base_hz(revs: f32) -> f32 {
        55.0 + 110.0 * revs.clamp(0.0, 1.0)
    }

    /// Reconstructs the rev curve with a sample-accurate first-order hold,
    /// clocked by the audio sample counter (not a wall-clock thread, so it can
    /// never be starved into stepping). Between the game's per-frame samples the
    /// road speed is linearly interpolated over the *measured* length of the
    /// previous frame, so the ramp always completes when the next frame arrives
    /// — whether the audio clock runs real-time, faster, or slower than the
    /// game. The result is one-pole smoothed into `smooth_rpm` (pitch) and
    /// `smooth_speed` (wind).
    fn advance(&mut self) {
        let frame = self.params.frame.load(Ordering::Relaxed);
        let speed = f32::from_bits(self.params.speed_mps.load(Ordering::Relaxed));
        let gear = self.params.gear.load(Ordering::Relaxed);
        if frame != self.last_frame {
            self.prev_speed = self.last_speed;
            if self.samples_in_frame > 0 {
                self.prev_frame_len = self.samples_in_frame;
            }
            self.last_frame = frame;
            self.samples_in_frame = 0;
        }
        self.last_speed = speed;

        let frac = (self.samples_in_frame as f32 / self.prev_frame_len as f32).clamp(0.0, 1.0);
        self.samples_in_frame += 1;

        let rpm_frac = interpolated_rpm_frac(self.prev_speed, speed, frac, gear);
        self.smooth_rpm += (rpm_frac - self.smooth_rpm) * self.rpm_alpha;
        let interp = self.prev_speed + (speed - self.prev_speed) * frac;
        self.smooth_speed += (interp * 3.6 - self.smooth_speed) * self.speed_alpha;
    }

    /// Advances the engine and returns the interpolated loop sample at the
    /// current pitch (before the runtime noise layers), so tests can measure
    /// pitch directly. The read position is a continuous `f64` phase ramp, so
    /// the pitch cannot quantize to discrete steps.
    fn next_osc(&mut self) -> f32 {
        self.advance();
        let hz = Self::base_hz(self.smooth_rpm);
        let len = self.loop_buf.len() as f64;
        self.read_pos += (hz / self.f0) as f64 * (len / SAMPLE_RATE as f64);
        if self.read_pos >= len {
            self.read_pos -= len;
        }
        let i = self.read_pos.floor() as usize;
        let frac = (self.read_pos - i as f64) as f32;
        let a = self.loop_buf[i];
        let b = self.loop_buf[(i + 1) % self.loop_buf.len()];
        let v = a + (b - a) * frac;

        // Diagnostic capture: publish the live pitch/revs and a 1 kHz trace.
        // All gated by one Relaxed load; no I/O or allocation happens here.
        if let Some(cap) = &self.capture {
            if cap.is_enabled() {
                self.params
                    .audio_hz
                    .store(hz.to_bits(), Ordering::Relaxed);
                self.params
                    .audio_smooth_rpm
                    .store(self.smooth_rpm.to_bits(), Ordering::Relaxed);
                self.params
                    .audio_sample_idx
                    .store(self.sample_count, Ordering::Relaxed);
                if self.sample_count.is_multiple_of(TRACE_EVERY_SAMPLES) {
                    cap.push(TraceSample {
                        sample_idx: self.sample_count,
                        frame: self.params.frame.load(Ordering::Relaxed),
                        samples_in_frame: self.samples_in_frame,
                        smooth_rpm: self.smooth_rpm,
                        hz,
                    });
                }
            }
        }
        self.sample_count += 1;
        v
    }

    fn next_value(&mut self) -> f32 {
        let osc = self.next_osc();
        let revs = self.smooth_rpm;
        let blown = self.params.blown.load(Ordering::Relaxed);

        // Idle lope: a slow amplitude wobble that fades out as revs rise.
        self.lope_time += 1.0 / SAMPLE_RATE as f32;
        let lope = 1.0
            + LOPE_DEPTH
                * (std::f32::consts::TAU * LOPE_RATE_HZ * self.lope_time).sin()
                * (1.0 - revs);

        // Wind rises with road speed.
        let wind = self.next_noise() * (self.smooth_speed / 340.0).clamp(0.0, 1.0) * 0.2;

        let amp = if blown { 0.05 } else { 0.5 + 0.3 * revs };

        (osc * amp * lope + wind).tanh()
    }

    fn next_noise(&mut self) -> f32 {
        let mut x = self.noise_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.noise_state = x;
        (x >> 33) as f32 / (1u64 << 31) as f32 - 1.0
    }
}

impl Iterator for EngineSound {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        Some(self.next_value())
    }
}

impl Source for EngineSound {
    fn current_span_len(&self) -> Option<usize> {
        None
    }

    fn channels(&self) -> ChannelCount {
        ChannelCount::new(1).unwrap()
    }

    fn sample_rate(&self) -> SampleRate {
        SampleRate::new(SAMPLE_RATE).unwrap()
    }

    fn total_duration(&self) -> Option<Duration> {
        None
    }
}

fn noise_rng(seed: &mut u64) -> f32 {
    let mut x = *seed;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *seed = x;
    ((x >> 33) as f32 / (1u64 << 31) as f32) - 1.0
}

/// Low-passed noise with an exponential decay envelope.
fn filtered_noise(dur_s: f32, cutoff_hz: f32, decay: f32, gain: f32) -> Vec<f32> {
    let n = (SAMPLE_RATE as f32 * dur_s) as usize;
    let mut out = Vec::with_capacity(n);
    let mut seed = 0x1234_5678_9ABC_DEF0u64;
    let mut lp = 0.0f32;
    let alpha = 1.0 - (-1.0 / (SAMPLE_RATE as f32 * (1.0 / (cutoff_hz * std::f32::consts::TAU))))
        .exp();
    for i in 0..n {
        let t = i as f32 / SAMPLE_RATE as f32;
        lp += alpha * (noise_rng(&mut seed) - lp);
        out.push(lp * (-t * decay).exp() * gain);
    }
    out
}

/// A sine sweep with an exponential decay envelope.
fn sweep(dur_s: f32, f0: f32, f1: f32, decay: f32, gain: f32) -> Vec<f32> {
    let n = (SAMPLE_RATE as f32 * dur_s) as usize;
    let mut out = Vec::with_capacity(n);
    let mut phase = 0.0f32;
    for i in 0..n {
        let t = i as f32 / n as f32;
        phase += (f0 + (f1 - f0) * t) / SAMPLE_RATE as f32;
        out.push((phase * std::f32::consts::TAU).sin() * (-t * decay).exp() * gain);
    }
    out
}

fn mix(a: &[f32], b: &[f32]) -> Vec<f32> {
    let n = a.len().max(b.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(a.get(i).copied().unwrap_or(0.0) + b.get(i).copied().unwrap_or(0.0));
    }
    out
}

/// All one-shot effects are synthesized in code, so they are trivial to tune.
/// `Test` is the UI confirm beep.
struct SfxBuffers {
    wreck: Vec<f32>,
    perfect: Vec<f32>,
    blow: Vec<f32>,
    gear: Vec<f32>,
    test: Vec<f32>,
}

impl SfxBuffers {
    fn new() -> Self {
        let wreck = mix(
            &sweep(0.6, 180.0, 45.0, 5.0, 0.7),
            &filtered_noise(0.6, 3200.0, 8.0, 0.55),
        );
        let perfect = mix(
            &sweep(0.28, 880.0, 1760.0, 3.0, 0.35),
            &sweep(0.28, 1760.0, 2640.0, 3.0, 0.18),
        );
        let blow = mix(
            &sweep(1.1, 140.0, 24.0, 2.2, 0.8),
            &filtered_noise(1.1, 900.0, 3.0, 0.45),
        );
        let gear = filtered_noise(0.07, 4000.0, 60.0, 0.4);
        let test = sweep(0.14, 880.0, 880.0, 3.0, 0.5);
        SfxBuffers {
            wreck,
            perfect,
            blow,
            gear,
            test,
        }
    }

    fn get(&self, sfx: Sfx) -> &[f32] {
        match sfx {
            Sfx::Wreck => &self.wreck,
            Sfx::PerfectShift => &self.perfect,
            Sfx::Blow => &self.blow,
            Sfx::Gear => &self.gear,
            Sfx::Test => &self.test,
        }
    }
}

fn output_devices() -> Vec<Device> {
    cpal::default_host()
        .output_devices()
        .map(|iter| iter.collect())
        .unwrap_or_default()
}

fn resolve_device(device_index: usize) -> Option<Device> {
    output_devices()
        .get(device_index)
        .cloned()
        .or_else(|| cpal::default_host().default_output_device())
}

/// Builds the per-stream players and applies the initial volume/toggle state.
fn open_players(
    stream: &MixerDeviceSink,
    params: &Arc<EngineParams>,
    loop_buf: &Arc<Vec<f32>>,
    capture: Option<SharedCapture>,
    settings: AudioSettings,
) -> (Player, Player, Player) {
    let mixer = stream.mixer();
    let engine = Player::connect_new(mixer);
    engine.append(
        EngineSound::new(params.clone(), loop_buf.clone(), LOOP_F0, capture)
            .repeat_infinite(),
    );
    let sfx = Player::connect_new(mixer);
    let music = Player::connect_new(mixer);

    let e = settings.sfx_gain();
    let m = settings.music_gain();
    engine.set_volume(e);
    sfx.set_volume(e);
    music.set_volume(m);
    if !settings.fx_enabled {
        engine.pause();
        sfx.pause();
    }
    if !settings.music_enabled {
        music.pause();
    }
    (engine, sfx, music)
}

/// Owns the audio output stream and the looping engine / SFX / music players.
/// `init` returns `None` when no usable output device exists, so the game keeps
/// running silently (e.g. on CI or headless machines).
pub struct AudioEngine {
    _stream: MixerDeviceSink,
    engine: Player,
    sfx: Player,
    music: Option<Player>,
    params: Arc<EngineParams>,
    loop_buf: Arc<Vec<f32>>,
    capture: Option<SharedCapture>,
    sfx_buffers: SfxBuffers,
    applied: AudioSettings,
}

impl AudioEngine {
    pub fn init(settings: AudioSettings, capture: Option<SharedCapture>) -> Option<AudioEngine> {
        let device = resolve_device(settings.device_index)?;
        let mut stream = DeviceSinkBuilder::from_device(device)
            .ok()?
            .with_buffer_size(cpal::BufferSize::Fixed(OUTPUT_PERIOD_FRAMES))
            .with_error_callback(ignore_benign_alsa_errors)
            .open_sink_or_fallback()
            .ok()?;
        stream.log_on_drop(false);
        let params = Arc::new(EngineParams::default());
        let loop_buf = Arc::new(render_engine_loop(LOOP_F0, LOOP_DURATION_S));
        let (engine, sfx, music) = open_players(&stream, &params, &loop_buf, capture.clone(), settings);
        Some(AudioEngine {
            _stream: stream,
            engine,
            sfx,
            music: Some(music),
            params,
            loop_buf,
            capture,
            sfx_buffers: SfxBuffers::new(),
            applied: settings,
        })
    }

    /// Current engine pitch state for the diagnostic capture: (smooth_rpm, hz).
    pub fn engine_state(&self) -> (f32, f32) {
        let rpm = f32::from_bits(self.params.audio_smooth_rpm.load(Ordering::Relaxed));
        let hz = f32::from_bits(self.params.audio_hz.load(Ordering::Relaxed));
        (rpm, hz)
    }

    /// Absolute sample index of the engine source (for capture alignment).
    pub fn engine_sample_idx(&self) -> u64 {
        self.params.audio_sample_idx.load(Ordering::Relaxed)
    }

    /// The device index currently in effect.
    pub fn active_device(&self) -> usize {
        self.applied.device_index
    }

    /// Applies new volumes/toggles without touching the output device.
    pub fn apply(&mut self, settings: AudioSettings) {
        let e = settings.sfx_gain();
        let m = settings.music_gain();
        self.engine.set_volume(e);
        self.sfx.set_volume(e);
        if let Some(music) = &self.music {
            music.set_volume(m);
        }
        if settings.fx_enabled {
            self.engine.play();
            self.sfx.play();
        } else {
            self.engine.pause();
            self.sfx.pause();
        }
        if let Some(music) = &self.music {
            if settings.music_enabled {
                music.play();
            } else {
                music.pause();
            }
        }
        self.applied = settings;
    }

    /// Reopens the output stream on `device_index`, falling back to the default
    /// device when the chosen one cannot be opened. Preserves the active
    /// settings (volumes/toggles) and the current engine parameters.
    pub fn switch_device(&mut self, device_index: usize) -> Result<(), ()> {
        let device = resolve_device(device_index).ok_or(())?;
        let mut stream = DeviceSinkBuilder::from_device(device)
            .map_err(|_| ())?
            .with_buffer_size(cpal::BufferSize::Fixed(OUTPUT_PERIOD_FRAMES))
            .with_error_callback(ignore_benign_alsa_errors)
            .open_sink_or_fallback()
            .map_err(|_| ())?;
        stream.log_on_drop(false);
        let (engine, sfx, music) =
            open_players(&stream, &self.params, &self.loop_buf, self.capture.clone(), self.applied);
        self._stream = stream;
        self.engine = engine;
        self.sfx = sfx;
        self.music = Some(music);
        self.applied.device_index = device_index;
        Ok(())
    }

    /// Publishes the live vehicle state for the engine source. `speed_mps`/`gear`
    /// are the raw physics values, `dt` the (physics-clamped) frame duration,
    /// and `frame` a monotonically increasing per-frame counter (bumped once per
    /// frame). The engine source reconstructs the rev curve itself.
    pub fn set_engine(&self, speed_mps: f32, gear: u32, dt: f32, frame: u32, blown: bool) {
        self.params
            .speed_mps
            .store(speed_mps.max(0.0).to_bits(), Ordering::Relaxed);
        self.params.gear.store(gear, Ordering::Relaxed);
        self.params
            .dt
            .store(dt.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
        self.params.frame.store(frame, Ordering::Relaxed);
        self.params.blown.store(blown, Ordering::Relaxed);
    }

    /// Queues a one-shot effect (subject to the FX channel volume/toggle).
    pub fn play_sfx(&self, sfx: Sfx) {
        let buffer = self.sfx_buffers.get(sfx).to_vec();
        let source = SamplesBuffer::new(
            ChannelCount::new(1).unwrap(),
            SampleRate::new(SAMPLE_RATE).unwrap(),
            buffer,
        );
        self.sfx.append(source);
    }
}

/// Sample-space engine runner for the headless `--drive --audio-capture` path.
///
/// Reproduces the interactive audio pipeline faithfully without a cpal device:
/// the game publishes state once per frame and the runner advances the engine
/// by exactly `dt * SAMPLE_RATE` samples per frame — the same sample-per-frame
/// ratio a real-time 48 kHz stream under a 60 Hz game would see. This keeps the
/// capture's timing identical to what the player hears, even when the host has
/// no real-time output device.
pub struct EngineRunner {
    params: Arc<EngineParams>,
    engine: EngineSound,
}

impl EngineRunner {
    pub fn new(capture: Option<SharedCapture>) -> EngineRunner {
        let params = Arc::new(EngineParams::default());
        let loop_buf = Arc::new(render_engine_loop(LOOP_F0, LOOP_DURATION_S));
        let engine = EngineSound::new(params.clone(), loop_buf, LOOP_F0, capture);
        EngineRunner { params, engine }
    }

    /// Publishes the live vehicle state (mirrors `AudioEngine::set_engine`).
    pub fn set_engine(&self, speed_mps: f32, gear: u32, dt: f32, frame: u32, blown: bool) {
        self.params
            .speed_mps
            .store(speed_mps.max(0.0).to_bits(), Ordering::Relaxed);
        self.params.gear.store(gear, Ordering::Relaxed);
        self.params
            .dt
            .store(dt.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
        self.params.frame.store(frame, Ordering::Relaxed);
        self.params.blown.store(blown, Ordering::Relaxed);
    }

    /// Advances one game frame's worth of samples at 48 kHz.
    pub fn advance_frame(&mut self) {
        let dt = f32::from_bits(self.params.dt.load(Ordering::Relaxed))
            .clamp(1.0 / 240.0, 1.0 / 15.0);
        let samples = (dt * SAMPLE_RATE as f32).round().max(1.0) as usize;
        for _ in 0..samples {
            self.engine.next_osc();
        }
    }

    /// Advances one frame, pushing the final output samples (after tanh) into
    /// `out` so the actual audible signal can be rendered to a WAV.
    pub fn advance_frame_into(&mut self, out: &mut Vec<f32>) {
        let dt = f32::from_bits(self.params.dt.load(Ordering::Relaxed))
            .clamp(1.0 / 240.0, 1.0 / 15.0);
        let samples = (dt * SAMPLE_RATE as f32).round().max(1.0) as usize;
        for _ in 0..samples {
            out.push(self.engine.next_value());
        }
    }

    /// Current pitch state as (smooth_rpm, hz).
    pub fn state(&self) -> (f32, f32) {
        let rpm = f32::from_bits(self.params.audio_smooth_rpm.load(Ordering::Relaxed));
        let hz = f32::from_bits(self.params.audio_hz.load(Ordering::Relaxed));
        (rpm, hz)
    }

    /// Absolute sample index of the engine source.
    pub fn sample_idx(&self) -> u64 {
        self.params.audio_sample_idx.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_defaults_are_sane() {
        let a = AudioSettings::default();
        assert_eq!(a.device_index, 0);
        assert_eq!(a.master, 80);
        assert_eq!(a.music, 70);
        assert_eq!(a.sfx, 80);
        assert!(a.fx_enabled);
        assert!(a.music_enabled);
    }

    #[test]
    fn gains_follow_the_volume_sliders() {
        let mut a = AudioSettings::default();
        assert!((a.master_gain() - 0.8).abs() < 1e-5);
        assert!((a.music_gain() - 0.8 * 0.7 * 0.7).abs() < 1e-4);
        assert!((a.sfx_gain() - 0.8 * 0.8 * 0.8).abs() < 1e-4);

        a.master = 0;
        assert_eq!(a.master_gain(), 0.0);
        assert_eq!(a.music_gain(), 0.0);
        assert_eq!(a.sfx_gain(), 0.0);
    }

    #[test]
    fn gains_clamp_out_of_range_volumes() {
        let mut a = AudioSettings::default();
        a.master = 150;
        a.sfx = 255;
        assert_eq!(a.master_gain(), 1.0);
        assert_eq!(a.sfx_gain(), 1.0);
    }

    #[test]
    fn engine_source_always_yields_and_never_ends() {
        let params = Arc::new(EngineParams::default());
        params
            .speed_mps
            .store(f32::to_bits(30.0), Ordering::Relaxed);
        params.gear.store(2, Ordering::Relaxed);
        params.dt.store(f32::to_bits(1.0 / 60.0), Ordering::Relaxed);
        params.frame.store(1, Ordering::Relaxed);
        let mut engine = EngineSound::new(params, test_loop(), 100.0, None);
        assert_eq!(engine.current_span_len(), None);
        assert_eq!(engine.channels().get(), 1);
        assert_eq!(engine.sample_rate().get(), SAMPLE_RATE);
        assert_eq!(engine.total_duration(), None);
        for _ in 0..4096 {
            let s = engine.next();
            assert!(s.is_some());
            let v = s.unwrap();
            assert!(v.is_finite());
            assert!((-1.5..=1.5).contains(&v));
        }
    }

    #[test]
    fn engine_fundamental_maps_idle_to_redline() {
        assert!((EngineSound::base_hz(0.0) - 55.0).abs() < 1e-5);
        assert!((EngineSound::base_hz(1.0) - 165.0).abs() < 1e-5);
        assert!((EngineSound::base_hz(0.5) - 110.0).abs() < 1e-5);
    }

    /// A small engine loop for tests (5 cycles, cheap to render).
    fn test_loop() -> Arc<Vec<f32>> {
        Arc::new(render_engine_loop(100.0, 0.05))
    }

    #[test]
    fn engine_loop_is_seamless() {
        let loop_buf = render_engine_loop(100.0, 0.05);
        // Integer number of cycles at f0, so the loop wraps seamlessly.
        let cycles = (LOOP_F0 * 0.05).fract().abs();
        assert!(cycles < 1e-4, "loop must contain whole cycles");
        assert!(!loop_buf.is_empty());
        assert!(loop_buf.iter().all(|v| v.is_finite()));
        // The waveform is non-trivial: it both crosses zero and has energy.
        assert!(loop_buf.iter().any(|&v| v > 0.5));
        assert!(loop_buf.iter().any(|&v| v < -0.5));
    }

    #[test]
    fn engine_loop_resample_tracks_revs() {
        // The read position advances in proportion to the fundamental (55 vs
        // 165 Hz => exactly 3x), so the pitch is a smooth, sample-accurate ramp.
        let measure = |rpm: f32| {
            let params = Arc::new(EngineParams::default());
            let redline = crate::game::vehicle::redline_speed(1);
            params
                .speed_mps
                .store(f32::to_bits(redline * rpm), Ordering::Relaxed);
            params.gear.store(1, Ordering::Relaxed);
            params.dt.store(f32::to_bits(1.0 / 60.0), Ordering::Relaxed);
            params.frame.store(1, Ordering::Relaxed);
            let mut engine = EngineSound::new(params, test_loop(), 100.0, None);
            for _ in 0..(SAMPLE_RATE as f32 * 0.15) as usize {
                engine.next_osc();
            }
            let start = engine.read_pos;
            for _ in 0..(SAMPLE_RATE as f32 * 0.1) as usize {
                engine.next_osc();
            }
            engine.read_pos - start
        };
        let idle = measure(0.0);
        let redline = measure(1.0);
        let ratio = redline / idle;
        assert!(
            (ratio - 3.0).abs() < 0.05,
            "revs ratio should be 3x, got {ratio}"
        );
    }

    #[test]
    fn engine_rpm_smooths_a_step_without_lagging() {
        // The one-pole should reach a step within a few time constants: after
        // ~100 ms (5x the 20 ms constant) it should be within 1% of target, and
        // mid-way it must still be moving (monotonic, bounded) — not snapped.
        let params = Arc::new(EngineParams::default());
        let redline = crate::game::vehicle::redline_speed(1);
        params
            .speed_mps
            .store(f32::to_bits(redline), Ordering::Relaxed);
        params.gear.store(1, Ordering::Relaxed);
        params.dt.store(f32::to_bits(1.0 / 60.0), Ordering::Relaxed);
        params.frame.store(1, Ordering::Relaxed);
        let mut engine = EngineSound::new(params, test_loop(), 100.0, None);
        let mid = (SAMPLE_RATE as f32 * 0.02) as usize;
        for _ in 0..mid {
            engine.next_osc();
        }
        assert!(
            engine.smooth_rpm > 0.3 && engine.smooth_rpm < 0.9,
            "after ~20 ms the step should be part-way there, got {}",
            engine.smooth_rpm
        );
        for _ in 0..(SAMPLE_RATE as f32 * 0.1) as usize {
            engine.next_osc();
        }
        assert!(
            (engine.smooth_rpm - 1.0).abs() < 0.01,
            "one-pole should converge within ~100 ms, got {}",
            engine.smooth_rpm
        );
    }

    #[test]
    fn engine_revs_rise_monotonically_under_constant_accel() {
        // Publish speed rising linearly (constant accel) across several frames
        // and confirm the sample-accurate hold yields a smoothly rising revs
        // curve with no frame-quantized zigzag or dip.
        let params = Arc::new(EngineParams::default());
        let redline = crate::game::vehicle::redline_speed(1);
        params.gear.store(1, Ordering::Relaxed);
        params.dt.store(f32::to_bits(1.0 / 60.0), Ordering::Relaxed);
        let mut engine = EngineSound::new(params.clone(), test_loop(), 100.0, None);
        let frame_len = (SAMPLE_RATE as f32 / 60.0) as usize;
        let mut prev = engine.smooth_rpm;
        for f in 1..=6u32 {
            let speed = redline * (f as f32 / 6.0);
            params.speed_mps.store(f32::to_bits(speed), Ordering::Relaxed);
            params.frame.store(f, Ordering::Relaxed);
            for _ in 0..frame_len {
                engine.next_osc();
                assert!(
                    engine.smooth_rpm >= prev - 1e-4,
                    "smooth_rpm must not fall during acceleration"
                );
                prev = engine.smooth_rpm;
            }
        }
        // Settle at the redline so the one-pole fully converges.
        params.speed_mps.store(f32::to_bits(redline), Ordering::Relaxed);
        for f in 7..=12u32 {
            params.frame.store(f, Ordering::Relaxed);
            for _ in 0..frame_len {
                engine.next_osc();
            }
        }
        assert!(
            (engine.smooth_rpm - 1.0).abs() < 0.02,
            "revs should converge to the redline, got {}",
            engine.smooth_rpm
        );
    }

    #[test]
    fn interpolated_rpm_tracks_speed_through_the_gear() {
        // First-order hold: halfway between two frame samples of road speed,
        // the revs should be exactly the midpoint mapped through the gear's
        // redline, and out-of-range speeds must clamp.
        let frac = interpolated_rpm_frac(0.0, 18.0, 0.5, 1);
        let expected = (9.0 / crate::game::vehicle::redline_speed(1)).clamp(0.0, 1.0);
        assert!((frac - expected).abs() < 1e-4, "midpoint revs wrong: {frac}");

        // Gear 5 is an overdrive; the same speed reads lower revs there.
        let low = interpolated_rpm_frac(0.0, 95.0, 1.0, 5);
        assert!((0.0..0.9).contains(&low), "5th gear revs should stay low: {low}");

        // Excessive speed (redline past) clamps to 1.0, negative to 0.0.
        assert_eq!(interpolated_rpm_frac(0.0, 10_000.0, 1.0, 1), 1.0);
        assert_eq!(interpolated_rpm_frac(-5.0, -10.0, 0.5, 1), 0.0);
    }

    #[test]
    fn sfx_buffers_are_finite_and_bounded() {
        let sfx = SfxBuffers::new();
        for kind in [Sfx::Wreck, Sfx::PerfectShift, Sfx::Blow, Sfx::Gear, Sfx::Test] {
            let buffer = sfx.get(kind);
            assert!(!buffer.is_empty(), "{kind:?} must synthesize samples");
            assert!(buffer.iter().all(|v| v.is_finite()));
            assert!(buffer.iter().all(|v| v.abs() <= 1.5));
        }
    }

    #[test]
    fn sfx_buffers_have_sensible_lengths() {
        let sfx = SfxBuffers::new();
        for kind in [Sfx::Wreck, Sfx::PerfectShift, Sfx::Blow, Sfx::Gear, Sfx::Test] {
            let secs = sfx.get(kind).len() as f32 / SAMPLE_RATE as f32;
            assert!(
                (0.03..=2.0).contains(&secs),
                "{kind:?} should be a short one-shot, got {secs}s"
            );
        }
    }
}