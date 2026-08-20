// SPDX-License-Identifier: MIT

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
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
struct EngineParams {
    rpm: AtomicU32,
    speed: AtomicU32,
    blown: AtomicBool,
}

impl Default for EngineParams {
    fn default() -> Self {
        EngineParams {
            rpm: AtomicU32::new(f32::to_bits(0.0)),
            speed: AtomicU32::new(f32::to_bits(0.0)),
            blown: AtomicBool::new(false),
        }
    }
}

const SAMPLE_RATE: u32 = 48_000;

/// How many harmonics the procedural engine stacks up (sawtooth-ish timbre).
const HARMONIC_COUNT: usize = 6;

/// Procedural engine source. A harmonic stack whose fundamental tracks the
/// (smoothed) RPM, layered with a touch of white-noise roughness and
/// speed-proportional wind, then saturated with a soft tanh. Runs forever on
/// the audio thread; the game writes `EngineParams`.
struct EngineSound {
    params: Arc<EngineParams>,
    /// Phase of each harmonic in 0..1.
    phases: [f32; HARMONIC_COUNT],
    noise_state: u64,
    smooth_rpm: f32,
    smooth_speed: f32,
}

impl EngineSound {
    fn new(params: Arc<EngineParams>) -> Self {
        EngineSound {
            params,
            phases: [0.0; HARMONIC_COUNT],
            noise_state: 0x9E37_79B9_7F4A_7C15,
            smooth_rpm: 0.0,
            smooth_speed: 0.0,
        }
    }

    /// Fundamental frequency (Hz): 55 at idle, 165 at the redline.
    fn base_hz(revs: f32) -> f32 {
        55.0 + 110.0 * revs.clamp(0.0, 1.0)
    }

    /// Applies the shared attack/release smoothing to RPM and speed. The engine
    /// target arrives once per frame, so the attack must span several frames or
    /// fast revs read back as a stepped staircase; 150 ms (~9 frames at 60 Hz)
    /// glides the pitch up smoothly, matching the release side. The oscillator
    /// phases accumulate continuously, so frequency changes still glissando.
    fn advance(&mut self) {
        let rpm = f32::from_bits(self.params.rpm.load(Ordering::Relaxed));
        let speed = f32::from_bits(self.params.speed.load(Ordering::Relaxed));
        let attack = 1.0 - (-1.0 / (SAMPLE_RATE as f32 * 0.15)).exp();
        let release = 1.0 - (-1.0 / (SAMPLE_RATE as f32 * 0.15)).exp();
        let alpha = if rpm > self.smooth_rpm { attack } else { release };
        self.smooth_rpm += (rpm - self.smooth_rpm) * alpha;
        let speed_alpha = 1.0 - (-1.0 / (SAMPLE_RATE as f32 * 0.2)).exp();
        self.smooth_speed += (speed - self.smooth_speed) * speed_alpha;
    }

    /// Advances the engine and returns the raw harmonic-stack sample (before
    /// the noise layers and final mix), so tests can measure pitch directly.
    fn next_osc(&mut self) -> f32 {
        self.advance();
        let hz = Self::base_hz(self.smooth_rpm);
        let mut v = 0.0f32;
        for n in 1..=HARMONIC_COUNT {
            let i = n - 1;
            self.phases[i] += (hz * n as f32) / SAMPLE_RATE as f32;
            if self.phases[i] >= 1.0 {
                self.phases[i] -= 1.0;
            }
            v += (self.phases[i] * std::f32::consts::TAU).sin() * (1.0 / n as f32);
        }
        v
    }

    fn next_value(&mut self) -> f32 {
        let osc = self.next_osc();
        let revs = self.smooth_rpm;
        let blown = self.params.blown.load(Ordering::Relaxed);
        // A touch of roughness keeps it organic; wind rises with road speed.
        let roughness = self.next_noise() * if blown { 0.12 } else { 0.03 };
        let wind = self.next_noise() * (self.smooth_speed / 340.0).clamp(0.0, 1.0) * 0.2;
        let amp = if blown { 0.05 } else { 0.5 + 0.3 * revs };

        (osc * amp + wind + roughness).tanh()
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
    settings: AudioSettings,
) -> (Player, Player, Player) {
    let mixer = stream.mixer();
    let engine = Player::connect_new(mixer);
    engine.append(EngineSound::new(params.clone()).repeat_infinite());
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
    sfx_buffers: SfxBuffers,
    applied: AudioSettings,
}

impl AudioEngine {
    pub fn init(settings: AudioSettings) -> Option<AudioEngine> {
        let device = resolve_device(settings.device_index)?;
        let mut stream = DeviceSinkBuilder::from_device(device)
            .ok()?
            .with_error_callback(ignore_benign_alsa_errors)
            .open_sink_or_fallback()
            .ok()?;
        stream.log_on_drop(false);
        let params = Arc::new(EngineParams::default());
        let (engine, sfx, music) = open_players(&stream, &params, settings);
        Some(AudioEngine {
            _stream: stream,
            engine,
            sfx,
            music: Some(music),
            params,
            sfx_buffers: SfxBuffers::new(),
            applied: settings,
        })
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
            .with_error_callback(ignore_benign_alsa_errors)
            .open_sink_or_fallback()
            .map_err(|_| ())?;
        stream.log_on_drop(false);
        let (engine, sfx, music) = open_players(&stream, &self.params, self.applied);
        self._stream = stream;
        self.engine = engine;
        self.sfx = sfx;
        self.music = Some(music);
        self.applied.device_index = device_index;
        Ok(())
    }

    /// Updates the procedural engine loop from the live vehicle state.
    pub fn set_engine(&self, rpm_frac: f32, speed_kmh: f32, blown: bool) {
        self.params
            .rpm
            .store(rpm_frac.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
        self.params
            .speed
            .store(speed_kmh.max(0.0).to_bits(), Ordering::Relaxed);
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
            .rpm
            .store(f32::to_bits(0.5), Ordering::Relaxed);
        params.speed.store(f32::to_bits(120.0), Ordering::Relaxed);
        let mut engine = EngineSound::new(params);
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

    #[test]
    fn engine_pitch_tracks_the_revs() {
        let crossings = |rpm: f32| {
            let params = Arc::new(EngineParams::default());
            params.rpm.store(f32::to_bits(rpm), Ordering::Relaxed);
            let mut engine = EngineSound::new(params);
            let mut count = 0usize;
            let mut prev = engine.next_osc();
            for _ in 1..SAMPLE_RATE as usize {
                let v = engine.next_osc();
                if (prev < 0.0) != (v < 0.0) {
                    count += 1;
                }
                prev = v;
            }
            count
        };
        let idle = crossings(0.0);
        let redline = crossings(1.0);
        assert!(
            redline > idle * 2,
            "redline ({redline}) should cross zero far more often than idle ({idle})"
        );
    }

    #[test]
    fn engine_rpm_attack_spans_multiple_frames() {
        // A hard 0 -> 1 rpm step (as if the player floored it between two
        // frames) must glide: after ~25 ms the smoothed value should still be
        // well short of the target, proving the attack spans many frames rather
        // than snapping to the frame-quantized input (which reads as "stepped").
        let params = Arc::new(EngineParams::default());
        let mut engine = EngineSound::new(params);
        let frames = (SAMPLE_RATE as f32 * 0.025) as usize;
        for _ in 0..frames {
            engine.next_osc();
        }
        let revs = engine.next_osc();
        assert!(
            (0.0..0.5).contains(&engine.smooth_rpm),
            "25 ms after a 0->1 rpm step the smoother should be <50% there, got {}",
            engine.smooth_rpm
        );
        assert!(revs.is_finite());
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