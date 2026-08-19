// SPDX-License-Identifier: MIT

use std::io::Cursor;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::Duration;

use rodio::buffer::SamplesBuffer;
use rodio::cpal::{self, traits::{DeviceTrait, HostTrait}, Device};
use rodio::source::Source;
use rodio::{
    ChannelCount, DeviceSinkBuilder, MixerDeviceSink, Player, SampleRate,
};

/// Embedded 16-bit mono PCM @ 48 kHz sound files (see assets/sfx/SOURCES.md).
const WAV_ENGINE_LOOP: &[u8] = include_bytes!("../../assets/sfx/engine_loop.wav");
const WAV_WRECK: &[u8] = include_bytes!("../../assets/sfx/wreck.wav");
const WAV_PERFECT: &[u8] = include_bytes!("../../assets/sfx/perfect_shift.wav");
const WAV_BLOW: &[u8] = include_bytes!("../../assets/sfx/blow.wav");
const WAV_GEAR: &[u8] = include_bytes!("../../assets/sfx/gear.wav");

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

/// One-shot sound effects. Wreck, PerfectShift, Blow and Gear play embedded
/// recordings; Test is synthesized.
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

/// Lowest playback rate of the real engine loop, reached at idle. Pitching a
/// real recording down reads it slower and lower; 0.25x keeps the low-end rumble
/// without collapsing into an inaudible sub-bass.
const MIN_PLAYBACK_RATE: f32 = 0.25;

/// Slow-AGC "de-pump" that flattens the level drift baked into the recorded
/// loop. The source was recorded on the move, not on a dyno, so its body bands
/// breathe a few dB over ~0.25-1 Hz cycles (plus a small step at the loop seam).
/// Pitched down, those cycles stretch into seconds-long pumping that reads as
/// "stepped". The AGC tracks the loop's slow RMS envelope at the *actual*
/// playback rate and gently normalizes it, so the engine holds a steady level
/// while its fast texture (firing ripple, harmonics) is untouched.
const AGC_TIME_S: f32 = 0.7;
const AGC_GAIN_MIN: f32 = 0.5;
const AGC_GAIN_MAX: f32 = 2.0;
const AGC_GAIN_SMOOTH_S: f32 = 0.05;

/// Decodes an embedded WAV to mono float samples at `SAMPLE_RATE`, downmixing
/// extra channels and peak-normalizing to 0.95. Returns `None` on any failure so
/// callers can fall back to a synthesized buffer.
fn load_wav(bytes: &'static [u8]) -> Option<Vec<f32>> {
    let decoder = rodio::Decoder::new_wav(Cursor::new(bytes)).ok()?;
    let channels = decoder.channels().get();
    let rate = decoder.sample_rate().get();
    let mut samples: Vec<f32> = decoder.collect();
    if samples.is_empty() {
        return None;
    }
    if channels > 1 {
        let channels = channels as usize;
        samples = samples
            .chunks(channels)
            .map(|c| c.iter().sum::<f32>() / channels as f32)
            .collect();
    }
    if rate != SAMPLE_RATE {
        let ratio = rate as f64 / SAMPLE_RATE as f64;
        let n = (samples.len() as f64 / ratio) as usize;
        let mut resampled = Vec::with_capacity(n);
        for i in 0..n {
            let pos = i as f64 * ratio;
            let i0 = (pos.floor() as usize).min(samples.len() - 1);
            let i1 = (i0 + 1).min(samples.len() - 1);
            let frac = (pos - pos.floor()) as f32;
            resampled.push(samples[i0] + (samples[i1] - samples[i0]) * frac);
        }
        samples = resampled;
    }
    let peak = samples.iter().fold(0.0f32, |m, s| m.max(s.abs())).max(1e-6);
    let scale = 0.95 / peak;
    for s in &mut samples {
        *s *= scale;
    }
    Some(samples)
}

/// Real recorded engine loop (a 4-cylinder car at ~7500 rpm, public under
/// CC-BY). If it cannot be decoded, a simple harmonic stack takes its place so
/// the game still has an engine.
fn load_engine_loop() -> Vec<f32> {
    load_wav(WAV_ENGINE_LOOP).unwrap_or_else(|| {
        let n = SAMPLE_RATE as usize;
        let mut out = Vec::with_capacity(n);
        let mut phase = 0.0f32;
        let tau = std::f32::consts::TAU;
        for _ in 0..n {
            phase += 87.0 / SAMPLE_RATE as f32;
            phase %= 1.0;
            let v = (phase * tau).sin() * 0.5 + (phase * 2.0 * tau).sin() * 0.3;
            out.push(v);
        }
        out
    })
}

/// Looping engine source built from a real recording. The recorded loop is read
/// back at a variable rate (`MIN_PLAYBACK_RATE` at idle, 1.0 at the redline) so
/// pitch and speed track the vehicle; the position accumulates continuously, so
/// rate changes glissando instead of stepping. Runs forever on the audio thread;
/// the game writes `EngineParams`.
struct LoopEngineSound {
    params: Arc<EngineParams>,
    samples: Vec<f32>,
    pos: f64,
    noise_state: u64,
    smooth_rpm: f32,
    smooth_speed: f32,
    /// Reference RMS level the AGC normalizes toward (the loop's own average).
    rms_ref: f32,
    /// One-pole RMS of the resampled loop output (slow).
    env_rms: f32,
    /// Smoothed AGC gain, starts at unity.
    smooth_gain: f32,
}

impl LoopEngineSound {
    fn new(params: Arc<EngineParams>) -> Self {
        let samples = load_engine_loop();
        let rms_ref = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
        LoopEngineSound {
            params,
            samples,
            pos: 0.0,
            noise_state: 0x9E37_79B9_7F4A_7C15,
            smooth_rpm: 0.0,
            smooth_speed: 0.0,
            rms_ref,
            env_rms: rms_ref * rms_ref,
            smooth_gain: 1.0,
        }
    }

    /// Pure mapping from revs to the real-loop playback rate.
    fn playback_rate(revs: f32) -> f32 {
        MIN_PLAYBACK_RATE + (1.0 - MIN_PLAYBACK_RATE) * revs.clamp(0.0, 1.0)
    }

    /// Applies the shared attack/release smoothing to RPM and speed. Short time
    /// constants (attack 40 ms, release 150 ms) so the pitch snaps onto the RPM
    /// needle; the resampler interpolates continuously, so no stepping returns.
    fn advance(&mut self) {
        let rpm = f32::from_bits(self.params.rpm.load(Ordering::Relaxed));
        let speed = f32::from_bits(self.params.speed.load(Ordering::Relaxed));
        let attack = 1.0 - (-1.0 / (SAMPLE_RATE as f32 * 0.04)).exp();
        let release = 1.0 - (-1.0 / (SAMPLE_RATE as f32 * 0.15)).exp();
        let alpha = if rpm > self.smooth_rpm { attack } else { release };
        self.smooth_rpm += (rpm - self.smooth_rpm) * alpha;
        let speed_alpha = 1.0 - (-1.0 / (SAMPLE_RATE as f32 * 0.2)).exp();
        self.smooth_speed += (speed - self.smooth_speed) * speed_alpha;
    }

    /// Advances the engine and returns the raw (unmixed, unamplified) loop
    /// sample, linearly interpolated at the current playback rate, then run
    /// through the slow-AGC de-pump. Kept separate from the final mix so tests
    /// can measure pitch and envelope without the noise layers.
    fn next_loop(&mut self) -> f32 {
        self.advance();
        let rate = Self::playback_rate(self.smooth_rpm) as f64;
        let n = self.samples.len() as f64;
        let pos = if self.pos >= n { self.pos - n } else { self.pos };
        let i0 = pos.floor();
        let i1 = if i0 + 1.0 < n { i0 + 1.0 } else { 0.0 };
        let frac = (pos - i0) as f32;
        let a = self.samples[i0 as usize];
        let b = self.samples[i1 as usize];
        self.pos = pos + rate;
        let raw = a + (b - a) * frac;

        // De-pump: track the slow level and normalize it toward the loop's own
        // average, so the recording's baked-in breathing and the seam step no
        // longer stretch into audible pumping when pitched down.
        let rms_alpha = 1.0 - (-1.0 / (SAMPLE_RATE as f32 * AGC_TIME_S)).exp();
        self.env_rms += (raw * raw - self.env_rms) * rms_alpha;
        let gain =
            (self.rms_ref / (self.env_rms.sqrt().max(1e-6))).clamp(AGC_GAIN_MIN, AGC_GAIN_MAX);
        let g_alpha = 1.0 - (-1.0 / (SAMPLE_RATE as f32 * AGC_GAIN_SMOOTH_S)).exp();
        self.smooth_gain += (gain - self.smooth_gain) * g_alpha;
        raw * self.smooth_gain
    }

    fn next_value(&mut self) -> f32 {
        let loop_sample = self.next_loop();
        let revs = self.smooth_rpm;
        let blown = self.params.blown.load(Ordering::Relaxed);
        // A touch of roughness keeps it organic; wind rises with road speed.
        let roughness = self.next_noise() * 0.03;
        let wind = self.next_noise() * (self.smooth_speed / 340.0).clamp(0.0, 1.0) * 0.2;
        // Near-constant level: volume shouldn't chase the revs (that felt like
        // lag). The AGC already keeps the loop level steady.
        let amp = if blown { 0.04 } else { 0.6 + 0.25 * revs };

        (loop_sample * amp + wind + roughness).tanh()
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

impl Iterator for LoopEngineSound {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        Some(self.next_value())
    }
}

impl Source for LoopEngineSound {
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

/// One-shot effects. Wreck, Blow, Gear and PerfectShift are embedded recordings;
/// each falls back to its old synthesized shape if the file cannot be decoded.
/// Test stays synthesized (it is a UI confirm beep).
struct SfxBuffers {
    wreck: Vec<f32>,
    perfect: Vec<f32>,
    blow: Vec<f32>,
    gear: Vec<f32>,
    test: Vec<f32>,
}

impl SfxBuffers {
    fn new() -> Self {
        let wreck = load_wav(WAV_WRECK).unwrap_or_else(|| filtered_noise(0.7, 2600.0, 10.0, 0.9));
        let perfect = load_wav(WAV_PERFECT).unwrap_or_else(|| {
            mix(
                &sweep(0.28, 880.0, 1760.0, 3.0, 0.35),
                &sweep(0.28, 1760.0, 2640.0, 3.0, 0.18),
            )
        });
        let blow = load_wav(WAV_BLOW).unwrap_or_else(|| {
            mix(
                &sweep(1.1, 140.0, 24.0, 2.2, 0.8),
                &filtered_noise(1.1, 900.0, 3.0, 0.45),
            )
        });
        let gear = load_wav(WAV_GEAR).unwrap_or_else(|| filtered_noise(0.05, 4000.0, 60.0, 0.4));
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
    engine.append(LoopEngineSound::new(params.clone()).repeat_infinite());
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
        let mut engine = LoopEngineSound::new(params);
        assert_eq!(engine.current_span_len(), None);
        assert_eq!(engine.channels().get(), 1);
        assert_eq!(engine.sample_rate().get(), SAMPLE_RATE);
        assert_eq!(engine.total_duration(), None);
        assert_eq!(engine.samples.len(), SAMPLE_RATE as usize * 4);
        for _ in 0..4096 {
            let s = engine.next();
            assert!(s.is_some());
            let v = s.unwrap();
            assert!(v.is_finite());
            assert!((-1.5..=1.5).contains(&v));
        }
    }

    #[test]
    fn playback_rate_maps_idle_to_redline() {
        assert!((LoopEngineSound::playback_rate(0.0) - MIN_PLAYBACK_RATE).abs() < 1e-6);
        assert!((LoopEngineSound::playback_rate(1.0) - 1.0).abs() < 1e-6);
        assert!((LoopEngineSound::playback_rate(0.5) - 0.625).abs() < 1e-6);
    }

    #[test]
    fn engine_pitch_tracks_the_revs() {
        let crossings = |rpm: f32| {
            let params = Arc::new(EngineParams::default());
            params.rpm.store(f32::to_bits(rpm), Ordering::Relaxed);
            let mut engine = LoopEngineSound::new(params);
            let mut count = 0usize;
            let mut prev = engine.next_loop();
            for _ in 1..SAMPLE_RATE as usize {
                let v = engine.next_loop();
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
            redline > idle * 3,
            "redline ({redline}) should cross zero far more often than idle ({idle})"
        );
    }

    #[test]
    fn engine_loop_de_pumps_slow_amplitude_modulation() {
        // A 200 Hz tone whose level breathes 0..=1 over a 4 s cycle, standing in
        // for the recording's baked-in level drift.
        let n = SAMPLE_RATE as usize * 8;
        let tau = std::f32::consts::TAU;
        let am: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / SAMPLE_RATE as f32;
                (t * 200.0 * tau).sin() * (0.5 + 0.5 * (t * 0.25 * tau).sin())
            })
            .collect();

        let input_p2p = {
            let mut out = Vec::new();
            let win = SAMPLE_RATE as usize / 20;
            for chunk in am.chunks(win) {
                let rms = (chunk.iter().map(|s| s * s).sum::<f32>() / chunk.len() as f32).sqrt();
                out.push(rms);
            }
            let min = out.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = out.iter().cloned().fold(0.0f32, f32::max);
            (max - min) / ((max + min) / 2.0)
        };
        assert!(input_p2p > 1.5, "test fixture must breathe hard (got {input_p2p})");

        let params = Arc::new(EngineParams::default());
        let mut engine = LoopEngineSound::new(params);
        engine.samples = am;
        engine.rms_ref = (engine.samples.iter().map(|s| s * s).sum::<f32>()
            / engine.samples.len() as f32)
            .sqrt();
        engine.env_rms = engine.rms_ref * engine.rms_ref;
        engine.smooth_gain = 1.0;

        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(engine.next_loop());
        }
        let win = SAMPLE_RATE as usize / 20;
        let mut env = Vec::new();
        for chunk in out.chunks(win) {
            let rms = (chunk.iter().map(|s| s * s).sum::<f32>() / chunk.len() as f32).sqrt();
            env.push(rms);
        }
        let min = env.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = env.iter().cloned().fold(0.0f32, f32::max);
        let output_p2p = (max - min) / ((max + min) / 2.0);
        assert!(
            output_p2p < input_p2p * 0.5,
            "AGC should halve the level drift (input {input_p2p}, output {output_p2p})"
        );
    }

    #[test]
    fn wav_assets_decode_to_embedded_buffers() {
        let sfx = SfxBuffers::new();
        for kind in [Sfx::Wreck, Sfx::Blow, Sfx::Gear, Sfx::PerfectShift] {
            let buffer = sfx.get(kind);
            assert!(buffer.len() > SAMPLE_RATE as usize / 20, "{kind:?} too short");
        }
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
}
