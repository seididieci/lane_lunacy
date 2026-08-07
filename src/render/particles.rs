// SPDX-License-Identifier: MIT

//! CPU billboard rain particles, drift dust, low-hanging mist, and traffic
//! lights.
//!
//! `RainSystem` keeps a fixed pool of streak drops inside a camera-following
//! box, advances them each frame (fast fall with wrap-around respawn), and
//! bakes the billboard quads into a per-frame vertex buffer. Streaks lean
//! along the apparent motion (gravity plus the player's forward speed) so the
//! rain reads as streaming toward the camera. `DustSystem` spawns clumped,
//! slow-drifting cloud puffs on hard steering/sideslip; `MistSystem` lays a
//! camera-following pool of large, soft cloud puffs along the road as local
//! ground mist, complementing the tile-based sky dome; `build_taillights`
//! and `build_headlights` are the night-time traffic glows. All share the same
//! camera-facing CPU-quad pass and a sprite atlas (cell 0 = rain gaussian,
//! cells 1..=3 = organic cloud shapes).

use std::time::{SystemTime, UNIX_EPOCH};

use glam::Vec3;

use crate::math::smoothstep;
use crate::surface::DustProfile;
use crate::vertex::ParticleVertex;

const MAX_DROPS: usize = 700;
const FALL_SPEED: f32 = 32.0;
const BOX_X: f32 = 12.0;
const BOX_Y_MIN: f32 = -3.5; // relative to the eye; reaches down to road level
const BOX_Y_MAX: f32 = 22.0;
const BOX_Z_NEAR: f32 = -38.0; // ahead of the camera
const BOX_Z_FAR: f32 = 10.0; // behind the camera

#[derive(Clone, Copy, Debug)]
struct Drop {
    pos: Vec3,
    axis: Vec3,
    width: f32,
    length: f32,
    alpha: f32,
    color: Vec3,
}

pub struct RainSystem {
    drops: Vec<Drop>,
    rng: Rng,
}

impl Default for RainSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl RainSystem {
    pub fn new() -> Self {
        let mut rng = Rng::new();
        let drops = (0..MAX_DROPS)
            .map(|_| spawn_drop(&mut rng, Vec3::ZERO))
            .collect();
        RainSystem { drops, rng }
    }

    /// Deterministic variant for the headless snapshot path: the drop field is
    /// spawned from the scenario seed instead of the clock.
    pub fn with_seed(seed: u64) -> Self {
        let mut rng = Rng::from_seed(seed);
        let drops = (0..MAX_DROPS)
            .map(|_| spawn_drop(&mut rng, Vec3::ZERO))
            .collect();
        RainSystem { drops, rng }
    }

    /// Advances all drops. `eye` anchors the respawn volume; `car_speed` (m/s)
    /// drives the apparent streak lean (the car drives toward -z).
    pub fn update(&mut self, dt: f32, eye: Vec3, car_speed: f32) {
        // Rain is world-stationary (only gravity moves it); the camera drives
        // past it, so the box wrap streams it behind. The streak axis is the
        // apparent motion relative to the camera (gravity + forward speed).
        let rel = Vec3::new(0.0, -FALL_SPEED, 0.0);
        let axis = Vec3::new(0.0, -FALL_SPEED, car_speed).normalize_or_zero();
        for drop in &mut self.drops {
            drop.pos += rel * dt;
            drop.axis = axis;
            let o = drop.pos - eye;
            let outside =
                o.y < BOX_Y_MIN || o.x.abs() > BOX_X || o.z < BOX_Z_NEAR || o.z > BOX_Z_FAR;
            if outside {
                *drop = spawn_drop(&mut self.rng, eye);
            }
        }
    }

    /// Bakes one camera-facing streak quad per drop. Drops just behind the
    /// camera are skipped so they don't pop into view at the respawn edge.
    /// `intensity` (0..1) scales each drop's alpha for smooth weather ramps.
    pub fn build_vertices(&self, eye: Vec3, right: Vec3, intensity: f32) -> Vec<ParticleVertex> {
        if intensity <= 0.001 {
            return Vec::new();
        }
        let right = right.normalize_or_zero();
        let mut out = Vec::with_capacity(self.drops.len() * 6);
        for d in &self.drops {
            if d.pos.z - eye.z > 5.0 {
                continue;
            }
            let color = [d.color.x, d.color.y, d.color.z, d.alpha * intensity];
            let half_len = d.length * 0.5;
            let half_w = d.width * 0.5;
            let top = d.pos + d.axis * half_len;
            let bottom = d.pos - d.axis * half_len;
            let l = -right * half_w;
            let r = right * half_w;
            push_quad(
                &mut out,
                top + l,
                top + r,
                bottom + r,
                bottom + l,
                color,
                0.0,
            );
        }
        out
    }
}

/// Ground-constrained dust clouds kicked up on hard steering/sideslip.
/// World-stationary (the camera drives past them), capped and recycled.
const MAX_PUFFS: usize = 384;
/// Emission ticks per second at full drift; each tick spawns a small clump.
const PUFFS_PER_SEC: f32 = 22.0;
/// Puffs spawned per emission tick, tightly clustered so they overlap into a
/// single cloud instead of reading as separate dots.
const PUFFS_PER_TICK: usize = 5;
/// Weak gravity so the cloud lingers and drifts gently upward ("evaporating")
/// instead of collapsing onto the asphalt or billowing like steam.
const PUFF_GRAVITY: f32 = -1.0;
/// Longest puff life in seconds; randomized below this so puffs desync. Long
/// enough that at low speed the cloud stays around the tires, rises into a
/// visible haze and fades in place, while at speed the car/camera drives past
/// it so it streams backward through the camera and culls behind the eye.
const PUFF_MAX_LIFE: f32 = 2.4;
/// Lowest height a puff's center keeps (the road plane is y=0.02 with depth
/// write on; sitting above it avoids being depth-culled by the asphalt).
const PUFF_MIN_Y: f32 = 0.08;

#[derive(Clone, Copy, Debug)]
struct Puff {
    pos: Vec3,
    vel: Vec3,
    life: f32,
    size: f32,
    color: Vec3,
    alpha: f32,
    variant: f32,
    up: f32,
}

pub struct DustSystem {
    puffs: Vec<Puff>,
    rng: Rng,
    spawn_accum: f32,
}

impl Default for DustSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl DustSystem {
    pub fn new() -> Self {
        DustSystem {
            puffs: Vec::with_capacity(MAX_PUFFS),
            rng: Rng::new(),
            spawn_accum: 0.0,
        }
    }

    /// Deterministic variant for the headless snapshot path.
    pub fn with_seed(seed: u64) -> Self {
        DustSystem {
            puffs: Vec::with_capacity(MAX_PUFFS),
            rng: Rng::from_seed(seed),
            spawn_accum: 0.0,
        }
    }

    /// Advances alive puffs and spawns fresh clumps from `drift` (0..1), the
    /// material-scaled drift metric. Puffs rise off the rear corners, drift
    /// slowly backward and upward, and fade out after lingering.
    pub fn update(
        &mut self,
        dt: f32,
        drift: f32,
        profile: &DustProfile,
        rear_points: [Vec3; 2],
        car_forward: Vec3,
    ) {
        for p in &mut self.puffs {
            p.life -= dt;
            p.vel.y += PUFF_GRAVITY * dt;
            p.pos += p.vel * dt;
            if p.pos.y < PUFF_MIN_Y {
                p.pos.y = PUFF_MIN_Y;
                p.vel.y = p.vel.y.abs() * 0.2;
            }
        }
        self.puffs.retain(|p| p.life > 0.0);

        if drift <= 0.001 {
            return;
        }
        self.spawn_accum += drift * PUFFS_PER_SEC * dt;
        while self.spawn_accum >= 1.0 {
            self.spawn_accum -= 1.0;
            self.spawn_clump(profile, &rear_points, car_forward);
        }
    }

    /// Spawns a tight clump at one rear corner: a few puffs within a few
    /// centimeters of each other, pushed outward off the tire and gently
    /// backward and up, so the cluster hangs together, spreads across the road
    /// behind the car and rises into a visible cloud before fading.
    fn spawn_clump(&mut self, profile: &DustProfile, rear_points: &[Vec3; 2], car_forward: Vec3) {
        let idx = (self.rng.next() & 1) as usize;
        let wheel = rear_points[idx];
        // The rear points are the left (index 0) and right (index 1) corners;
        // outward means away from the car's centerline along the right vector.
        let outward = car_forward.cross(Vec3::Y) * if idx == 1 { 1.0 } else { -1.0 };
        for _ in 0..PUFFS_PER_TICK {
            if self.puffs.len() >= MAX_PUFFS {
                self.puffs.remove(0);
            }
            let kick_back = -car_forward * self.rng.range(0.4, 1.0);
            let vel = kick_back
                + outward * self.rng.range(0.3, 0.7)
                + Vec3::new(
                    self.rng.range(-0.15, 0.15),
                    self.rng.range(0.6, 1.2),
                    self.rng.range(-0.15, 0.15),
                );
            self.puffs.push(Puff {
                pos: wheel
                    + Vec3::new(
                        self.rng.range(-0.05, 0.05),
                        self.rng.range(0.10, 0.25),
                        self.rng.range(-0.05, 0.05),
                    ),
                vel,
                life: self.rng.range(1.5, PUFF_MAX_LIFE),
                size: 0.42 * profile.puff_scale * self.rng.range(0.85, 1.15),
                color: Vec3::from(profile.color),
                alpha: profile.alpha * 0.28,
                variant: 1.0 + (self.rng.next() % 3) as f32,
                up: self.rng.range(0.7, 1.4),
            });
        }
    }

    /// Bakes one camera-facing cloud quad per puff. Puffs behind the camera
    /// are skipped; size swells slowly and alpha pops in fast, holds, then
    /// fades out in the last part of the puff's life.
    pub fn build_vertices(&self, eye: Vec3, right: Vec3) -> Vec<ParticleVertex> {
        let right = right.normalize_or_zero();
        let mut out = Vec::with_capacity(self.puffs.len() * 6);
        for p in &self.puffs {
            if p.pos.z - eye.z > 5.0 {
                continue;
            }
            let t = 1.0 - p.life / PUFF_MAX_LIFE;
            let fade = smoothstep(0.0, 0.12, t) * (1.0 - smoothstep(0.55, 1.0, t));
            let size = p.size * (0.9 + t * 0.7);
            let color = [p.color.x, p.color.y, p.color.z, p.alpha * fade];
            let side = right * size;
            let up = Vec3::Y * size * 0.7 * p.up;
            let tl = p.pos - side + up;
            let tr = p.pos + side + up;
            let br = p.pos + side - up;
            let bl = p.pos - side - up;
            push_quad(&mut out, tl, tr, br, bl, color, p.variant);
        }
        out
    }
}

/// Low-hanging ground mist: a fixed, camera-following pool of large soft cloud
/// puffs at road level, recycled like the rain box so the car drives through a
/// persistent bank. Unlike the tile-based dome (task 2), these are local
/// 2.5D billboards, so they read as "hybrid" clouds near the camera.
const MAX_MIST_PUFFS: usize = 96;
/// Half-width of the mist volume around the camera (metres).
const MIST_X: f32 = 14.0;
/// Vertical band of the mist volume in world space. The pool is anchored to a
/// road-level point under the camera (y≈0), so the bank hugs the asphalt.
const MIST_Y_MIN: f32 = 0.15;
const MIST_Y_MAX: f32 = 1.6;
/// Depth range of the mist volume ahead of / behind the camera.
const MIST_Z_NEAR: f32 = -45.0;
const MIST_Z_FAR: f32 = 8.0;
/// Slow horizontal churn so the bank drifts instead of looking frozen.
const MIST_DRIFT: f32 = 1.2;

#[derive(Clone, Copy, Debug)]
struct MistPuff {
    pos: Vec3,
    vel: Vec3,
    size: f32,
    alpha: f32,
    variant: f32,
}

pub struct MistSystem {
    puffs: Vec<MistPuff>,
    rng: Rng,
}

impl Default for MistSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl MistSystem {
    pub fn new() -> Self {
        let mut rng = Rng::new();
        let puffs = (0..MAX_MIST_PUFFS)
            .map(|_| spawn_mist_puff(&mut rng, Vec3::ZERO))
            .collect();
        MistSystem { puffs, rng }
    }

    /// Deterministic variant for the headless snapshot path.
    pub fn with_seed(seed: u64) -> Self {
        let mut rng = Rng::from_seed(seed);
        let puffs = (0..MAX_MIST_PUFFS)
            .map(|_| spawn_mist_puff(&mut rng, Vec3::ZERO))
            .collect();
        MistSystem { puffs, rng }
    }

    /// Advances the pool: puffs drift slowly in place and recycle whenever the
    /// car moves past them. `anchor` is the road-level point (y≈0) the volume
    /// follows; the bank stays grounded there.
    pub fn update(&mut self, dt: f32, anchor: Vec3) {
        for p in &mut self.puffs {
            p.pos += p.vel * dt;
            let o = p.pos - anchor;
            let outside = o.y < MIST_Y_MIN
                || o.y > MIST_Y_MAX
                || o.x.abs() > MIST_X
                || o.z < MIST_Z_NEAR
                || o.z > MIST_Z_FAR;
            if outside {
                *p = spawn_mist_puff(&mut self.rng, anchor);
            }
        }
    }

    /// Bakes one camera-facing cloud quad per puff. `intensity` (0..1) scales
    /// the alpha so the bank fades in/out with the weather; `tint` dresses the
    /// puffs with the current fog color so they blend with the lighting.
    pub fn build_vertices(
        &self,
        eye: Vec3,
        right: Vec3,
        intensity: f32,
        tint: Vec3,
    ) -> Vec<ParticleVertex> {
        if intensity <= 0.001 {
            return Vec::new();
        }
        let right = right.normalize_or_zero();
        let mut out = Vec::with_capacity(self.puffs.len() * 6);
        for p in &self.puffs {
            if p.pos.z - eye.z > 5.0 {
                continue; // behind the camera
            }
            let color = [tint.x, tint.y, tint.z, p.alpha * intensity];
            let side = right * p.size;
            // Flatten the quads so the mist hugs the road as a low bank.
            let up = Vec3::Y * p.size * 0.4;
            let tl = p.pos - side + up;
            let tr = p.pos + side + up;
            let br = p.pos + side - up;
            let bl = p.pos - side - up;
            push_quad(&mut out, tl, tr, br, bl, color, p.variant);
        }
        out
    }
}

fn spawn_mist_puff(rng: &mut Rng, anchor: Vec3) -> MistPuff {
    let pos = anchor
        + Vec3::new(
            rng.range(-MIST_X, MIST_X),
            rng.range(MIST_Y_MIN, MIST_Y_MAX),
            rng.range(MIST_Z_NEAR, MIST_Z_FAR),
        );
    MistPuff {
        pos,
        vel: Vec3::new(
            rng.range(-MIST_DRIFT, MIST_DRIFT),
            rng.range(0.05, MIST_DRIFT * 0.5),
            rng.range(-MIST_DRIFT, MIST_DRIFT),
        ),
        size: rng.range(2.5, 5.0),
        alpha: rng.range(0.10, 0.18),
        variant: 1.0 + (rng.next() % 3) as f32,
    }
}

/// Drift intensity (0..1) driving dust emission, scaled by the surface's
/// dustiness (`emission`). Combines three physical cues plus a minimal
/// baseline so dustier materials always trail a little:
///   - sideslip (`lateral_velocity`), the world-space lateral offset rate;
///   - hard steering (`steer`), so yanking the wheel bursts dust instantly
///     even before slip builds up;
///   - launch/traction (`throttle` at low speed), so take-off kicks a cloud.
pub fn drift_intensity(
    speed: f32,
    lateral_velocity: f32,
    steer: f32,
    throttle: bool,
    emission: f32,
) -> f32 {
    if speed <= 0.0 || emission <= 0.0 {
        return 0.0;
    }
    let speed_gate = smoothstep(6.0, 18.0, speed);
    let slip = (lateral_velocity.abs() / 4.0).clamp(0.0, 1.0) * speed_gate;
    let steer_kick = smoothstep(0.6, 0.9, steer.abs()) * speed_gate;
    let launch = if throttle {
        (1.0 - speed / 6.0).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let combined = (slip + steer_kick + launch + 0.12 * speed_gate).clamp(0.0, 1.0);
    combined * emission
}

/// Bakes a soft radial RGBA sprite (gaussian blob, white core fading to clear)
/// used as the particle texture. Stretched over a thin quad it forms a soft
/// streak; the gaussian profile has no hard silhouette at the rim.
pub fn generate_soft_sprite(size: u32) -> Vec<u8> {
    let n = size as usize;
    let center = (n as f32 - 1.0) * 0.5;
    let radius = n as f32 * 0.5;
    let falloff = 4.0 / (radius * radius);
    let mut out = Vec::with_capacity(n * n * 4);
    for py in 0..n {
        for px in 0..n {
            let dx = px as f32 - center;
            let dy = py as f32 - center;
            let d2 = dx * dx + dy * dy;
            let a = (-d2 * falloff).exp();
            out.push(255);
            out.push(255);
            out.push(255);
            out.push((a * 255.0) as u8);
        }
    }
    out
}

/// Bakes an organic, irregular cloud sprite (RGBA) for dust puffs. A large
/// central blob plus a couple of near-center lobes guarantee a dense core,
/// while scattered outer blobs shape the billowy rim. Alpha uses a soft
/// saturating falloff so overlapping plateaus merge seamlessly into one
/// churning cloud instead of separate dots. Different seeds reshape the
/// scattered lobes while keeping the guaranteed core.
pub fn generate_cloud_sprite(size: u32, seed: u64) -> Vec<u8> {
    let n = size as usize;
    let mut s = seed | 1;
    let mut next = move || {
        let mut x = s;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s = x;
        x as f32 / u64::MAX as f32
    };
    // Blob kernels: (offset_x, offset_y, radius) in -1..1 cell units.
    let blobs: Vec<(f32, f32, f32)> = (0..8)
        .map(|i| match i {
            0 => (0.0, 0.0, 0.30 + next() * 0.08),
            1 => (
                (next() - 0.5) * 0.36,
                (next() - 0.5) * 0.36,
                0.18 + next() * 0.10,
            ),
            _ => (
                (next() - 0.5) * 0.8,
                (next() - 0.5) * 0.8,
                0.15 + next() * 0.20,
            ),
        })
        .collect();
    let center = (n as f32 - 1.0) * 0.5;
    let radius = n as f32 * 0.5;
    let mut out = Vec::with_capacity(n * n * 4);
    for py in 0..n {
        for px in 0..n {
            let nx = (px as f32 - center) / radius; // -1..1
            let ny = (py as f32 - center) / radius;
            // Hard-transparent rim so atlas cells never bleed under bilinear
            // filtering.
            if nx.abs() > 0.92 || ny.abs() > 0.92 {
                out.extend_from_slice(&[255, 255, 255, 0]);
                continue;
            }
            let mut d = 0.0;
            for &(bx, by, br) in &blobs {
                let dx = nx - bx;
                let dy = ny - by;
                d += (-(dx * dx + dy * dy) / (br * br)).exp();
            }
            // Soft saturating falloff: dense interiors reach full alpha, the
            // single rim fades to clear.
            let a = 1.0 - (-d * 2.2).exp();
            // Low-frequency internal shading (gently darker lobes): kept low in
            // contrast so overlapping puffs read as one smooth haze rather than
            // bright spots/rings when they blend together.
            let shade = 0.72 + 0.28 * d / (d + 0.8);
            let c = (255.0 * shade).clamp(0.0, 255.0);
            out.push(c as u8);
            out.push(c as u8);
            out.push(c as u8);
            out.push((a * 255.0) as u8);
        }
    }
    out
}

/// Bakes camera-facing red taillight quads for each rear-light center.
/// Lights behind the camera are skipped. `intensity` (0..1) scales the glow;
/// callers pass the effective night darkness so lights switch off by day.
pub fn build_taillights(
    centers: &[Vec3],
    eye: Vec3,
    right: Vec3,
    intensity: f32,
) -> Vec<ParticleVertex> {
    if intensity <= 0.001 {
        return Vec::new();
    }
    let right = right.normalize_or_zero();
    let half_w = 0.14;
    let half_h = 0.19;
    let color = [1.0, 0.10, 0.10, intensity * 0.9];
    let mut out = Vec::with_capacity(centers.len() * 6);
    for c in centers {
        if c.z - eye.z > 5.0 {
            continue; // behind the camera
        }
        let up = Vec3::Y * half_h;
        let side = right * half_w;
        let tl = c - side + up;
        let tr = c + side + up;
        let br = c + side - up;
        let bl = c - side - up;
        push_quad(&mut out, tl, tr, br, bl, color, 0.0);
    }
    out
}

/// Bakes the visible headlights of oncoming traffic: a warm-white disc with a
/// faint halo per light. Road illumination is projected in the mesh shader for
/// uniformity, so this pass renders only the visible lamp glow. `lights` is a
/// list of `(center, forward_dir)` pairs: the light
/// at its real anchor position (lateral offset and height baked in by the
/// caller), and the direction its beams shine. Everything behind the camera is
/// skipped. `intensity` (0..1) scales the glow.
pub fn build_headlights(
    lights: &[(Vec3, Vec3)],
    eye: Vec3,
    right: Vec3,
    intensity: f32,
) -> Vec<ParticleVertex> {
    if intensity <= 0.001 {
        return Vec::new();
    }
    let right = right.normalize_or_zero();
    let mut out = Vec::with_capacity(lights.len() * 12);
    for (center, _forward) in lights {
        if center.z - eye.z > 5.0 {
            continue; // behind the camera
        }
        let disc = [1.0, 0.98, 0.90, intensity * 0.95];
        let halo = [1.0, 0.92, 0.75, intensity * 0.25];
        let up = Vec3::Y * 0.22;
        let side = right * 0.18;
        let halo_up = Vec3::Y * 0.36;
        let halo_side = right * 0.30;
        // Disc.
        let tl = *center - side + up;
        let tr = *center + side + up;
        let br = *center + side - up;
        let bl = *center - side - up;
        push_quad(&mut out, tl, tr, br, bl, disc, 0.0);
        // Soft halo around the disc.
        let tl = *center - halo_side + halo_up;
        let tr = *center + halo_side + halo_up;
        let br = *center + halo_side - halo_up;
        let bl = *center - halo_side - halo_up;
        push_quad(&mut out, tl, tr, br, bl, halo, 0.0);
    }
    out
}

fn spawn_drop(rng: &mut Rng, eye: Vec3) -> Drop {
    let pos = eye
        + Vec3::new(
            rng.range(-BOX_X, BOX_X),
            rng.range(BOX_Y_MIN, BOX_Y_MAX),
            rng.range(BOX_Z_NEAR, BOX_Z_FAR),
        );
    Drop {
        pos,
        axis: Vec3::NEG_Y,
        width: rng.range(0.035, 0.07),
        length: rng.range(0.6, 1.3),
        alpha: rng.range(0.25, 0.5),
        color: Vec3::new(0.55, 0.62, 0.74),
    }
}

fn push_quad(
    out: &mut Vec<ParticleVertex>,
    tl: Vec3,
    tr: Vec3,
    br: Vec3,
    bl: Vec3,
    color: [f32; 4],
    sprite_variant: f32,
) {
    out.push(ParticleVertex {
        position: tl.to_array(),
        uv: [0.0, 0.0],
        color,
        sprite_variant,
    });
    out.push(ParticleVertex {
        position: tr.to_array(),
        uv: [1.0, 0.0],
        color,
        sprite_variant,
    });
    out.push(ParticleVertex {
        position: br.to_array(),
        uv: [1.0, 1.0],
        color,
        sprite_variant,
    });
    out.push(ParticleVertex {
        position: tl.to_array(),
        uv: [0.0, 0.0],
        color,
        sprite_variant,
    });
    out.push(ParticleVertex {
        position: br.to_array(),
        uv: [1.0, 1.0],
        color,
        sprite_variant,
    });
    out.push(ParticleVertex {
        position: bl.to_array(),
        uv: [0.0, 1.0],
        color,
        sprite_variant,
    });
}

/// Small xorshift PRNG (no external rand dependency).
struct Rng(u64);

impl Rng {
    fn new() -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before epoch")
            .as_nanos() as u64;
        Rng(nanos | 1)
    }

    /// Deterministic RNG for the headless snapshot path, derived from the
    /// scenario seed so identical seeds produce identical rain/dust.
    fn from_seed(seed: u64) -> Self {
        Rng(seed | 1)
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        let r = (self.next() >> 11) as f32 / (1u64 << 53) as f32;
        lo + r * (hi - lo)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rain_with_seed_is_deterministic() {
        let fingerprint = |system: &RainSystem| {
            system
                .build_vertices(Vec3::ZERO, Vec3::X, 1.0)
                .iter()
                .map(|v| v.position[0] * 1e3 + v.position[1] * 1e1 + v.position[2])
                .fold(0.0, |acc, f| acc + f)
        };
        assert_eq!(
            fingerprint(&RainSystem::with_seed(42)),
            fingerprint(&RainSystem::with_seed(42))
        );
        assert_ne!(
            fingerprint(&RainSystem::with_seed(42)),
            fingerprint(&RainSystem::with_seed(7))
        );
    }

    #[test]
    fn taillights_emit_quads_and_skip_behind_the_camera() {
        let centers = vec![
            Vec3::new(2.0, 0.8, -10.0),
            Vec3::new(2.0, 0.8, 20.0), // behind the eye at origin
        ];
        let verts = build_taillights(&centers, Vec3::ZERO, Vec3::X, 1.0);
        assert_eq!(verts.len(), 6, "one in-front light only");
        assert!(verts.iter().all(|v| v.color[0] > 0.9 && v.color[1] < 0.2));
    }

    #[test]
    fn taillights_turn_off_with_zero_intensity() {
        let centers = vec![Vec3::new(0.0, 0.0, -5.0)];
        assert!(build_taillights(&centers, Vec3::ZERO, Vec3::X, 0.0).is_empty());
    }

    #[test]
    fn traffic_lights_use_the_rain_sprite_cell() {
        let centers = vec![Vec3::new(0.0, 0.0, -5.0)];
        let verts = build_taillights(&centers, Vec3::ZERO, Vec3::X, 1.0);
        assert!(verts.iter().all(|v| v.sprite_variant == 0.0));
        let lights = vec![(Vec3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0))];
        let verts = build_headlights(&lights, Vec3::ZERO, Vec3::X, 1.0);
        assert!(verts.iter().all(|v| v.sprite_variant == 0.0));
    }

    #[test]
    fn cloud_sprites_are_opaque_in_the_middle_and_transparent_at_the_rim() {
        let sprite = generate_cloud_sprite(128, 42);
        let n = 128usize;
        let max_alpha = sprite.chunks_exact(4).map(|px| px[3]).max().unwrap_or(0) as f32 / 255.0;
        assert!(max_alpha > 0.9, "the cloud core is dense: {max_alpha}");
        // Rim texels must be fully transparent so atlas cells don't bleed.
        for (px, py) in [(0usize, 0usize), (n - 1, 0), (0, n - 1), (n - 1, n - 1)] {
            let idx = (py * n + px) * 4 + 3;
            assert_eq!(sprite[idx], 0, "rim texel must be transparent");
        }
    }

    #[test]
    fn headlights_emit_discs_and_halos() {
        let lights = vec![(Vec3::new(2.0, 0.8, -10.0), Vec3::new(0.0, 0.0, 1.0))];
        let verts = build_headlights(&lights, Vec3::ZERO, Vec3::X, 1.0);
        assert_eq!(
            verts.len(),
            12,
            "1 disc + 1 halo per light = 2 quads = 12 verts"
        );
        assert!(
            verts.iter().all(|v| v.color[0] > 0.9 && v.color[1] > 0.9),
            "headlights are warm white"
        );
    }

    #[test]
    fn headlights_turn_off_with_zero_intensity() {
        let lights = vec![(Vec3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0))];
        assert!(build_headlights(&lights, Vec3::ZERO, Vec3::X, 0.0).is_empty());
    }

    #[test]
    fn drift_intensity_gates_on_speed_slip_steer_launch_and_material() {
        // Non-dusty surface or standstill -> no dust.
        assert_eq!(drift_intensity(40.0, 6.0, 1.0, false, 0.0), 0.0);
        assert_eq!(drift_intensity(0.0, 6.0, 1.0, true, 1.0), 0.0);
        // Parked just past the stop threshold (engine blow) with no throttle
        // must not trail dust either: the ambient baseline is speed-gated.
        assert_eq!(
            drift_intensity(0.3, 6.0, 1.0, false, 1.0),
            0.0,
            "no dust parked after a blow"
        );

        // Straight cruise on a dusty surface: only the minimal ambient trail.
        let ambient = drift_intensity(40.0, 0.0, 0.0, false, 1.0);
        assert!(
            (0.0..0.3).contains(&ambient),
            "minimal ambient baseline: {ambient}"
        );

        // Each cue adds dust over the baseline.
        let slip = drift_intensity(40.0, 6.0, 0.0, false, 1.0);
        let steer = drift_intensity(40.0, 0.0, 1.0, false, 1.0);
        let launch = drift_intensity(3.0, 0.0, 0.0, true, 1.0);
        assert!(slip > ambient, "sideslip adds dust");
        assert!(steer > ambient, "hard steering adds dust");
        assert!(launch > ambient, "launch adds dust");

        // Emission scales the whole thing.
        let dusty = drift_intensity(40.0, 6.0, 1.0, false, 1.0);
        let worn = drift_intensity(40.0, 6.0, 1.0, false, 0.6);
        assert!(worn > 0.0 && worn < dusty, "dust scales with emission");
    }

    #[test]
    fn dust_spawns_on_drift_and_builds_camera_facing_quads() {
        let mut dust = DustSystem::new();
        let profile = DustProfile {
            emission: 1.0,
            color: [0.5, 0.45, 0.4],
            puff_scale: 1.0,
            alpha: 0.6,
        };
        let rear = [Vec3::new(-0.9, 0.0, -2.0), Vec3::new(0.9, 0.0, -2.0)];
        dust.update(0.5, 1.0, &profile, rear, Vec3::NEG_Z);
        assert!(!dust.puffs.is_empty(), "drift spawns puffs");
        let verts = dust.build_vertices(Vec3::ZERO, Vec3::X);
        assert_eq!(verts.len(), dust.puffs.len() * 6, "one quad per puff");
        assert!(
            verts.iter().all(|v| v.position[1] < 1.5),
            "dust stays a low cloud near the road surface"
        );
        assert!(
            verts
                .iter()
                .all(|v| (1.0..=3.0).contains(&v.sprite_variant)),
            "dust uses cloud sprite cells"
        );
    }

    #[test]
    fn dust_culls_puffs_behind_the_camera() {
        let mut dust = DustSystem::new();
        let profile = DustProfile {
            emission: 1.0,
            color: [0.5, 0.45, 0.4],
            puff_scale: 1.0,
            alpha: 0.6,
        };
        // Rear points behind the eye (z > 5); driving toward +Z keeps them there.
        let rear = [Vec3::new(-0.9, 0.0, 10.0), Vec3::new(0.9, 0.0, 10.0)];
        dust.update(0.5, 1.0, &profile, rear, Vec3::Z);
        assert!(
            !dust.puffs.is_empty(),
            "a full second of drift must spawn puffs"
        );
        assert!(dust.build_vertices(Vec3::ZERO, Vec3::X).is_empty());
    }

    #[test]
    fn dust_puffs_age_out_and_stop_spawning_without_drift() {
        let mut dust = DustSystem::new();
        let profile = DustProfile {
            emission: 1.0,
            color: [0.5, 0.45, 0.4],
            puff_scale: 1.0,
            alpha: 0.6,
        };
        let rear = [Vec3::new(-0.9, 0.0, -2.0), Vec3::new(0.9, 0.0, -2.0)];
        dust.update(1.0, 1.0, &profile, rear, Vec3::NEG_Z);
        assert!(!dust.puffs.is_empty());
        // All puffs die within 2.5s of life; no drift spawns replacements.
        dust.update(2.5, 0.0, &profile, rear, Vec3::NEG_Z);
        assert!(dust.puffs.is_empty());
        assert!(dust.build_vertices(Vec3::ZERO, Vec3::X).is_empty());
    }

    #[test]
    fn mist_with_seed_is_deterministic() {
        let fingerprint = |system: &MistSystem| {
            system
                .build_vertices(Vec3::ZERO, Vec3::X, 1.0, Vec3::ONE)
                .iter()
                .map(|v| v.position[0] * 1e3 + v.position[1] * 1e1 + v.position[2])
                .fold(0.0, |acc, f| acc + f)
        };
        assert_eq!(
            fingerprint(&MistSystem::with_seed(42)),
            fingerprint(&MistSystem::with_seed(42))
        );
        assert_ne!(
            fingerprint(&MistSystem::with_seed(42)),
            fingerprint(&MistSystem::with_seed(7))
        );
    }

    #[test]
    fn mist_emits_quads_only_with_intensity() {
        let mist = MistSystem::with_seed(42);
        assert!(
            mist.build_vertices(Vec3::ZERO, Vec3::X, 0.0, Vec3::ONE)
                .is_empty(),
            "no mist at zero intensity"
        );
        let verts = mist.build_vertices(Vec3::ZERO, Vec3::X, 1.0, Vec3::ONE);
        assert!(!verts.is_empty(), "mist must render");
        assert_eq!(verts.len() % 6, 0, "complete quads only");
        assert!(
            verts.len() <= MAX_MIST_PUFFS * 6,
            "pool is capped (some puffs culled behind the camera)"
        );
        assert!(
            verts
                .iter()
                .all(|v| (1.0..=3.0).contains(&v.sprite_variant)),
            "mist uses cloud sprite cells"
        );
    }

    #[test]
    fn mist_stays_a_low_near_camera_bank() {
        let mut mist = MistSystem::with_seed(42);
        // Anchor the bank at road level; the eye sits 4m above it.
        let anchor = Vec3::ZERO;
        let eye = Vec3::new(0.0, 4.0, 0.0);
        mist.update(1.0, anchor);
        let verts = mist.build_vertices(eye, Vec3::X, 1.0, Vec3::ONE);
        assert!(
            verts.iter().all(|v| v.position[1] < eye.y),
            "mist hugs the road below the eye"
        );
    }

    #[test]
    fn mist_culls_puffs_behind_the_camera() {
        let mist = MistSystem::with_seed(42);
        // Puffs ahead of the eye have z < 0; a camera looking along -Z sees
        // them, while anything with z > eye.z + 5 is behind and culled.
        let verts = mist.build_vertices(Vec3::ZERO, Vec3::X, 1.0, Vec3::ONE);
        assert!(verts.iter().all(|v| v.position[2] <= 5.0));
    }
}
