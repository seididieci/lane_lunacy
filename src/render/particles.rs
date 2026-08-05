// SPDX-License-Identifier: MIT

//! CPU billboard rain particles.
//!
//! `RainSystem` keeps a fixed pool of streak drops inside a camera-following
//! box, advances them each frame (fast fall with wrap-around respawn), and
//! bakes the billboard quads into a per-frame vertex buffer. Streaks lean
//! along the apparent motion (gravity plus the player's forward speed) so the
//! rain reads as streaming toward the camera. The billboard build is the
//! reusable piece: the same CPU-quad pass can later feed dust puffs or mist.

use std::time::{SystemTime, UNIX_EPOCH};

use glam::Vec3;

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

impl RainSystem {
    pub fn new() -> Self {
        let mut rng = Rng::new();
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
            let outside = o.y < BOX_Y_MIN
                || o.x.abs() > BOX_X
                || o.z < BOX_Z_NEAR
                || o.z > BOX_Z_FAR;
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
            push_quad(&mut out, top + l, top + r, bottom + r, bottom + l, color);
        }
        out
    }
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
        push_quad(&mut out, tl, tr, br, bl, color);
    }
    out
}

/// Bakes the visible headlights of oncoming traffic: a warm-white disc with a
/// faint halo per light, plus a soft light pool cast on the asphalt in front of
/// each light. `lights` is a list of `(center, forward_dir)` pairs: the light
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
    let mut out = Vec::with_capacity(lights.len() * 18);
    for (center, forward) in lights {
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
        push_quad(&mut out, tl, tr, br, bl, disc);
        // Soft halo around the disc.
        let tl = *center - halo_side + halo_up;
        let tr = *center + halo_side + halo_up;
        let br = *center + halo_side - halo_up;
        let bl = *center - halo_side - halo_up;
        push_quad(&mut out, tl, tr, br, bl, halo);
        // Light pool cast on the road ahead of the light, following the full
        // horizontal beam direction (so it stays on the asphalt around curves).
        let pool_center = Vec3::new(center.x, 0.02, center.z);
        if pool_center.z - eye.z <= 5.0 {
            let across = forward.cross(Vec3::Y).normalize_or_zero() * 1.7;
            let along = Vec3::new(forward.x, 0.0, forward.z).normalize_or_zero() * 5.0;
            let pool = [1.0, 0.95, 0.82, intensity * 0.18];
            let tl = pool_center + along + across;
            let tr = pool_center + along - across;
            let br = pool_center - along - across;
            let bl = pool_center - along + across;
            push_quad(&mut out, tl, tr, br, bl, pool);
        }
    }
    out
}

fn spawn_drop(rng: &mut Rng, eye: Vec3) -> Drop {
    let pos = eye + Vec3::new(
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
) {
    out.push(ParticleVertex { position: tl.to_array(), uv: [0.0, 0.0], color });
    out.push(ParticleVertex { position: tr.to_array(), uv: [1.0, 0.0], color });
    out.push(ParticleVertex { position: br.to_array(), uv: [1.0, 1.0], color });
    out.push(ParticleVertex { position: tl.to_array(), uv: [0.0, 0.0], color });
    out.push(ParticleVertex { position: br.to_array(), uv: [1.0, 1.0], color });
    out.push(ParticleVertex { position: bl.to_array(), uv: [0.0, 1.0], color });
}

/// Small xorshift PRNG (no external rand dependency).
struct Rng(u64);

#[cfg(test)]
mod tests {
    use super::*;

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
    fn headlights_emit_discs_halos_and_a_road_pool() {
        let lights = vec![(Vec3::new(2.0, 0.8, -10.0), Vec3::new(0.0, 0.0, 1.0))];
        let verts = build_headlights(&lights, Vec3::ZERO, Vec3::X, 1.0);
        assert_eq!(verts.len(), 18, "1 disc + 1 halo + 1 pool per light = 3 quads = 18 verts");
        assert!(
            verts.iter().all(|v| v.color[0] > 0.9 && v.color[1] > 0.9),
            "headlights are warm white"
        );
        assert!(
            verts.iter().any(|v| v.position[1] < 0.1),
            "the road pool lies flat on the asphalt"
        );
    }

    #[test]
    fn headlights_turn_off_with_zero_intensity() {
        let lights = vec![(Vec3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0))];
        assert!(build_headlights(&lights, Vec3::ZERO, Vec3::X, 0.0).is_empty());
    }
}

impl Rng {
    fn new() -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before epoch")
            .as_nanos() as u64;
        Rng(nanos | 1)
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
