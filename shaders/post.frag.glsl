#version 450
// SPDX-License-Identifier: MIT
// Post-processing composite pass. Reads the HDR offscreen scene image (and, when
// BLOOM is on, the lowest-downsampled bloom image), the resolved depth
// attachment (for screen-space puddle reflections), and applies the enabled FX
// chain. All effects are gated by bits in the PostSettings `flags` uniform, so
// with everything off this is an identity passthrough.
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform PostSettings {
    uint flags;
    float time;
    float vignette_strength;
    float grain_amount;
    float saturation_boost;
    float bloom_strength;
    float chroma_strength;
    float texel_x;
    float texel_y;
    float wet_fac;
    // Puddle-reflection quality: 0 = off (skip SSR entirely), 1 = low,
    // 2 = high. Lives where the old padding began, so std140 layout is stable.
    float puddle_quality;
    float _pad1;
    float _pad2;
    float _pad3;
    float _pad4;
    float _pad5;
    mat4 inv_view_proj;
    mat4 view_proj;
    vec4 eye;
    vec4 fog_color;
};

layout(set = 0, binding = 1) uniform sampler2D scene;
layout(set = 0, binding = 2) uniform sampler2D bloom;
layout(set = 0, binding = 3) uniform sampler2D depth;

const uint FLAG_FXAA = 1u << 0;
const uint FLAG_BLOOM = 1u << 1;
const uint FLAG_VIGNETTE = 1u << 2;
const uint FLAG_GRAIN = 1u << 3;
const uint FLAG_SATURATION = 1u << 4;
const uint FLAG_CHROMA = 1u << 5;
const uint FLAG_RAINDROPS = 1u << 6;
const uint FLAG_REFLECT = 1u << 7;

const vec3 LUMA = vec3(0.299, 0.587, 0.114);

float luma(vec3 c) {
    return dot(c, LUMA);
}

// Deterministic hash for film grain; animates with `time`.
float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// FXAA-style edge blending: on a detected edge, blend the center sample toward
// its two neighbors across the dominant edge direction.
vec3 fxaa(vec3 color, vec2 uv) {
    vec2 inv = vec2(texel_x, texel_y);
    vec3 e = texture(scene, uv + vec2(-1.0, -1.0) * inv).rgb;
    vec3 f = texture(scene, uv + vec2(0.0, -1.0) * inv).rgb;
    vec3 g = texture(scene, uv + vec2(1.0, -1.0) * inv).rgb;
    vec3 b = texture(scene, uv + vec2(-1.0, 0.0) * inv).rgb;
    vec3 d = texture(scene, uv + vec2(1.0, 0.0) * inv).rgb;
    vec3 h = texture(scene, uv + vec2(-1.0, 1.0) * inv).rgb;
    vec3 i = texture(scene, uv + vec2(0.0, 1.0) * inv).rgb;
    vec3 j = texture(scene, uv + vec2(1.0, 1.0) * inv).rgb;

    float lC = luma(color);
    float lE = luma(e);
    float lF = luma(f);
    float lG = luma(g);
    float lB = luma(b);
    float lD = luma(d);
    float lH = luma(h);
    float lI = luma(i);
    float lJ = luma(j);

    // Horizontal vs vertical edge energy from the middle row / middle column.
    float edgeH = abs(lB + lD - 2.0 * lC) + abs(lE + lG - 2.0 * lF) + abs(lH + lJ - 2.0 * lI);
    float edgeV = abs(lF + lI - 2.0 * lC) + abs(lE + lH - 2.0 * lB) + abs(lG + lJ - 2.0 * lD);
    float edge = edgeH + edgeV;
    if (edge < 0.01) {
        return color;
    }

    // dir -> 1 blends horizontally (across a vertical edge), dir -> 0 blends
    // vertically (across a horizontal edge).
    float dir = 0.5 + 0.5 * clamp((edgeV - edgeH) / max(edge, 1e-6), -1.0, 1.0);
    vec3 blend = mix((f + i) * 0.5, (b + d) * 0.5, dir);
    float alpha = clamp(edge, 0.0, 0.75) * 0.35;
    return mix(color, blend, alpha);
}

// Reconstructs the world-space position of the surface seen at `uv`, given the
// depth attachment value `d` (Vulkan clip-space z, y-down framebuffer UVs).
vec3 reconstruct_world(vec2 uv, float d) {
    vec4 clip = vec4(uv * 2.0 - 1.0, d, 1.0);
    vec4 w = inv_view_proj * clip;
    return w.xyz / max(w.w, 1e-6);
}

// Deterministic value noise (quintic interpolation over the hash grid), so the
// puddle patches are continuous blobs instead of a hard-edged cell checkerboard.
float value_noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    vec2 a = i + vec2(0.0, 0.0);
    vec2 b = i + vec2(1.0, 0.0);
    vec2 c = i + vec2(0.0, 1.0);
    vec2 d = i + vec2(1.0, 1.0);
    float v = mix(
        mix(hash12(a * 7.13 + vec2(1.7, 3.1)), hash12(b * 7.13 + vec2(1.7, 3.1)), u.x),
        mix(hash12(c * 7.13 + vec2(1.7, 3.1)), hash12(d * 7.13 + vec2(1.7, 3.1)), u.x),
        u.y);
    return v;
}

// Fractal sum of `value_noise` (2 or 3 octaves depending on quality), normalized
// to ~[0, 1]. `oct` drives the detail: high quality gets a third octave.
float puddle_noise(vec2 p, int oct) {
    float amp = 0.55;
    float v = 0.0;
    float norm = 0.0;
    vec2 q = p;
    for (int i = 0; i < 3; ++i) {
        v += amp * value_noise(q);
        norm += amp;
        q = q * 2.13 + vec2(7.3, 3.7);
        if (i + 1 >= oct) {
            break;
        }
    }
    return v / norm;
}

// Deterministic puddle patches on the asphalt ribbon, driven by the wet factor
// so puddles appear as rain ramps up and stay stable as the car drives past.
// Mirrors `road_curve` (12 * sin(0.02 * s)) and ROAD_HALF + shoulder from the
// Rust side so the mask lines up with the actual road geometry.
//
// `quality` (0/1/2) trades detail for cost: higher levels add a domain-warped
// third noise octave for more organic, irregular puddle outlines.
float puddle_mask(vec3 world_pos, float wet, float quality) {
    if (wet <= 0.001) {
        return 0.0;
    }
    // The road is a flat ribbon near y = 0; reject everything else quickly.
    if (abs(world_pos.y - 0.02) > 0.05) {
        return 0.0;
    }
    float s = -world_pos.z;
    float lat = world_pos.x - 12.0 * sin(s * 0.02);
    float half_road = 4.8 + 0.55; // ROAD_HALF + shoulder strip
    if (abs(lat) > half_road) {
        return 0.0;
    }
    // Continuous, road-aligned noise space: low frequency along the road (long
    // shallow pools), higher across it, so patches melt into the asphalt rather
    // than snapping to a block grid.
    int oct = quality > 1.5 ? 3 : 2;
    vec2 q = vec2(s * 0.11, lat * 0.45);
    // Domain warp: bend the noise coordinates so the puddle outlines meander
    // instead of following a straight grid. Higher quality warps harder.
    float warp_amp = quality > 1.5 ? 0.6 : 0.35;
    vec2 w = warp_amp * vec2(
        puddle_noise(q + vec2(0.0, 1.7), 2),
        puddle_noise(q + vec2(5.3, 2.9), 2));
    float n = puddle_noise(q + w, oct);
    float pat = smoothstep(0.48, 0.60, n);
    if (pat <= 0.001) {
        return 0.0;
    }
    // Taper at the road edges so puddles melt into the shoulder.
    float edge = smoothstep(half_road, half_road - 0.7, abs(lat));
    return clamp(pat * edge * wet, 0.0, 1.0);
}

// Screen-space reflection for the puddle at `world_pos`/`view_dir`. Marches the
// reflected ray in world space, projects each sample to the screen, and accepts
// a hit when the reconstructed surface sits at roughly the ray's height (or the
// sample is sky/far, which gives the classic wet-road sky sheen).
// `quality` (1 = low, 2 = high) trades ray steps for cost.
vec3 ssr_reflection(vec3 world_pos, vec3 view_dir, float quality) {
    vec3 rd = normalize(vec3(view_dir.x, -view_dir.y, view_dir.z));
    if (rd.y <= 0.01) {
        return vec3(0.0); // looking straight down: nothing to mirror
    }
    int steps = quality > 1.5 ? 28 : 14;
    vec3 ro = world_pos;
    vec3 hit = vec3(0.0);
    bool found = false;
    float hit_t = 0.0;
    for (int i = 1; i <= steps; ++i) {
        float t = 1.2 + float(i) * 0.55;
        if (t > 70.0) {
            break;
        }
        vec3 p = ro + rd * t;
        if (p.y > 40.0) {
            break;
        }
        vec4 clip = view_proj * vec4(p, 1.0);
        if (clip.w <= 0.0) {
            break;
        }
        vec2 sp = clip.xy / clip.w * 0.5 + 0.5;
        if (sp.x < 0.0 || sp.x > 1.0 || sp.y < 0.0 || sp.y > 1.0) {
            continue;
        }
        float d = texture(depth, sp).r;
        // Sky / far plane: sample the sky directly for the classic sheen.
        if (d >= 0.999) {
            hit = texture(scene, sp).rgb;
            hit_t = t;
            found = true;
            break;
        }
        vec3 sw = reconstruct_world(sp, d);
        // Height match: the ray has reached roughly the surface height, and the
        // surface is meaningfully above the puddle (skip the road itself).
        if (sw.y >= p.y - 0.6 && sw.y <= p.y + 0.15 && sw.y > world_pos.y + 0.2) {
            hit = texture(scene, sp).rgb;
            hit_t = t;
            found = true;
            break;
        }
    }
    if (!found) {
        // Miss: fade toward whatever sits at the far end of the reflected ray
        // (roughly the horizon/sky), else the fog color.
        hit = fog_color.rgb;
        vec4 far_clip = view_proj * vec4(ro + rd * 60.0, 1.0);
        if (far_clip.w > 0.0) {
            vec2 far_sp = far_clip.xy / far_clip.w * 0.5 + 0.5;
            if (far_sp.x >= 0.0 && far_sp.x <= 1.0 && far_sp.y >= 0.0 && far_sp.y <= 1.0) {
                hit = texture(scene, far_sp).rgb;
            }
        }
        hit *= 0.55; // keep misses faint so puddles stay readable
        hit_t = 70.0;
    }
    // Fade reflections with distance so far puddles melt into the fog.
    return hit * exp(-hit_t * 0.03);
}

// Camera rain droplets: a multi-octave hash grid of drops that fall with `time`
// and refract the scene behind them, like water on the windshield. Drawn last
// (over the whole FX chain) so the lens reads as a physical surface.
vec3 rain_droplets(vec3 color, vec2 uv, float t, float wet) {
    if (wet <= 0.001) {
        return color;
    }
    vec2 refr = vec2(0.0);
    float cov = 0.0;
    for (int layer = 0; layer < 3; ++layer) {
        float f = float(layer);
        float scale = mix(9.0, 46.0, f / 2.0);
        float speed = mix(0.28, 0.7, f / 2.0);
        vec2 p = uv * scale;
        vec2 id = floor(p);
        vec2 gv = fract(p) - 0.5;
        float h = hash12(id * 7.13 + vec2(1.7, 3.1));
        // Only a fraction of cells carry a drop; a wetter lens gets more drops.
        float dens = smoothstep(0.66, 0.90, h) * mix(0.35, 1.0, wet);
        if (dens <= 0.001) {
            continue;
        }
        // The drop falls through its cell over time (recycled by `fract`).
        float fall = fract(t * speed + id.y * 0.133 + h * 1.7);
        float drop_y = fall - 0.5;
        // Elliptical streak, size from the cell hash; faster layers stretch more.
        float rx = 0.16 + 0.15 * fract(h * 3.31);
        float ry = rx * (1.4 + 3.2 * fract(h * 7.17) * speed);
        vec2 dp = vec2(gv.x, gv.y - drop_y);
        float d = length(dp / vec2(rx, ry));
        float mask = smoothstep(1.0, 0.72, d) * dens;
        // Fade at the cell top/bottom so drops enter and leave smoothly.
        float edge = smoothstep(0.0, 0.25, fall) * (1.0 - smoothstep(0.75, 1.0, fall));
        mask *= edge;
        cov += mask;
        // Refraction offset, scaled by the drop size for a lens pinch.
        refr += dp * mask * (0.035 + 0.05 * fract(h * 5.7));
    }
    if (cov <= 0.001) {
        return color;
    }
    vec3 refracted = texture(scene, uv + refr).rgb;
    vec3 col = mix(color, refracted, clamp(cov, 0.0, 1.0) * 0.85);
    col *= 1.0 - 0.16 * clamp(cov, 0.0, 1.0);
    return col;
}

void main() {
    vec2 uv = v_uv;
    vec3 color = texture(scene, uv).rgb;

    if ((flags & FLAG_CHROMA) != 0u) {
        // Chromatic aberration: shift red/blue samples radially from center.
        vec2 dir = (uv - 0.5) * 2.0;
        vec2 off = dir * chroma_strength;
        float r = texture(scene, uv + off).r;
        float bl = texture(scene, uv - off).b;
        color = vec3(r, color.g, bl);
    }

    // Puddle reflections: reconstruct the surface and reflect the scene into
    // the wet asphalt. Runs before FXAA/bloom so the reflections are
    // antialiased and get the same glow treatment as the rest of the frame.
    if ((flags & FLAG_REFLECT) != 0u && wet_fac > 0.001) {
        float d = texture(depth, uv).r;
        if (d < 1.0) {
                vec3 world_pos = reconstruct_world(uv, d);
                float pm = puddle_mask(world_pos, wet_fac, puddle_quality);
                if (pm > 0.001) {
                    vec3 view_dir = normalize(world_pos - eye.xyz);
                    vec3 refl = ssr_reflection(world_pos, view_dir, puddle_quality);
                // Fresnel-ish pickup: puddles reflect most at grazing angles
                // (exactly how a chase camera sees the road ahead).
                float fres = pow(1.0 - max(dot(normalize(view_dir), vec3(0.0, 1.0, 0.0)), 0.0), 3.0);
                float blend = pm * (0.35 + 0.65 * fres);
                color = mix(color, refl, clamp(blend, 0.0, 1.0));
                color *= 1.0 - 0.22 * clamp(blend, 0.0, 1.0);
            }
        }
    }

    if ((flags & FLAG_FXAA) != 0u) {
        color = fxaa(color, uv);
    }

    if ((flags & FLAG_BLOOM) != 0u) {
        color += texture(bloom, uv).rgb * bloom_strength;
    }

    if ((flags & FLAG_SATURATION) != 0u) {
        float l = luma(color);
        color = mix(vec3(l), color, saturation_boost);
    }

    if ((flags & FLAG_VIGNETTE) != 0u) {
        vec2 ndc = (uv - 0.5) * 2.0;
        float d = dot(ndc, ndc) * 0.5;
        color *= 1.0 - vignette_strength * smoothstep(0.4, 1.6, d);
    }

    if ((flags & FLAG_GRAIN) != 0u) {
        float n = hash12(uv * 1920.0 + vec2(time * 17.0, time * 13.0));
        color += (n - 0.5) * grain_amount;
    }

    // Wet-lens droplets last: water on the glass sits on top of everything
    // (the HUD pass composites above, so text stays readable).
    if ((flags & FLAG_RAINDROPS) != 0u) {
        color = rain_droplets(color, uv, time, wet_fac);
    }

    f_color = vec4(color, 1.0);
}
