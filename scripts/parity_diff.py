#!/usr/bin/env python3
"""Compares a snapshot against its golden reference: probe JSON + PNG.

Usage: parity_diff.py <baseline-path> <current-path>

Each path is the base name without extension; the tool appends .json and .png.
The probe JSON is compared field-by-field with an absolute tolerance and the
PNG is compared pixel-by-pixel. Exits nonzero on any mismatch.
"""

import json
import sys

import numpy as np
from PIL import Image

# Probes are written with 6 decimal places; allow tiny float churn only.
PROBE_TOL = 1e-4


def diff_json(baseline, current):
    a = json.load(open(baseline))
    b = json.load(open(current))
    diffs = []
    for key in a:
        va, vb = a[key], b[key]
        if va is None and vb is None:
            continue
        if va is None or vb is None:
            diffs.append((key, va, vb))
            continue
        # "sun_ndc" is a [x, y] array; compare element-wise.
        if isinstance(va, list):
            if len(va) != len(vb) or any(
                abs(x - y) > PROBE_TOL for x, y in zip(va, vb)
            ):
                diffs.append((key, va, vb))
        elif abs(va - vb) > PROBE_TOL:
            diffs.append((key, va, vb))
    return diffs


def diff_png(baseline, current):
    a = np.asarray(Image.open(baseline).convert("RGB"), dtype=np.int16)
    b = np.asarray(Image.open(current).convert("RGB"), dtype=np.int16)
    if a.shape != b.shape:
        return "size mismatch: %s vs %s" % (a.shape, b.shape)
    d = np.abs(a - b)
    max_channel = d.max(axis=2)
    return (float(d.mean()), int(d.max()), int((max_channel > 0).sum()))


def main():
    base, cur = sys.argv[1], sys.argv[2]
    ok = True

    diffs = diff_json(base + ".json", cur + ".json")
    if diffs:
        ok = False
        for key, va, vb in diffs:
            print("  probe %s: baseline=%s current=%s" % (key, va, vb))

    result = diff_png(base + ".png", cur + ".png")
    if isinstance(result, str):
        ok = False
        print("  png: %s" % result)
    else:
        mean, mx, n_diff = result
        print(
            "  png: mean|d|=%f max|d|=%d differing_pixels=%d" % (mean, mx, n_diff)
        )
        if mean != 0.0 or n_diff != 0:
            ok = False

    print("%s %s" % ("PASS" if ok else "FAIL", base))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
