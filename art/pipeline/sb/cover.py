"""COVER shader: the painted modern teen-comic cover look.

Traits reproduced (from the reference cover): flat colour with ONE soft-edged shadow shape
whose edge is broken by brush texture; warm key + coloured neon rim lights (magenta on one
side, teal on the other); glossy hair with sharp light streaks; cool reflected light on dark
cloth; tapered dark-brown brush ink that is heavy on silhouettes, light and broken inside;
quick hatch strokes in cloth shadows; no halftone, no plate slip.
"""
import math

import cv2
import numpy as np

from .comicfx import smoothstep, to_srgb, lum
from .inkdot import edges, saturate

STYLE = dict(
    term=0.26, soft=0.045, brush=0.10,
    key_tint=(1.0, 0.95, 0.86),
    shadow_cloth=(0.56, 0.52, 0.74), shadow_skin=(0.80, 0.50, 0.50), shadow_fur=(0.60, 0.56, 0.72),
    dark_lift=(0.035, 0.040, 0.065),
    hi_skin=(1.0, 0.86, 0.70), hi_amt=0.32,
    rim_left=(1.0, 0.36, 0.62), rim_right=(0.30, 0.95, 0.92), rim_w=0.40, rim_amt=0.85,
    ink=(0.13, 0.07, 0.09), ink_rel=0.0034, ink_inner=0.55, ink_break=0.55,
    hatch=0.55,
)


def brush_field(h, w, H, angle=35.0, length=0.05, width=0.004, seed=1):
    """Anisotropic noise that looks like overlapping brush strokes, in [-1, 1]."""
    rng = np.random.default_rng(seed)
    n = rng.normal(0, 1, (h, w)).astype(np.float32)
    sx, sy = max(1.0, length * H / 2.5), max(0.8, width * H)
    k = cv2.getGaussianKernel(int(sx * 6) | 1, sx) @ cv2.getGaussianKernel(int(sy * 6) | 1, sy).T
    k = k.astype(np.float32)
    M = cv2.getRotationMatrix2D((k.shape[1] / 2, k.shape[0] / 2), angle, 1.0)
    side = int(max(k.shape) * 1.5) | 1
    kk = np.zeros((side, side), np.float32)
    y0, x0 = (side - k.shape[0]) // 2, (side - k.shape[1]) // 2
    kk[y0:y0 + k.shape[0], x0:x0 + k.shape[1]] = k
    M = cv2.getRotationMatrix2D((side / 2, side / 2), angle, 1.0)
    kk = cv2.warpAffine(kk, M, (side, side))
    kk /= kk.sum()
    f = cv2.filter2D(n, -1, kk)
    return np.clip(f / (f.std() + 1e-6) / 2.0, -1, 1)


def shade(P, meta, style=None, H=None):
    st = dict(STYLE)
    if style:
        st.update(style)
    h, w = P['depth'].shape
    H = H or h
    a = P['alpha']
    mats = meta.get('materials', {})
    maxid = max([int(k) for k in mats] + [1]) + 1

    def table(key):
        t = np.zeros(maxid + 1, np.float32)
        for k, v in mats.items():
            val = v.get(key, 0)
            t[int(k)] = float(val) if not isinstance(val, bool) else float(val)
        return t
    mid = np.clip(P['mat'], 0, maxid)
    is_skin = table('skin')[mid] > 0.5
    is_fur = table('fur')[mid] > 0.5
    is_hair = table('hair')[mid] > 0.5
    is_glass = table('glass')[mid] > 0.5
    is_metal = table('metal')[mid] > 0.5
    is_cloth = table('cloth')[mid] > 0.5
    is_flat = table('flat')[mid] > 0.5
    is_ink = table('ink')[mid] > 0.5
    spec_s = table('spec')[mid]
    emit_m = table('emit')[mid] > 0

    base = saturate(to_srgb(P['albedo']), 1.06)
    from .pixclass import classify
    PC = classify(P, base)
    if PC is not None:   # textured model: per-pixel classes override the per-material flags
        base = PC['base']
        is_skin, is_fur, is_hair = PC['skin'], PC['fur'], PC['hair']
        is_glass, is_metal, is_cloth = PC['glass'], PC['metal'], PC['cloth']
        spec_s = np.where(is_hair, 0.7, np.where(PC['region'] == 2, 0.25, 0.0)).astype(np.float32)
    ke = meta.get('key_energy', 3.0)
    light = np.clip(lum(P['direct']) / (ke / math.pi), 0, 1.3)
    nrm = P['normal']
    cam = meta.get('camera', {})
    V = -np.array(cam.get('forward', (0, 1, 0)), np.float32)
    Rt = np.array(cam.get('right', (1, 0, 0)), np.float32)
    L = -np.array(meta.get('key_dir', (0.5, -0.5, -0.7)), np.float32)
    ndv = np.sum(nrm * V, -1)
    ndl = np.sum(nrm * L, -1)
    nx = np.sum(nrm * Rt, -1)

    # ---- one soft, brush-broken shadow shape
    B = brush_field(h, w, H, 35, 0.05, 0.004, seed=2)
    B2 = brush_field(h, w, H, -20, 0.03, 0.003, seed=3)
    thr = st['term'] + st['brush'] * B
    lit = smoothstep(thr - st['soft'], thr + st['soft'], light)
    lb = lum(base)[..., None]
    dark = lb[..., 0] < 0.22
    tint = np.where(is_skin[..., None], np.array(st['shadow_skin']),
                    np.where(is_fur[..., None], np.array(st['shadow_fur']), np.array(st['shadow_cloth'])))
    shadow = base * tint
    shadow = np.where(dark[..., None], base * 0.62 + np.array([0.02, 0.02, 0.05]), shadow)
    lit_col = base * np.array(st['key_tint'])
    lit_col = np.where(dark[..., None], base + np.array(st['dark_lift']) * (0.6 + 0.4 * np.clip(ndl, 0, 1))[..., None], lit_col)
    # dark cloth: big calm light shapes instead of fold blotches
    light_s = cv2.GaussianBlur(light.astype(np.float32), (0, 0), 0.007 * H)
    lit_dark = smoothstep(thr - 0.09, thr + 0.09, light_s)
    lit = np.where(dark, lit_dark, lit)
    lit_col = np.where(dark[..., None], base * 1.05 + np.array([0.022, 0.026, 0.045]), lit_col)
    shadow = np.where(dark[..., None], base * 0.70, shadow)
    col = shadow * (1 - lit[..., None]) + lit_col * lit[..., None]
    # painterly variation inside the shapes
    col = col * (1 + 0.035 * B2[..., None])
    # warm highlight plane on skin (forehead, nose, cheekbones)
    hi = smoothstep(0.80 + 0.06 * B, 0.86 + 0.06 * B, light) * is_skin
    col = col * (1 - hi[..., None] * st['hi_amt']) + np.array(st['hi_skin']) * hi[..., None] * st['hi_amt']
    # deep occlusion accents (painted darker pockets)
    ao = cv2.GaussianBlur(P['ao'], (0, 0), 0.002 * H)
    pocket = np.clip((0.70 - ao) / 0.4, 0, 1) * (1 - lit)
    col = col * (1 - 0.28 * pocket[..., None])

    # ---- coloured rim lights (neon from the environment)
    fres = 1 - np.abs(ndv)
    rim = smoothstep(1 - st['rim_w'], 1 - st['rim_w'] + 0.10, fres + 0.06 * B)
    left = rim * smoothstep(0.05, 0.35, -nx)
    right = rim * smoothstep(0.05, 0.35, nx)
    solid = (a > 0.99) & ~is_ink & ~is_glass
    for m, c in ((left, st['rim_left']), (right, st['rim_right'])):
        m = m * solid * st['rim_amt']
        col = col * (1 - m[..., None]) + np.array(c) * m[..., None]

    # ---- glossy highlights: hair streaks, metal, leather-ish cloth sheen
    Hh = (L + V) / np.linalg.norm(L + V)
    ndh = np.clip(np.sum(nrm * Hh, -1), 0, 1)
    streak = brush_field(h, w, H, 80, 0.02, 0.0012, seed=7)
    hair_hi = smoothstep(0.86, 0.90, ndh + 0.06 * streak) * (streak > 0.05) * is_hair * (lit > 0.5)
    col = col * (1 - hair_hi[..., None] * 0.85) + np.array([0.92, 0.90, 0.96]) * hair_hi[..., None] * 0.85
    met = is_metal & (a > 0.5)
    mcol = np.where((ndh > 0.82)[..., None], np.array([1.0, 0.97, 0.82]),
                    np.where((lit > 0.5)[..., None], base * 1.05, base * np.array([0.55, 0.40, 0.42])))
    col = np.where(met[..., None], mcol, col)
    sheen = smoothstep(0.93, 0.96, ndh) * is_cloth * (spec_s > 0.1) * (lit > 0.5)
    col = col + np.array([0.20, 0.22, 0.30]) * sheen[..., None]

    # ---- aviator lenses: dark olive gradient + diagonal reflections
    if is_glass.any():
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        ys = np.where(is_glass.any(1))[0]
        y0, y1 = ys.min(), ys.max()
        tv = np.clip((yy - y0) / max(1, y1 - y0), 0, 1)
        lens = np.array([0.08, 0.10, 0.08]) * (1 - tv[..., None]) + np.array([0.30, 0.32, 0.22]) * tv[..., None]
        band = ((xx * 0.8 + yy) / (0.022 * H)) % 4.0
        refl = ((band > 0.25) & (band < 0.75)) | ((band > 1.3) & (band < 1.45))
        lens = np.where(refl[..., None], np.array([0.80, 0.86, 0.80]), lens)
        lens = lens + np.array(st['rim_left']) * 0.20 * smoothstep(0.6, 1.0, xx / w)[..., None]
        col = np.where(is_glass[..., None], lens, col)

    col = np.where(is_flat[..., None], np.where((lit > 0.5)[..., None], base, base * 0.82), col)
    em = to_srgb(P['emit'])
    col = np.where(emit_m[..., None], np.clip(em * 0.5 + 0.6, 0, 1), col)

    # ---- ink: heavy outside, light and broken inside, more on the shadow side
    E = edges(dict(P, mat=PC['cls']) if PC is not None else P, n_thresh=0.42, d_thresh=0.025)
    s1080 = H / 1080.0
    wob = cv2.GaussianBlur(np.random.default_rng(5).normal(0, 1, (h, w)).astype(np.float32), (0, 0), 2.5 * s1080)
    wob = wob / (wob.std() + 1e-6) * 0.5 * s1080
    gaps = brush_field(h, w, H, 0, 0.012, 0.012, seed=11)
    ink = np.zeros((h, w), np.float32)
    base_w = st['ink_rel'] * H
    for k, wt in (('sil', 1.45), ('obj', 1.2), ('depth', 1.0), ('mat', 0.75), ('crease', 0.55)):
        m = E[k]
        if k in ('crease',):
            m = m & ~(is_fur & ~is_hair) & ~is_glass
        if k in ('mat', 'crease'):
            m = m & (gaps > -st['ink_break'])
        if k != 'sil' and m.any():   # drop fragments too short to read as a stroke (they print as dots)
            n, lbl, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
            keep = np.zeros(n, bool)
            keep[1:] = stats[1:, cv2.CC_STAT_AREA] > max(6, int(0.012 * H))
            m = keep[lbl]
        if not m.any():
            continue
        dist = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
        r = base_w * wt / 2 * (1.0 + 0.6 * (1 - lit) * (k != 'sil')) * (0.75 + 0.25 * (B2 + 1))
        ink = np.maximum(ink, np.clip(r - (dist + wob) + 0.5, 0, 1))
    # quick hatch strokes in cloth shadow (the painter's dry-brush marks)
    if st['hatch'] > 0:
        hb = brush_field(h, w, H, 60, 0.025, 0.0010, seed=13)
        hatch = (hb > 0.55) * (lit < 0.3) * is_cloth * (a > 0.99) * (B > 0.0)
        ink = np.maximum(ink, hatch.astype(np.float32) * 0.85 * st['hatch'])
    ink = np.where(emit_m, 0, ink)
    alpha = np.maximum(a, ink * (cv2.dilate((a > 0.01).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0))
    glow = np.zeros_like(col)
    if emit_m.any():
        e = np.clip(P['emit'], 0, 4)
        glow = to_srgb(cv2.GaussianBlur(e, (0, 0), 0.006 * H) * 0.6 + cv2.GaussianBlur(e, (0, 0), 0.02 * H) * 0.8)
    out = np.clip(col, 0, 1)
    out = out * (1 - ink[..., None]) + np.array(st['ink']) * ink[..., None]
    return dict(color=np.clip(out, 0, 1), alpha=np.clip(alpha, 0, 1), ink=ink, glow=glow, lit=lit)


def layer(exr_path, style=None, out_size=None):
    from . import exr
    P, meta = exr.load(exr_path)
    S = shade(P, meta, style)
    rgba = np.dstack([1 - (1 - S['color']) * (1 - S['glow']), np.clip(S['alpha'] + lum(S['glow']), 0, 1)]).astype(np.float32)
    if out_size is not None:
        rgba = cv2.resize(rgba, out_size, interpolation=cv2.INTER_AREA)
    return rgba, P, meta


def paper(img, amt=1.0, seed=4):
    h, w = img.shape[:2]
    s = h / 1080.0
    rng = np.random.default_rng(seed)
    n = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 0.8 * s)
    m = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 12 * s)
    m /= m.std() + 1e-6
    return np.clip(img + (n * 0.012 + m * 0.010)[..., None] * amt, 0, 1)
