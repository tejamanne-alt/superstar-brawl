"""INK-DOT comic shader (v2 look): FLAT COLOUR -> LIGHT AS DOTS -> INK LINE -> SLIP + PAPER.

Matches the reference direction: saturated flat colour, light described only with dot
screens and a hard terminator, deep violet shadows, heavy wobbly ink outlines, colour plates
slipped off-register under a crisp black plate, printed on grainy paper.

All sizes are relative to image height so supersampled renders scale correctly.
"""
import math

import cv2
import numpy as np

from .comicfx import halftone_mask, smoothstep, to_srgb, lum

PRESETS = dict(
    sunset=dict(key_tint=(1.0, 0.97, 0.90), shadow_tint=(0.48, 0.34, 0.84), light_shadow=(0.66, 0.56, 0.96),
                skin_shadow_tint=(0.80, 0.48, 0.64), dark_lift=(0.06, 0.05, 0.14), rim=(1.0, 0.62, 0.30),
                rim_amt=0.0, fill=(0.40, 0.30, 0.95)),
    night=dict(key_tint=(0.86, 0.90, 1.0), shadow_tint=(0.34, 0.28, 0.80), light_shadow=(0.50, 0.46, 0.95),
               skin_shadow_tint=(0.62, 0.40, 0.72), dark_lift=(0.05, 0.06, 0.16), rim=(0.30, 1.0, 0.95),
               rim_amt=1.0, fill=(1.0, 0.25, 0.75)),
    neon=dict(key_tint=(1.0, 0.93, 0.84), shadow_tint=(0.42, 0.32, 0.80), light_shadow=(0.60, 0.52, 0.95),
              skin_shadow_tint=(0.78, 0.44, 0.66), dark_lift=(0.06, 0.05, 0.15), rim=(1.0, 0.30, 0.70),
              rim_amt=1.0, fill=(0.25, 0.95, 0.85)),
)

BASE = dict(
    sat=1.12,
    terminator=0.17,       # N.L (normalised) where light flips to shadow
    form_start=0.80,       # below this, dots of shadow colour start creeping into the light
    form_max=0.62,         # max dot coverage just before the terminator
    dot_rel=0.0068,        # dot period / image height
    dot_angle=45.0,
    ao_dots=0.75,
    ink_rel=0.0040,        # base ink width / image height
    ink=(0.07, 0.04, 0.13),
    wobble=0.55,           # ink edge wobble in px at 1080p
    slip_rel=0.0021,
    paper=1.0,
    line_fade=(30.0, 160.0, 0.55),
    spec_hi=0.55,
)


def _noise(h, w, sigma, seed):
    rng = np.random.default_rng(seed)
    n = rng.normal(0, 1, (h, w)).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), sigma)
    return n / (n.std() + 1e-6)


def saturate(rgb, k):
    g = lum(rgb)[..., None]
    return np.clip(g + (rgb - g) * k, 0, 1)


def edges(P, n_thresh=0.30, d_thresh=0.03):
    a = P['alpha'] > 0.5
    obj, mat = P['obj'], P['mat']
    n = P['normal']
    d = P['depth'].copy()
    d[~a] = 1e5

    def nd(arr, f):
        out = np.zeros(arr.shape[:2], bool)
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            sh = np.roll(np.roll(arr, dy, 0), dx, 1)
            out |= f(arr, sh)
        return out
    sil = nd(a.astype(np.int8), lambda x, y: x != y)
    ob = nd(obj, lambda x, y: x != y) & a
    dz = nd(d, lambda x, y: np.abs(x - y) / np.minimum(x, y) > d_thresh) & a
    cr = nd(n, lambda x, y: (1 - np.sum(x * y, -1)) > n_thresh) & a
    me = nd(mat, lambda x, y: x != y) & a
    return dict(sil=sil, obj=ob, depth=dz, crease=cr, mat=me)


def ink_alpha(P, st, H, skip=None, light=None):
    E = edges(P)
    h, w = P['depth'].shape
    base = st['ink_rel'] * H
    s1080 = H / 1080.0
    wob = _noise(h, w, 3.0 * s1080, 11) * st['wobble'] * s1080
    wvar = 0.82 + 0.28 * np.clip(_noise(h, w, 25 * s1080, 12), -1.5, 1.5) / 1.5
    out = np.zeros((h, w), np.float32)
    weights = dict(sil=1.35, obj=1.2, depth=1.0, mat=0.72, crease=0.62)
    for k, wt in weights.items():
        m = E[k]
        if skip is not None and k != 'sil':
            m = m & ~skip
        if not m.any():
            continue
        dist = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
        r = base * wt * wvar / 2
        if light is not None and k != 'sil':
            r = r * (1.0 + 0.35 * (light < st['terminator']))
        out = np.maximum(out, np.clip(r - (dist + wob) + 0.5, 0, 1))
    if st.get('line_fade'):
        n0, n1, amt = st['line_fade']
        out *= 1 - smoothstep(n0, n1, P['depth']) * amt
    return out


def shade(P, meta, preset='sunset', style=None, H=None, steps=4):
    """Returns dict(color, ink, alpha, glow, flat). steps<4 is used for the breakdown board."""
    st = dict(BASE)
    st.update(PRESETS[preset])
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
    is_ink = table('ink')[mid] > 0.5
    is_flat = table('flat')[mid] > 0.5
    spec_s = table('spec')[mid]
    is_glass = table('glass')[mid] > 0.5
    is_metal = table('metal')[mid] > 0.5
    emit_m = table('emit')[mid] > 0

    base = saturate(to_srgb(P['albedo']), st['sat'])
    flat = base.copy()
    if steps == 1:
        return dict(color=flat, ink=np.zeros((h, w), np.float32), alpha=a, glow=np.zeros_like(flat), flat=flat)

    ke = meta.get('key_energy', 3.0)
    light = np.clip(lum(P['direct']) / (ke / math.pi), 0, 1.2)
    nrm = P['normal']
    cam = meta.get('camera', {})
    V = -np.array(cam.get('forward', (0, 1, 0)), np.float32)
    L = -np.array(meta.get('key_dir', (0.5, -0.5, -0.7)), np.float32)
    ndv = np.sum(nrm * V, -1)
    ndl = np.sum(nrm * L, -1)

    # --- colours
    lb = lum(base)[..., None]
    tint = np.array(st['shadow_tint'], np.float32) * (1 - lb) + np.array(st['light_shadow'], np.float32) * lb
    tint = np.where(is_skin[..., None], np.array(st['skin_shadow_tint'], np.float32), tint)
    shadow = base * tint
    dark = lb[..., 0] < 0.10
    lit = base * np.array(st['key_tint'], np.float32)
    lit = np.where(dark[..., None], base + np.array(st['dark_lift'], np.float32), lit)
    deep = shadow * 0.62

    # --- light as dots
    period = st['dot_rel'] * H
    term = st['terminator']
    in_light = light > term
    cov_form = smoothstep(st['form_start'], term, light) * st['form_max']
    dots_form = halftone_mask(h, w, cov_form, period, st['dot_angle'])
    col = np.where(in_light[..., None], lit * (1 - dots_form[..., None]) + shadow * dots_form[..., None], shadow)
    ao = cv2.GaussianBlur(P['ao'], (0, 0), 0.0015 * H)
    cov_ao = np.clip((0.9 - ao) / 0.5, 0, 1) * st['ao_dots']
    cov_ao = np.maximum(cov_ao, np.where(in_light, 0, smoothstep(-0.05, -0.6, ndl) * 0.35))
    dots_ao = halftone_mask(h, w, cov_ao, period * 0.85, st['dot_angle'] + 30)
    col = col * (1 - dots_ao[..., None]) + np.where(in_light[..., None], shadow, deep) * dots_ao[..., None]
    # fill-light dots on the shadow side (coloured bounce from the opposite sky)
    fill_cov = smoothstep(-0.15, -0.75, ndl) * (~in_light) * 0.30 * (1 - dark)
    dfill = halftone_mask(h, w, fill_cov, period * 1.1, st['dot_angle'] - 30)
    col = col * (1 - dfill[..., None] * 0.6) + (shadow * 0.5 + np.array(st['fill']) * 0.5) * dfill[..., None] * 0.6

    # hard spec shapes + light dots around them (hair, leather, glass, metal)
    Hh = (L + V) / np.linalg.norm(L + V)
    ndh = np.clip(np.sum(nrm * Hh, -1), 0, 1)
    shiny = np.clip(spec_s + is_glass + is_metal * 0.8, 0, 1)
    core = (ndh > 0.955) * in_light * shiny
    halo = smoothstep(0.86, 0.955, ndh) * in_light * shiny
    hi_col = lit + (1 - lit) * st['spec_hi']
    hdots = halftone_mask(h, w, halo * 0.55, period * 0.8, st['dot_angle'])
    col = col * (1 - hdots[..., None]) + hi_col * hdots[..., None]
    col = np.where(core[..., None] > 0.5, hi_col, col)
    glass_band = (np.abs(ndv) < 0.55) * is_glass * (ndh > 0.75)
    col = np.where(glass_band[..., None], np.clip(lit + 0.55, 0, 1), col)

    # neon rim on the side away from the key (cover-art flourish, off for the sunset street)
    if st['rim_amt'] > 0:
        rim = (np.abs(ndv) < 0.32) * (ndl < 0.2) * (a > 0.99) * (~is_ink)
        rim = cv2.morphologyEx(rim.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)).astype(np.float32)
        col = col * (1 - rim[..., None] * st['rim_amt']) + np.array(st['rim'], np.float32) * rim[..., None] * st['rim_amt']

    col = np.where(is_flat[..., None], np.where(in_light[..., None], base, base * np.array(st['skin_shadow_tint'])), col)
    col = np.where(is_ink[..., None], np.array(st['ink'], np.float32), col)
    em = to_srgb(P['emit'])
    col = np.where(emit_m[..., None], np.clip(em * 0.5 + 0.6, 0, 1), col)

    glow = np.zeros_like(col)
    emit_any = lum(P['emit']) > 0.05
    if emit_any.any():
        e = np.clip(P['emit'], 0, 4)
        glow = to_srgb(cv2.GaussianBlur(e, (0, 0), 0.006 * H) * 0.7 + cv2.GaussianBlur(e, (0, 0), 0.02 * H) * 0.9)
    ink = np.zeros((h, w), np.float32)
    if steps >= 3:
        ink = ink_alpha(P, st, H, skip=emit_any | is_ink, light=light)
    alpha = np.maximum(a, ink * (cv2.dilate((a > 0.01).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0))
    return dict(color=np.clip(col, 0, 1), ink=ink, alpha=np.clip(alpha, 0, 1), glow=glow, flat=flat, light=light,
                style=st)


# ----------------------------------------------------------------------------- print finish

def slip(img, px):
    """Colour plates off-register (red right/down, blue left/up)."""
    if px <= 0:
        return img
    h, w = img.shape[:2]
    out = img.copy()
    M1 = np.float32([[1, 0, px], [0, 1, px * 0.45]])
    M2 = np.float32([[1, 0, -px], [0, 1, -px * 0.45]])
    out[..., 0] = cv2.warpAffine(img[..., 0], M1, (w, h), borderMode=cv2.BORDER_REPLICATE)
    out[..., 2] = cv2.warpAffine(img[..., 2], M2, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return out


def paper(img, amt=1.0, seed=3):
    h, w = img.shape[:2]
    s = h / 1080.0
    rng = np.random.default_rng(seed)
    fine = rng.normal(0, 1, (h, w)).astype(np.float32)
    fine = cv2.GaussianBlur(fine, (0, 0), 0.7 * s)
    mott = _noise(h, w, 9 * s, seed + 1) * 0.6 + _noise(h, w, 40 * s, seed + 2) * 0.4
    fib = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), sigmaX=8 * s, sigmaY=0.7 * s)
    fib /= fib.std() + 1e-6
    t = (fine * 0.022 + mott * 0.015 + fib * 0.005) * amt
    out = img + t[..., None]
    # sparse speckles: dark flecks and paper-white flecks
    sp = rng.random((h, w))
    dark = (sp > 1 - 0.00010 * amt).astype(np.float32)
    light = (sp < 0.00025 * amt).astype(np.float32)
    k = max(1, int(round(1.4 * s)))
    dark = cv2.dilate(dark, np.ones((k, k), np.uint8))
    light = cv2.dilate(light, np.ones((k, k), np.uint8))
    out = out * (1 - dark[..., None] * 0.40) + light[..., None] * 0.25
    out = out * np.array([1.0, 0.985, 0.955]) + np.array([0.012, 0.01, 0.0])
    return np.clip(out, 0, 1)


def finish(color, ink, slip_px, ink_col=(0.07, 0.04, 0.13), paper_amt=1.0, glow=None, glow_alpha=None):
    img = slip(color, slip_px)
    if glow is not None:
        img = 1 - (1 - img) * (1 - glow)
    img = img * (1 - ink[..., None]) + np.array(ink_col, np.float32) * ink[..., None]
    if paper_amt:
        img = paper(img, paper_amt)
    return img


def render_frame(exr_path, sky, preset='sunset', style=None, out_size=None, haze=None, slip_scale=1.0,
                 paper_amt=1.0):
    """EXR passes + 2D background -> finished frame (sky must match out_size or render size)."""
    from . import exr
    from .paint import haze as apply_haze
    P, meta = exr.load(exr_path)
    h, w = P['depth'].shape
    S = shade(P, meta, preset, style)
    col = S['color']
    if haze is not None:
        hc, n0, n1, amt = haze
        col = apply_haze(col, P['depth'], S['alpha'], hc, n0, n1, amt)
    bg = cv2.resize(sky.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC) if sky.shape[:2] != (h, w) else sky
    img = col * S['alpha'][..., None] + bg * (1 - S['alpha'][..., None])
    ink = S['ink']
    glow = S['glow']
    if out_size is not None and (w, h) != tuple(out_size):
        img = cv2.resize(img.astype(np.float32), out_size, interpolation=cv2.INTER_AREA)
        ink = cv2.resize(ink, out_size, interpolation=cv2.INTER_AREA)
        glow = cv2.resize(glow, out_size, interpolation=cv2.INTER_AREA)
        an = meta.get('meta', {}).get('anchors')
        if an:
            f = out_size[0] / w
            meta['meta']['anchors'] = {k: [v[0] * f, v[1] * f, v[2]] for k, v in an.items()}
        S = dict(S, alpha=cv2.resize(S['alpha'], out_size, interpolation=cv2.INTER_AREA))
    Ho = img.shape[0]
    out = finish(img, ink, BASE['slip_rel'] * Ho * slip_scale, paper_amt=paper_amt, glow=glow)
    return out, P, meta, S


def layer(exr_path, preset='neon', style=None, out_size=None, slip_scale=1.0):
    """Character cut-out as RGBA (for cards / select screen), with ink and slipped colour."""
    from . import exr
    P, meta = exr.load(exr_path)
    h, w = P['depth'].shape
    S = shade(P, meta, preset, style)
    a = S['alpha']
    col = S['color'] * a[..., None]
    sp = BASE['slip_rel'] * h * slip_scale
    pm = np.dstack([col, a])
    M1 = np.float32([[1, 0, sp], [0, 1, sp * 0.45]])
    M2 = np.float32([[1, 0, -sp], [0, 1, -sp * 0.45]])
    r = cv2.warpAffine(pm[..., [0, 3]], M1, (w, h))
    b = cv2.warpAffine(pm[..., [2, 3]], M2, (w, h))
    A = np.maximum(a, np.maximum(r[..., 1], b[..., 1]))
    C = np.dstack([r[..., 0], pm[..., 1], b[..., 0]]) / np.clip(A[..., None], 1e-4, None)
    C = 1 - (1 - C) * (1 - S['glow'])
    A = np.clip(A + lum(S['glow']) * 0.9, 0, 1)
    ink = S['ink']
    C = C * (1 - ink[..., None]) + np.array(BASE['ink'], np.float32) * ink[..., None]
    A = np.maximum(A, ink)
    rgba = np.dstack([np.clip(C, 0, 1), A]).astype(np.float32)
    if out_size is not None:
        rgba = cv2.resize(rgba, out_size, interpolation=cv2.INTER_AREA)
    return rgba
