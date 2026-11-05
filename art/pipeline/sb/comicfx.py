"""Comic-book NPR shader ("Filmverse look") applied to Blender render passes.

Pipeline per layer:
  1. cel bands from the key light (Diffuse Direct pass = N.L x shadows, noise free)
  2. hue-shifted shadows (cool purple / warm magenta on skin) instead of plain darkening
  3. halftone dot screen in the light->shadow transition, hatching in deep shadow / AO creases
  4. painted fill + coloured rim light from the Normal pass, hard spec shapes on leather / glass / metal
  5. variable-weight ink lines from depth, normal, object and material discontinuities
  6. emission -> glow; printed-paper misregistration (chromatic offset by depth) and grain
All of this is what the Unity URP shader will reproduce in real time (see ART_BIBLE.md).
"""
import math
import numpy as np
import cv2

DEFAULT_STYLE = dict(
    key_tint=(1.0, 0.95, 0.88),        # multiply on lit areas (sunlight colour)
    shadow_tint=(0.50, 0.46, 0.78),    # cool purple shadow multiply
    skin_shadow_tint=(0.70, 0.46, 0.55),
    dark_lift=(0.16, 0.18, 0.30),      # how black cloth shows form (blue sheen)
    rim_color=(1.0, 0.35, 0.65),       # magenta rim (Spider-Verse style)
    rim_width=0.62,
    fill_color=(0.35, 0.55, 1.0),      # sky bounce from the opposite side
    fill_strength=0.10,
    band_hi=0.32, band_lo=0.16,
    halftone_px=7.0, halftone_angle=45.0,
    hatch_px=6.0, hatch_angle=-35.0,
    line_px=1.8, line_color=(0.05, 0.03, 0.10),
    misregister_px=2.0,
    glow=1.0,
    line_fade_near=1e9, line_fade_far=2e9, line_fade_amt=0.75,
    use_halftone=True, use_hatch=True, use_lines=True, use_rim=True, use_spec=True,
)


def to_srgb(lin):
    lin = np.clip(lin, 0, None)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def lum(c):
    return c[..., 0] * 0.2126 + c[..., 1] * 0.7152 + c[..., 2] * 0.0722


# ----------------------------------------------------------------------------- screens

def halftone_mask(h, w, coverage, period, angle_deg, ox=0.0, oy=0.0):
    """Rotated dot screen. coverage in 0..1 -> fraction of cell covered by the dot."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    a = math.radians(angle_deg)
    u = ((xx + ox) * math.cos(a) + (yy + oy) * math.sin(a)) / period
    v = (-(xx + ox) * math.sin(a) + (yy + oy) * math.cos(a)) / period
    fu = u - np.floor(u) - 0.5
    fv = v - np.floor(v) - 0.5
    d = np.sqrt(fu * fu + fv * fv)
    r = np.sqrt(np.clip(coverage, 0, 1) / math.pi) * 1.05
    aa = 0.8 / period
    return np.clip((r - d) / aa + 0.5, 0, 1)


def hatch_mask(h, w, density, period, angle_deg, ox=0.0):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    a = math.radians(angle_deg)
    u = ((xx + ox) * math.cos(a) + yy * math.sin(a)) / period
    f = np.abs(u - np.floor(u) - 0.5) * 2  # 0 at line centre .. 1 between lines
    width = np.clip(density, 0, 1) * 0.8
    aa = 1.2 / period
    return np.clip((width - f) / aa + 0.5, 0, 1)


def stipple(h, w, density, seed=3):
    rng = np.random.default_rng(seed)
    n = rng.random((h, w)).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), 0.7)
    n = (n - n.min()) / (n.max() - n.min() + 1e-6)
    return (n < density * 0.55).astype(np.float32)


# ----------------------------------------------------------------------------- lines

def _edge_soft(edge, radius):
    """Antialiased stroke of given radius (px) around binary edge pixels."""
    inv = (edge < 0.5).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5).astype(np.float32)
    return np.clip(radius - dist + 0.5, 0, 1), dist


def ink_lines(P, style, light, scale=1.0, skip=None, seam_skip=None):
    h, w = P['depth'].shape
    a = P['alpha']
    obj, mat = P['obj'], P['mat']
    n = P['normal']
    d = P['depth'].copy()
    big = 1e4
    d[a < 0.5] = big
    lw = style['line_px'] * scale

    def neigh_diff(arr, f):
        out = np.zeros(arr.shape[:2], np.float32)
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            sh = np.roll(np.roll(arr, dy, 0), dx, 1)
            out = np.maximum(out, f(arr, sh))
        return out

    # silhouette vs background
    sil = neigh_diff((a > 0.5).astype(np.float32), lambda x, y: np.abs(x - y)) > 0.5
    # object boundaries (different fighters / props)
    ob_e = neigh_diff(obj.astype(np.float32), lambda x, y: (np.abs(x - y) > 0.5).astype(np.float32)) > 0.5
    # depth discontinuities (relative)
    dz = neigh_diff(d, lambda x, y: np.abs(x - y) / np.minimum(x, y)) > 0.035
    # normal creases
    nd = neigh_diff(n, lambda x, y: 1.0 - np.sum(x * y, -1)) > 0.32
    # material seams
    me = neigh_diff(mat.astype(np.float32), lambda x, y: (np.abs(x - y) > 0.5).astype(np.float32)) > 0.5

    inside = a > 0.01
    sil &= True
    dz &= inside
    nd &= inside & (a > 0.5)
    me &= inside & (a > 0.5)
    if skip is not None:
        for m in (dz, nd, me, ob_e):
            m &= ~skip
    if seam_skip is not None:
        ss = cv2.dilate(seam_skip.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        me &= ~ss
        nd &= ~ss
    shadow_w = 1.0 + 0.6 * (1.0 - np.clip(light, 0, 1))
    s_sil, _ = _edge_soft(sil | ob_e, lw * 1.25)
    s_dz, _ = _edge_soft(dz, lw * 1.0)
    s_me, _ = _edge_soft(me, lw * 0.6)
    s_nd, _ = _edge_soft(nd, lw * 0.55)
    s_inner = np.maximum(s_dz, np.maximum(s_me, s_nd))
    # thicker strokes on the shadow side: blur-in weight
    s_inner = np.clip(s_inner * shadow_w, 0, 1)
    return np.maximum(s_sil, s_inner)


# ----------------------------------------------------------------------------- main shader

def shade(P, meta, style=None, scale=1.0, seed=0):
    st = dict(DEFAULT_STYLE)
    if style:
        st.update(style)
    h, w = P['depth'].shape
    a = P['alpha']
    mats = meta.get('materials', {})
    maxid = max([int(k) for k in mats] + [1]) + 1

    def table(key, default=0.0):
        t = np.full(maxid + 1, default, np.float32)
        for k, v in mats.items():
            val = v.get(key, default)
            t[int(k)] = float(val) if not isinstance(val, bool) else (1.0 if val else 0.0)
        return t

    mid = np.clip(P['mat'], 0, maxid)
    is_skin = table('skin')[mid] > 0.5
    is_ink = table('ink')[mid] > 0.5
    is_flat = table('flat')[mid] > 0.5
    spec_s = table('spec')[mid]
    is_metal = table('metal')[mid] > 0.5
    is_glass = table('glass')[mid] > 0.5
    is_hatch = table('hatch')[mid] > 0.5
    is_stip = table('stipple')[mid] > 0.5
    emit_m = table('emit')[mid] > 0

    base = to_srgb(P['albedo'])
    ke = meta.get('key_energy', 3.0)
    light = np.clip(lum(P['direct']) / (ke / math.pi), 0, 1.2)  # Cycles direct = E*cos/pi

    nrm = P['normal']
    cam = meta.get('camera', {})
    V = -np.array(cam.get('forward', (0, 1, 0)), np.float32)
    L = -np.array(meta.get('key_dir', (0.5, -0.5, -0.7)), np.float32)
    ndv = np.clip(np.sum(nrm * V, -1), -1, 1)
    ndl = np.sum(nrm * L, -1)

    # --- cel bands with AA
    hi, lo = st['band_hi'], st['band_lo']
    lit_w = smoothstep(hi - 0.03, hi + 0.03, light)
    sh_w = 1.0 - smoothstep(lo - 0.03, lo + 0.03, light)
    mid_cov = np.clip((hi - light) / (hi - lo), 0, 1)  # 0 lit .. 1 shadow

    tint = np.where(is_skin[..., None], np.array(st['skin_shadow_tint'], np.float32),
                    np.array(st['shadow_tint'], np.float32))
    shadow_col = base * tint
    dark = lum(base) < 0.12
    lift = np.array(st['dark_lift'], np.float32)
    lit_col = base * np.array(st['key_tint'], np.float32)
    lit_col = np.where(dark[..., None], base + lift * 0.55, lit_col)
    shadow_col = np.where(dark[..., None], base * 0.6, shadow_col)
    mid_col = lit_col * 0.5 + shadow_col * 0.5

    # halftone dots in the transition
    dots = halftone_mask(h, w, mid_cov * 0.9, st['halftone_px'] * scale, st['halftone_angle'])
    if not st['use_halftone']:
        dots = (mid_cov > 0.5).astype(np.float32)
    trans = np.clip(1.0 - lit_w - sh_w, 0, 1)
    col = lit_col * lit_w[..., None]
    col += (lit_col * (1 - dots[..., None]) + shadow_col * dots[..., None]) * trans[..., None]
    col += shadow_col * sh_w[..., None]

    # deep shadow hatching + AO creases
    ao = cv2.GaussianBlur(P['ao'], (0, 0), 1.2 * scale)
    crease = np.clip((0.75 - ao) / 0.5, 0, 1)
    hatch_den = np.clip((lo - light) / lo, 0, 1) * 0.55 + crease * 0.6
    hatch_den = np.where(is_hatch, hatch_den, hatch_den * 0.5)
    hatch_den = np.where(P['obj'] >= 40, hatch_den * st.get('env_hatch', 0.25), hatch_den)
    hatch_den = hatch_den * (1 - smoothstep(st['line_fade_near'], st['line_fade_far'], P['depth']))
    hat = hatch_mask(h, w, hatch_den, st['hatch_px'] * scale, st['hatch_angle'])
    hat_col = shadow_col * 0.55
    if not st['use_hatch']:
        hat = hat * 0
    col = col * (1 - hat[..., None] * 0.85) + hat_col * hat[..., None] * 0.85

    # stubble / beard stipple
    stp = stipple(h, w, 0.55, seed + 7) * is_stip * (1.0 if st['use_hatch'] else 0.0)
    col = col * (1 - stp[..., None] * 0.5) + (base * 0.35)[..., :] * stp[..., None] * 0.5

    # fill light from the sky side (painted, not rendered)
    fill = np.clip(-ndl, 0, 1) * st['fill_strength']
    col = col + np.array(st['fill_color'], np.float32) * fill[..., None] * (1 - lit_w[..., None])

    # rim light: grazing angle on the side away from the key
    rim = smoothstep(st['rim_width'], st['rim_width'] + 0.1, 1 - np.abs(ndv)) * (ndl < 0.35)
    rim = rim * (a > 0.99) * (~is_ink) * (1.0 if st['use_rim'] else 0.0)
    rim = cv2.GaussianBlur(rim.astype(np.float32), (0, 0), 0.6)
    rc = np.array(st['rim_color'], np.float32)
    col = col + rc * rim[..., None] * 0.65

    # hard specular shapes
    Hh = (L + V)
    Hh = Hh / np.linalg.norm(Hh)
    ndh = np.clip(np.sum(nrm * Hh, -1), 0, 1)
    spec = (smoothstep(0.93, 0.95, ndh) * spec_s) * (light > 0.2)
    spec_glass = smoothstep(0.80, 0.82, ndh) * is_glass
    metal_band = smoothstep(0.55, 0.57, ndh) * is_metal * 0.5
    spec_all = np.clip(spec + spec_glass + metal_band, 0, 1) * (1.0 if st['use_spec'] else 0.0)
    col = col * (1 - spec_all[..., None]) + np.array([1.0, 0.98, 0.95], np.float32) * spec_all[..., None]

    # flats (eyes, ink pieces) stay graphic
    col = np.where(is_flat[..., None], np.where(light[..., None] > lo, base, base * 0.75), col)
    col = np.where(is_ink[..., None], np.array(st['line_color'], np.float32), col)

    # emission
    em = to_srgb(P['emit'])
    col = np.where(emit_m[..., None], np.clip(em * 0.6 + 0.55, 0, 1), col)

    # ink
    emit_any = lum(P['emit']) > 0.05
    lines = ink_lines(P, st, light, scale, skip=emit_any, seam_skip=is_ink | is_flat)
    lines = lines * (1 - smoothstep(st['line_fade_near'], st['line_fade_far'], P['depth']) * st['line_fade_amt'])
    if not st['use_lines']:
        lines = lines * 0
    lc = np.array(st['line_color'], np.float32)
    col = col * (1 - lines[..., None]) + lc * lines[..., None]
    alpha = np.maximum(a, lines * (cv2.dilate((a > 0.01).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0))

    # glow layer (added after compositing)
    glow = np.zeros_like(col)
    if emit_any.any():
        e = np.clip(P['emit'], 0, 4)
        g1 = cv2.GaussianBlur(e, (0, 0), 6 * scale)
        g2 = cv2.GaussianBlur(e, (0, 0), 22 * scale)
        glow = to_srgb(g1 * 0.8 + g2 * 1.0) * st['glow']
    return dict(color=np.clip(col, 0, 1), alpha=np.clip(alpha, 0, 1), glow=glow, light=light,
                depth=P['depth'], lines=lines)


# ----------------------------------------------------------------------------- compositing

def over(bg, fg, alpha):
    return fg * alpha[..., None] + bg * (1 - alpha[..., None])


def screen(a, b):
    return 1 - (1 - a) * (1 - b)


def misregister(img, px, mask=None):
    """Offset red/blue plates like an off-register print (Spider-Verse defocus)."""
    if px <= 0:
        return img
    M1 = np.float32([[1, 0, px], [0, 1, px * 0.3]])
    M2 = np.float32([[1, 0, -px], [0, 1, -px * 0.3]])
    h, w = img.shape[:2]
    r = cv2.warpAffine(img[..., 0], M1, (w, h), borderMode=cv2.BORDER_REFLECT)
    b = cv2.warpAffine(img[..., 2], M2, (w, h), borderMode=cv2.BORDER_REFLECT)
    out = img.copy()
    out[..., 0] = r
    out[..., 2] = b
    if mask is not None:
        out = img * (1 - mask[..., None]) + out * mask[..., None]
    return out


def misregister_layer(color, alpha, px):
    """Misregister a premultiplied layer, keeping alpha union so fringes show on the bg."""
    pm = color * alpha[..., None]
    rgba = np.dstack([pm, alpha])
    h, w = alpha.shape
    M1 = np.float32([[1, 0, px], [0, 1, px * 0.3]])
    M2 = np.float32([[1, 0, -px], [0, 1, -px * 0.3]])
    r = cv2.warpAffine(rgba[..., [0, 3]], M1, (w, h))
    b = cv2.warpAffine(rgba[..., [2, 3]], M2, (w, h))
    out_a = np.maximum(alpha, np.maximum(r[..., 1], b[..., 1]))
    pm2 = np.dstack([r[..., 0], pm[..., 1], b[..., 0]])
    col = pm2 / np.clip(out_a[..., None], 1e-4, None)
    return np.clip(col, 0, 1), out_a


def paper(img, strength=0.05, seed=1):
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)
    n = rng.normal(0, 1, (h, w)).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), 0.8) * 0.6 + cv2.GaussianBlur(n, (0, 0), 3.0) * 2.0
    fib = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), sigmaX=6, sigmaY=0.6)
    t = (n * 0.6 + fib * 0.8) * strength
    return np.clip(img + t[..., None] * 0.5, 0, 1)


def vignette(img, k=0.25):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    v = 1 - k * smoothstep(0.6, 1.5, r)
    return img * v[..., None]


def save(img, path):
    from PIL import Image
    Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path)
