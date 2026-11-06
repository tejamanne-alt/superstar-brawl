"""BALA SIMHA: hero-quality sculpted model built from the reference photos.

Signature read (from the references): big wavy swept-back salt-and-pepper hair with a high
front wave; full trimmed salt-and-pepper beard (greyest at the chin) joined to a thick dark
mustache that droops at the corners; heavy straight brows; red tilak dot on the forehead;
gold-wire aviators with dark olive lenses; broad face, fleshy cheeks, strong straight nose,
thick neck and stocky build; black shirt with open collar and chest pocket; cream dhoti;
gold watch and rings.

Head-local space: metres, origin between the ears at eye level, face toward -Y, +Z up,
+X = character's left. Body space: metres, origin on the ground between the feet.
"""
import math

import numpy as np

from .sdf import Grid, smin, smax, sub, smoothstep, mesh, to_blender, sample
from . import geo

SKIN = np.array([0.80, 0.56, 0.42], np.float32)
EXPR = {
    'smirk': dict(open=0.0, smile=(0.15, 0.55), brow_in=0.45, glasses=True),
    'roar': dict(open=1.0, smile=(0.0, 0.0), brow_in=1.0, glasses=True),
    'glare': dict(open=0.0, smile=(0.0, 0.1), brow_in=0.9, glasses=False),
    'laugh': dict(open=0.55, smile=(0.9, 0.9), brow_in=-0.3, glasses=False),
    'calm': dict(open=0.0, smile=(0.25, 0.25), brow_in=0.0, glasses=False),
}
C_SKULL = np.array([0.0, 0.012, 0.032], np.float32)
HEAD_TO_BODY = (0.0, -0.012, 1.595)   # head origin in body space (eye level)


# ============================================================================= head fields

def head_fields(g, ex):
    o = ex['open']
    jd = 0.017 * o
    E = g.ellipsoid
    d = smin(E((0, 0.012, 0.032), (0.080, 0.100, 0.090)), E((0, 0.040, 0.0), (0.070, 0.070, 0.075)), 0.02)
    d = smin(d, E((0, -0.030, 0.046), (0.068, 0.066, 0.054)), 0.02)                     # forehead
    for s in (-1, 1):
        d = smin(d, E((s * 0.051, -0.058, -0.010), (0.031, 0.034, 0.026)), 0.016)        # cheekbones
        d = smin(d, E((s * 0.046, -0.042, -0.054 - jd * 0.5), (0.034, 0.044, 0.041)), 0.02)  # fleshy cheeks / jowls
        d = smin(d, E((s * 0.081, 0.012, -0.006), (0.011, 0.021, 0.031)), 0.006)          # ears
    d = smin(d, E((0, -0.030, -0.073 - jd), (0.059, 0.066, 0.050)), 0.02)                 # jaw
    d = smin(d, E((0, -0.071, -0.105 - jd), (0.031, 0.026, 0.025)), 0.014)                # chin
    d = smin(d, E((0, -0.079, -0.050 - jd * 0.4), (0.038, 0.032, 0.036)), 0.016)          # muzzle
    d = smin(d, g.capsule((-0.054, -0.083, 0.021), (0.054, -0.083, 0.021), 0.012), 0.012)  # brow ridge
    for s in (-1, 1):
        d = sub(d, E((s * 0.033, -0.093, 0.002), (0.020, 0.016, 0.012)), 0.006)           # eye sockets
        d = smin(d, g.sphere((s * 0.033, -0.083, 0.0), 0.0118), 0.003)                     # eyeballs
        d = smin(d, E((s * 0.033, -0.090, 0.009), (0.019, 0.010, 0.0065)), 0.004)          # heavy upper lids
        d = smin(d, E((s * 0.035, -0.088, -0.014), (0.016, 0.008, 0.006)), 0.006)          # eye bags
    # nose: strong, straight, broad tip
    d = smin(d, g.capsule((0, -0.094, 0.013), (0, -0.118, -0.027), 0.0085, 0.012), 0.008)
    d = smin(d, g.sphere((0, -0.119, -0.031), 0.0155), 0.006)
    for s in (-1, 1):
        d = smin(d, g.sphere((s * 0.0175, -0.106, -0.036), 0.0115), 0.005)                 # alae
        d = sub(d, E((s * 0.010, -0.113, -0.046), (0.0055, 0.006, 0.0035)), 0.002)         # nostrils
    # lips
    d = smin(d, E((0, -0.104, -0.053), (0.026, 0.012, 0.009)), 0.006)
    d = smin(d, E((0, -0.100, -0.067 - jd), (0.022, 0.012, 0.008)), 0.005)
    cav = None
    if o > 0.05:
        cav = E((0, -0.100, -0.061 - jd * 0.5), (0.023, 0.030, 0.003 + 0.013 * o))
        d = sub(d, cav, 0.004)
    # neck + double chin
    d = smin(d, g.capsule((0, 0.015, -0.06), (0, 0.020, -0.215), 0.060), 0.03)
    d = smin(d, E((0, -0.030, -0.118 - jd), (0.051, 0.048, 0.030)), 0.02)
    return d, cav


def angles(g):
    px, py, pz = g.X - C_SKULL[0], g.Y - C_SKULL[1], g.Z - C_SKULL[2]
    az = np.degrees(np.arctan2(px, -py))
    el = np.degrees(np.arctan2(pz, np.hypot(px, py)))
    return az, el


HL_A = [0, 40, 62, 72, 80, 88, 96, 125, 150, 180]
HL_E = [29, 25, 16, 5, -31, -31, 7, 5, -18, -28]


def hair_field(g, d_head, az, el):
    a = np.abs(az)
    elh = np.interp(a, HL_A, HL_E).astype(np.float32)
    mask = (elh - el) * 0.00175                                          # >0 below the hairline
    front = np.exp(-(a / 58.0) ** 2)
    top = smoothstep(16, 38, el) * (1 - 0.35 * smoothstep(70, 100, el))
    side = np.exp(-((a - 88) / 32.0) ** 2) * smoothstep(-6, 22, el) * (1 - smoothstep(55, 88, el))
    t = 0.013 + 0.034 * front * top + 0.014 * smoothstep(35, 70, el) + 0.030 * side + 0.010 * smoothstep(110, 160, a) * smoothstep(-15, 20, el)
    t = t * smoothstep(elh - 1, elh + 13, el)
    # clumps swept straight back, wavy, wiggling as they travel over the head
    elr = np.radians(el)
    phase = az / 16.0 + 0.80 * np.sin(elr * 4.0 + az / 30.0) + 0.55 * g.noise(0.035, seed=6)
    groove = np.abs(np.sin(np.pi * phase)) ** 0.45
    t = t * (0.64 + 0.36 * groove)
    t = t + 0.0055 * np.sin(elr * 9.0 + np.radians(az) * 2.0) * (0.4 + 0.6 * front) * smoothstep(10, 30, el)
    t = t + 0.0025 * g.noise(0.018, seed=5)
    h = smax(d_head - t, mask, 0.003)
    # the high front wave (pompadour) swept back
    wave = g.ellipsoid((0.004, -0.040, 0.130), (0.072, 0.062, 0.044))
    wave = smax(wave, -(g.Y + 0.106), 0.004)                            # front face of the wave
    wave_g = 0.82 + 0.18 * groove
    h = smin(h, wave / np.maximum(wave_g, 0.5), 0.022)
    return h


def beard_field(g, d_head, ex, cav):
    o = ex['open']
    X, Y, Z = g.X, g.Y, g.Z
    ax = np.abs(X)
    zb = np.interp(ax, [0, 0.02, 0.035, 0.05, 0.065, 0.078, 0.095], [-0.042, -0.042, -0.036, -0.024, -0.011, 0.004, 0.004])
    yb = 0.010 - 0.034 * smoothstep(-0.07, -0.13, Z)          # stop at the jaw angle, then follow the jaw line
    region = np.maximum(Z - zb, Y - yb)
    region = np.maximum(region, -(Z + 0.150))
    lip_z = -0.067 - 0.017 * o
    hole = ((X / (0.025 + 0.004 * o)) ** 2 + ((Z - lip_z) / (0.010 + 0.012 * o)) ** 2 - 1.0) * 0.012
    region = np.maximum(region, np.where(Y < -0.06, -hole, -1.0))
    t = 0.0040 + 0.0058 * smoothstep(-0.05, -0.125, Z) + 0.0012 * smoothstep(0.03, 0.07, ax)
    t = t * smoothstep(0.0, 0.012, -region)                     # taper into the skin at every edge
    t = t * (0.86 + 0.14 * np.abs(np.sin(X * 2 * np.pi / 0.0065 + Z * 55.0)) ** 0.5)
    t = t + (0.0018 * g.noise(0.0065, seed=9) + 0.0010 * g.noise(0.003, seed=10)) * smoothstep(0.0, 0.006, -region)
    b = smax(d_head - t, region, 0.002)
    if cav is not None:
        b = sub(b, cav, 0.003)
    return b


def mustache_field(g, ex):
    sl, sr = ex['smile']
    parts = []
    for s, lift in ((1, sl), (-1, sr)):
        L = lift * 0.008
        pts = [(0, -0.118, -0.045), (s * 0.018, -0.117, -0.046 + L * 0.3), (s * 0.032, -0.108, -0.054 + L * 0.6),
               (s * 0.040, -0.098, -0.068 + L), (s * 0.043, -0.089, -0.083 + L)]
        parts.append(g.tube(pts, [0.0100, 0.0108, 0.0098, 0.0082, 0.0060], k=0.004))
    m = smin(parts[0], parts[1], 0.004)
    m = smin(m, g.ellipsoid((0, -0.114, -0.049), (0.038, 0.0125, 0.0120)), 0.005)
    m = m + 0.0012 * g.noise(0.004, seed=21)
    return m


def brow_field(g, ex):
    bi = ex['brow_in']
    parts = []
    for s in (-1, 1):
        pts = [(s * 0.013, -0.098, 0.022 - 0.006 * bi), (s * 0.029, -0.100, 0.027 - 0.002 * bi),
               (s * 0.047, -0.094, 0.029), (s * 0.061, -0.083, 0.023)]
        parts.append(g.tube(pts, [0.0048, 0.0054, 0.0048, 0.0030], k=0.002))
    return smin(parts[0], parts[1], 0.001) + 0.0007 * g.noise(0.003, seed=33)


# ============================================================================= colouring

def _vnoise(V, scale, seed=0):
    rng = np.random.default_rng(seed)
    k = rng.normal(size=(24, 3)) / scale
    ph = rng.uniform(0, 2 * np.pi, 24)
    return (np.sin(V @ k.T + ph).sum(1) / np.sqrt(12)).astype(np.float32)


def col_skin(V):
    x, y, z = V[:, 0], V[:, 1], V[:, 2]
    c = np.tile(SKIN, (len(V), 1))
    flush = np.exp(-(((np.abs(x) - 0.048) / 0.025) ** 2 + ((z + 0.020) / 0.025) ** 2)) * (y < -0.05)
    nose = np.exp(-((x / 0.018) ** 2 + ((z + 0.030) / 0.02) ** 2)) * (y < -0.10)
    c = c + (np.array([0.06, -0.04, -0.04]) * (0.6 * flush + 0.7 * nose)[:, None])
    fore = smoothstep(0.03, 0.07, z) * (y < -0.05)
    c = c + np.array([0.03, 0.03, 0.02]) * fore[:, None]
    lips = np.exp(-((x / 0.022) ** 2 + ((z + 0.067) / 0.007) ** 2)) * (y < -0.09)
    c = c * (1 - lips[:, None] * 0.6) + np.array([0.62, 0.32, 0.30]) * lips[:, None] * 0.6
    return np.clip(c, 0, 1)


def _mix(dark, light, g):
    g = np.clip(g, 0, 1)[:, None]
    return (np.array(dark) * (1 - g) + np.array(light) * g).astype(np.float32)


def col_hair(V):
    p = V - C_SKULL
    az = np.degrees(np.arctan2(p[:, 0], -p[:, 1]))
    el = np.degrees(np.arctan2(p[:, 2], np.hypot(p[:, 0], p[:, 1])))
    g = 0.08 + 0.40 * smoothstep(55, 95, np.abs(az)) * (1 - smoothstep(30, 62, el))
    streak = _vnoise(V * np.array([2.5, 0.5, 0.8]), 0.006, 3)
    g = g + 0.45 * np.clip(streak - 0.45, 0, 1)
    return _mix((0.12, 0.10, 0.11), (0.62, 0.61, 0.64), g)


def col_beard(V):
    x, z = V[:, 0], V[:, 2]
    g = 0.34 + 0.36 * smoothstep(-0.075, -0.135, z) - 0.10 * smoothstep(0.050, 0.078, np.abs(x))
    strand = _vnoise(V * np.array([1.0, 1.0, 0.35]), 0.0018, 7)
    g = g + 0.16 * strand
    return _mix((0.17, 0.155, 0.165), (0.86, 0.86, 0.87), g)


def col_mustache(V):
    strand = _vnoise(V * np.array([0.5, 1.0, 1.0]), 0.0018, 11)
    return _mix((0.10, 0.085, 0.09), (0.55, 0.55, 0.57), 0.12 + 0.12 * strand)


def col_brow(V):
    return _mix((0.12, 0.10, 0.11), (0.55, 0.55, 0.57), 0.10 + 0.10 * _vnoise(V, 0.002, 13))


# ============================================================================= face decal

DX0, DX1, DZ0, DZ1 = -0.10, 0.10, -0.14, 0.13


def decal_uv(V_local):
    u = (V_local[:, 0] - DX0) / (DX1 - DX0)
    v = (V_local[:, 2] - DZ0) / (DZ1 - DZ0)
    front = V_local[:, 1] < -0.035
    u = np.where(front, u, 0.002)
    v = np.where(front, v, 0.998)
    return np.stack([u, v], 1)


def paint_decal(path, ex, ppm=9000):
    """Ink + paint marks in head-local metric space (front projection)."""
    from PIL import Image, ImageDraw
    from .facepaint import _cr
    ss = 2
    k = ppm * ss
    W, H = int((DX1 - DX0) * k), int((DZ1 - DZ0) * k)
    im = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)

    def P(x, z):
        return ((x - DX0) * k, (DZ1 - z) * k)

    def stroke(pts, wfn, col):
        q = np.array(_cr(pts, 14), float)
        seg = np.diff(q, axis=0)
        L = np.concatenate([[0], np.cumsum(np.linalg.norm(seg, axis=1))])
        L = L / max(L[-1], 1e-9)
        tan = np.gradient(q, axis=0)
        tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-12
        nrm = np.stack([-tan[:, 1], tan[:, 0]], 1)
        w = np.array([wfn(t) for t in L]) / 2
        poly = [P(*p) for p in q + nrm * w[:, None]] + [P(*p) for p in (q - nrm * w[:, None])[::-1]]
        d.polygon(poly, fill=col)

    def ell(cx, cz, rx, rz, col):
        x0, y0 = P(cx - rx, cz + rz)
        x1, y1 = P(cx + rx, cz - rz)
        d.ellipse([x0, y0, x1, y1], fill=col)

    INK = (40, 20, 28, 255)
    LINE = (95, 45, 40, 190)
    taper = lambda w: (lambda t: w * (0.25 + 0.75 * math.sin(math.pi * t)))
    # forehead lines, crow's feet, eye bags, nasolabial folds above the beard
    for z0 in (0.056, 0.065):
        stroke([(-0.034, z0 - 0.002), (0, z0 + 0.002), (0.034, z0 - 0.002)], taper(0.0010), LINE)
    for s in (-1, 1):
        for a in (-0.006, 0.0, 0.007):
            stroke([(s * 0.060, a), (s * 0.072, a * 1.8 + 0.002)], taper(0.0009), LINE)
        stroke([(s * 0.018, -0.018), (s * 0.034, -0.023), (s * 0.050, -0.017)], taper(0.0011), LINE)
        stroke([(s * 0.022, -0.020), (s * 0.034, -0.032), (s * 0.040, -0.040)], taper(0.0012), LINE)
        # nostril darks
        ell(s * 0.0105, -0.0455, 0.0045, 0.0026, (45, 18, 20, 230))
        stroke([(s * 0.026, -0.030), (s * 0.028, -0.038), (s * 0.020, -0.044)], taper(0.0014), INK)
    # tilak
    ell(0.0, 0.040, 0.0050, 0.0064, (205, 22, 30, 255))
    ell(-0.0012, 0.0418, 0.0016, 0.0020, (240, 90, 80, 255))
    # mouth line / lower lip shadow (only when closed)
    if ex['open'] < 0.05:
        sl, sr = ex['smile']
        stroke([(-0.022, -0.0595 + 0.004 * sr), (-0.010, -0.0605), (0.010, -0.0605), (0.022, -0.0595 + 0.004 * sl)],
               lambda t: 0.0014 + 0.0010 * math.sin(math.pi * t), INK)
        stroke([(-0.012, -0.0755), (0, -0.0765), (0.012, -0.0755)], taper(0.0012), LINE)
    if not ex['glasses']:
        bi = ex['brow_in']
        for s in (-1, 1):
            ex0, ez0 = s * 0.033, 0.001
            ew, eh = 0.0135, 0.0052 * (0.8 if bi > 0.5 else 1.0)
            upper = [(ex0 - s * ew, ez0 - 0.001), (ex0 - s * ew * 0.4, ez0 + eh - 0.002 * bi), (ex0 + s * ew * 0.45, ez0 + eh),
                     (ex0 + s * ew, ez0 + 0.0005)]
            lower = [(ex0 - s * ew, ez0 - 0.001), (ex0, ez0 - eh * 0.85), (ex0 + s * ew, ez0 + 0.0005)]
            alm = _cr(upper) + _cr(lower)[::-1]
            d.polygon([P(*p) for p in alm], fill=(245, 236, 222, 255))
            r = eh * 0.95
            lay = Image.new('RGBA', im.size, (0, 0, 0, 0))
            ld = ImageDraw.Draw(lay)
            for (cx, cz, rr, col) in ((ex0 - s * 0.001, ez0 - 0.0005, r, (92, 54, 30, 255)),
                                      (ex0 - s * 0.001, ez0 - 0.0005, r * 0.5, (20, 10, 12, 255)),
                                      (ex0 - s * 0.001 + r * 0.35, ez0 + r * 0.35, r * 0.22, (255, 255, 255, 255))):
                x0, y0 = P(cx - rr, cz + rr)
                x1, y1 = P(cx + rr, cz - rr)
                ld.ellipse([x0, y0, x1, y1], fill=col)
            m = Image.new('L', im.size, 0)
            ImageDraw.Draw(m).polygon([P(*p) for p in alm], fill=255)
            lay.putalpha(Image.fromarray(np.minimum(np.asarray(lay)[..., 3], np.asarray(m))))
            im.alpha_composite(lay)
            d = ImageDraw.Draw(im)
            stroke(upper, lambda t: 0.0016 + 0.0016 * math.sin(math.pi * min(1, t * 1.2)), INK)
            stroke([(x, z + 0.0045) for x, z in upper[1:]], taper(0.0010), LINE)
            stroke(lower[1:], taper(0.0008), LINE)
    out = im.resize((W // ss, H // ss), Image.LANCZOS)
    out.save(path)
    return path


# ============================================================================= materials

def vcol_material(ml, name, fallback, decal=None, **props):
    """Material reading the 'Col' vertex colour, optionally overlaid by a decal texture."""
    import bpy
    if name in ml.mats:
        return ml.mats[name]
    m = ml.get(name, fallback, **props)
    nt = m.node_tree
    bsdf = [n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'][0]
    vc = nt.nodes.new('ShaderNodeVertexColor')
    vc.layer_name = 'Col'
    if decal is None:
        nt.links.new(vc.outputs['Color'], bsdf.inputs['Base Color'])
        return m
    tex = nt.nodes.new('ShaderNodeTexImage')
    tex.image = bpy.data.images.load(decal, check_existing=False)
    tex.image.alpha_mode = 'STRAIGHT'
    tex.extension = 'CLIP'
    tex.interpolation = 'Cubic'
    mix = nt.nodes.new('ShaderNodeMix')
    mix.data_type = 'RGBA'
    cols = [s for s in mix.inputs if s.type == 'RGBA']
    fac = [s for s in mix.inputs if s.type == 'VALUE'][0]
    out = [s for s in mix.outputs if s.type == 'RGBA'][0]
    nt.links.new(vc.outputs['Color'], cols[0])
    nt.links.new(tex.outputs['Color'], cols[1])
    nt.links.new(tex.outputs['Alpha'], fac)
    nt.links.new(out, bsdf.inputs['Base Color'])
    return m


# ============================================================================= head build

def head_matrix(turn=0.0, tilt=0.0, roll=0.0):
    """Head-local -> body. turn: + looks toward character's left; tilt: + chin up."""
    from mathutils import Matrix, Vector
    piv = Vector((0, 0.015, -0.11))                     # pivot at the top of the neck (head-local)
    R = (Matrix.Rotation(math.radians(turn), 4, 'Z') @ Matrix.Rotation(math.radians(-tilt), 4, 'X') @
         Matrix.Rotation(math.radians(roll), 4, 'Y'))
    S = Matrix.Diagonal((1.07, 1.07, 1.07, 1.0))
    return Matrix.Translation(Vector(HEAD_TO_BODY)) @ Matrix.Translation(piv) @ R @ S @ Matrix.Translation(-piv)


def build_head(ml, build_dir, expr='smirk', M=None, step=0.0013, pass_index=1, tag='bala'):
    import os
    from mathutils import Matrix
    ex = EXPR[expr]
    M = M or Matrix.Identity(4)
    Mn = np.array(M, np.float32)

    def xf(V):
        return (V @ Mn[:3, :3].T + Mn[:3, 3]).astype(np.float32)

    g = Grid((-0.108, -0.150, -0.220), (0.108, 0.138, 0.182), step)
    d_head, cav = head_fields(g, ex)
    az, el = angles(g)
    os.makedirs(os.path.join(build_dir, 'bala'), exist_ok=True)
    decal = paint_decal(os.path.join(build_dir, 'bala', f'decal_{expr}.png'), ex)
    skin_m = vcol_material(ml, f'bala.skin.{expr}', SKIN, decal=decal, skin=True, char='bala')
    hair_m = vcol_material(ml, 'bala.hair', (0.12, 0.10, 0.10), hair=True, spec=0.7, fur=True, char='bala')
    beard_m = vcol_material(ml, 'bala.beard', (0.5, 0.5, 0.5), fur=True, beard=True, char='bala')
    brow_m = vcol_material(ml, 'bala.brow', (0.12, 0.10, 0.11), fur=True, char='bala')
    obs = []
    V, F = mesh(g, d_head)
    C = col_skin(V)
    if cav is not None:
        inside = sample(g, cav, V) < 0.0025
        C[inside] = np.array([0.32, 0.07, 0.09])
    obs.append(to_blender(f'{tag}_head', xf(V), F, skin_m, C, uv_fn=lambda _V, V0=V: decal_uv(V0), pass_index=pass_index))
    for name, field, colf, mat in (('hair', hair_field(g, d_head, az, el), col_hair, hair_m),
                                   ('beard', beard_field(g, d_head, ex, cav), col_beard, beard_m),
                                   ('mustache', mustache_field(g, ex), col_mustache, beard_m),
                                   ('brows', brow_field(g, ex), col_brow, brow_m)):
        V, F = mesh(g, field)
        if V is None:
            continue
        obs.append(to_blender(f'{tag}_{name}', xf(V), F, mat, colf(V), pass_index=pass_index, smooth_iters=1))
    if ex['open'] > 0.05:
        tm = ml.get('bala.teeth', (0.97, 0.95, 0.90), flat=True, char='bala')
        from mathutils import Matrix as Mx, Vector
        for zz, rz in ((-0.057, 0.0045), (-0.068 - 0.017 * ex['open'], 0.0035)):
            T = M @ Mx.Translation(Vector((0, -0.098, zz))) @ Mx.Diagonal((0.019, 0.009, rz, 1))
            ob = geo.blob(f'{tag}_teeth{zz}', lambda p: p, segments=16, rings=8, mats=[tm], matrix=T, subsurf=1,
                          pass_index=pass_index)
            obs.append(ob)
    if ex['glasses']:
        obs += aviators(ml, M, pass_index, tag)
    return obs


def aviators(ml, M, pass_index=1, tag='bala'):
    from mathutils import Vector
    gold = ml.get('bala.gold', (0.95, 0.76, 0.30), metal=True, spec=1.0, char='bala')
    lens = ml.get('bala.lens', (0.16, 0.19, 0.14), glass=True, spec=1.0, char='bala')
    outline = [(-0.022, 0.018), (0.0, 0.0205), (0.022, 0.019), (0.0295, 0.009), (0.0285, -0.010), (0.019, -0.026),
               (0.003, -0.0295), (-0.013, -0.023), (-0.0235, -0.009)]
    obs = []
    for s in (-1, 1):
        cx, cz, cy = s * 0.034, -0.003, -0.121
        ring = [Vector((cx + s * u, cy - 0.0025 * (u / 0.03), cz + v)) for u, v in outline]
        pts = geo.catmull_rom(ring + [ring[0], ring[1]], 6)[:-6]
        obs.append(geo.loft(f'{tag}_frame{s}', [M @ p for p in pts], 0.0011, ring=6, samples=1, mats=[gold], caps=(False, False),
                            subsurf=0, pass_index=pass_index))
        # lens: fan triangulation, slightly bulged
        cen = Vector((cx, cy - 0.003, cz - 0.002))
        verts = [M @ cen] + [M @ (p + Vector((0, -0.0012, 0))) for p in pts]
        faces = [(0, i + 1, (i + 1) % len(pts) + 1) for i in range(len(pts))]
        obs.append(geo.make_object(f'{tag}_lens{s}', verts, faces, [lens], [0] * len(faces), subsurf=1, pass_index=pass_index))
        # temple arm back to the ear, hooking down behind it
        t0 = Vector((cx + s * 0.0285, cy + 0.001, cz + 0.012))
        arm = [t0, Vector((s * 0.074, -0.075, 0.012)), Vector((s * 0.085, -0.02, 0.010)), Vector((s * 0.086, 0.020, 0.004)),
               Vector((s * 0.083, 0.032, -0.012))]
        obs.append(geo.loft(f'{tag}_temple{s}', [M @ p for p in arm], 0.0012, ring=6, samples=6, mats=[gold], subsurf=0,
                            pass_index=pass_index))
        pad = Vector((s * 0.011, -0.110, -0.010))
        obs.append(geo.blob(f'{tag}_pad{s}', lambda p: p, segments=8, rings=5, mats=[gold], subsurf=0, pass_index=pass_index,
                            matrix=M @ geo.Matrix.Translation(pad) @ geo.Matrix.Diagonal((0.0025, 0.002, 0.004, 1))))
    for z, sag in ((0.0175, 0.0), (0.004, 0.004)):
        br = [Vector((-0.012, -0.121, z)), Vector((0, -0.124, z + sag)), Vector((0.012, -0.121, z))]
        obs.append(geo.loft(f'{tag}_bridge{z}', [M @ p for p in br], 0.0011, ring=6, samples=6, mats=[gold], subsurf=0,
                            pass_index=pass_index))
    return obs


# ============================================================================= body (crossed-arms "mass" pose)

def _n(v):
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v)


ARMS = dict(
    S_L=(0.185, 0.012, 1.385), E_L=(0.200, -0.180, 1.170), W_L=(-0.090, -0.222, 1.235),
    S_R=(-0.185, 0.012, 1.385), E_R=(-0.200, -0.115, 1.125), W_R=(0.085, -0.152, 1.190),
)
SHIRT = (0.17, 0.17, 0.21)   # "black" shirt as a printable charcoal
DHOTI = (0.95, 0.91, 0.80)


def torso_core(g):
    E = g.ellipsoid
    d = smin(E((0, 0.005, 1.27), (0.168, 0.120, 0.150)), E((0, 0.015, 1.37), (0.182, 0.100, 0.072)), 0.045)
    for s in (-1, 1):
        d = smin(d, g.capsule((s * 0.05, 0.03, 1.46), (s * 0.165, 0.02, 1.405), 0.045), 0.04)
        d = smin(d, E((s * 0.072, -0.083, 1.30), (0.078, 0.048, 0.062)), 0.03)
    d = smin(d, E((0, -0.022, 1.10), (0.158, 0.128, 0.150)), 0.05)
    d = smin(d, E((0, 0.0, 0.95), (0.160, 0.112, 0.110)), 0.05)
    d = smin(d, g.capsule((0, 0.022, 1.40), (0, 0.024, 1.52), 0.060), 0.03)
    return d


def arm_core(g, S, E, W):
    d = g.sphere(S, 0.064)
    d = smin(d, g.capsule(S, E, 0.057, 0.046), 0.02)
    d = smin(d, g.sphere(E, 0.046), 0.01)
    d = smin(d, g.capsule(E, W, 0.048, 0.036), 0.015)
    return d


def arm_frame(S, E):
    A = _n(np.subtract(E, S))
    f = np.array([0, -1.0, 0])
    ef = _n(f - A * (A @ f))
    eo = _n(np.cross(A, ef))
    return A, ef, eo


def hand_field(g, W, S_o, E_o, outer_sign, t_grip=0.50):
    """Hand on the opposite upper arm (S_o->E_o): palm on its front, fingers wrapping to the
    outer side (outer_sign = sign of x of that arm's outer side)."""
    A, ef, eo = arm_frame(S_o, E_o)
    if np.sign(eo[0]) != outer_sign:
        eo = -eo
    C = np.add(S_o, np.subtract(E_o, S_o) * t_grip)
    R = 0.057 + (0.046 - 0.057) * t_grip + 0.009
    d_th = lambda th: ef * math.cos(th) + eo * math.sin(th)
    th_p = math.radians(-8)
    n = d_th(th_p)
    Pc = C + n * (R + 0.016)
    tng = _n(np.subtract(Pc, W))
    tng = _n(tng - n * (tng @ n))
    side = _n(np.cross(n, tng))
    d = g.ellipsoid_rot(tuple(Pc), [tng, side, n], (0.046, 0.038, 0.019))
    d = smin(d, g.capsule(tuple(W), tuple(Pc - tng * 0.02), 0.032, 0.036), 0.012)
    for k, (ax_off, ln) in enumerate(((-0.0225, 0.95), (-0.0075, 1.05), (0.0075, 1.0), (0.0225, 0.82))):
        pts, rad = [], []
        for j in range(5):
            th = th_p + math.radians(30 + 58 * ln * j / 4)
            p = C + A * ((0.0 if True else 0) + np.dot(side, A) * 0) + d_th(th) * (R + 0.010)
            p = p + side * ax_off
            pts.append(tuple(p))
            rad.append(0.0112 - 0.0008 * j)
        d = smin(d, g.tube(pts, rad, k=0.003), 0.006)
    # thumb along the front of the arm toward the shoulder
    tb = [tuple(Pc - tng * 0.03 + side * 0.035 + n * 0.006), tuple(Pc - tng * 0.005 + side * 0.060 + n * 0.004),
          tuple(Pc + tng * 0.012 + side * 0.075 + n * 0.0)]
    d = smin(d, g.tube(tb, [0.012, 0.010, 0.0085], k=0.003), 0.006)
    return d, dict(C=C, A=A, side=side, n=n, R=R, th_p=th_p, eo=eo, ef=ef)


def body_meshes(ml, full=True, pass_index=1, tag='bala', skin_mat=None):
    from mathutils import Vector
    obs = []
    shirt_m = vcol_material(ml, 'bala.shirt', SHIRT, char='bala', cloth=True, spec=0.25)
    skin_m = skin_mat or vcol_material(ml, 'bala.skinbody', SKIN, skin=True, char='bala')
    gold = ml.get('bala.gold', (0.95, 0.76, 0.30), metal=True, spec=1.0, char='bala')
    a = {k: np.array(v) for k, v in ARMS.items()}

    # ---- torso: skin core (neck + V) and shirt shell
    g = Grid((-0.25, -0.22, 0.84 if full else 0.98), (0.25, 0.18, 1.54), 0.003)
    core = torso_core(g)
    X, Y, Z = g.X, g.Y, g.Z
    V, F = mesh(g, smax(core, -(Z - 1.24), 0.0))
    if V is not None:
        obs.append(to_blender(f'{tag}_torso_skin', V, F, skin_m, np.tile(SKIN, (len(V), 1)),
                              uv_fn=lambda _V: np.tile([0.002, 0.998], (len(_V), 1)), pass_index=pass_index))
    folds = 0.0030 * g.noise(0.09, seed=41) + 0.0008 * g.noise(0.03, seed=42)
    shirt = core - 0.009 - folds - 0.010 * smoothstep(0.93, 0.87, Z)
    collar_z = 1.452 - 0.012 * smoothstep(0.0, -0.08, Y)
    vw = 0.062 * smoothstep(1.30, 1.44, Z)
    vcut = np.where((Y < -0.02) & (Z > 1.30), vw - np.abs(X), -1.0)
    shirt = np.maximum(shirt, Z - collar_z)
    shirt = np.maximum(shirt, vcut * 0.5)
    shirt = np.maximum(shirt, -(Z - 0.87))
    V, F = mesh(g, shirt)
    obs.append(to_blender(f'{tag}_shirt', V, F, shirt_m, np.tile(np.array(SHIRT, np.float32), (len(V), 1)),
                          pass_index=pass_index))
    # placket, buttons and the chest pocket follow the shirt surface
    def front_y(x, z):
        ys = np.linspace(-0.22, 0.0, 400)
        P = np.stack([np.full_like(ys, x), ys, np.full_like(ys, z)], 1).astype(np.float32)
        v = sample(g, shirt, P)
        i = np.argmax(v < 0)
        return float(ys[max(i - 1, 0)])
    plk = [Vector((0.006, front_y(0.006, z) - 0.0015, z)) for z in np.linspace(0.90, 1.30, 9)]
    obs.append(geo.loft(f'{tag}_placket', plk, (0.013, 0.0016), ring=8, samples=2, mats=[shirt_m], subsurf=0,
                        side_hint=(1, 0, 0), pass_index=pass_index))
    btn = ml.get('bala.button', (0.10, 0.10, 0.13), spec=0.8, char='bala')
    for z in (0.98, 1.10, 1.22):
        y = front_y(0.006, z) - 0.004
        obs.append(geo.blob(f'{tag}_btn{z}', lambda p: p, segments=10, rings=6, mats=[btn], subsurf=0,
                            pass_index=pass_index,
                            matrix=geo.Matrix.Translation((0.006, y, z)) @ geo.Matrix.Diagonal((0.0055, 0.0025, 0.0055, 1))))
    pk = [(0.045, 1.215), (0.115, 1.215), (0.115, 1.30), (0.045, 1.30), (0.045, 1.215)]
    pts = []
    for (x0, z0), (x1, z1) in zip(pk, pk[1:]):
        for t in np.linspace(0, 1, 6, endpoint=False):
            x, z = x0 + (x1 - x0) * t, z0 + (z1 - z0) * t
            pts.append(Vector((x, front_y(x, z) - 0.0012, z)))
    pts.append(pts[0])
    obs.append(geo.loft(f'{tag}_pocket', pts, 0.0016, ring=6, samples=1, mats=[shirt_m], subsurf=0, caps=(False, False),
                        pass_index=pass_index))
    # ---- collar: band behind the neck + open points lying on the chest
    nb = Vector((0, 0.024, 1.452))
    arc = [nb + Vector((math.sin(th) * 0.074, math.cos(th) * 0.074 - 0.004, 0.012 * (1 - math.cos(th))))
           for th in np.radians(np.linspace(-118, 118, 17))]
    obs.append(geo.loft(f'{tag}_collarband', arc, (0.0035, 0.019), ring=8, samples=3, mats=[shirt_m], subsurf=1,
                        side_hint=lambda t, arc=arc: (Vector((arc[min(int(t * 16), 16)].x, arc[min(int(t * 16), 16)].y - 0.02, 0))).normalized(),
                        pass_index=pass_index))
    for s in (-1, 1):
        pts = [Vector((s * 0.060, -0.038, 1.472)), Vector((s * 0.078, -0.080, 1.440)), Vector((s * 0.096, -0.112, 1.398))]
        nrm = Vector((s * 0.35, -1.0, 0.55)).normalized()
        obs.append(geo.loft(f'{tag}_collar{s}', pts, lambda t: (0.0035, 0.030 * (1 - 0.85 * t) + 0.003), ring=8,
                            samples=5, mats=[shirt_m], subsurf=1, side_hint=nrm, pass_index=pass_index))
    # gold chain in the open collar
    ch = [Vector((s * 0.055 * math.cos(t), -0.06 - 0.075 * math.sin(t) ** 1.5, 1.44 - 0.095 * math.sin(t) ** 2))
          for s in (-1,) for t in np.linspace(0, math.pi, 1)]
    chain = [Vector((math.cos(t) * 0.058, -0.070 - 0.06 * math.sin(t), 1.43 - 0.10 * math.sin(t) ** 2))
             for t in np.linspace(0.15, math.pi - 0.15, 24)]
    obs.append(geo.loft(f'{tag}_chain', chain, 0.0017, ring=6, samples=2, mats=[gold], subsurf=0, pass_index=pass_index))

    # ---- arms (sleeves) and hands
    for s, side in ((1, 'L'), (-1, 'R')):
        S, E, W = a['S_' + side], a['E_' + side], a['W_' + side]
        pts = np.array([S, E, W])
        lo, hi = pts.min(0) - 0.085, pts.max(0) + 0.085
        g = Grid(lo, hi, 0.003)
        core = arm_core(g, S, E, W)
        sl = core - 0.009 - 0.0035 * g.noise(0.05, seed=50 + s) - 0.0008 * g.noise(0.02, seed=60 + s)
        # cuff ring + sleeve end a little before the wrist
        A = _n(W - E)
        tw = (g.X - W[0]) * A[0] + (g.Y - W[1]) * A[1] + (g.Z - W[2]) * A[2]
        sl = sl - 0.003 * smoothstep(-0.065, -0.050, tw) * (1 - smoothstep(-0.018, -0.012, tw))
        sl = np.maximum(sl, tw + 0.012)
        V, F = mesh(g, sl)
        obs.append(to_blender(f'{tag}_sleeve{side}', V, F, shirt_m, np.tile(np.array(SHIRT, np.float32), (len(V), 1)),
                              pass_index=pass_index))
        # wrist skin between cuff and hand
        V, F = mesh(g, np.maximum(core, -(tw + 0.02)))
        if V is not None:
            obs.append(to_blender(f'{tag}_wrist{side}', V, F, skin_m, np.tile(SKIN, (len(V), 1)),
                                  uv_fn=lambda _V: np.tile([0.002, 0.998], (len(_V), 1)), pass_index=pass_index))
    # hands: left hand grips the right upper arm, right hand grips the left one
    for side, o_side, outer in (('L', 'R', -1), ('R', 'L', 1)):
        W = a['W_' + side]
        S_o, E_o = a['S_' + o_side], a['E_' + o_side]
        Cg = np.add(S_o, np.subtract(E_o, S_o) * 0.5)
        lo, hi = np.minimum(W, Cg) - 0.12, np.maximum(W, Cg) + 0.12
        g = Grid(lo, hi, 0.0016)
        hd, info = hand_field(g, W, S_o, E_o, outer, t_grip=0.48 if side == 'L' else 0.62)
        V, F = mesh(g, hd)
        obs.append(to_blender(f'{tag}_hand{side}', V, F, skin_m, np.tile(SKIN, (len(V), 1)),
                              uv_fn=lambda _V: np.tile([0.002, 0.998], (len(_V), 1)), pass_index=pass_index))
        if side == 'L':   # ring on the ring finger
            C, A2, sd, R = info['C'], info['A'], info['side'], info['R']
            th = info['th_p'] + math.radians(40)
            c = C + (info['ef'] * math.cos(th) + info['eo'] * math.sin(th)) * (R + 0.010) + sd * 0.009
            ring = [Vector(tuple(c + (info['ef'] * math.cos(th) + info['eo'] * math.sin(th)) * 0.0105 * math.cos(u) + sd * 0.0105 * math.sin(u)))
                    for u in np.linspace(0, 2 * math.pi, 13)]
            obs.append(geo.loft(f'{tag}_ring', ring, 0.0022, ring=6, samples=2, mats=[gold], subsurf=0, caps=(False, False),
                                pass_index=pass_index))
    # gold watch on the left wrist
    E, W = a['E_L'], a['W_L']
    A = _n(W - E)
    u = _n(np.cross(A, [0, 0, 1.0]))
    v = _n(np.cross(A, u))
    c = W - A * 0.030
    r = 0.036 + (0.048 - 0.036) * 0.1 + 0.004
    band = [Vector(tuple(c + (u * math.cos(t) + v * math.sin(t)) * r)) for t in np.linspace(0, 2 * math.pi, 25)]
    obs.append(geo.loft(f'{tag}_watchband', band, (0.004, 0.011), ring=8, samples=2, mats=[gold], subsurf=1,
                        caps=(False, False), side_hint=tuple(A), pass_index=pass_index))
    up = -v if v[2] < 0 else v
    face_c = c + up * (r + 0.003)
    obs.append(geo.blob(f'{tag}_watchface', lambda p: p, segments=16, rings=8, mats=[gold], subsurf=1, pass_index=pass_index,
                        matrix=geo.basis(tuple(A), tuple(u), tuple(face_c)) @ geo.Matrix.Diagonal((0.016, 0.016, 0.005, 1))))
    if not full:
        return obs

    # ---- dhoti + legs + sandals
    g = Grid((-0.27, -0.24, 0.05), (0.27, 0.22, 1.00), 0.004)
    legs = None
    for s in (-1, 1):
        t = g.tube([(s * 0.095, 0.0, 0.92), (s * 0.110, -0.012, 0.50), (s * 0.120, 0.015, 0.09)], [0.112, 0.098, 0.106], k=0.02)
        legs = t if legs is None else smin(legs, t, 0.10)
    legs = smin(legs, g.ellipsoid((0, 0.0, 0.93), (0.182, 0.130, 0.100)), 0.06)
    pleat = 0.004 * np.abs(np.sin(g.X * 2 * np.pi / 0.034)) * smoothstep(-0.04, -0.12, g.Y) * smoothstep(0.88, 0.40, g.Z)
    legs = legs + pleat - 0.003 * g.noise(0.06, seed=71) - 0.0015 * g.noise(0.02, seed=72)
    legs = np.maximum(legs, -(g.Z - 0.075 - 0.006 * g.noise(0.05, seed=73)))
    legs = np.maximum(legs, g.Z - 0.98)
    V, F = mesh(g, legs)
    cz = V[:, 2]
    col = np.tile(np.array(DHOTI, np.float32), (len(V), 1))
    border = (cz < 0.105) & (cz > 0.088)
    col[border] = np.array([0.86, 0.62, 0.20])
    dhoti_m = vcol_material(ml, 'bala.dhoti', DHOTI, char='bala', cloth=True)
    obs.append(to_blender(f'{tag}_dhoti', V, F, dhoti_m, col, pass_index=pass_index))
    sole = ml.get('bala.sandal', (0.45, 0.25, 0.12), char='bala', spec=0.3)
    for s in (-1, 1):
        foot = geo.Matrix.Translation((s * 0.122, -0.055, 0.040)) @ geo.Matrix.Rotation(math.radians(-s * 8), 4, 'Z') @ \
            geo.Matrix.Diagonal((0.046, 0.118, 0.034, 1))
        obs.append(geo.blob(f'{tag}_foot{s}', lambda p: Vector((p.x, p.y, p.z * (0.6 if p.y < 0 else 1.0))), segments=20,
                            rings=12, mats=[skin_m], matrix=foot, subsurf=1, pass_index=pass_index))
        sm = geo.Matrix.Translation((s * 0.122, -0.060, 0.010)) @ geo.Matrix.Rotation(math.radians(-s * 8), 4, 'Z') @ \
            geo.Matrix.Diagonal((0.056, 0.132, 0.010, 1))
        obs.append(geo.blob(f'{tag}_sole{s}', lambda p: p, segments=20, rings=8, mats=[sole], matrix=sm, subsurf=1,
                            pass_index=pass_index))
        for yy in (-0.10, -0.03):
            strap = [Vector((s * 0.122 + dx, -0.06 + yy, 0.022 + 0.030 * (1 - (dx / 0.05) ** 2))) for dx in np.linspace(-0.05, 0.05, 7)]
            obs.append(geo.loft(f'{tag}_strap{s}{yy}', strap, (0.012, 0.003), ring=6, samples=2, mats=[sole], subsurf=0,
                                side_hint=(0, 1, 0), pass_index=pass_index))
    return obs


def build(ml, build_dir, expr='smirk', full=True, turn=8.0, tilt=5.0, roll=0.0, head_step=0.0013, tag='bala'):
    """Whole character in body space. Returns (objects, head_matrix)."""
    M = head_matrix(turn, tilt, roll)
    obs = build_head(ml, build_dir, expr, M=M, step=head_step, tag=tag)
    skin_m = ml.mats[f'bala.skin.{expr}']
    obs += body_meshes(ml, full=full, tag=tag, skin_mat=skin_m)
    return obs, M
