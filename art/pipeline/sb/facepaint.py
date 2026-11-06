"""2D comic face artwork, projected onto the 3D skull (front planar projection).

Faces are drawn the way a comic penciller would ink them (New Riverdale / modern teen-comic
look): heavy tapered upper lash line with a flick, small iris under the lid, crease line,
tapered brows, a nose described only by nostril wings and one shadow-side line, big readable
mouths (grin / smirk / shout / gritted teeth) and stylised pointed mustaches.

Coordinates are the head's local deformed space: x right (character's left = viewer's right
seen from the front), z up, units of the head radius. The texture covers x in [-0.9, 0.9] and
z in [-1.3, 1.2]; head.py maps UVs with exactly the same window.
"""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

X0, X1, Z0, Z1 = -0.9, 0.9, -1.3, 1.2
INK = (18, 10, 33, 255)


def uv_of(x, z):
    return ((x - X0) / (X1 - X0), (z - Z0) / (Z1 - Z0))


def _cr(pts, n=10):
    """Catmull-Rom through 2D points."""
    p = [np.array(q, float) for q in pts]
    if len(p) < 3:
        return [tuple(p[0] + (p[1] - p[0]) * t) for t in np.linspace(0, 1, n + 1)]
    ext = [p[0] + (p[0] - p[1])] + p + [p[-1] + (p[-1] - p[-2])]
    out = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        for t in np.linspace(0, 1, n, endpoint=False):
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t +
                              (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(p[-1])
    return [tuple(q) for q in out]


def ss(e0, e1, x):
    t = min(1.0, max(0.0, (x - e0) / (e1 - e0)))
    return t * t * (3 - 2 * t)


def rgba(c, a=255):
    if len(c) == 4:
        return tuple(int(v) for v in c)
    return tuple(int(v * 255) if v <= 1.0 else int(v) for v in c) + (a,)


class FacePainter:
    def __init__(self, ppu=620, supersample=2):
        self.k = ppu * supersample
        self.ssf = supersample
        self.W = int((X1 - X0) * self.k)
        self.H = int((Z1 - Z0) * self.k)
        self.img = Image.new('RGBA', (self.W, self.H), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.img)

    # -------------------------------------------------------------- primitives
    def P(self, x, z):
        return ((x - X0) * self.k, (Z1 - z) * self.k)

    def poly(self, pts, fill, smooth=True, outline=None, width=0.0, layer=None):
        q = _cr(pts) if smooth else pts
        d = ImageDraw.Draw(layer) if layer is not None else self.d
        d.polygon([self.P(*p) for p in q], fill=rgba(fill))
        if outline is not None and width > 0:
            self.stroke(q + [q[0]], lambda t: width, outline, smooth=False)

    def stroke(self, pts, wfn, col=INK, smooth=True, caps=True, layer=None):
        """Tapered stroke; wfn(t) gives the full width in head units."""
        q = np.array(_cr(pts, 14) if smooth else pts, float)
        if len(q) < 2:
            return
        seg = np.diff(q, axis=0)
        L = np.concatenate([[0], np.cumsum(np.linalg.norm(seg, axis=1))])
        L = L / max(L[-1], 1e-9)
        tan = np.gradient(q, axis=0)
        tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-12
        nrm = np.stack([-tan[:, 1], tan[:, 0]], 1)
        w = np.array([wfn(t) for t in L]) / 2
        left = q + nrm * w[:, None]
        right = q - nrm * w[:, None]
        d = ImageDraw.Draw(layer) if layer is not None else self.d
        pts_px = [self.P(*p) for p in left] + [self.P(*p) for p in right[::-1]]
        d.polygon(pts_px, fill=rgba(col))
        if caps:
            for p, r in ((q[0], w[0]), (q[-1], w[-1])):
                if r * self.k > 0.6:
                    cx, cy = self.P(*p)
                    rr = r * self.k
                    d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=rgba(col))

    def ellipse(self, c, rx, rz, fill, layer=None):
        d = ImageDraw.Draw(layer) if layer is not None else self.d
        x, y = self.P(*c)
        d.ellipse([x - rx * self.k, y - rz * self.k, x + rx * self.k, y + rz * self.k], fill=rgba(fill))

    def clipped(self, mask_pts, draw_fn):
        """Draw with draw_fn(layer) and keep only what falls inside mask polygon."""
        lay = Image.new('RGBA', self.img.size, (0, 0, 0, 0))
        draw_fn(lay)
        m = Image.new('L', self.img.size, 0)
        ImageDraw.Draw(m).polygon([self.P(*p) for p in _cr(mask_pts)], fill=255)
        lay.putalpha(Image.fromarray(np.minimum(np.asarray(lay)[..., 3], np.asarray(m))))
        self.img.alpha_composite(lay)
        self.d = ImageDraw.Draw(self.img)

    def soft(self, c, rx, rz, col, alpha, blur=0.05):
        lay = Image.new('RGBA', self.img.size, (0, 0, 0, 0))
        self.ellipse(c, rx, rz, tuple(col[:3]) + (alpha,) if max(col) > 1 else rgba(col, alpha), layer=lay)
        lay = lay.filter(ImageFilter.GaussianBlur(blur * self.k))
        self.img.alpha_composite(lay)
        self.d = ImageDraw.Draw(self.img)

    # -------------------------------------------------------------- features
    def eye(self, side, f, expr):
        ex, ez = f.get('eye_x', 0.29), f.get('eye_z', 0.11)
        ew = 0.172 * f.get('eye_w', 1.0)
        op = f.get('eye_open', 1.0)
        eh = 0.092 * op
        drop_in = {'angry': 0.030, 'grit': 0.032, 'shout': 0.012, 'smirk': 0.006}.get(expr, 0.0)
        drop_out = 0.0
        if expr == 'smirk' and side < 0:
            eh *= 0.78
            drop_out = 0.01
        if expr == 'grin':
            eh *= 0.88  # happy eyes squint a little from the cheeks
        s = side
        inner = (s * (ex - ew), ez - 0.006)
        outer = (s * (ex + ew), ez + 0.010)
        upper = [inner, (s * (ex - ew * 0.62), ez + eh * 0.70 - drop_in), (s * (ex - ew * 0.12), ez + eh - drop_in * 0.45),
                 (s * (ex + ew * 0.52), ez + eh * 0.82 - drop_out), outer]
        lower = [inner, (s * (ex - ew * 0.35), ez - eh * 0.58), (s * (ex + ew * 0.35), ez - eh * 0.62 + (0.012 if expr == 'grin' else 0)),
                 outer]
        almond = _cr(upper) + _cr(lower)[::-1]
        self.poly(almond, (0.98, 0.96, 0.93), smooth=False)
        look = f.get('look', (0.0, 0.0))
        ic = (s * ex + look[0] * ew * 0.35 - s * 0.006, ez - eh * 0.02 + look[1] * eh * 0.3)
        r = eh * 0.98 + 0.012
        iris = f.get('iris_col', (0.30, 0.18, 0.10))

        def draw_iris(lay):
            self.ellipse(ic, r, r, iris, layer=lay)
            self.ellipse((ic[0], ic[1] + r * 0.18), r * 0.98, r * 0.42, tuple(c * 0.55 for c in iris), layer=lay)
            self.ellipse(ic, r * 0.46, r * 0.46, (0.05, 0.03, 0.06), layer=lay)
            self.ellipse((ic[0] + r * 0.32, ic[1] + r * 0.34), r * 0.24, r * 0.24, (1, 1, 1), layer=lay)
            # top lash shadow on the eyeball
            self.stroke(_cr(upper), lambda t: eh * 0.55, (158, 140, 160, 255), layer=lay)
        self.clipped(almond, draw_iris)
        # heavy upper lash line with a flick at the outer corner
        self.stroke(upper, lambda t: 0.014 + 0.032 * ss(0.10, 0.70, t) - 0.012 * ss(0.92, 1.0, t))
        self.stroke([outer, (s * (ex + ew + 0.030), ez + 0.034), (s * (ex + ew + 0.058), ez + 0.058)],
                    lambda t: 0.028 * (1 - t) + 0.003)
        # lower lid (outer two thirds) and crease
        lw = _cr(lower)
        n = len(lw)
        self.stroke(lw[int(n * 0.40):], lambda t: 0.003 + 0.007 * math.sin(math.pi * t), (70, 40, 60, 255))
        up = _cr(upper)
        crease = [(x, z + 0.040 + 0.010 * math.sin(math.pi * i / (len(up) - 1))) for i, (x, z) in enumerate(up)]
        self.stroke(crease[int(len(crease) * 0.22):int(len(crease) * 0.86)], lambda t: 0.0025 + 0.006 * math.sin(math.pi * t))
        if f.get('eye_bags'):
            self.stroke([(s * (ex - ew * 0.3), ez - eh - 0.045), (s * (ex + ew * 0.2), ez - eh - 0.065),
                         (s * (ex + ew * 0.8), ez - eh - 0.035)], lambda t: 0.002 + 0.006 * math.sin(math.pi * t))
        # cheekbone accent (the clean modern-comic cheek line)
        if f.get('cheek_line', True):
            self.stroke([(s * (ex + ew * 0.75), ez - 0.17), (s * (ex + ew * 1.10), ez - 0.26), (s * (ex + ew * 1.15), ez - 0.33)],
                        lambda t: 0.002 + 0.008 * math.sin(math.pi * t), (95, 50, 60, 255))
        if 'crow' in f.get('lines', []):
            for k in range(3):
                a = -0.5 + k * 0.45
                self.stroke([(s * (ex + ew + 0.04), ez + 0.0 + a * 0.05),
                             (s * (ex + ew + 0.10), ez + a * 0.11)], lambda t: 0.006 * (1 - t) + 0.001)

    def brow(self, side, f, expr):
        bz = f.get('brow_z', 0.29)
        th = f.get('brow_thick', 1.0)
        col = f.get('brow_col', (0.08, 0.05, 0.06))
        a_in = {'angry': -0.075, 'grit': -0.085, 'shout': -0.06, 'smirk': -0.02, 'grin': 0.015, 'calm': 0.0}.get(expr, 0)
        raise_ = 0.045 if (expr == 'smirk' and side > 0) else 0.0
        if expr == 'grin':
            raise_ = 0.02
        s = side
        pts = [(s * 0.065, bz + a_in + raise_), (s * 0.16, bz + 0.035 + a_in * 0.5 + raise_),
               (s * 0.30, bz + 0.062 + raise_ + a_in * 0.1), (s * 0.45, bz + 0.018 + raise_)]
        self.stroke(pts, lambda t: (0.078 * th) * (1 - t) ** 0.85 + 0.013, col)
        # top edge ink to keep pale brows readable
        if sum(col[:3]) > 1.5:
            self.stroke([(x, z + 0.028 * th * (1 - i / 3)) for i, (x, z) in enumerate(pts)],
                        lambda t: 0.006 * (1 - t) + 0.002, (120, 110, 120, 255))
        if expr in ('angry', 'grit', 'shout') and side > 0:
            for x in (-0.035, 0.035):
                self.stroke([(x, bz - 0.04), (x * 0.6, bz + 0.03)], lambda t: 0.007 * math.sin(math.pi * t) + 0.002)

    def nose(self, f):
        nl, nw = f.get('nose_len', 1.0), f.get('nose_w', 1.0)
        tip = 0.11 - 0.30 * nl
        sd = f.get('shade_side', -1)
        # nostril wings
        for s in (-1, 1):
            self.stroke([(s * 0.115 * nw, tip + 0.055), (s * 0.135 * nw, tip + 0.012), (s * 0.090 * nw, tip - 0.020),
                         (s * 0.040, tip - 0.018)], lambda t: 0.004 + 0.010 * math.sin(math.pi * t))
        # under-tip and the shadow-side bridge line
        self.stroke([(-0.045, tip - 0.040), (0.0, tip - 0.050), (0.045, tip - 0.040)], lambda t: 0.003 + 0.006 * math.sin(math.pi * t))
        self.stroke([(sd * 0.055, 0.17), (sd * 0.075, 0.02), (sd * 0.095, tip + 0.075)],
                    lambda t: 0.002 + 0.008 * math.sin(math.pi * t) ** 0.8)

    def mouth(self, f, expr):
        mz, mw = f.get('mouth_z', -0.42), f.get('mouth_w', 0.25)
        nw = f.get('nose_w', 1.0)
        tip = 0.11 - 0.30 * f.get('nose_len', 1.0)
        lipc = f.get('lip_col', (0.62, 0.30, 0.30))
        inner = (0.28, 0.05, 0.09)
        teeth = (0.99, 0.98, 0.94)

        def smile_lines(sides=(-1, 1), strength=1.0):
            for s in sides:
                self.stroke([(s * 0.16 * nw, tip + 0.03), (s * (mw + 0.06), mz + 0.02), (s * (mw + 0.07), mz - 0.07)],
                            lambda t: (0.002 + 0.010 * math.sin(math.pi * t)) * strength)

        if expr in ('grin', 'shout', 'grit'):
            if expr == 'grin':
                top = [(-mw, mz + 0.035), (-mw * 0.55, mz + 0.010), (0, mz + 0.018), (mw * 0.55, mz + 0.010), (mw, mz + 0.035)]
                bot = [(-mw, mz + 0.035), (-mw * 0.62, mz - 0.075), (0, mz - 0.115), (mw * 0.62, mz - 0.075), (mw, mz + 0.035)]
                tooth_h = 0.050
            elif expr == 'shout':
                mw2 = mw * 0.82
                top = [(-mw2, mz + 0.03), (-mw2 * 0.5, mz + 0.055), (0, mz + 0.06), (mw2 * 0.5, mz + 0.055), (mw2, mz + 0.03)]
                bot = [(-mw2, mz + 0.03), (-mw2 * 0.75, mz - 0.11), (0, mz - 0.20), (mw2 * 0.75, mz - 0.11), (mw2, mz + 0.03)]
                tooth_h = 0.045
            else:  # grit: clenched teeth
                top = [(-mw, mz + 0.000), (-mw * 0.5, mz + 0.022), (0, mz + 0.026), (mw * 0.5, mz + 0.022), (mw, mz + 0.000)]
                bot = [(-mw, mz + 0.000), (-mw * 0.5, mz - 0.060), (0, mz - 0.066), (mw * 0.5, mz - 0.060), (mw, mz + 0.000)]
                tooth_h = 1.0
            shape = _cr(top) + _cr(bot)[::-1]
            self.poly(shape, inner, smooth=False)

            def inside(lay):
                ttop = _cr(top)
                band = ttop + [(x, z - tooth_h) for x, z in ttop[::-1]]
                ImageDraw.Draw(lay).polygon([self.P(*p) for p in band], fill=rgba(teeth))
                if expr == 'shout':
                    bb = _cr(bot)
                    band2 = bb + [(x, z + 0.04) for x, z in bb[::-1]]
                    ImageDraw.Draw(lay).polygon([self.P(*p) for p in band2], fill=rgba(teeth))
                    self.ellipse((0, mz - 0.13), mw * 0.38, 0.045, (0.85, 0.32, 0.38), layer=lay)
                elif expr == 'grin':
                    self.ellipse((0, mz - 0.105), mw * 0.45, 0.035, (0.86, 0.34, 0.40), layer=lay)
                if expr == 'grit':
                    self.stroke([(-mw * 0.95, mz - 0.018), (0, mz - 0.012), (mw * 0.95, mz - 0.018)], lambda t: 0.006,
                                (120, 100, 110, 255), layer=lay)
                    for k in range(-3, 4):
                        x = k * mw * 0.27
                        self.stroke([(x, mz + 0.03), (x * 1.02, mz - 0.07)], lambda t: 0.004, (150, 130, 140, 255),
                                    layer=lay)
            self.clipped(shape, inside)
            self.stroke(top, lambda t: 0.014 + 0.012 * (1 - math.sin(math.pi * t)))
            self.stroke(bot, lambda t: 0.006 + 0.009 * math.sin(math.pi * t))
            low = min(z for _, z in _cr(bot))
            self.stroke([(-mw * 0.38, low - 0.040), (0, low - 0.052), (mw * 0.38, low - 0.040)],
                        lambda t: 0.003 + 0.008 * math.sin(math.pi * t), (95, 45, 55, 255))
            for s in (-1, 1):  # corner ticks
                c = top[0] if s < 0 else top[-1]
                self.stroke([(c[0] + s * 0.005, c[1] + 0.025), (c[0] + s * 0.03, c[1] - 0.01)],
                            lambda t: 0.009 * (1 - t) + 0.002)
            smile_lines(strength=1.2 if expr != 'grit' else 1.4)
        else:
            if expr == 'smirk':
                line = [(-mw * 0.85, mz - 0.002), (-mw * 0.3, mz - 0.012), (mw * 0.35, mz - 0.002), (mw * 0.95, mz + 0.040)]
            elif expr == 'calm':
                line = [(-mw * 0.9, mz + 0.012), (0, mz - 0.010), (mw * 0.9, mz + 0.012)]
            else:  # angry / determined
                line = [(-mw * 0.9, mz - 0.022), (-mw * 0.3, mz - 0.002), (mw * 0.3, mz - 0.002), (mw * 0.9, mz - 0.022)]
            self.stroke(line, lambda t: 0.010 + 0.012 * math.sin(math.pi * t))
            self.poly([(-mw * 0.45, mz - 0.035), (0, mz - 0.025), (mw * 0.45, mz - 0.035), (mw * 0.3, mz - 0.075),
                       (0, mz - 0.085), (-mw * 0.3, mz - 0.075)], lipc)
            self.stroke([(-mw * 0.30, mz - 0.095), (0, mz - 0.102), (mw * 0.30, mz - 0.095)],
                        lambda t: 0.003 + 0.006 * math.sin(math.pi * t), (95, 45, 55, 255))
            if expr == 'smirk':
                self.stroke([(mw * 0.95, mz + 0.06), (mw * 1.05, mz + 0.02)], lambda t: 0.008 * (1 - t) + 0.002)
                smile_lines(sides=(1,), strength=0.9)
            else:
                smile_lines(strength=0.7)
        # chin line + cleft
        self.stroke([(-0.10, mz - 0.27), (0, mz - 0.255), (0.10, mz - 0.27)], lambda t: 0.002 + 0.006 * math.sin(math.pi * t),
                    (95, 50, 60, 255))
        if f.get('chin_cleft'):
            self.stroke([(0, mz - 0.33), (0.003, mz - 0.40)], lambda t: 0.002 + 0.006 * math.sin(math.pi * t))

    def mustache(self, m, f):
        style = m.get('style', 'thick')
        mz = f.get('mouth_z', -0.42)
        tip = 0.11 - 0.30 * f.get('nose_len', 1.0)
        col = m.get('col', (0.06, 0.04, 0.05))
        hi = m.get('hi', tuple(min(1, c + 0.28) for c in col[:3]))
        w = m.get('width', 0.27)
        th = m.get('thick', 1.0)
        for s in (-1, 1):
            if style == 'handlebar':
                up = [(0, tip - 0.028), (s * 0.11, tip - 0.045), (s * 0.22, mz + 0.095 * th), (s * (w + 0.05), mz + 0.12),
                      (s * (w + 0.12), mz + 0.23)]
                lo = [(s * (w + 0.12), mz + 0.23), (s * (w + 0.03), mz + 0.07), (s * 0.21, mz + 0.000), (s * 0.10, mz + 0.012),
                      (0, mz + 0.032)]
            elif style == 'walrus':
                up = [(0, tip - 0.030), (s * 0.11, tip - 0.045), (s * 0.22, mz + 0.05), (s * (w + 0.02), mz - 0.10)]
                lo = [(s * (w + 0.02), mz - 0.10), (s * (w - 0.04), mz - 0.03), (s * 0.15, mz - 0.005), (0, mz + 0.005)]
            elif style == 'trim':
                up = [(0, mz + 0.085), (s * 0.12, mz + 0.080), (s * (w * 0.85), mz + 0.035)]
                lo = [(s * (w * 0.85), mz + 0.035), (s * 0.12, mz + 0.045), (0, mz + 0.052)]
            else:  # thick chevron
                up = [(0, tip - 0.030), (s * 0.12, tip - 0.048), (s * 0.23, mz + 0.05 * th), (s * (w + 0.03), mz - 0.045)]
                lo = [(s * (w + 0.03), mz - 0.045), (s * (w - 0.04), mz + 0.000), (s * 0.12, mz + 0.012), (0, mz + 0.032)]
            shape = _cr(up) + _cr(lo)[1:]
            self.poly(shape, col, smooth=False)
            self.stroke(shape + [shape[0]], lambda t: 0.007, smooth=False)
            # highlight hatches following the hair direction
            for k in range(2):
                a = up[1][0] * (0.6 + 0.3 * k), up[1][1] - 0.012 - 0.012 * k
                b = up[2][0] * (0.95 + 0.05 * k), up[2][1] - 0.008 * k
                self.stroke([a, b], lambda t: 0.009 * math.sin(math.pi * t) + 0.001, rgba(hi, 200))

    def beard(self, b, f):
        style = b.get('style', 'french')
        col = b.get('col', (0.9, 0.9, 0.88))
        mz, mw = f.get('mouth_z', -0.42), f.get('mouth_w', 0.25)
        if style == 'french':
            shape = [(-mw - 0.02, mz + 0.03), (-mw - 0.07, mz - 0.12), (-0.12, mz - 0.40), (0, mz - 0.62),
                     (0.12, mz - 0.40), (mw + 0.07, mz - 0.12), (mw + 0.02, mz + 0.03), (mw * 0.8, mz - 0.10),
                     (0, mz - 0.14), (-mw * 0.8, mz - 0.10)]
            self.poly(shape, col)
            self.stroke(_cr(shape) + [_cr(shape)[0]], lambda t: 0.007, smooth=False)
            for k in range(4):
                x = -0.08 + k * 0.055
                self.stroke([(x, mz - 0.20), (x * 0.6, mz - 0.45 + 0.03 * abs(k - 1.5))],
                            lambda t: 0.006 * math.sin(math.pi * t) + 0.001, (150, 150, 160, 255))
        elif style in ('stubble', 'full'):
            rng = np.random.default_rng(7)
            lay = Image.new('RGBA', self.img.size, (0, 0, 0, 0))
            d = ImageDraw.Draw(lay)
            dens = 1.0 if style == 'stubble' else 2.6
            for _ in range(int(5200 * dens)):
                x = rng.uniform(-0.70, 0.70)
                z = rng.uniform(-1.25, -0.25)
                jaw_top = -0.30 - 0.45 * max(0, 0.50 - abs(x))
                if z > jaw_top:
                    continue
                if abs(x) < mw * 1.05 and mz - 0.12 < z < mz + 0.04:
                    continue  # keep the mouth clear
                if abs(x) < 0.16 and z > mz + 0.04 and z > 0.11 - 0.30 * f.get('nose_len', 1.0) - 0.06:
                    continue
                r = rng.uniform(0.0025, 0.0055) * (1.6 if style == 'full' else 1.0)
                cx, cy = self.P(x, z)
                rr = r * self.k
                d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=rgba(col, 200 if style == 'stubble' else 255))
            self.img.alpha_composite(lay)
            self.d = ImageDraw.Draw(self.img)

    def lines(self, f):
        L = f.get('lines', [])
        if 'forehead' in L:
            for i, z in enumerate((0.47, 0.54, 0.61)):
                self.stroke([(-0.24, z - 0.01), (0, z + 0.012), (0.24, z - 0.01)],
                            lambda t: 0.002 + 0.006 * math.sin(math.pi * t), (95, 50, 60, 255))

    def blush(self, f):
        bc = f.get('blush', (0.92, 0.42, 0.42))
        for s in (-1, 1):
            self.soft((s * 0.36, -0.10), 0.17, 0.09, bc, 46, blur=0.05)
        self.soft((0, 0.11 - 0.30 * f.get('nose_len', 1.0) + 0.01), 0.06, 0.04, bc, 40, blur=0.03)

    # -------------------------------------------------------------- whole face
    def paint(self, face, expr):
        f = dict(face)
        self.blush(f)
        self.lines(f)
        if f.get('beard_2d'):
            self.beard(f['beard_2d'], f)
        for s in (-1, 1):
            self.eye(s, f, expr)
            self.brow(s, f, expr)
        self.nose(f)
        self.mouth(f, expr)
        if f.get('mustache_2d'):
            self.mustache(f['mustache_2d'], f)
        out = self.img.resize((self.W // self.ssf, self.H // self.ssf), Image.LANCZOS)
        return out


def paint_face(face, expr, path, ppu=620):
    fp = FacePainter(ppu)
    im = fp.paint(face, expr)
    im.save(path)
    return path
