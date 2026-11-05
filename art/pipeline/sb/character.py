"""Procedural stylized fighter: rig -> lofted body -> clothes -> head -> props.

Coordinates: Blender Z-up. In character-local space the fighter faces -Y, their LEFT side
is +X and RIGHT side is -X. Pose directions are given in that local space and the whole
figure is rotated by `yaw` (degrees) and placed at `root`.
Lengths are fractions of the character height H so one pose fits every body type.
"""
import math
from mathutils import Vector, Matrix, Quaternion

from .geo import loft, blob, make_object, lerp, smoothstep, look_matrix, basis
from .head import Head


def N(d):
    return Vector(d).normalized()


def _profile(ts, vals):
    def f(t):
        if t <= ts[0]:
            return vals[0]
        for i in range(len(ts) - 1):
            if t <= ts[i + 1]:
                u = (t - ts[i]) / max(ts[i + 1] - ts[i], 1e-9)
                u = u * u * (3 - 2 * u)
                a, b = vals[i], vals[i + 1]
                if isinstance(a, tuple):
                    return tuple(lerp(x, y, u) for x, y in zip(a, b))
                return lerp(a, b, u)
        return vals[-1]
    return f


def _ctrl_ts(points):
    d = [0.0]
    for a, b in zip(points, points[1:]):
        d.append(d[-1] + (Vector(b) - Vector(a)).length)
    return [x / d[-1] for x in d]


class Character:
    def __init__(self, spec, pose, matlib, index=1, root=(0, 0, 0)):
        self.spec = spec
        self.pose = pose
        self.ml = matlib
        self.index = index
        self.root = Vector(root)
        self.objects = []
        self.H = spec['H']
        self.cid = spec['id']

    # ------------------------------------------------------------------ materials
    def nm(self, part):
        return f"{self.cid}_{part}"

    def mat(self, key):
        """Resolve a material key through the spec palette (char-specific colors)."""
        pal = self.spec['palette']
        base = {
            'ink': ((0.02, 0.02, 0.04), dict(ink=True)),
            'eyewhite': ((0.95, 0.93, 0.88), dict(flat=True)),
            'iris': (pal.get('iris', (0.10, 0.06, 0.03)), dict(flat=True)),
            'mouth': ((0.18, 0.03, 0.04), dict(flat=True)),
            'lip': (pal.get('lip', tuple(c * 0.72 for c in pal['skin'])), {}),
            'brow': (pal.get('brow', pal.get('hair', (0.03, 0.03, 0.03))), dict(hatch=True)),
            'hair': (pal.get('hair', (0.03, 0.03, 0.035)), dict(hatch=True, spec=0.5)),
            'beard': (pal.get('beard', pal.get('hair', (0.03, 0.03, 0.03))), dict(stipple=True)),
            'gold': ((0.95, 0.70, 0.18), dict(spec=1.0, metal=True)),
            'steel': ((0.62, 0.66, 0.72), dict(spec=1.0, metal=True)),
            'frame': (pal.get('frame', (0.85, 0.65, 0.2)), dict(spec=1.0, metal=True)),
            'lens': (pal.get('lens', (0.04, 0.05, 0.07)), dict(spec=1.0, glass=True)),
        }
        if key in base:
            col, props = base[key]
        else:
            entry = pal[key]
            if isinstance(entry, dict):
                col = entry['color']
                props = {k: val for k, val in entry.items() if k != 'color'}
            else:
                col, props = entry, {}
        emit = props.pop('emit', 0.0) if 'emit' in props else 0.0
        if key == 'skin':
            props = dict(props, skin=True)
        return self.ml.get(f"{self.cid}.{key}", col, emit=emit, char=self.cid, **props)

    def track(self, ob):
        ob.pass_index = self.index
        self.objects.append(ob)
        return ob

    # ------------------------------------------------------------------ rig
    def solve(self):
        H, sp, po = self.H, self.spec, self.pose
        hs = 0.081 * sp.get('head_scale', 1.0)
        L = dict(spine=0.30 * H, neck=0.045 * H, head=1.12 * hs * H, sw=0.13 * H * sp.get('shoulders', 1.0),
                 hw=0.058 * H * sp.get('hips', 1.0), ua=0.165 * H, fa=0.15 * H, th=0.245 * H,
                 sh=0.24 * H, ankle=0.045 * H)
        self.L = L
        J = {}
        P = Vector((0, 0, 0))
        spine = N(po.get('spine', (0, 0.03, 1)))
        J['pelvis'] = P
        J['chest'] = P + spine * L['spine']
        neck = N(po.get('neck', spine * 1.0 + Vector((0, -0.15, 0))))
        J['neck'] = J['chest'] + neck * L['neck']
        hd = N(po.get('head_dir', neck))
        J['head'] = J['neck'] + hd * L['head']
        lat_p = Vector((1, 0, 0))
        lat_p = (lat_p - spine * lat_p.dot(spine)).normalized()
        tw = math.radians(po.get('twist', 0.0))
        lat_c = Quaternion(spine, tw) @ lat_p
        self.lat_p, self.lat_c, self.spine_dir, self.neck_dir = lat_p, lat_c, spine, neck
        for s, side in (('L', 1), ('R', -1)):
            S = J['chest'] + lat_c * (side * L['sw']) - spine * (0.02 * H)
            J['sh' + s] = S
            a = po.get('arm' + s, {})
            up = N(a.get('upper', (side * 0.25, 0.0, -1)))
            fo = N(a.get('fore', (side * 0.1, -0.15, -1)))
            J['el' + s] = S + up * L['ua']
            J['wr' + s] = J['el' + s] + fo * L['fa']
            hp = J['pelvis'] + lat_p * (side * L['hw']) - spine * (0.04 * H)
            J['hip' + s] = hp
            g = po.get('leg' + s, {})
            th = N(g.get('thigh', (side * 0.08, 0.0, -1)))
            sh = N(g.get('shin', (side * 0.04, 0.03, -1)))
            J['kn' + s] = hp + th * L['th']
            J['an' + s] = J['kn' + s] + sh * L['sh']
        # world transform
        yaw = math.radians(po.get('yaw', 0.0))
        R = Matrix.Rotation(yaw, 4, 'Z')
        self.R = R
        self.R3 = R.to_3x3()
        J = {k: R @ val for k, val in J.items()}
        self.lat_p = self.R3 @ self.lat_p
        self.lat_c = self.R3 @ self.lat_c
        self.spine_dir = self.R3 @ self.spine_dir
        self.neck_dir = self.R3 @ self.neck_dir
        self.fwd = self.R3 @ Vector((0, -1, 0))
        if po.get('ground', True):
            low = min(J['anL'].z, J['anR'].z) - L['ankle']
            lift = -low + po.get('air', 0.0) * H
        else:
            lift = 0.0
        off = self.root + Vector((0, 0, lift))
        self.J = {k: val + off for k, val in J.items()}
        return self.J

    def local_dir(self, d):
        return (self.R3 @ Vector(d)).normalized()

    # ------------------------------------------------------------------ body
    def build(self):
        self.solve()
        self.torso()
        for s in ('L', 'R'):
            self.arm(s)
            self.leg(s)
            self.hand(s)
            self.foot(s)
        o = self.spec['outfit']
        if o.get('jacket'):
            self.jacket(o['jacket'])
        if o.get('skirt'):
            self.skirt(o['skirt'])
        if o.get('belt'):
            self.belt(o['belt'])
        if o.get('collar', True):
            self.collar(o)
        self.make_head()
        from . import props as P
        for pr in self.spec.get('props', []) + self.pose.get('props', []):
            getattr(P, pr['type'])(self, **{k: val for k, val in pr.items() if k != 'type'})
        for ob in self.objects:
            ob.pass_index = self.index
        return self.objects

    def torso(self):
        H, J, sp = self.H, self.J, self.spec
        b = sp.get('build', 1.0)
        belly = sp.get('belly', 0.0)
        chestb = sp.get('chest', 1.0)
        P, C, Nk, Hd = J['pelvis'], J['chest'], J['neck'], J['head']
        sv = C - P
        pts = [P - self.spine_dir * 0.075 * H, P, P + sv * 0.40, P + sv * 0.72, C,
               C + self.neck_dir * 0.03 * H, Nk, Hd - self.neck_dir * 0.035 * H]
        ts = _ctrl_ts(pts)
        neckr = 0.037 * sp.get('neck_w', 1.0)
        rad = _profile(ts, [(0.080 * b, 0.062), (0.100 * b * sp.get('hips', 1), 0.072),
                            (0.084 * b * (1 + belly * 0.25), 0.064 * (1 + belly * 0.5)),
                            (0.120 * b * chestb, 0.080 * chestb), (0.126 * b * sp.get('shoulders', 1), 0.068),
                            (0.078 * b, 0.052), (neckr, neckr), (neckr * 0.95, neckr * 0.95)])
        rad_H = lambda t: tuple(r * H for r in rad(t))
        o = sp['outfit']
        top_t = _profile([0, 1], [0, 1])
        t_belt = ts[1] + (ts[2] - ts[1]) * o.get('waist', 0.55)
        t_neck = ts[5] + (ts[6] - ts[5]) * 0.4
        vneck = o.get('vneck', 0.0)
        fwd = self.fwd
        mats = [self.mat('bottom'), self.mat('top'), self.mat('skin')]
        if o.get('top_trim'):
            mats.append(self.mat('top_trim'))

        def mat_fn(t, a, d):
            if t < t_belt:
                return 0
            if t > t_neck:
                return 2
            if vneck and t > ts[3] + (ts[4] - ts[3]) * (1 - vneck) and d.dot(fwd) > math.cos(
                    math.radians(18 + 30 * smoothstep(ts[3], t_neck, t))):
                return 2
            if o.get('top_trim') and t > ts[4] - 0.03 and t < t_neck:
                return 3
            return 1

        cloth = 0.006 * H

        def offset(t, a, d):
            f = d.dot(fwd)
            m = 1.0
            if belly:
                m += belly * 0.35 * smoothstep(0.0, 1.0, f) * math.exp(-((t - ts[2]) / 0.12) ** 2)
            m += 0.10 * chestb * max(0.0, f) ** 2 * math.exp(-((t - ts[3]) / 0.08) ** 2) * b
            m += 0.08 * max(0.0, -f) ** 2 * math.exp(-((t - ts[0] - 0.04) / 0.08) ** 2)
            if t > t_belt:  # blouse folds just above the waistline
                m += 0.035 * math.exp(-((t - t_belt - 0.05) / 0.05) ** 2) * max(0.0, math.sin(a * 9 + 0.7))
            return m

        ob = loft(self.nm('torso'), pts, lambda t: (rad_H(t)[0] + cloth * (t < t_neck), rad_H(t)[1] + cloth * (t < t_neck)),
                  ring=20, samples=6, side_hint=self.lat_p, mats=mats, mat_fn=mat_fn,
                  twist=math.radians(self.pose.get('twist', 0.0)), offset_fn=offset,
                  caps=(True, False))
        self.track(ob)
        self.torso_pts, self.torso_ts, self.torso_rad = pts, ts, rad_H

    def arm(self, s):
        H, J, sp = self.H, self.J, self.spec
        b = sp.get('build', 1.0) * sp.get('arms', 1.0)
        S, E, Wr = J['sh' + s], J['el' + s], J['wr' + s]
        side = 1 if s == 'L' else -1
        inner = S - self.lat_c * (side * 0.035 * H) + self.spine_dir * 0.01 * H
        pts = [inner, S, S.lerp(E, 0.5), E, E.lerp(Wr, 0.35), Wr]
        ts = _ctrl_ts(pts)
        rad = _profile(ts, [0.050 * b, 0.055 * b, 0.045 * b, 0.033 * b, 0.039 * b, 0.025 * b])
        o = sp['outfit']
        sl = o.get('sleeve', 1.0)
        a_over = self.pose.get('arm' + s, {}).get('sleeve')
        if a_over is not None:
            sl = a_over
        if s == 'R' and o.get('steel_arm'):
            sl = min(sl, 0.5)
        sleeve_t = ts[1] + (1 - ts[1]) * sl
        cloth = 0.007 * H
        cuff = o.get('cuff', 0.0) * H
        steel = s == 'R' and o.get('steel_arm')
        mats = [self.mat('skin'), self.mat('sleeve' if 'sleeve' in sp['palette'] else 'top')]
        if steel:
            mats.append(self.mat('steel'))
            mats.append(self.mat('steel_dark'))

        def radius(t):
            r = rad(t) * H
            if t < sleeve_t:
                r += cloth + (cuff if t > sleeve_t - 0.06 else 0)
            if steel and t >= sleeve_t:
                r += 0.008 * H
            return r

        def mat_fn(t, a, d):
            if t < sleeve_t:
                return 1
            if steel:
                return 2 if int((t - sleeve_t) * 28) % 2 == 0 else 3
            return 0

        te = ts[3]

        def offset(t, a, d):
            if steel and t >= sleeve_t:
                return 1.0 + 0.10 * (int((t - sleeve_t) * 28) % 2 == 0)
            m = 1.0
            if t < sleeve_t:  # cloth bunching at the elbow and cuff
                m += 0.07 * math.exp(-((t - te) / 0.07) ** 2) * max(0.0, math.sin((t - te) * 90 + a * 1.5))
                m += 0.05 * math.exp(-((t - sleeve_t + 0.03) / 0.05) ** 2) * max(0.0, math.sin(a * 3 + t * 40))
            return m

        ob = loft(self.nm('arm' + s), pts, radius, ring=16, samples=10, side_hint=self.spine_dir,
                  mats=mats, mat_fn=mat_fn, offset_fn=offset, caps=(True, True))
        self.track(ob)

    def leg(self, s):
        H, J, sp = self.H, self.J, self.spec
        b = sp.get('build', 1.0) * sp.get('legs', 1.0)
        Hp, K, A = J['hip' + s], J['kn' + s], J['an' + s]
        pts = [Hp + self.spine_dir * 0.06 * H, Hp, Hp.lerp(K, 0.45), K, K.lerp(A, 0.3), A]
        ts = _ctrl_ts(pts)
        rad = _profile(ts, [0.068 * b, 0.071 * b, 0.060 * b, 0.043 * b, 0.048 * b, 0.029 * b])
        o = sp['outfit']
        loose = o.get('loose_pants', 0.0)
        cloth = (0.008 + loose * 0.03) * H
        hem = o.get('pant_len', 1.0)
        hem_t = ts[1] + (1 - ts[1]) * hem
        mats = [self.mat('legs' if 'legs' in sp['palette'] else 'bottom'), self.mat('skin')]

        def radius(t):
            r = rad(t) * H + (cloth * (0.6 + 0.4 * t) if t <= hem_t else 0)
            return r

        tk = ts[3]

        def offset(t, a, d):
            if t > hem_t:
                return 1.0
            m = 1.0 + 0.06 * math.exp(-((t - tk) / 0.08) ** 2) * max(0.0, math.sin((t - tk) * 70 + a * 2))
            m += 0.06 * math.exp(-((t - hem_t + 0.05) / 0.06) ** 2) * max(0.0, math.sin(t * 60 + a * 2.5))
            m += 0.03 * math.exp(-((t - ts[2]) / 0.1) ** 2) * max(0.0, math.sin(a * 2 + t * 25))
            return m

        ob = loft(self.nm('leg' + s), pts, radius, ring=16, samples=10, side_hint=self.lat_p,
                  mats=mats, mat_fn=lambda t, a, d: 0 if t <= hem_t else 1, caps=(True, True), offset_fn=offset)
        self.track(ob)

    # ------------------------------------------------------------------ hands
    def hand(self, s):
        H, J = self.H, self.J
        a = self.pose.get('arm' + s, {})
        kind = a.get('hand', 'fist')
        side = 1 if s == 'L' else -1
        Wr, E = J['wr' + s], J['el' + s]
        dirv = N(self.local_dir(a['hand_dir'])) if 'hand_dir' in a else (Wr - E).normalized()
        palm = self.local_dir(a.get('palm', (0, 0, -1) if kind != 'fist' else (-side, 0, 0)))
        # hand frame: z along fingers, y = palm normal, x = y cross z
        z = dirv
        y = (palm - z * palm.dot(z)).normalized()
        if y.length < 0.5:
            y = z.orthogonal().normalized()
        x = y.cross(z)
        thumb = -side  # thumb on -x for left hand with this construction
        M = Matrix((x, y, z)).transposed().to_4x4()
        M.translation = Wr
        u = 0.1 * H  # hand length unit
        gm = self.spec['outfit'].get('gloves')
        mat = self.mat('gloves') if gm else self.mat('skin')
        if s == 'R' and self.spec['outfit'].get('steel_arm'):
            mat = self.mat('steel')
        W = lambda p: M @ Vector(p)

        def palm_blob(name, c, size, deform=None):
            loc = Matrix.Translation(Vector(c)) @ Matrix.Diagonal((size[0], size[1], size[2], 1))
            return self.track(blob(self.nm(name), deform or (lambda p: p), segments=16, rings=10,
                                   mats=[mat], matrix=M @ loc, subsurf=1))

        def finger(name, pts, r):
            self.track(loft(self.nm(name), [W(p) for p in pts], r, ring=8, samples=4, mats=[mat],
                            side_hint=x, subsurf=1))

        if kind in ('fist', 'grip'):
            def fd(p):
                q = p.copy()
                if q.z > 0.3:
                    q.y *= 0.9
                q.z += 0.15 * max(0, q.y) * q.z
                return q
            palm_blob(s + '_fist', (0, -0.03 * u, 0.42 * u), (0.30 * u, 0.30 * u, 0.40 * u), fd)
            for i in range(4):
                xx = (i - 1.5) * 0.15 * u
                c = W((xx, -0.12 * u, 0.72 * u))
                self.track(blob(self.nm(f'{s}_knuckle{i}'), lambda p: p, segments=10, rings=6, mats=[mat],
                                matrix=Matrix.Translation(c) @ Matrix.Diagonal((0.08 * u, 0.08 * u, 0.08 * u, 1)),
                                subsurf=1))
            finger(s + '_thumb', [(thumb * 0.22 * u, -0.05 * u, 0.2 * u), (thumb * 0.20 * u, -0.27 * u, 0.45 * u),
                                  (thumb * 0.02 * u, -0.30 * u, 0.55 * u)], 0.075 * u)
        else:
            palm_blob(s + '_palm', (0, 0, 0.38 * u), (0.30 * u, 0.13 * u, 0.40 * u))
            ext = {'open': [1, 1, 1, 1], 'point': [1, 0, 0, 0], 'two': [1, 1, 0, 0], 'claw': [0.5] * 4,
                   'gun': [1, 0, 0, 0], 'wave': [1, 1, 1, 1]}[kind]
            spread = 0.12 if kind in ('open', 'wave') else 0.03
            for i in range(4):
                fx = thumb * (0.22 - i * 0.145) * u
                e = ext[i]
                base = Vector((fx, 0, 0.72 * u))
                ln = (0.55 - abs(i - 1.2) * 0.07) * u
                sp_ = (fx / u) * spread * 3
                if e >= 1:
                    pts = [base, base + Vector((sp_ * u, -0.05 * u, ln * 0.5)), base + Vector((sp_ * 1.8 * u, -0.10 * u, ln))]
                elif e > 0:
                    pts = [base, base + Vector((0, -0.15 * u, ln * 0.4)), base + Vector((0, -0.30 * u, ln * 0.2))]
                else:
                    pts = [base, base + Vector((0, -0.18 * u, 0.12 * u)), base + Vector((0, -0.20 * u, -0.08 * u))]
                finger(f'{s}_f{i}', pts, 0.065 * u)
            if kind == 'gun':
                tp = [(thumb * 0.25 * u, -0.05 * u, 0.3 * u), (thumb * 0.42 * u, -0.10 * u, 0.55 * u),
                      (thumb * 0.40 * u, -0.16 * u, 0.75 * u)]
            else:
                tp = [(thumb * 0.25 * u, -0.05 * u, 0.25 * u), (thumb * 0.45 * u, -0.12 * u, 0.42 * u),
                      (thumb * 0.52 * u, -0.18 * u, 0.62 * u)]
            finger(s + '_thumb', tp, 0.075 * u)
        self.hand_frames = getattr(self, 'hand_frames', {})
        self.hand_frames[s] = M

    # ------------------------------------------------------------------ feet
    def foot(self, s):
        H, J = self.H, self.J
        g = self.pose.get('leg' + s, {})
        side = 1 if s == 'L' else -1
        A, K = J['an' + s], J['kn' + s]
        fdir = self.local_dir(g.get('foot', (side * 0.15, -1, 0)))
        shin = (A - K).normalized()
        up = -shin if g.get('foot_follow_shin') else Vector((0, 0, 1))
        fdir = (fdir - up * fdir.dot(up)).normalized()
        x = fdir.cross(up).normalized()
        M = Matrix((x, fdir, up)).transposed().to_4x4()
        M.translation = A
        o = self.spec['outfit']
        kind = o.get('shoes', 'sneaker')
        u = H
        if kind == 'sandal':
            def fdf(p):
                q = Vector((p.x * 0.042 * u, (p.y * 0.5 + 0.45) * 0.12 * u * 1.0, (p.z * 0.5 + 0.1) * 0.05 * u))
                if p.y > 0.2:
                    q.z *= 0.6
                return q - Vector((0, 0, 0.032 * u))
            self.track(blob(self.nm('foot' + s), fdf, segments=16, rings=10, mats=[self.mat('skin')], matrix=M))
            def sole(p):
                return Vector((p.x * 0.05 * u, (p.y * 0.5 + 0.43) * 0.13 * u, p.z * 0.006 * u - 0.044 * u))
            self.track(blob(self.nm('sole' + s), sole, segments=16, rings=6, mats=[self.mat('shoe')], matrix=M))
            self.track(loft(self.nm('strap' + s), [M @ Vector((-0.04 * u, 0.05 * u, -0.035 * u)),
                                                   M @ Vector((0, 0.06 * u, -0.012 * u)),
                                                   M @ Vector((0.04 * u, 0.05 * u, -0.035 * u))],
                            0.009 * u, ring=6, samples=3, mats=[self.mat('shoe')], side_hint=fdir))
            return
        h = {'sneaker': 0.075, 'boot': 0.11, 'formal': 0.06}.get(kind, 0.07)

        def shoe(p):
            q = Vector((p.x * 0.043 * u, (p.y * 0.5 + 0.38) * 0.155 * u, (p.z * 0.5 + 0.2) * h * u))
            if p.y > 0.0:
                q.z = lerp(q.z, min(q.z, 0.035 * u), smoothstep(0.0, 0.8, p.y))
            q.x *= 1.0 - 0.3 * smoothstep(0.5, 1.0, p.y)
            return q - Vector((0, 0, 0.042 * u))
        self.track(blob(self.nm('shoe' + s), shoe, segments=20, rings=12, mats=[self.mat('shoe')], matrix=M))
        if kind in ('sneaker', 'boot'):
            def sole(p):
                q = Vector((p.x * 0.054 * u, (p.y * 0.5 + 0.36) * 0.158 * u, p.z * 0.012 * u - 0.046 * u))
                return q
            self.track(blob(self.nm('sole' + s), sole, segments=20, rings=8, mats=[self.mat('sole')], matrix=M))
        if kind == 'boot':
            self.track(loft(self.nm('bootshaft' + s), [A - shin * 0.0, A - shin * 0.09 * u], 0.040 * u, ring=12,
                            samples=2, mats=[self.mat('shoe')], side_hint=x))

    # ------------------------------------------------------------------ clothing extras
    def jacket(self, jk):
        H = self.H
        pts, ts, rad = self.torso_pts, self.torso_ts, self.torso_rad
        t0 = ts[1] + (ts[2] - ts[1]) * jk.get('hem', 0.35)
        t1 = ts[5]
        sub = [p for p, t in zip(pts, ts) if t0 - 0.05 <= t <= t1 + 0.01]
        sub_ts = [t for t in ts if t0 - 0.05 <= t <= t1 + 0.01]
        sub_ts = [(t - sub_ts[0]) / (sub_ts[-1] - sub_ts[0]) for t in sub_ts]
        g = jk.get('gap', 22)
        fwd = self.fwd
        thick = jk.get('thick', 0.018) * H

        def radius(t):
            tt = lerp(sub_ts[0], 1, t)
            tt_full = lerp(t0, t1, t)
            r = rad(tt_full)
            return (r[0] + thick, r[1] + thick * 1.2)

        def open_fn(t, a, d):
            if not jk.get('open', True):
                return False
            gap = g + 14 * t
            return d.dot(fwd) > math.cos(math.radians(gap))

        mats = [self.mat('jacket'), self.mat('jacket_trim') if 'jacket_trim' in self.spec['palette'] else self.mat('jacket')]

        def mat_fn(t, a, d):
            return 1 if t < 0.12 else 0
        ob = loft(self.nm('jacket'), sub, radius, ring=24, samples=6, side_hint=self.lat_p, mats=mats,
                  mat_fn=mat_fn, open_fn=open_fn, caps=(False, False),
                  twist=math.radians(self.pose.get('twist', 0.0)) * 0.9)
        self.track(ob)
        sol = ob.modifiers.new('solid', 'SOLIDIFY')
        sol.thickness = 0.008 * H
        ob.modifiers.move(1, 0)
        # standing collar around the back of the neck + flat lapels on the chest
        if jk.get('collar', True):
            C, Nk = self.J['chest'], self.J['neck']
            nb = C + self.neck_dir * 0.035 * H
            back = -fwd
            arc = []
            for k in range(9):
                th = math.radians(-115 + 230 * k / 8)
                arc.append(nb + (self.lat_c * math.sin(th) + back * math.cos(th)) * 0.062 * H + self.neck_dir * 0.012 * H)
            self.track(loft(self.nm('collar_stand'), arc, lambda t: (0.026 * H, 0.006 * H), ring=8, samples=4,
                            side_hint=self.neck_dir, mats=[self.mat('jacket')]))
            for side in (-1, 1):
                p0 = nb + self.lat_c * (side * 0.055 * H) + fwd * 0.035 * H
                p1 = C + self.lat_c * (side * 0.062 * H) + fwd * 0.095 * H - self.spine_dir * 0.035 * H
                p2 = C + self.lat_c * (side * 0.045 * H) + fwd * 0.10 * H - self.spine_dir * 0.12 * H
                self.track(loft(self.nm(f'lapel{side}'), [p0, p1, p2],
                                lambda t: (0.026 * H * (1 - 0.5 * t), 0.005 * H), ring=8, samples=4,
                                side_hint=self.lat_c, mats=[self.mat('jacket')]))
        if jk.get('tails'):
            self.coat_tails(jk)

    def coat_tails(self, jk):
        """Long coat below the waist: open in front, flowing with `flow` (world vector)."""
        H, J = self.H, self.J
        P = J['pelvis']
        flow = Vector(jk.get('flow', (0, 0.3, 0)))
        flow = self.R3 @ flow
        kL, kR = J['knL'], J['knR']
        length = jk.get('length', 0.30) * H
        down = Vector((0, 0, -1))
        mid_k = (kL + kR) / 2
        legdir = (mid_k - P).normalized()
        pts = [P + self.spine_dir * 0.08 * H, P - self.spine_dir * 0.01 * H,
               P + legdir * length * 0.5 + flow * H * 0.15,
               P + legdir * length + flow * H * 0.4]
        spread = (kL - kR).length

        def radius(t):
            rx = lerp(0.10 * H, max(0.11 * H, spread * 0.55 + 0.07 * H), smoothstep(0, 1, t)) * self.spec.get('build', 1)
            ry = lerp(0.08 * H, 0.10 * H, t)
            return (rx, ry)
        fwd = self.fwd
        ob = loft(self.nm('coattail'), pts, radius, ring=24, samples=6, side_hint=self.lat_p,
                  mats=[self.mat('jacket')], open_fn=lambda t, a, d: d.dot(fwd) > math.cos(math.radians(30 + 25 * t)),
                  caps=(False, False),
                  offset_fn=lambda t, a, d: 1.0 + 0.06 * t * math.sin(a * 5 + 1.0))
        self.track(ob)
        sol = ob.modifiers.new('solid', 'SOLIDIFY')
        sol.thickness = 0.008 * H
        ob.modifiers.move(1, 0)

    def skirt(self, sk):
        """Dhoti / lungi / kurta tail: closed wrap from waist to hem."""
        H, J = self.H, self.J
        P = J['pelvis']
        kL, kR = J['knL'], J['knR']
        aL, aR = J['anL'], J['anR']
        hem = sk.get('length', 0.75)  # 0 = hips, 1 = ankles
        t_knee = 0.5
        flow = self.R3 @ Vector(sk.get('flow', (0, 0.05, 0)))
        lo_mid = lerp((kL + kR) / 2, (aL + aR) / 2, max(0, (hem - t_knee) / (1 - t_knee))) if hem > t_knee else \
            P.lerp((kL + kR) / 2, hem / t_knee)
        top = P + self.spine_dir * sk.get('top', 0.10) * H
        pts = [top, P - self.spine_dir * 0.02 * H, P.lerp(lo_mid, 0.55) + flow * H * 0.1, lo_mid + flow * H * 0.25]
        spread_lo = (lerp(kL, aL, max(0, hem - 0.5) * 2) - lerp(kR, aR, max(0, hem - 0.5) * 2)).length

        def radius(t):
            rx = lerp(0.098 * H * self.spec.get('build', 1) * (1 + self.spec.get('belly', 0) * 0.15),
                      max(0.12 * H, spread_lo * 0.5 + 0.07 * H), smoothstep(0.1, 1, t))
            ry = lerp(0.075 * H * (1 + self.spec.get('belly', 0) * 0.4), max(0.09 * H, rx * 0.65), smoothstep(0.1, 1, t))
            return (rx, ry)
        mats = [self.mat('skirt'), self.mat('skirt_trim') if 'skirt_trim' in self.spec['palette'] else self.mat('skirt')]
        ob = loft(self.nm('skirt'), pts, radius, ring=24, samples=6, side_hint=self.lat_p, mats=mats,
                  mat_fn=lambda t, a, d: 1 if t > 0.93 else 0, caps=(False, False),
                  offset_fn=lambda t, a, d: 1.0 + 0.05 * t * math.sin(a * 7 + 0.5),
                  open_fn=(lambda t, a, d: d.dot(self.fwd) > math.cos(math.radians(14 * t))) if sk.get('slit') else None)
        self.track(ob)
        sol = ob.modifiers.new('solid', 'SOLIDIFY')
        sol.thickness = 0.008 * H
        ob.modifiers.move(1, 0)

    def belt(self, bt):
        H = self.H
        pts, ts, rad = self.torso_pts, self.torso_ts, self.torso_rad
        t = ts[1] + (ts[2] - ts[1]) * self.spec['outfit'].get('waist', 0.55)
        # interpolate the waist point along the torso control polyline
        i = max(k for k in range(len(ts)) if ts[k] <= t)
        u = (t - ts[i]) / max(ts[i + 1] - ts[i], 1e-6)
        c = pts[i].lerp(pts[i + 1], u)
        r = rad(t)
        p0 = c - self.spine_dir * 0.012 * H
        p1 = c + self.spine_dir * 0.012 * H
        self.track(loft(self.nm('belt'), [p0, p1], (r[0] + 0.011 * H, r[1] + 0.011 * H), ring=20, samples=2,
                        side_hint=self.lat_p, mats=[self.mat('belt')], caps=(False, False)))
        if bt.get('buckle', True):
            bc = c + self.fwd * (r[1] + 0.012 * H)
            m = look_matrix(self.fwd, self.spine_dir, bc) @ Matrix.Diagonal((0.022 * H, 0.006 * H, 0.016 * H, 1))
            self.track(blob(self.nm('buckle'), lambda p: p, segments=12, rings=8, mats=[self.mat('gold')], matrix=m))

    def collar(self, o):
        """Spread shirt collar + placket + buttons + chest pocket (ink detail lines)."""
        if not o.get('shirt_collar'):
            return
        H, J = self.H, self.J
        C = J['chest']
        nb = C + self.neck_dir * 0.03 * H
        fwd, up = self.fwd, self.spine_dir
        cmat = self.mat('collar') if 'collar' in self.spec['palette'] else self.mat('top')
        for side in (-1, 1):
            lat = self.lat_c * side
            p0 = nb - fwd * 0.04 * H + lat * 0.02 * H + up * 0.012 * H
            p1 = nb + lat * 0.055 * H + up * 0.004 * H
            p2 = nb + fwd * 0.05 * H + lat * 0.045 * H - up * 0.02 * H
            p3 = nb + fwd * 0.075 * H + lat * 0.02 * H - up * 0.065 * H
            self.track(loft(self.nm(f'collar{side}'), [p0, p1, p2, p3],
                            lambda t: (0.02 * H * (1 - 0.35 * t), 0.004 * H), ring=8, samples=4,
                            side_hint=up, mats=[cmat]))
        if o.get('placket', True) and not o.get('jacket'):
            pts, ts, rad = self.torso_pts, self.torso_ts, self.torso_rad
            t_belt = ts[1] + (ts[2] - ts[1]) * o.get('waist', 0.55)
            line = []
            for k in range(6):
                t = t_belt + (ts[4] - 0.04 - t_belt) * k / 5
                i = max(j for j in range(len(ts)) if ts[j] <= t)
                u = (t - ts[i]) / max(ts[i + 1] - ts[i], 1e-6)
                c = pts[i].lerp(pts[i + 1], u)
                r = rad(t)
                line.append(c + fwd * (r[1] + 0.009 * H) * (1.12 if 0.2 < k / 5 < 0.8 else 1.04))
            self.track(loft(self.nm('placket'), line, 0.0022 * H, ring=6, samples=3, mats=[self.mat('ink')]))
            for k in range(1, 5):
                p = line[k] + fwd * 0.002 * H
                m = look_matrix(fwd, up, p) @ Matrix.Diagonal((0.006 * H, 0.002 * H, 0.006 * H, 1))
                self.track(blob(self.nm(f'button{k}'), lambda q: q, segments=8, rings=4, mats=[self.mat('ink')],
                                matrix=m, subsurf=0))

    # ------------------------------------------------------------------ head
    def make_head(self):
        H, J, po = self.H, self.J, self.pose
        hy, hp, hr = [math.radians(a) for a in po.get('head_rot', (0, 0, 0))]
        fwd = self.local_dir((0, -1, 0))
        up = self.neck_dir.copy()
        Rh = Matrix.Rotation(hy, 3, up) @ Matrix.Rotation(hp, 3, fwd.cross(up).normalized())
        f2 = Rh @ fwd
        u2 = Rh @ up
        u2 = Matrix.Rotation(hr, 3, f2) @ u2
        s = 0.081 * H * self.spec.get('head_scale', 1.0)
        M = look_matrix(f2, u2, J['head']) @ Matrix.Diagonal((s, s, s, 1))
        self.head_matrix = M
        face = dict(self.spec['face'])
        face['expr'] = po.get('expr', face.get('expr', 'angry'))
        if 'look' in po:
            face['look'] = po['look']
        hd = Head(self, M, face, self.index)
        for ob in hd.build():
            self.track(ob)
        self.head = hd
