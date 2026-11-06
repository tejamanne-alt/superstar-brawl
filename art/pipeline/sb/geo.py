"""Geometry helpers for the procedural Superstar Brawl characters and props.

Everything is built from a few primitives:
  * loft()       - sweep elliptical rings along a smooth spline (limbs, torso, hair strands,
                   mustaches, cloth tails ...). Each ring band can carry its own material, which
                   is how clothing seams (sleeves, cuffs, trouser hems) are made.
  * blob()       - deformed UV sphere driven by a Python function (heads, hands, shoes ...).
  * shell()      - offset a region of an existing surface (hair caps, beards, masks, bandages).
All objects get a subdivision modifier and smooth shading so the comic ink pass sees
clean silhouettes.
"""
import math
import bmesh
import bpy
from mathutils import Vector, Matrix

UP = Vector((0, 0, 1))


def v(*a):
    if len(a) == 1:
        return Vector(a[0])
    return Vector(a)


def lerp(a, b, t):
    return a + (b - a) * t


def smoothstep(e0, e1, x):
    if e0 == e1:
        return 0.0 if x < e0 else 1.0
    t = max(0.0, min(1.0, (x - e0) / (e1 - e0)))
    return t * t * (3 - 2 * t)


# ----------------------------------------------------------------------------- splines

def catmull_rom(points, samples=8):
    """Centripetal-ish Catmull-Rom through points; returns dense polyline."""
    pts = [Vector(p) for p in points]
    if len(pts) == 2:
        return [pts[0].lerp(pts[1], i / samples) for i in range(samples + 1)]
    ext = [pts[0] + (pts[0] - pts[1])] + pts + [pts[-1] + (pts[-1] - pts[-2])]
    out = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        for s in range(samples):
            t = s / samples
            t2, t3 = t * t, t * t * t
            q = 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
                       (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
            out.append(q)
    out.append(pts[-1].copy())
    return out


def arc_params(poly):
    d = [0.0]
    for a, b in zip(poly, poly[1:]):
        d.append(d[-1] + (b - a).length)
    L = d[-1] if d[-1] > 0 else 1.0
    return [x / L for x in d], L


# ----------------------------------------------------------------------------- materials

def srgb_to_linear(c):
    return tuple((x / 12.92) if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c)


class MatLib:
    """Material registry. pass_index on each material is read back by the comic shader
    (Material Index pass) to look up per-material ink / halftone / spec settings."""

    def __init__(self):
        self.mats = {}
        self.props = {}
        self.next_id = 1

    def get(self, name, color=(0.8, 0.8, 0.8), emit=0.0, **props):
        if name in self.mats:
            return self.mats[name]
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        nt = m.node_tree
        bsdf = None
        for n in nt.nodes:
            if n.type == 'BSDF_PRINCIPLED':
                bsdf = n
        srgb = tuple(color[:3])
        c = srgb_to_linear(srgb) + (1.0,)
        bsdf.inputs['Base Color'].default_value = c
        bsdf.inputs['Roughness'].default_value = 1.0
        try:
            bsdf.inputs['Specular IOR Level'].default_value = 0.0
        except KeyError:
            pass
        if emit > 0:
            bsdf.inputs['Emission Color'].default_value = c
            bsdf.inputs['Emission Strength'].default_value = emit
            try:
                m.cycles.emission_sampling = 'NONE'  # FX glow must not light the scene
            except AttributeError:
                pass
        m.pass_index = self.next_id
        m.diffuse_color = c
        p = dict(name=name, color=list(srgb), emit=emit)
        p.update(props)
        self.props[self.next_id] = p
        self.next_id += 1
        self.mats[name] = m
        return m

    def table(self):
        return {str(k): v for k, v in self.props.items()}


# ----------------------------------------------------------------------------- mesh building

def make_object(name, verts, faces, mats=None, face_mats=None, subsurf=2, smooth=True,
                collection=None, pass_index=0):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(x) for x in verts], [], faces)
    me.validate(clean_customdata=False)
    me.update()
    ob = bpy.data.objects.new(name, me)
    (collection or bpy.context.scene.collection).objects.link(ob)
    if mats:
        for m in mats:
            me.materials.append(m)
        if face_mats:
            for poly, mi in zip(me.polygons, face_mats):
                poly.material_index = mi
    if smooth:
        for poly in me.polygons:
            poly.use_smooth = True
    if subsurf:
        mod = ob.modifiers.new('subd', 'SUBSURF')
        mod.levels = subsurf
        mod.render_levels = subsurf
    ob.pass_index = pass_index
    return ob


def loft(name, points, radius, ring=14, samples=6, side_hint=None, mats=None, mat_fn=None,
         caps=(True, True), cap_rings=3, open_fn=None, subsurf=2, collection=None,
         pass_index=0, twist=0.0, offset_fn=None, cap_depth=1.0):
    """Sweep rings along a Catmull-Rom spline.

    radius(t) -> float or (rx, ry); rx is along the side axis, ry along the 'front' axis.
    mat_fn(t, angle, dir) -> material slot index (dir = world-space ring direction).
    open_fn(t, angle, dir) -> True to drop the face (used for open jackets / coats).
    offset_fn(t, angle, dir) -> scalar radial multiplier (wrinkles, bellies, clumps).
    """
    poly = catmull_rom(points, samples)
    ts, L = arc_params(poly)
    n = len(poly)
    tangents = []
    for i in range(n):
        a = poly[max(i - 1, 0)]
        b = poly[min(i + 1, n - 1)]
        t = (b - a)
        tangents.append(t.normalized() if t.length > 1e-9 else Vector((0, 0, 1)))
    hint_fn = side_hint if callable(side_hint) else None
    hint = Vector(side_hint) if (side_hint is not None and hint_fn is None) else None
    frames = []
    prev_side = None
    for i, T in enumerate(tangents):
        if hint_fn is not None:
            hint = Vector(hint_fn(ts[i]))
        if hint is not None:
            side = hint - T * hint.dot(T)
            if side.length < 1e-4:
                side = prev_side if prev_side is not None else T.orthogonal()
        else:
            if prev_side is None:
                side = T.orthogonal()
            else:
                side = prev_side - T * prev_side.dot(T)
                if side.length < 1e-6:
                    side = T.orthogonal()
        side.normalize()
        front = T.cross(side).normalized()
        if twist:
            ang = twist * ts[i]
            s2 = side * math.cos(ang) + front * math.sin(ang)
            front = T.cross(s2).normalized()
            side = s2
        frames.append((side, front))
        prev_side = side

    def rad(t):
        r = radius(t) if callable(radius) else radius
        if isinstance(r, (int, float)):
            return (r, r)
        return r

    verts, faces, fm = [], [], []
    rings = []

    def add_ring(center, side, front, rx, ry, t):
        idx = []
        for k in range(ring):
            a = 2 * math.pi * k / ring
            d = side * math.cos(a) + front * math.sin(a)
            m = offset_fn(t, a, d) if offset_fn else 1.0
            p = center + side * (math.cos(a) * rx * m) + front * (math.sin(a) * ry * m)
            idx.append(len(verts))
            verts.append(p)
        rings.append((idx, t, side, front))

    # start cap rings (hemispherical)
    if caps[0]:
        rx, ry = rad(0.0)
        side, front = frames[0]
        for c in range(cap_rings, 0, -1):
            ang = (math.pi / 2) * c / (cap_rings + 1)
            s = math.cos(ang)
            add_ring(poly[0] - tangents[0] * (math.sin(ang) * min(rx, ry) * cap_depth), side, front,
                     max(rx * s, 1e-4), max(ry * s, 1e-4), 0.0)
    for i in range(n):
        rx, ry = rad(ts[i])
        side, front = frames[i]
        add_ring(poly[i], side, front, rx, ry, ts[i])
    if caps[1]:
        rx, ry = rad(1.0)
        side, front = frames[-1]
        for c in range(1, cap_rings + 1):
            ang = (math.pi / 2) * c / (cap_rings + 1)
            s = math.cos(ang)
            add_ring(poly[-1] + tangents[-1] * (math.sin(ang) * min(rx, ry) * cap_depth), side, front,
                     max(rx * s, 1e-4), max(ry * s, 1e-4), 1.0)

    for r in range(len(rings) - 1):
        a, ta, sa, fa = rings[r]
        b, tb, _, _ = rings[r + 1]
        tm = 0.5 * (ta + tb)
        for k in range(ring):
            k2 = (k + 1) % ring
            ang = 2 * math.pi * (k + 0.5) / ring
            d = sa * math.cos(ang) + fa * math.sin(ang)
            if open_fn and open_fn(tm, ang, d):
                continue
            faces.append((a[k], a[k2], b[k2], b[k]))
            fm.append(mat_fn(tm, ang, d) if mat_fn else 0)
    # close caps with a center vertex
    if caps[0] and not open_fn:
        c0 = len(verts)
        verts.append(poly[0] - tangents[0] * min(rad(0.0)) * cap_depth)
        a = rings[0][0]
        for k in range(ring):
            faces.append((c0, a[(k + 1) % ring], a[k]))
            fm.append(mat_fn(0.0, 0, rings[0][2]) if mat_fn else 0)
    if caps[1] and not open_fn:
        c1 = len(verts)
        verts.append(poly[-1] + tangents[-1] * min(rad(1.0)) * cap_depth)
        a = rings[-1][0]
        for k in range(ring):
            faces.append((c1, a[k], a[(k + 1) % ring]))
            fm.append(mat_fn(1.0, 0, rings[-1][2]) if mat_fn else 0)
    return make_object(name, verts, faces, mats, fm, subsurf=subsurf, collection=collection,
                       pass_index=pass_index)


def blob(name, deform, segments=32, rings=20, mats=None, mat_fn=None, matrix=None, subsurf=1,
         collection=None, pass_index=0, keep_fn=None):
    """UV sphere of radius 1 run through deform(Vector)->Vector. mat_fn(p_orig)->slot.
    keep_fn(p_orig)->bool drops faces (used for shells)."""
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=segments, v_segments=rings, radius=1.0)
    bm.verts.ensure_lookup_table()
    orig = [vv.co.copy() for vv in bm.verts]
    verts = [deform(o.copy()) for o in orig]
    faces, fm = [], []
    for f in bm.faces:
        idx = [vv.index for vv in f.verts]
        c = sum((orig[i] for i in idx), Vector()) / len(idx)
        if keep_fn and not keep_fn(c):
            continue
        faces.append(tuple(idx))
        fm.append(mat_fn(c) if mat_fn else 0)
    bm.free()
    if matrix is not None:
        verts = [matrix @ p for p in verts]
    ob = make_object(name, verts, faces, mats, fm, subsurf=subsurf, collection=collection,
                     pass_index=pass_index)
    return ob


def box(name, size, center=(0, 0, 0), bevel=0.0, mats=None, matrix=None, collection=None,
        pass_index=0, subsurf=0):
    sx, sy, sz = size
    cx, cy, cz = center
    vs = []
    for z in (-1, 1):
        for y in (-1, 1):
            for x in (-1, 1):
                vs.append(Vector((cx + x * sx / 2, cy + y * sy / 2, cz + z * sz / 2)))
    fs = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    if matrix is not None:
        vs = [matrix @ p for p in vs]
    ob = make_object(name, vs, fs, mats, [0] * 6, subsurf=subsurf, smooth=False,
                     collection=collection, pass_index=pass_index)
    if bevel > 0:
        b = ob.modifiers.new('bevel', 'BEVEL')
        b.width = bevel
        b.segments = 2
        b.limit_method = 'NONE'
        ob.modifiers.move(len(ob.modifiers) - 1, 0)
        for p in ob.data.polygons:
            p.use_smooth = False
    return ob


def cylinder(name, p0, p1, r, sides=16, mats=None, collection=None, pass_index=0, caps=True):
    p0, p1 = Vector(p0), Vector(p1)
    T = (p1 - p0).normalized()
    s = T.orthogonal().normalized()
    f = T.cross(s)
    vs, fs = [], []
    for end in (p0, p1):
        for k in range(sides):
            a = 2 * math.pi * k / sides
            vs.append(end + s * math.cos(a) * r + f * math.sin(a) * r)
    for k in range(sides):
        k2 = (k + 1) % sides
        fs.append((k, k2, sides + k2, sides + k))
    if caps:
        fs.append(tuple(range(sides))[::-1])
        fs.append(tuple(range(sides, 2 * sides)))
    ob = make_object(name, vs, fs, mats, [0] * len(fs), subsurf=0, smooth=True,
                     collection=collection, pass_index=pass_index)
    if caps:
        ob.data.polygons[-1].use_smooth = False
        ob.data.polygons[-2].use_smooth = False
    return ob


def basis(x_axis, y_axis, origin=(0, 0, 0)):
    """Matrix from two (roughly orthogonal) axes; z = x cross y."""
    x = Vector(x_axis).normalized()
    y = Vector(y_axis)
    y = (y - x * y.dot(x)).normalized()
    z = x.cross(y)
    m = Matrix((x, y, z)).transposed().to_4x4()
    m.translation = Vector(origin)
    return m


def look_matrix(forward, up, origin):
    """Local -Y faces 'forward', +Z ~ up (our head / body convention)."""
    f = Vector(forward).normalized()
    u = Vector(up)
    u = (u - f * u.dot(f)).normalized()
    x = (-f).cross(u)  # local +X
    y = -f
    m = Matrix((x, y, u)).transposed().to_4x4()
    m.translation = Vector(origin)
    return m


def solidify(ob, thickness=0.01):
    m = ob.modifiers.new('solid', 'SOLIDIFY')
    m.thickness = thickness
    m.offset = 0
    return m


def parent_keep(child, parent):
    child.parent = parent
    child.matrix_parent_inverse = parent.matrix_world.inverted()
