"""Code sculpting: signed-distance fields on a voxel grid, smooth-blended primitives, noise,
marching cubes -> Blender mesh with per-vertex colours and front-projected UVs.

All coordinates are metres in the space the caller chooses. Fields are float32 arrays on a
Grid; negative = inside.
"""
import math

import numpy as np
from scipy import ndimage


class Grid:
    def __init__(self, lo, hi, step):
        self.lo = np.asarray(lo, np.float32)
        self.hi = np.asarray(hi, np.float32)
        self.step = float(step)
        n = np.ceil((self.hi - self.lo) / step).astype(int) + 1
        self.shape = tuple(int(v) for v in n)
        xs = self.lo[0] + np.arange(n[0], dtype=np.float32) * step
        ys = self.lo[1] + np.arange(n[1], dtype=np.float32) * step
        zs = self.lo[2] + np.arange(n[2], dtype=np.float32) * step
        self.X, self.Y, self.Z = np.meshgrid(xs, ys, zs, indexing='ij')

    # ------------------------------------------------------------------ primitives
    def ellipsoid(self, c, r):
        x = (self.X - c[0]) / r[0]
        y = (self.Y - c[1]) / r[1]
        z = (self.Z - c[2]) / r[2]
        k0 = np.sqrt(x * x + y * y + z * z)
        k1 = np.sqrt((x / r[0]) ** 2 + (y / r[1]) ** 2 + (z / r[2]) ** 2)
        return (k0 * (k0 - 1.0) / np.maximum(k1, 1e-6)).astype(np.float32)

    def ellipsoid_rot(self, c, axes, r):
        """Ellipsoid with orthonormal axes (rows) and radii r along them."""
        a = np.asarray(axes, np.float32)
        px, py, pz = self.X - c[0], self.Y - c[1], self.Z - c[2]
        q = [(px * a[i, 0] + py * a[i, 1] + pz * a[i, 2]) / r[i] for i in range(3)]
        k0 = np.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2)
        k1 = np.sqrt((q[0] / r[0]) ** 2 + (q[1] / r[1]) ** 2 + (q[2] / r[2]) ** 2)
        return (k0 * (k0 - 1.0) / np.maximum(k1, 1e-6)).astype(np.float32)

    def sphere(self, c, r):
        return (np.sqrt((self.X - c[0]) ** 2 + (self.Y - c[1]) ** 2 + (self.Z - c[2]) ** 2) - r).astype(np.float32)

    def capsule(self, a, b, ra, rb=None):
        rb = ra if rb is None else rb
        a = np.asarray(a, np.float32)
        b = np.asarray(b, np.float32)
        ba = b - a
        L2 = float(ba @ ba) + 1e-12
        px, py, pz = self.X - a[0], self.Y - a[1], self.Z - a[2]
        h = np.clip((px * ba[0] + py * ba[1] + pz * ba[2]) / L2, 0, 1)
        dx, dy, dz = px - ba[0] * h, py - ba[1] * h, pz - ba[2] * h
        return (np.sqrt(dx * dx + dy * dy + dz * dz) - (ra + (rb - ra) * h)).astype(np.float32)

    def tube(self, pts, radii, k=0.0):
        """Union of capsules along a polyline (radii per point)."""
        d = None
        for i in range(len(pts) - 1):
            c = self.capsule(pts[i], pts[i + 1], radii[i], radii[i + 1])
            d = c if d is None else (smin(d, c, k) if k > 0 else np.minimum(d, c))
        return d

    def plane(self, n, d):
        """Signed distance to plane n.p = d (positive on the +n side)."""
        n = np.asarray(n, np.float32)
        n = n / np.linalg.norm(n)
        return (self.X * n[0] + self.Y * n[1] + self.Z * n[2] - d).astype(np.float32)

    def noise(self, scale, seed=0, octaves=1):
        """Smooth value noise in [-1, 1] with feature size ~scale (metres)."""
        out = np.zeros(self.shape, np.float32)
        amp, tot = 1.0, 0.0
        for o in range(octaves):
            s = scale / (2 ** o)
            cells = np.maximum(2, (np.array(self.shape) * self.step / s).astype(int) + 3)
            rng = np.random.default_rng(seed + 31 * o)
            lat = rng.uniform(-1, 1, tuple(cells)).astype(np.float32)
            f = [(self.shape[i] - 1) / (cells[i] - 3) for i in range(3)]
            idx = np.meshgrid(*[np.arange(self.shape[i], dtype=np.float32) / f[i] + 1 for i in range(3)], indexing='ij')
            out += amp * ndimage.map_coordinates(lat, idx, order=3, mode='nearest').astype(np.float32)
            tot += amp
            amp *= 0.5
        return out / tot

    def coords(self):
        return self.X, self.Y, self.Z


def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return (b * (1 - h) + a * h - k * h * (1 - h)).astype(np.float32)


def smax(a, b, k):
    return -smin(-a, -b, k)


def sub(a, b, k=0.0):
    """a minus b."""
    return smax(a, -b, k) if k > 0 else np.maximum(a, -b)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


# ----------------------------------------------------------------------------- meshing

def mesh(grid, field, step_skip=1):
    from skimage.measure import marching_cubes
    f = field[::step_skip, ::step_skip, ::step_skip] if step_skip > 1 else field
    if f.min() >= 0 or f.max() <= 0:
        return None, None
    v, faces, _, _ = marching_cubes(f, level=0.0, spacing=(grid.step * step_skip,) * 3)
    v = v + grid.lo
    return v.astype(np.float32), faces[:, ::-1].astype(np.int32)  # flip winding -> outward normals


def to_blender(name, V, F, mat, colors=None, uv_fn=None, pass_index=0, smooth_iters=2, transform=None):
    """Create a Blender object from arrays. colors: (N,3) sRGB per vertex. uv_fn(V)->(N,2)."""
    import bpy
    from mathutils import Matrix
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(V))
    me.vertices.foreach_set('co', V.ravel())
    nl = len(F) * 3
    me.loops.add(nl)
    me.loops.foreach_set('vertex_index', F.ravel())
    me.polygons.add(len(F))
    me.polygons.foreach_set('loop_start', np.arange(0, nl, 3, dtype=np.int32))
    me.update(calc_edges=True)
    me.polygons.foreach_set('use_smooth', np.ones(len(F), bool))
    if colors is not None:
        lin = np.where(colors <= 0.04045, colors / 12.92, ((colors + 0.055) / 1.055) ** 2.4)
        ca = me.color_attributes.new('Col', 'FLOAT_COLOR', 'POINT')
        rgba = np.concatenate([lin, np.ones((len(V), 1), np.float32)], 1).astype(np.float32)
        ca.data.foreach_set('color', rgba.ravel())
    if uv_fn is not None:
        uv = me.uv_layers.new(name='UVMap')
        per_v = uv_fn(V).astype(np.float32)
        uv.data.foreach_set('uv', per_v[F.ravel()].ravel())
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    ob.pass_index = pass_index
    if transform is not None:
        ob.matrix_world = transform
    if smooth_iters:
        m = ob.modifiers.new('smooth', 'SMOOTH')
        m.factor = 0.5
        m.iterations = smooth_iters
    return ob


def sample(grid, field, P):
    """Trilinear sample of a field at points P (N,3)."""
    idx = ((P - grid.lo) / grid.step).T
    return ndimage.map_coordinates(field, idx, order=1, mode='nearest')
