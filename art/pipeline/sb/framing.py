"""Camera framing helpers."""
import math
import bpy
from mathutils import Vector


FX_TAGS = ('energy', 'varma', 'link', 'lathi', 'fg_', 'chain', 'trishul')


def body_only(objects):
    return [o for o in objects if not any(t in o.name for t in FX_TAGS)]


def bbox(objects):
    lo = Vector((1e9, 1e9, 1e9))
    hi = Vector((-1e9, -1e9, -1e9))
    for ob in objects:
        for c in ob.bound_box:
            p = ob.matrix_world @ Vector(c)
            lo = Vector((min(lo.x, p.x), min(lo.y, p.y), min(lo.z, p.z)))
            hi = Vector((max(hi.x, p.x), max(hi.y, p.y), max(hi.z, p.z)))
    return lo, hi


def fit_camera(objects, yaw_deg=20.0, pitch_deg=6.0, lens=50.0, margin=1.12, aspect=0.8, center_bias=(0, 0)):
    """Place a camera so the objects' bbox fills the frame. yaw rotates around the fighter
    (0 = straight on from -Y), pitch > 0 looks up from below (heroic angle)."""
    from . import scene
    lo, hi = bbox(body_only(objects))
    c = (lo + hi) / 2 + Vector((center_bias[0], 0, center_bias[1]))
    hgt = hi.z - lo.z
    wid = max(hi.x - lo.x, hi.y - lo.y)
    fov_long = 2 * math.atan(18.0 / lens)
    if aspect < 1:  # portrait: sensor fits height
        fov_v = fov_long
        fov_h = 2 * math.atan(math.tan(fov_v / 2) * aspect)
    else:
        fov_h = fov_long
        fov_v = 2 * math.atan(math.tan(fov_h / 2) / aspect)
    dist = max((hgt / 2 * margin) / math.tan(fov_v / 2), (wid / 2 * margin) / math.tan(fov_h / 2)) + wid * 0.3
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    d = Vector((math.sin(y) * math.cos(p), -math.cos(y) * math.cos(p), -math.sin(p)))
    loc = c + d * dist
    return scene.add_camera(loc, c, lens=lens)
