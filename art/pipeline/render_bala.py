"""Bala Simha hero renders (cover look). Usage: python render_bala.py <mode> [expr]
modes: test | hero | turn | heads | ingame"""
import sys, os, math, time
sys.path.insert(0, os.path.dirname(__file__))
import bpy  # noqa
from mathutils import Vector, Matrix
from sb import scene, bala
from sb.geo import MatLib

BUILD = os.path.join(os.path.dirname(__file__), '../build')
OUT = os.path.join(BUILD, 'bala')
os.makedirs(OUT, exist_ok=True)
mode = sys.argv[1] if len(sys.argv) > 1 else 'test'


def cam_at(target, yaw, pitch, dist, lens, shift=(0, 0)):
    for o in list(bpy.data.objects):
        if o.type == 'CAMERA':
            bpy.data.objects.remove(o)
    a, p = math.radians(yaw), math.radians(pitch)
    t = Vector(target)
    loc = t + Vector((math.sin(a) * math.cos(p), -math.cos(a) * math.cos(p), math.sin(p))) * dist
    cam = scene.add_camera(loc, t, lens=lens)
    cam.data.shift_x, cam.data.shift_y = shift
    return cam


def anchors():
    pts = {}
    for name in ('bala_head', 'bala_handL', 'bala_watchface', 'bala_lens1', 'bala_beard', 'bala_hair', 'bala_dhoti',
                 'bala_sole1', 'bala_shirt', 'bala_pocket', 'bala_chain'):
        o = bpy.data.objects.get(name)
        if o:
            bb = [o.matrix_world @ Vector(c) for c in o.bound_box]
            pts[name] = sum(bb, Vector()) / 8
    return scene.project(pts)


t0 = time.time()
if mode in ('test', 'hero', 'turn'):
    W, H = (1500, 2000) if mode != 'test' else (900, 1200)
    scene.reset(W, H, samples=16, ao_distance=0.15)
    ml = MatLib()
    obs, M = bala.build(ml, BUILD, 'smirk', full=True, turn=8, tilt=4)
    print('built', round(time.time() - t0, 1), 's', sum(len(o.data.vertices) for o in obs if o.type == 'MESH'))
    scene.add_sun((-0.50, 0.62, -0.60), 3.0)        # warm key from camera-left, high
    if mode == 'test':
        cam_at((0, 0, 0.90), 22, 4, 4.6, 50)
        scene.render(f'{OUT}/test_full.exr', ml, meta=dict(anchors=anchors()))
        cam_at((0, -0.02, 1.36), 24, -6, 1.75, 50)
        scene.render(f'{OUT}/test_bust.exr', ml, meta=dict(anchors=anchors()))
    elif mode == 'hero':
        cam_at((0.0, -0.02, 1.33), 26, -9, 1.95, 50, shift=(0.0, 0.02))
        scene.render(f'{OUT}/hero.exr', ml, meta=dict(anchors=anchors()))
    else:
        for yaw in (0, 35, 90, 180):
            cam_at((0, 0, 0.90), yaw, 2, 5.0, 50)
            scene.render(f'{OUT}/turn_{yaw}.exr', ml, meta=dict(anchors=anchors()))
elif mode == 'ingame':
    from sb import street
    scene.reset(3510, 1620, samples=16, ao_distance=0.3)
    ml = MatLib()
    street.street(ml, BUILD)
    obs, M = bala.build(ml, BUILD, 'smirk', full=True, turn=-6, tilt=4)
    R = Matrix.Translation((0.35, 1.2, 0.0)) @ Matrix.Rotation(math.radians(-22), 4, 'Z')
    for o in obs:
        o.matrix_world = R @ o.matrix_world
    scene.add_sun((0.45, 0.55, -0.70), 3.0)
    scene.add_camera((-0.6, -4.6, 1.05), (0.2, 4.0, 1.62), lens=24)
    an = anchors()
    an['vanish'] = scene.project(dict(v=Vector((0, 150, 2.0))))['v']
    scene.render(f'{OUT}/ingame.exr', ml, meta=dict(anchors=an))
elif mode == 'heads':
    for expr, yaw, pitch in (('smirk', 0, 0), ('smirk', 35, 2), ('smirk', 90, 0), ('glare', 18, 0), ('roar', 20, -4),
                             ('laugh', -20, 0)):
        scene.reset(900, 1000, samples=16, ao_distance=0.08)
        ml = MatLib()
        obs = bala.build_head(ml, BUILD, expr, M=Matrix.Identity(4))
        scene.add_sun((-0.50, 0.62, -0.60), 3.0)
        cam_at((0, -0.02, 0.0), yaw, pitch, 1.05, 60)
        scene.render(f'{OUT}/head_{expr}_{yaw}.exr', ml)
print('done', round(time.time() - t0, 1))
