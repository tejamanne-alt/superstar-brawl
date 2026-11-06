"""Bala Simha from the supplied FBX files (texture transferred onto the sculpt): hero, turnaround,
head close-ups, face close-up, in-game street shot."""
import sys, os, math, time
sys.path.insert(0, os.path.dirname(__file__))
import bpy  # noqa
from mathutils import Vector, Matrix
from sb import scene, fbxtex
from sb.geo import MatLib

BUILD = os.path.join(os.path.dirname(__file__), '../build')
OUT = os.path.join(BUILD, 'bala')
os.makedirs(OUT, exist_ok=True)
modes = sys.argv[1].split(',') if len(sys.argv) > 1 else ['hero', 'turn', 'heads', 'face', 'ingame']


def cam_at(target, yaw, pitch, dist, lens):
    for o in list(bpy.data.objects):
        if o.type == 'CAMERA':
            bpy.data.objects.remove(o)
    a, p = math.radians(yaw), math.radians(pitch)
    t = Vector(target)
    return scene.add_camera(t + Vector((math.sin(a) * math.cos(p), -math.cos(a) * math.cos(p), math.sin(p))) * dist, t, lens=lens)


def setup(w, h, street=False):
    scene.reset(w, h, samples=16, ao_distance=0.12 if not street else 0.3)
    ml = MatLib()
    if street:
        from sb import street as st
        st.street(ml, BUILD)
    obs, an, lab = fbxtex.load(ml)
    return ml, obs, an


t0 = time.time()
if 'hero' in modes or 'turn' in modes or 'heads' in modes or 'face' in modes:
    ml, obs, an = setup(1500, 2000)
    scene.add_sun((-0.50, 0.62, -0.60), 3.0)
    sc = bpy.context.scene
    if 'hero' in modes:
        cam_at((0.03, 0.0, 1.30), 20, -8, 1.95, 50)
        scene.render(f'{OUT}/fbx_hero.exr', ml, meta=dict(anchors=scene.project(an)))
    if 'turn' in modes:
        for yaw in (0, 35, 90, 180):
            cam_at((0.0, 0.0, 0.90), yaw, 2, 5.3, 50)
            scene.render(f'{OUT}/fbx_turn_{yaw}.exr', ml, meta=dict(anchors=scene.project(an)))
    if 'heads' in modes:
        sc.render.resolution_x, sc.render.resolution_y = 1000, 1000
        for yaw in (0, 35, 90):
            cam_at((0.06, 0.07, 1.60), yaw, 0, 1.25, 85)
            scene.render(f'{OUT}/fbx_head_{yaw}.exr', ml, meta=dict(anchors=scene.project(an)))
    if 'face' in modes:     # reference angle: head turned to his right, camera a touch low
        sc.render.resolution_x, sc.render.resolution_y = 2400, 3000
        cam_at((0.065, 0.06, 1.565), 30, -3, 1.30, 85)
        scene.render(f'{OUT}/fbx_face.exr', ml, meta=dict(anchors=scene.project(an)))
if 'ingame' in modes:
    ml, obs, an = setup(3510, 1620, street=True)
    R = Matrix.Translation((0.2, 0.6, 0.0)) @ Matrix.Rotation(math.radians(-12), 4, 'Z')
    for o in obs:
        o.matrix_world = R @ o.matrix_world
    an = {k: R @ v for k, v in an.items()}
    an['vanish'] = Vector((0, 150, 2.0))
    scene.add_sun((0.45, 0.55, -0.70), 3.0)
    scene.add_camera((-0.5, -4.3, 0.85), (0.15, 4.0, 1.45), lens=24)
    scene.render(f'{OUT}/fbx_ingame.exr', ml, meta=dict(anchors=scene.project(an)))
print('done', round(time.time() - t0, 1))
