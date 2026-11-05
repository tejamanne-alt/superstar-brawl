"""Key art: the full roster on the Stage 1 rooftop, low heroic wide angle."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import bpy  # noqa
from mathutils import Vector
from sb import scene, roster, environment as env
from sb.geo import MatLib
from sb.character import Character
W, H = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (3840, 2160)  # 1.5x supersampled
build = os.path.join(os.path.dirname(__file__), '../build')
scene.reset(W, H, samples=16, ao_distance=0.3)
ml = MatLib()
env.rooftop(ml, build)
env.city(ml)
env.kites(ml)
LAYOUT = [  # id, x, y, yaw, z (back row stands on a riser)
    ('raja', 0.0, -2.5, -4, 0), ('bala', -1.85, -2.0, 18, 0), ('srk', 2.05, -1.9, -14, 0),
    ('hriday', -3.55, -1.5, 30, 0), ('chiru', 3.7, -1.3, -26, 0),
    ('ambar', -1.30, 0.2, 8, 1.2), ('nag', 1.40, 0.3, -12, 1.2), ('venky', -3.55, 0.4, 16, 1.2),
    ('kamal', 3.75, 0.4, -26, 1.2)]
riser = ml.get('env.riser', (0.80, 0.66, 0.56))
from sb.environment import MeshAcc
acc = MeshAcc('riser', [riser, ml.get('env.riser_top', (0.92, 0.80, 0.66))])
acc.box(-5.0, -0.6, 0, 5.0, 1.4, 1.2, 0, top_mi=1)
acc.build(40)
anchors = {}
for i, (cid, x, y, yaw, z) in enumerate(LAYOUT):
    spec = roster.BY_ID[cid]
    pose = dict(spec['pose'], yaw=yaw + spec['pose'].get('yaw', 0))
    ch = Character(spec, pose, ml, i + 1, root=(x, y, z))
    ch.build()
    anchors[cid + '_head'] = ch.J['head']
    anchors[cid + '_feet'] = (ch.J['anL'] + ch.J['anR']) / 2
scene.add_sun((-0.62, 0.50, -0.60), 3.0)
cam = scene.add_camera((0.0, -7.2, 0.45), (0.0, 0.0, 1.78), lens=19)
cam.data.shift_y = 0.07
scene.render(os.path.join(build, 'keyart.exr'), ml, meta=dict(shot='keyart', anchors=scene.project(anchors)))
