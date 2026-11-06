"""Key art v2: the full roster in an arrowhead formation on Cinema Street (1.5x supersampled)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import bpy  # noqa
from mathutils import Vector
from sb import scene, roster, street
from sb.geo import MatLib
from sb.character import Character
W, H = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (3840, 2160)
build = os.path.join(os.path.dirname(__file__), '../build')
scene.reset(W, H, samples=16, ao_distance=0.3)
ml = MatLib()
street.street(ml, build)
LAYOUT = [('raja', 0.0, -2.2, -4), ('bala', -1.75, -1.3, 18), ('srk', 1.85, -1.2, -14), ('hriday', -3.55, 0.0, 30),
          ('chiru', 3.65, 0.1, -26), ('ambar', -5.3, 1.5, 14), ('nag', 5.4, 1.6, -16), ('venky', -7.0, 3.0, 22),
          ('kamal', 7.1, 3.1, -28)]
anchors = {}
for i, (cid, x, y, yaw) in enumerate(LAYOUT):
    spec = roster.BY_ID[cid]
    pose = dict(spec['pose'], yaw=yaw + spec['pose'].get('yaw', 0))
    ch = Character(spec, pose, ml, i + 1, root=(x, y, 0))
    ch.build()
    anchors[cid + '_head'] = ch.J['head']
    anchors[cid + '_feet'] = (ch.J['anL'] + ch.J['anR']) / 2
anchors['vanish'] = Vector((0, 150, 2.0))
scene.add_sun((0.45, 0.55, -0.70), 3.0)
scene.add_camera((0.0, -5.4, 0.80), (0.0, 4.0, 1.62), lens=17)
scene.render(os.path.join(build, 'keyart.exr'), ml, meta=dict(shot='keyart', anchors=scene.project(anchors)))
