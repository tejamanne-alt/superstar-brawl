"""2.5D gameplay shot on Stage 1 v2 (Cinema Street). Camera looks down the street; the fight
plane runs across the road. 1.5x supersampled."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import bpy  # noqa
from mathutils import Vector
from sb import scene, roster, street
from sb.geo import MatLib
from sb.character import Character
W, H = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (3510, 1620)
build = os.path.join(os.path.dirname(__file__), '../build')
scene.reset(W, H, samples=16, ao_distance=0.3)
ml = MatLib()
street.street(ml, build)
p1 = dict(roster.ALT_POSES['bala_block'], yaw=62)
p2 = dict(roster.ALT_POSES['raja_throw'], yaw=-58)
c1 = Character(roster.BY_ID['bala'], p1, ml, 1, root=(-2.2, 0.0, 0))
c1.build()
c2 = Character(roster.BY_ID['raja'], p2, ml, 2, root=(2.1, -0.2, 0))
c2.build()
gl = bpy.data.objects['raja_fg_bridge']
anchor_pts = dict(glasses=sum((Vector(c) for c in gl.bound_box), Vector()) / 8,
                  raja_hand=c2.J['wrR'], bala_fists=(c1.J['wrR'] + c1.J['wrL']) / 2,
                  bala_head=c1.J['head'], raja_head=c2.J['head'], bala_feet=c1.J['anL'], raja_feet=c2.J['anR'],
                  vanish=Vector((0, 150, 2.0)))
scene.add_sun((0.45, 0.55, -0.70), 3.0)
scene.add_camera((0.0, -6.4, 1.45), (0.0, 20.0, 2.55), lens=24)
scene.render(os.path.join(build, 'gameplay.exr'), ml, meta=dict(shot='gameplay', anchors=scene.project(anchor_pts)))
