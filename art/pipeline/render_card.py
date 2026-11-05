"""Render one fighter for a character card: python render_card.py <id> [width height yaw pitch]."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from sb import scene, roster, framing
from sb.geo import MatLib
from sb.character import Character
cid = sys.argv[1]
W = int(sys.argv[2]) if len(sys.argv) > 2 else 2250
Hh = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
yaw = float(sys.argv[4]) if len(sys.argv) > 4 else 24
pitch = float(sys.argv[5]) if len(sys.argv) > 5 else 11
out = os.path.join(os.path.dirname(__file__), f'../build/cards/{cid}.exr')
spec = roster.BY_ID[cid]
scene.reset(W, Hh, samples=16)
ml = MatLib()
ch = Character(spec, spec['pose'], ml, 1)
objs = ch.build()
scene.add_sun(spec.get('key', (-0.85, 0.22, -0.48)), 3.0)
cam = framing.fit_camera(objs, yaw_deg=yaw, pitch_deg=pitch, lens=35, margin=spec.get('card_margin', 1.36), aspect=W / Hh)
cam.data.shift_x = -0.10
cam.data.shift_y = -0.115
scene.render(out, ml, meta=dict(id=cid))
