"""Quick geometry check: all fighters in a row, plain shading preview."""
import sys, os, math
sys.path.insert(0, os.path.dirname(__file__))
from sb import scene, roster
from sb.geo import MatLib
from sb.character import Character
import bpy

only = sys.argv[1].split(',') if len(sys.argv) > 1 and sys.argv[1] != 'all' else None
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), '../build/debug_lineup.exr')
W = int(sys.argv[3]) if len(sys.argv) > 3 else 1920
sc = scene.reset(W, int(W * 9 / 16) if not only or len(only) > 2 else int(W * 1.1), samples=8)
ml = MatLib()
chars = [c for c in roster.ROSTER if not only or c['id'] in only]
n = len(chars)
gap = 1.25
for i, spec in enumerate(chars):
    x = (i - (n - 1) / 2) * gap
    ch = Character(spec, spec['pose'], ml, index=i + 1, root=(x, 0, 0))
    ch.build()
scene.add_sun((-0.5, 0.6, -0.65), 3.0)
span = max(n * gap, 2.4)
if n <= 2:
    scene.add_camera((0, -6.5, 1.15), (0, 0, 1.0), lens=50)
else:
    scene.add_camera((0, -span * 1.55, 1.2), (0, 0, 1.0), lens=50)
scene.render(out, ml, meta=dict(chars=[c['id'] for c in chars]))
