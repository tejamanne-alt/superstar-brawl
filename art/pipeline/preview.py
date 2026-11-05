import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image
from sb import exr
P, meta = exr.load(sys.argv[1])
lin = P['albedo'] * (0.35 + 0.65 * np.clip(P['direct'] / meta.get('key_energy', 3), 0, 1)) + P['emit']
srgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(np.clip(lin, 0, None), 1 / 2.4) - 0.055)
a = P['alpha'][..., None]
img = srgb * a + np.array([0.55, 0.6, 0.7]) * (1 - a)
Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(sys.argv[2])
print('saved', sys.argv[2])
