"""Per-pixel material classes for the textured Bala model (one texture, several meanings).

Inside the head region the texture holds skin, hair, beard, brows, mustache, lenses and the gold
frames; the comic shaders need to treat each differently, so pixels are classified from the
render passes: base colour (saturation / warmth), metallic and gloss AOVs, and world height.
Skin is also re-graded from the texture's pale beige toward his warm tan.
"""
import numpy as np

SKIN_TARGET = np.array([0.80, 0.57, 0.44])     # mean skin colour after grading (sRGB)
HAIR_LINE = 1.628                              # above this, dark head pixels are scalp hair


def classify(P, base):
    """base: sRGB albedo (H, W, 3). Returns dict of boolean masks + graded base, or None."""
    if 'region' not in P:
        return None
    a = np.clip(P['alpha'], 1e-4, None)
    reg = np.rint(P['region'] / a).astype(np.int32) * (P['alpha'] > 0.5)
    metal = P.get('metal', np.zeros_like(a)) / a > 0.5
    gloss = P.get('gloss', np.zeros_like(a)) / a
    hgt = P.get('height', np.zeros_like(a)) / a
    mx, mn = base.max(-1), base.min(-1)
    sat = (mx - mn) / np.maximum(mx, 1e-4)
    warm = base[..., 0] - base[..., 2]
    head = reg == 1
    skin_px = head & (sat > 0.10) & (warm > 0.045) & (mx > 0.30)
    glass = head & (gloss > 0.55) & (mx < 0.35) & ~metal & (hgt > 1.590) & (hgt < 1.650)
    # lenses are two big blobs: drop small dark-glossy fragments (nostrils, mouth line, beard shadow)
    import cv2
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(glass.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] > glass.size * 0.0006
    glass = keep[lbl]
    metal_px = metal & (reg > 0)
    fur = head & ~skin_px & ~glass & ~metal_px
    hair = fur & (hgt > HAIR_LINE)
    # chest skin in the open collar (the region split calls it shirt; the texture says skin)
    vskin = (reg == 2) & (sat > 0.10) & (warm > 0.045) & (mx > 0.35) & (hgt > 1.30) & ~metal_px
    skin_all = skin_px | vskin | ((reg == 4) & ~metal_px)
    graded = base.copy()
    if skin_all.sum() > 50:
        mean = base[skin_all].mean(0)
        k = SKIN_TARGET / np.maximum(mean, 1e-3)
        graded[skin_all] = np.clip(base[skin_all] * k, 0, 1)
    # beard / hair: a little extra contrast so the salt-and-pepper reads after shading
    if fur.any():
        g = graded[fur]
        graded[fur] = np.clip((g - 0.42) * 1.15 + 0.42, 0, 1)
    # class map for ink seams: skin and hair/beard share a class (soft beard edge, no outline)
    import cv2
    cls = np.zeros(reg.shape, np.uint8)
    cls[reg == 2] = 4
    cls[reg == 3] = 5
    cls[reg == 5] = 6
    cls[skin_all | fur] = 1
    cls[glass] = 8
    cls[metal_px] = 7
    cls = cv2.medianBlur(cls, 5)
    return dict(skin=skin_all, fur=fur, hair=hair, glass=glass, metal=metal_px, base=graded,
                cloth=((reg == 2) | (reg == 3)) & ~vskin, region=reg, cls=cls.astype(np.int32))
