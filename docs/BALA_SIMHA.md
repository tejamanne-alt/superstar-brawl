# Bala Simha: character spec (hero-quality pass)

Boards: `art/concept/10_bala_hero.png` (cover look), `11_bala_model_sheet.png` (turnaround, face views, callouts,
palette), `12_bala_ingame.png` (same model in the in-game ink-dot look), `13_bala_face.png` (face study at the
reference angle).
Model: the supplied textured Meshy FBX `art/source/bala_desert_king_textured.fbx` (309k triangles, own UVs, 2048 px
base colour / normal / roughness / metallic maps). Loader: `art/pipeline/sb/fbxtex.py` · per-pixel classes:
`sb/pixclass.py` · shaders: `sb/cover.py` (cover look), `sb/inkdot.py` (in game) · renders: `render_fbx.py` ·
boards: `compose_bala.py`.

## How the textured FBX is used

The model is rendered as delivered, with its own UVs and base-colour texture, so the beard, mustache, hair, tilak and
aviators are exactly as painted in the texture. Coarse body regions (head, shirt, dhoti, hands/feet, sandals) become
material slots. The comic shaders then tell skin from beard/hair, lenses and gold per pixel. They use the base colour,
the metallic and gloss maps (written as extra render passes) and world height. The texture's pale skin is graded to
his tan, and ink seams follow those per-pixel classes, so the beard edge stays soft with no outline.

If a low-poly textured file is supplied instead, `fbxtex.load` transfers its texture onto the untextured sculpt
`art/source/bala_desert_king.fbx`: every sculpt triangle borrows UVs from the nearest low-poly triangle.
The earlier code-built versions (`sb/bala.py`, `sb/fbxbala.py`) are kept for reference.

## Likeness checklist (from the reference photos)

| Feature | Spec |
|---------|------|
| Hair | Big, wavy, swept straight back; a high front wave rising up and back from a tall forehead; full above the ears; dark on top, grey at the sides and temples with grey streaks |
| Tilak | Round red dot on the lower forehead, between and above the brows |
| Brows | Heavy, straight, dark with a little grey; pulled down at the inner ends when angry |
| Eyewear | Gold-wire aviators (double bridge), dark olive lenses that are darker at the top |
| Beard | Full but trimmed close; salt-and-pepper, dark on the cheeks and sideburns, white-grey at the chin; follows the jaw (from the texture) |
| Mustache | Thick and dark, covering the upper lip and drooping at the corners into the beard |
| Face | Broad; fleshy cheeks and jowls; strong straight nose with a broad tip; heavy upper lids; light forehead lines |
| Build | About 1.78 m (FBX) and stocky: broad shoulders, thick neck, a slight belly; head about 7% larger than standard for presence |
| Outfit | Black shirt with an open collar, chest pocket, sleeves rolled to the elbow; chain; khaki dhoti (as in the texture); sandals |
| Jewellery | Gold watch on the left wrist, gold rings |
| Signature pose | The "mass walk": striding forward, right hand gathering the dhoti (from the FBX); arms-crossed intro as an alternate |

## Rendering rules (cover look)

One soft, brush-broken shadow shape per form; warm key light; magenta and teal neon rim lights;
sharp light streaks on the hair; black cloth painted as deep charcoal with large calm light shapes and a
cool sheen; tapered dark-brown brush ink, heavy on the silhouette and light and broken inside; dry-brush
hatch strokes in cloth shadows. In game, the same model switches to the ink-dot look (`sb/inkdot.py`).

## Expressions

The model has one fixed expression (stern, shades on). Smirk / glare / roar / laugh need facial blend shapes on a
rigged version of the head.

## Known gaps before production

- The model is one posed mesh: for mobile it needs a lighter retopology (about 25k triangles), a rig and blend
  shapes before it can animate in Unity. The texture can be re-baked onto that mesh.
- The model was AI-generated to resemble a real actor; the personality-rights caveat in `LEGAL_NOTE.md` applies.
