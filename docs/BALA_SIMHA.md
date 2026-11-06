# Bala Simha: character spec (hero-quality pass)

Boards: `art/concept/10_bala_hero.png` (cover look), `11_bala_model_sheet.png` (turnaround, expressions,
callouts, palette), `12_bala_ingame.png` (same model in the in-game ink-dot look).
Model: the supplied FBX sculpt `art/source/bala_desert_king.fbx` (walking pose, geometry only), segmented and painted
by `art/pipeline/sb/fbxbala.py` · shader: `sb/cover.py` (cover look) and `sb/inkdot.py` (in game) ·
renders: `render_fbx.py` · boards: `compose_bala.py`.

## How the FBX is used

The FBX has 503k vertices, no UVs, no colours and no rig. In Blender it is scaled to 1.78 m and stood on the ground,
then every vertex is classified into a region from measured positions on orthographic views: skin, hair, beard,
mustache, aviator lenses, gold (frames, watch, bracelet), shirt, dhoti and sandals. The aviators are found by ray
casting (a thin shell in front of the face). Each region becomes its own material slot, so the comic shaders ink the
region borders and shade skin, hair, cloth, metal and glass differently. Hair and beard get salt-and-pepper vertex
colours, and a red tilak disc is ray-cast onto the forehead.

The earlier code-sculpted version (`sb/bala.py`, `render_bala.py`) is kept for reference; it is not used by these boards.

## Likeness checklist (from the reference photos)

| Feature | Spec |
|---------|------|
| Hair | Big, wavy, swept straight back; a high front wave rising up and back from a tall forehead; full above the ears; dark on top, grey at the sides and temples with grey streaks |
| Tilak | Round red dot on the lower forehead, between and above the brows |
| Brows | Heavy, straight, dark with a little grey; pulled down at the inner ends when angry |
| Eyewear | Gold-wire aviators (double bridge), dark olive lenses that are darker at the top |
| Beard | Full but trimmed close; salt-and-pepper, darker on the cheeks and greyest at the chin; follows the jaw and stops at it |
| Mustache | Thick and dark, covering the upper lip and drooping at the corners into the beard |
| Face | Broad; fleshy cheeks and jowls; strong straight nose with a broad tip; heavy upper lids; light forehead lines |
| Build | About 1.78 m (FBX) and stocky: broad shoulders, thick neck, a slight belly; head about 7% larger than standard for presence |
| Outfit | Black shirt with an open collar, chest pocket and full sleeves; gold chain; cream dhoti with a gold border; kolhapuri sandals |
| Jewellery | Gold watch on the left wrist, gold rings |
| Signature pose | The "mass walk": striding forward, right hand gathering the dhoti (from the FBX); arms-crossed intro as an alternate |

## Rendering rules (cover look)

One soft, brush-broken shadow shape per form; warm key light; magenta and teal neon rim lights;
sharp light streaks on the hair; black cloth painted as deep charcoal with large calm light shapes and a
cool sheen; tapered dark-brown brush ink, heavy on the silhouette and light and broken inside; dry-brush
hatch strokes in cloth shadows. In game, the same model switches to the ink-dot look (`sb/inkdot.py`).

## Expressions

The FBX has one fixed expression (stern, shades on). Smirk / glare / roar / laugh exist only on the earlier
code-sculpted head (`sb/bala.py`); for the game they need facial blend shapes on the retopologised FBX head.

## Known gaps before production

- The FBX is a single posed mesh: it needs retopology (about 25k triangles for mobile), UVs and a rig before it can
  animate in Unity. Region painting should then move into a proper texture.
- Region borders come from measured boxes and angles, so a few edges (sleeve cuffs, beard line under the chin) are
  approximate; a texture-paint pass would clean them up.
- The FBX was AI-generated to resemble a real actor; the personality-rights caveat in `LEGAL_NOTE.md` applies to it.
