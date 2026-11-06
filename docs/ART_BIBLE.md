# Art Bible v2: the "Ink-Dot" look

Reference board: `art/concept/09_style_guide.png`. Implementation: `art/pipeline/sb/inkdot.py` (shader),
`sb/facepaint.py` + `sb/head.py` (characters), `sb/fxart.py` (comic FX), `sb/street.py` (Stage 1).

Direction: real 3D printed like a comic, matching the approved reference screenshots (purple streets, saturated
facades, halftone sky, heavy ink, off-register colour on grainy paper). Characters are drawn in a modern teen-comic
style (clean fashion, about 7.5 heads tall, sculpted hair, expressive inked faces), in the spirit of the
New Riverdale covers, applied to our own nine superstars. No Archie characters, logos or trade dress are used;
only the general illustration style, which is not protected.

## The four steps (every frame, every asset)

| Step | Rule |
|------|------|
| **1 Flat colour** | Saturated albedo only, no painted gradients on objects. "Black" is deep indigo `(0.13,0.12,0.26)` so dot screens still read; whites are warm cream `(0.99,0.96,0.88)`. |
| **2 Light as dots** | One hard key. Lit = flat colour; surfaces turning away get **growing dots of the shadow colour** (from N·L 0.80 down to the 0.17 terminator); shadow = flat violet (`× (0.48,0.34,0.84)`, lighter lavender on pale colours, rose `(0.80,0.48,0.64)` on skin). AO creases get a second, darker dot screen; shiny materials get a hard highlight shape ringed by light dots. Dot period = 0.68 % of image height, 45°. |
| **3 Ink line** | 0.40 % of image height (4.3 px at 1080p). Silhouettes ×1.35, object edges ×1.2, depth breaks ×1.0, material seams ×0.72, creases ×0.62, ×1.35 on the shadow side. Slight hand wobble (noise-displaced edge) and slow width variation. Fades 55 % between 30 and 160 m. |
| **4 Slip + paper** | Colour plates off-register (red right/down, blue left/up, 0.21 % of height) under a crisp ink plate; paper grain, mottling, faint fibres, rare dark/light flecks, warm print cast. |

## Lighting presets

| Preset | Use | Shadow | Rim | Bounce dots |
|--------|-----|--------|-----|-------------|
| `sunset` | Stage 1, key art, gameplay | violet | off | violet-blue |
| `night` | Stage 2 | deep indigo | cyan neon | magenta |
| `neon` | character cards, select, portraits | violet | fighter accent | mint |

## Characters (Archie-styled superstars)

- **Proportions:** head radius 0.071 × height (about 7.5 heads), V-taper torso, long legs, slim neck, smaller hands.
- **Faces are drawn, not modelled:** `facepaint.py` inks the face as 2D comic artwork projected onto the skull
  (heavy tapered upper lash line with a flick, crease line, small iris under the lid, tapered brows, nostril wings plus
  one shadow-side nose line, cheekbone accent, big readable mouths: grin / smirk / shout / gritted teeth).
  Mustaches and beards are drawn shapes with pointed ends. The nose is real 3D for the silhouette and light split.
- **Hair is sculpted:** one hair volume with a rising front wave, a steep rounded front face and V-grooves between
  locks (the ink pass turns them into lock lines), plus fat pointed spikes and forelock flips for the silhouette.
- **Fashion:** each superstar's signature outfit, tailored clean (rolled sleeves, open jackets, slim jeans, sneakers).
- One **accent colour** per fighter drives the card, HUD ring, neon sign and super FX.

## Environments

- Stage 1 **Cinema Street**: the street runs away from camera and the fight plane crosses it, giving the reference's
  one-point canyon. Pastel-saturated facades, cyan shop glass, red/teal/yellow awnings, signboards in English, Telugu
  and Hindi, black-and-yellow kerbs, auto-rickshaws, low-poly crowd, bunting, wires, the single-screen cinema at the end.
- Sky = flat magenta→orange gradient with a big halftone screen that swells toward the sun glow; far planes get haze.

## FX language

Halftone-gradient SFX lettering (yellow→orange→red or mint→teal→blue, dots inside the letters, ink outline,
offset drop shadow, slanted) · starburst hits with cream core and radiating strokes · outlined smoke balls with
lavender halftone undersides · dark speed lines converging on the vanishing point · triangle debris.

## Unity implementation plan (URP)

| Step | Unity feature |
|------|---------------|
| Flat colour + light as dots | Custom Lit Shader Graph: main-light N·L × shadow → terminator step; screen-space dot screen whose coverage = ramp; AO from SSAO drives the crease screen |
| Drawn faces | Face decal texture per expression on a front-projected UV set; swap textures for expressions (blend shapes only for jaw / brows silhouette) |
| Ink line | ScriptableRendererFeature: edge detect on depth + normals + object/material ID buffer; width × shadow mask; noise offset for wobble |
| Slip + paper | Final full-screen pass: R/B plate offset before the ink composite, paper texture multiply |
| Animation | Characters on 2s (12 fps holds) for the comic snap; camera, FX and UI at full rate |
