# Art Bible — the "Filmverse" comic look

Reference board: `art/concept/07_style_guide.png`. Implementation: `art/pipeline/sb/comicfx.py` (offline);
a Unity URP port of the same steps is planned for the build phase.

The target is the *feeling* of the Spider-Verse films: a printed comic over real 3D. It uses our own motifs (film strips,
single-screen cinema posters, Indian street colour), not any of their characters, logos or designs.

## The five steps

1. **Render real 3D** with one hard key light. In Blender the Diffuse Direct pass gives N·L × shadows with no noise.
2. **Cel bands:** three tones at N·L = 0.32 / 0.16. Shadows are *hue-shifted*: purple `(0.52,0.42,0.76)` multiply on cloth,
   magenta-brown `(0.72,0.44,0.52)` on skin. Never plain grey. Black cloth keeps form with a navy sheen.
3. **Print texture:** 45° halftone dots (7 px @1080p) only in the light→shadow terminator; −35° hatching (6 px) in deep shadow
   and ambient-occlusion creases. Stubble and beards use stipple.
4. **Ink:** edges from depth jumps, normal creases, object IDs and material seams. 1.8 px @1080p; silhouettes ×1.25, inner
   seams ×0.6, +60 % weight on the shadow side; ink fades with distance (25 → 120 m) so skylines stay painterly.
5. **Finish:** coloured rim light (stage preset), hard spec shapes on leather, glass and metal, emission glow, print
   misregistration instead of depth-of-field blur, paper grain, light vignette.

## Lighting presets

| Preset | Key tint | Shadow tint | Rim | Fill |
|--------|----------|-------------|-----|------|
| SUNSET | warm `(1.0,0.90,0.76)` | purple | coral `(1.0,0.45,0.55)` | sky blue |
| NIGHT  | cool `(0.70,0.78,1.0)` | deep indigo | cyan `(0.25,0.95,1.0)` | neon magenta |

## Characters

- Proportions: about 6.5 heads tall, broad V-taper, slightly oversized hands and heads for readability on phones.
- Every fighter owns **one accent colour** used on the card, HUD, hit sparks and super FX (see the palette rows in the board).
- Caricature knobs: brow weight, mustache style (handlebar / walrus / trim / thick), nose length and width, jaw, age lines,
  hair volume and quiff. Signature props carry identity as much as faces do (towel, chain, lathi, shades, steel sleeve, bandage, mask).
- Cloth gets fold ridges at elbows, knees and waist so ink lines describe the material.

## Environments

- Painted sky layer (brush streaks + Ben-Day dots) behind real 3D geometry.
- Indian street vocabulary: black water tanks, overhead wires, bunting, kites, hand-painted cinema banners with Telugu,
  Hindi and Tamil lettering, marigold garlands, pastel facades.
- Far layers get haze + misregistration, never blur.

## Motion and FX (for the build)

- Characters animate **on 2s** (12 fps holds) for the comic snap; camera, FX and UI run at full rate.
- Smears and multiples on fast attacks; speed lines and impact bursts are 2D cards drawn in the same ink.
- Hit text: DHISHUM! family of onomatopoeia, localised per language.

## Unity implementation plan (URP)

| Step | Unity feature |
|------|---------------|
| Cel bands + hue shift | Custom Lit Shader Graph: main-light N·L × shadow attenuation → 3-step ramp → shadow tint per material |
| Halftone and hatching | Screen-space UV pattern in the same shader, masked by ramp position and SSAO |
| Ink | ScriptableRendererFeature: Sobel on `_CameraDepthTexture` + `_CameraNormalsTexture` + an ID buffer; width scaled by shadow mask |
| Rim / spec shapes | Fresnel step + Blinn step in the toon shader, colour from a global lighting-preset ScriptableObject |
| Misregistration, grain | Final full-screen pass: depth-masked R/B offset + paper texture |
| On-2s animation | Animator `speed` stepping or a sampled-pose component on the character root |
