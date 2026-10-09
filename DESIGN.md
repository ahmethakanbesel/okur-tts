---
name: Okur
description: A Turkish F-keyboard office typewriter on a desk that reads back what you type.
colors:
  ground: "#ecedea"
  enamel: "#3f5f56"
  enamel-hi: "#557a6f"
  enamel-lo: "#2c4640"
  enamel-ink: "#e9efeb"
  enamel-error: "#ffd0d6"
  white: "#ffffff"
  paper: "#fbfbf8"
  paper-shade: "#e4e7e0"
  rule: "#c9cdc5"
  field-edge: "#aab0aa"
  ink: "#151515"
  ink-soft: "#3c403e"
  ink-muted: "#5b615e"
  ghost: "#666c69"
  red: "#c8102e"
  red-rest: "#b04a5a"
  red-deep: "#7d0a1d"
  red-spent: "#6d4a50"
  key: "#1b1d1c"
  legend: "#eef0ea"
  platen: "#121413"
  chrome-face: "#bfc5c8"
  chrome-edge: "#e3e7e9"
  chrome-letter: "#23282a"
  figure-model: "#e3ebe7"
  figure-decoder: "#ece8df"
typography:
  display:
    fontFamily: "Courier Prime, Courier New, ui-monospace, monospace"
    fontSize: "clamp(32px, 4.4vw, 50px)"
    fontWeight: 700
    lineHeight: 1.08
    letterSpacing: "-0.02em"
  headline:
    fontFamily: "Archivo, Helvetica Neue, Arial, sans-serif"
    fontSize: "clamp(36px, 5vw, 56px)"
    fontWeight: 800
    lineHeight: 0.95
    letterSpacing: "0.01em"
    fontVariation: "'wdth' 68"
  title:
    fontFamily: "Archivo, Helvetica Neue, Arial, sans-serif"
    fontSize: "15px"
    fontWeight: 800
    lineHeight: 1.2
    letterSpacing: "0.08em"
    fontVariation: "'wdth' 80"
  typed-input:
    fontFamily: "Courier Prime, Courier New, ui-monospace, monospace"
    fontSize: "clamp(17px, 1.6vw, 20px)"
    fontWeight: 400
    lineHeight: 1.55
  typed-reading:
    fontFamily: "Courier Prime, Courier New, ui-monospace, monospace"
    fontSize: "clamp(15px, 1.3vw, 17px)"
    fontWeight: 400
    lineHeight: 1.7
  typed-body:
    fontFamily: "Courier Prime, Courier New, ui-monospace, monospace"
    fontSize: "clamp(16px, 1.4vw, 18px)"
    fontWeight: 400
    lineHeight: 1.55
  typed-status:
    fontFamily: "Courier Prime, Courier New, ui-monospace, monospace"
    fontSize: "13.5px"
    fontWeight: 400
    lineHeight: 1.45
  typed-data:
    fontFamily: "Courier Prime, Courier New, ui-monospace, monospace"
    fontSize: "15px"
    fontWeight: 400
    lineHeight: 1.4
    fontFeature: "'tnum'"
  body:
    fontFamily: "Archivo, Helvetica Neue, Arial, sans-serif"
    fontSize: "17px"
    fontWeight: 400
    lineHeight: 1.55
  body-sm:
    fontFamily: "Archivo, Helvetica Neue, Arial, sans-serif"
    fontSize: "15px"
    fontWeight: 400
    lineHeight: 1.6
  label:
    fontFamily: "Archivo, Helvetica Neue, Arial, sans-serif"
    fontSize: "11px"
    fontWeight: 700
    lineHeight: 1.2
    letterSpacing: "0.2em"
  key-legend:
    fontFamily: "Archivo, Helvetica Neue, Arial, sans-serif"
    fontSize: "12.5px"
    fontWeight: 650
    lineHeight: 1
    letterSpacing: "0.12em"
    fontVariation: "'wdth' 82"
  key-oku:
    fontFamily: "Archivo, Helvetica Neue, Arial, sans-serif"
    fontSize: "20px"
    fontWeight: 800
    lineHeight: 1
    letterSpacing: "0.2em"
    fontVariation: "'wdth' 75"
  nameplate:
    fontFamily: "Archivo, Helvetica Neue, Arial, sans-serif"
    fontSize: "15px"
    fontWeight: 800
    lineHeight: 1
    letterSpacing: "0.38em"
    fontVariation: "'wdth' 70"
  engraving:
    fontFamily: "Archivo, Helvetica Neue, Arial, sans-serif"
    fontSize: "12px"
    fontWeight: 700
    lineHeight: 1.3
    letterSpacing: "0.26em"
    fontVariation: "'wdth' 78"
  figure-label:
    fontFamily: "Archivo, Helvetica Neue, Arial, sans-serif"
    fontSize: "13.5px"
    fontWeight: 650
    lineHeight: 1.2
rounded:
  paper: "0px"
  plate: "3px"
  figure-box: "4px"
  cap-sm: "7px"
  key: "8px"
  field: "8px"
  tray: "10px"
  oku: "12px"
  panel: "22px"
  panel-phone: "16px"
  roller: "23px"
spacing:
  hair: "6px"
  xs: "8px"
  sm: "14px"
  md: "22px"
  lg: "30px"
  footer: "96px"
  band: "104px"
  band-phone: "72px"
  machine: "min(1160px, calc(100vw - 32px))"
components:
  keycap:
    backgroundColor: "{colors.key}"
    textColor: "{colors.legend}"
    typography: "{typography.key-legend}"
    rounded: "{rounded.key}"
    padding: "11px 14px"
    height: "40px"
  keycap-hover:
    backgroundColor: "#262928"
    textColor: "{colors.legend}"
  keycap-nav:
    backgroundColor: "{colors.key}"
    textColor: "{colors.legend}"
    rounded: "{rounded.key}"
    padding: "9px 12px"
    height: "34px"
  key-oku:
    backgroundColor: "{colors.red}"
    textColor: "{colors.white}"
    typography: "{typography.key-oku}"
    rounded: "{rounded.oku}"
    padding: "0 28px"
    height: "58px"
    width: "148px"
  key-oku-disabled:
    backgroundColor: "{colors.red-spent}"
    textColor: "{colors.white}"
  nameplate:
    backgroundColor: "{colors.chrome-face}"
    textColor: "{colors.chrome-letter}"
    typography: "{typography.nameplate}"
    rounded: "{rounded.plate}"
    padding: "10px 12px 10px 17px"
  speed-tray:
    backgroundColor: "{colors.key}"
    rounded: "{rounded.tray}"
    padding: "4px"
  speed-option-on:
    backgroundColor: "{colors.legend}"
    textColor: "{colors.ink}"
    rounded: "{rounded.cap-sm}"
    padding: "11px 13px"
    height: "36px"
  sheet:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    rounded: "{rounded.paper}"
  text-field:
    backgroundColor: "{colors.white}"
    textColor: "{colors.ink}"
    typography: "{typography.typed-input}"
    rounded: "{rounded.field}"
    padding: "11px 14px"
  status-line:
    textColor: "{colors.enamel-ink}"
    typography: "{typography.typed-status}"
    width: "100%"
  status-line-error:
    textColor: "{colors.enamel-error}"
  sample-card:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    rounded: "{rounded.paper}"
    padding: "20px 24px 22px"
  figure-plate:
    backgroundColor: "{colors.white}"
    textColor: "{colors.ink}"
    typography: "{typography.figure-label}"
    rounded: "{rounded.paper}"
  code-block:
    backgroundColor: "{colors.paper-shade}"
    textColor: "{colors.ink}"
    padding: "14px 16px"
  mark:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper}"
    rounded: "{rounded.plate}"
    padding: "3px 6px"
---

# Design System: Okur

## Overview

**Creative North Star: "The Daktilo"**

The page is a desk with one Turkish F-keyboard office typewriter on it. The ground is a whitish desk top with a faint grain; on it sit the machine (a black rubber platen with chrome knobs, then a front panel in hammertone green enamel) and loose sheets of bond paper. Whatever is typed (headline, lede, input, the reading, sample sentences, the numbers, the status) is ribbon ink in Courier Prime. Whatever is part of the machine or names a section (nameplate, keys, section titles, engraved facts, labels) is condensed Archivo caps, the lettering stamped onto metal and keycaps.

The machine is matte and mechanical. Keys have physical depth (an inset bottom edge) that shrinks when pressed; the chrome nameplate and roller knobs are flat satin metal with a single bright top edge; paper has square corners and casts a soft drop onto the desk. The enamel survives only where the machine is: the front panel and, by colour kinship, the keycap tray. Density is that of a well-kept desk: generous sheets, one machine width (1160px), and wide bands of open desk between sections.

Red ribbon is the system's one signal. It marks where the visitor can act (the OKU key, focus, the caret) and where the text frontend acted (restored circumflexes, spelled numbers). Everything else is ink on desk or paper, or light on enamel.

**Key Characteristics:**
- Desk ground, enamel machine body, paper sheets, black keycaps; white only for the text field and the printed figures.
- Courier Prime for anything typed, condensed Archivo caps for anything engraved or printed on a key.
- Red only for action, frontend change, and focus.
- Text waits in ghost ink and strikes forward as it is read.
- Turkish first, English as a full equal; every label is translatable and the layout holds both.

## Colors

A pale, cool desk carrying a green-enamel machine, warm-white paper, and black ink, with one ribbon red.

### Primary
- **Ribbon Red** (`red`): the OKU key face, the global focus ring, the textarea caret, text selection, restored or changed letters once struck, and Okur's own play key in each sample card.
- **Resting Ribbon** (`red-rest`): changed letters before the voice reaches them; a softer red so the reading still reads as "waiting" while marking what the frontend will change.
- **Ribbon Shadow** (`red-deep`): the depth edge under the OKU key and Okur's play caps. Never a text color.
- **Spent Ribbon** (`red-spent`): the OKU key face while disabled (model loading).

### Neutral
- **Desk** (`ground`): the page ground and theme-color, with a faint SVG noise grain (5% alpha). Section titles, ledes, facts, the language lever and the footer sit directly on it in the ink scale.
- **Hammertone Enamel** (`enamel`): the machine body, i.e. the front panel, lit top to bottom from **Enamel Highlight** (`enamel-hi`) through enamel at 30% to **Enamel Shadow** (`enamel-lo`). **Enamel Ink** (`enamel-ink`): the status line on the panel. **Enamel Error** (`enamel-error`): the status line's error state, a pale red that stays legible on enamel where ribbon red would not.
- **White** (`white`): the OKU legend, the text field, and the ground of the printed figures.
- **Bond Paper** (`paper`): sheets, sample cards, the skip link. **Paper Shade** (`paper-shade`): the bottom 3% fade of the sheet in the platen, and code blocks. **Ruling** (`rule`): dashed dividers, tab rack ticks, example underlines. **Field Edge** (`field-edge`): the text field's 1px border at rest.
- **Ribbon Ink** (`ink`): all typed text and struck letters; section titles on the desk; the 2px rule above each system in a sample card and under table headers; the text field's focused border. **Ink Soft** (`ink-soft`): ledes on paper and on the desk, prose on notes. **Ink Muted** (`ink-muted`): labels, table headers, engraved facts, and footer credits.
- **Ghost Ink** (`ghost`): letters not yet read, separators between examples, and the faint former spelling drawn behind a changed word (at 35% opacity).
- **Keycap Black** (`key`) with **Legend** (`legend`): every key and the speed tray; legend is also the selected speed option's face and the status line's feed fill. **Platen Rubber** (`platen`): the roller.
- **Chrome Face** (`chrome-face`), **Chrome Edge** (`chrome-edge`), **Chrome Letter** (`chrome-letter`): the nameplate, the roller knobs, and the language lever knob.
- **Figure Model** (`figure-model`) and **Figure Decoder** (`figure-decoder`): module fills inside the printed figures only, acoustic model and decoder respectively. They never appear as page surfaces.

### Named Rules
**The Ribbon Rule.** Full-strength red appears only where the visitor can act, where the frontend changed the text, or on focus. The one decorative red is the margin stop, a 1px rule at 30% alpha on paper, which is part of the sheet's own ruling. Errors on enamel use enamel-error, not ribbon red.

**The Materials Rule.** Every page surface is desk, enamel, paper, or keycap (with chrome only for the nameplate and knobs). White is not a surface material: it is reserved for the text field and the figure plates. Enamel is the machine body only; it never returns as a section ground.

**The Reading Floor Rule.** Reading text holds at least 4.5:1 against its actual ground. On the desk: ink about 15.5:1, ink-soft about 9.0:1, ink-muted about 5.4:1. On paper: ghost about 5.2:1, resting red about 5.1:1, ink-muted about 6.1:1. On the panel: enamel-ink about 6.0:1 on enamel and enamel-error about 5.1:1, both only because the status line sits in the lower 70% of the panel; on enamel-hi they drop to about 4.1:1 and 3.5:1, so never move panel text into its top band. Inside figures, ink-muted holds about 5.2:1 on the module fills.

## Typography

**Display Font:** Courier Prime (with Courier New, ui-monospace), weights 400 and 700, self-hosted latin and latin-ext.
**Body Font:** Archivo (with Helvetica Neue, Arial), variable weight 400 to 800 and width 62% to 125%, self-hosted.

**Character:** Typewriter ink against stamped machine lettering. Courier Prime is what the machine prints; Archivo, squeezed with `font-stretch` and tracked wide in caps, is what is engraved on its body and printed on its keys.

### Hierarchy
- **Display** (Courier Prime 700, clamp 32 to 50px, 1.08, -0.02em, max 24ch, balanced): the typed headline on the sheet in the platen.
- **Headline** (Archivo 800, clamp 36 to 56px, 0.95, stretch 68%, uppercase, ink on the desk): section titles.
- **Title** (Archivo 800, 15px, 1.2, stretch 80%, 0.08em caps): note titles.
- **Typed input** (Courier Prime 400, clamp 17 to 20px, 1.55): the text field.
- **Typed reading** (Courier Prime 400, clamp 15 to 17px, 1.7): the reading line, deliberately a step smaller than the input so it reads as the machine's echo rather than a second field.
- **Typed body** (Courier Prime 400, clamp 16 to 18px, 1.55, max 60ch): the lede; sample sentences use clamp 17 to 21px; examples 15px.
- **Typed status** (Courier Prime 400, 13.5px, 1.45; 12.5px on phones): the status line on the panel.
- **Typed data** (Courier Prime 400, 15px, tabular figures): the ledger table.
- **Body** (Archivo 400, 17px, 1.55, max 68ch): section ledes and figure captions on the desk. **Body small** (Archivo 400, 14.5 to 15px, 1.6): prose on notes and the honesty note (max 78ch). Footer credits are 13.5px, max 90ch.
- **Label** (Archivo 700, 11px, 0.2em, uppercase, ink-muted): field labels, table headers (0.16em), card categories. The reading legend is Archivo 400 at 12px.
- **Key legend** (Archivo 650, 12.5px, stretch 82%, 0.12em caps): keycaps and speed options; nav keys at 11.5px. **OKU** is Archivo 800 at 20px, stretch 75%, 0.2em.
- **Nameplate** (Archivo 800, 15px, stretch 70%, 0.38em caps). **Engraving** (Archivo 700, 12px, stretch 78%, 0.26em caps, ink-muted with ink figures): the facts row under the panel.
- **Figure label** (Archivo 650, 13.5px): module names inside figures; sub-lines Archivo 11.5px ink-soft; math in a serif italic at 14px; parameter counts in Courier Prime 11px ink-muted; group captions Archivo 700 11px at 0.12em caps.

### Named Rules
**The Typed Versus Engraved Rule.** If the text is something written or read (content, numbers, input, status), it is Courier Prime. If it names a part of the machine or a control, it is condensed Archivo caps. Archivo in sentence case is reserved for explanatory prose and figure labels.

**The Squeeze Rule.** Machine lettering is always narrowed with `font-stretch` (68% to 82%) and tracked open; never set Archivo caps at normal width on the page. (Figure group captions are drawn, not engraved, and are exempt.)

## Layout

One machine width, `min(1160px, 100vw - 32px)`, carries the cover, the machine, every band, and the footer. The first viewport stacks top to bottom: cover (nameplate left, keycap nav and TR/EN lever right, all on the desk), the platen (a 46px roller with chrome knobs overhanging its ends by 12px), the sheet rolled out beneath it (inset up to 64px each side, tucked 18px under the roller), the enamel front panel (OKU, speed tray, NEW TAKE, DOWNLOAD in a wrapping row, the status line as a full-width row beneath), then the engraved facts centred under it on the desk.

Below, sections are bands on the desk separated by 104px (72px on phones): an ink headline, an ink-soft lede, then paper or figures. Sample cards stack in a single column with a 14px gap; the ledger is one sheet; "How it works" is two figures stacked, each a full-width plate with its caption below; code and notes are three sheets in a 1.3/1/1 grid. The footer is separated by a 1px rule at 12% black.

Responsive rules, as built:
- **1000px:** notes stack, sample systems go 2 by 2.
- **760px:** the nav keys become a single horizontally scrolling row under the nameplate (scrollbar hidden, bleeding to the screen edges). The front panel becomes a sticky dock at the bottom of the viewport; OKU grows to fill its row; the status line is clamped to two lines. The platen shrinks to 34px, the margin stop on the hero sheet is removed, card label tabs turn into a header row.
- **480px:** sample systems become one column.

Spacing rhythm in use: 6, 8, 14, 22, 30px inside the machine; 96 to 104px between bands.

## Elevation & Depth

Depth is physical, not ambient. The desk is the ground; paper lies on it and casts a soft, downward, negatively spread drop; the machine (platen and panel) casts a heavier shadow; keys and knobs are raised parts with an inset depth edge and a small contact shadow; recessed parts (the speed tray, the lever track, the text field) are inset. Lighting comes from above: the panel's top-to-bottom enamel gradient and a 1px light top edge on chrome and keys. Figures are printed plates and carry no shadow.

### Shadow Vocabulary
- **Sheet drop** (`0 24px 40px -18px rgb(0 0 0 / 0.55), 0 1px 0 rgb(0 0 0 / 0.08)`): the sheet in the platen, the ledger, and notes.
- **Card drop** (`0 14px 28px -14px rgb(0 0 0 / 0.55)`): sample cards.
- **Keycap** (`inset 0 -3px 0 #000, inset 0 1px 0 rgb(255 255 255 / 0.12), 0 3px 6px rgb(0 0 0 / 0.35)`); pressed: `inset 0 -1px 0 #000, 0 1px 2px rgb(0 0 0 / 0.35)` with `translateY(2px)`.
- **OKU** (`inset 0 -5px 0 red-deep, inset 0 1px 0 rgb(255 255 255 / 0.3), 0 5px 9px rgb(0 0 0 / 0.45)`); pressed shrinks the edge to 2px with `translateY(3px)`.
- **Recess** (`inset 0 2px 4px #000`): speed tray and lever track.
- **Field** (`inset 0 1px 2px rgb(0 0 0 / 0.07)`); focused adds `0 0 0 3px rgb(21 21 21 / 0.12)`.
- **Chrome** (`0 2px 3px rgb(0 0 0 / 0.35)` plus a 1px chrome-edge top border): nameplate and knobs.
- **Panel / platen** (`0 18px 30px rgb(0 0 0 / 0.4)`, `0 8px 14px rgb(0 0 0 / 0.45)`).

### Named Rules
**The Matte Parts Rule.** Chrome and keys are flat fills with one light top edge and a soft contact shadow; no metallic sheen gradients, no glossy highlights. Gradients belong only to light falling on the enamel panel and the paper's own fade into the platen.

**The Pressed Depth Rule.** A key's depth edge is its state: it shrinks and the key drops 2 to 3px on press. Hover never lifts a key.

## Shapes

Paper is always square-cornered; machine parts are rounded according to their size: nameplate and marks 3px, play caps and speed options 7px, keycaps and the text field 8px, the speed tray 10px, OKU 12px, the front panel's bottom corners 22px (16px on phones), the platen a full 23px pill with round knobs. Dividers on paper are dashed ruling; structural rules (above each system, under headers) are 2px solid ink. The margin stop is a 1px vertical red rule at 30% alpha inset from a sheet's left edge. The tab rack is a strip of 1px ticks every 9px with a solid ink triangle marking each numeric column's right edge.

Inside figures the language is the paper diagram: modules are 4px-radius boxes stroked in 1.1px ink, arrows 1.2px ink with a small filled head, host-code modules and conditioning arrows dashed (5/4 and 4/3), operators as 12px-radius white circles.

## Components

### Keycaps
Black, square-shouldered keys with off-white condensed legends.
- **Shape:** gently rounded (8px), minimum 40px tall (34px in the nav, 38px in the phone dock).
- **Default:** keycap black with legend, depth edge and contact shadow.
- **Hover:** face lightens slightly (#262928); no lift. **Active:** drops 2px, depth edge shrinks to 1px over 120ms.
- **Disabled:** 55% opacity with a progress cursor (while the model loads).
- **Uses:** section nav, NEW TAKE, DOWNLOAD, show-all-sentences, LIVE READ on each card.

### OKU Key
The single primary action, the only full-red surface at rest.
- **Shape:** 12px radius, at least 148 by 58px; on phones it flexes to fill its row at 42px.
- **Default:** ribbon red face, white legend with an SVG play triangle; red-deep 5px depth edge.
- **Speaking:** the legend becomes DUR with an SVG stop square (`data-speaking`). **Hover:** brightness 1.06. **Active:** drops 3px.
- **Disabled:** spent-ribbon face with a darker depth edge, progress cursor.

### Speed Lever and Language Lever
- **Speed:** a recessed keycap-black tray (10px, 4px padding) holding three radio options (slow 0.85, normal 1, fast 1.15). Selected option is a legend face with ink text; unselected options are dim legend (#b9c2bd) that whitens on hover.
- **Language lever:** a `role="switch"` on the desk with TR and EN in ink either side of a 46 by 22px recessed track; the chrome knob slides 24px over 220ms. The inactive side's label drops to 60% opacity.

### Nameplate
Chrome face, chrome-letter caps (OKUR) at 70% width and 0.38em tracking, 3px radius, one bright top edge. It is the home link.

### Sheet, Platen, and Margin Stop
- **Platen:** a black rubber roller with chrome knobs, decorative (`aria-hidden`).
- **Sheet:** bond paper with a fine SVG grain, square corners, the sheet drop, and a margin stop on the left. In the machine it fades to paper-shade in its last 3% and slides under the roller.

### Text Field
A real input set into the sheet.
- **Style:** white fill, 1px field-edge border, 8px radius, faint inset shadow, 11px by 14px padding, red caret, field-sizing to content (3.6em to 12em, vertical resize).
- **Hover:** border darkens to #7f8681. **Focus:** border turns ink with a 3px ink ring at 12% alpha, replacing the global red outline; 120ms transition.

### Reading Line
Each letter is its own span in ghost ink and strikes to ink as its sound begins. A changed word rests in resting-red and strikes to full red; its original spelling is drawn behind it from `data-was` as a 35% ghost pseudo-element, so it is never part of the text or its selection. A legend line with an 18 by 3px red swatch explains the red.

### Status Line and Paper Feed
A transparent, full-width row under the keys on the enamel panel, typed in Courier Prime in enamel-ink and announced as `role="status"`. A thin 2px feed bar (white at 20%, max 360px) sits above the text and fills with legend from left to right (`scaleX`, 200ms linear) as the model downloads, then disappears. Error state turns the text enamel-error.

### Sample Card and Label Grid
Every hard sentence is one card with the same grid: a 200px label tab (category in label caps, LIVE READ keycap) behind a dashed vertical rule, then the sentence typed large and four system columns. Each system column starts with a 2px ink rule and a play key (30px keycap cap with SVG play or stop glyph, red for Okur) followed by the system name in caps; the playing system is underlined. On phones the tab becomes a header row.

### Ledger and Tab Rack
The metrics table is typed on one sheet in Courier Prime with tabular figures. Headers are label caps over a 2px ink rule; rows are separated by dashed ruling; best values are bold; Okur's row (labelled with its release, e.g. Okur v1.0) is bold with a small ink-on-paper mark. Above the header sits a tab rack whose triangles mark each numeric column's stop. The table scrolls horizontally inside the sheet below 820px.

### Figures
"How it works" is told in numbered paper-style figures, generated in both languages by `scripts/figures.py` and inlined so they use the page's typefaces.
- **Plate:** white ground, square, no shadow, scaled to the machine width.
- **Modules:** figure-model fill for acoustic-model parts, figure-decoder fill for the decoder, white for rule-based and input/output boxes, white with a dashed stroke for host code; a legend row names the fills with their parameter counts.
- **Lettering:** see Figure label in Typography; shapes see Shapes.
- **Caption:** below the plate, Archivo in ink, opening with a bold "Şekil n." / "Figure n."; it explains every symbol the figure uses.

### Notes
Sheets with a margin stop holding condensed caps titles, Archivo prose, and code blocks on paper-shade with no radius.

### Footer
Archivo 13.5px ink-muted credits on the desk; links in ink at weight 650. The first line credits the text normalizer (normalizer-tr) with a link; the second lists comparison-sample licences, typefaces and runtime.

### Motion
- **Strike:** when a letter's sound begins (timed from the model's own per-letter durations), it turns from ghost to ink and strikes in from 3px above at 20% opacity over 160ms on `cubic-bezier(0.16, 1, 0.3, 1)`.
- **Keys:** 120ms press on the same curve; text field border and ring 120ms; lever knob 220ms; paper feed 200ms linear.
- **Reduced motion:** strike animation, key and knob transitions, feed transition, and smooth scrolling are all removed; letters still change from ghost to ink.

### Language
Turkish is the default (`lang="tr"`, `?lang=en` or the lever switches); English is a complete alternative, figures included. All strings live in one dictionary per language; controls must accommodate the longer of the two (keycaps never wrap).

## Do's and Don'ts

### Do:
- **Do** type every piece of content in Courier Prime and letter every machine part in condensed Archivo caps (stretch 68% to 82%, tracking 0.12em or more).
- **Do** put reading content on paper (square, grained, dropped) and set titles, ledes, facts and credits directly on the desk in the ink scale.
- **Do** keep the enamel to the machine body: the front panel and its parts.
- **Do** let letters wait in ghost ink and strike forward to ink as they are read; changed words rest in resting-red and strike to full red.
- **Do** give every key a depth edge that shrinks on press, and keep keys at least 34px tall.
- **Do** keep reading text at 4.5:1 or better against its actual ground, and check panel text against the lighter top of the enamel.
- **Do** draw new explanatory diagrams in the figure grammar (white plate, 1.1px ink strokes, the two module fills, numbered caption) and generate them in both languages.
- **Do** use SVG glyphs for play and stop, inheriting the key's legend color.
- **Do** respect reduced motion by removing the strike and press transitions, never the ghost-to-ink state change.

### Don't:
- **Don't** use red for decoration, headings, errors on enamel, or emphasis; it is for action, frontend change, and focus only (the 30% margin stop excepted).
- **Don't** return the enamel to the page ground or use it behind sections.
- **Don't** add metallic sheen gradients, glossy highlights, or chrome beyond the nameplate and knobs.
- **Don't** round paper corners or give paper a border.
- **Don't** lift keys on hover or use hard offset shadows; depth is a soft contact shadow and an inset edge.
- **Don't** set content in Archivo caps or set machine labels in Courier.
- **Don't** introduce a new surface material or a second accent hue; the figure fills stay inside figures.
