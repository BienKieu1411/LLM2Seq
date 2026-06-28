# LLM2Seq UI Theme Specification

This file describes the visual style, layout, and component rules for the LLM2Seq web demo. It is written so another LLM or developer can recreate a UI with the same theme without seeing the original CSS.

## Product Context

LLM2Seq is a text summarization demo. The interface should feel like a compact academic/engineering tool, not a marketing landing page. The first screen should be the usable summarization interface: source input, decode mode, max-token control, summarize button, output, and runtime metrics.

The visual direction is playful but still technical. Use a light neo-brutalist style: thick dark borders, hard offset shadows, flat color fills, clear panels, and compact controls.

## Overall Layout

- Use a single-column app layout.
- The page fills the viewport height.
- Header is sticky at the top.
- Main content is centered with a maximum width of `900px`.
- Main content padding: `2rem 1.5rem`.
- Vertical gap between major blocks: `1.5rem`.
- The main interaction block is a bordered white card.
- The result block appears below the input block only after generation starts or completes.
- Footer is simple and centered.

Suggested structure:

```text
App
  Sticky Header
    Logo
    Status badges
  Main
    Input Card
      Source Text header
      Textarea
      Character/word count
      Decode Mode segmented control
      Max Tokens slider
      Summarize button
    Result Card
      Summary header + mode badge
      Summary text box
      Performance Metrics badges
  Footer
```

## Theme Name

Use the internal theme name: `neo-academic-tool`.

Keywords:

- light neo-brutalist
- compact research demo
- playful technical
- thick border
- offset shadow
- flat accents
- readable summary workspace

Avoid:

- gradient backgrounds
- glassmorphism
- soft blurred cards
- dark dashboard theme
- marketing hero sections
- oversized decorative illustrations
- excessive animation

## Color Tokens

Use these exact colors when possible.

```css
:root {
  --bg-cream: #fef3e2;
  --bg-cream-dark: #f5e6cc;
  --surface: #ffffff;
  --border: #1a1a2e;
  --text-primary: #1a1a2e;
  --text-secondary: #4a4a5e;
  --text-muted: #7a7a8e;

  --accent-red: #ff4d4d;
  --accent-red-hover: #e63939;
  --accent-green: #2ec4b6;
  --accent-green-hover: #1fa898;
  --accent-yellow: #ffe66d;
  --accent-blue: #4361ee;
  --accent-blue-hover: #3651d4;
  --accent-purple: #9b5de5;
  --accent-orange: #ff9f1c;
}
```

Usage rules:

- Page background: `--bg-cream`.
- Main cards and header: `--surface`.
- Borders and hard shadows: `--border`.
- Primary action button: `--accent-red` with white text.
- Active segmented control: `--accent-yellow`.
- Logo accent: red text on yellow background.
- Model-ready badge: pale green background with green status dot.
- Autoregressive badge: pale blue.
- MTP Verified badge: pale purple.
- Latency metric: blue.
- Tokens metric: green.
- Speed metric: purple.
- Acceptance metric: orange.
- Steps metric: yellow.

## Typography

Use:

```css
--font-main: "Space Grotesk", system-ui, -apple-system, sans-serif;
--font-mono: "JetBrains Mono", "Fira Code", monospace;
```

Rules:

- Main font is used for labels, body text, buttons, headings, and controls.
- Mono font is used for badges, metrics, model names, and numeric values.
- Body line-height: `1.6`.
- Summary text line-height: `1.8`.
- Text should be short, direct, and functional.
- Do not use ornate writing or marketing copy.

Approximate sizes:

- App logo title: `1.8rem`, weight `700`.
- Card title: `1.2rem`, weight `700`.
- Control label: `1.1rem`, weight `700`.
- Button: `1rem` to `1.15rem`, weight `600`.
- Textarea: `0.95rem`, line-height `1.7`.
- Badge: `0.8rem`, mono.
- Metric label: `0.72rem`, uppercase.
- Metric value: `0.9rem`, mono, weight `600`.

## Borders, Radius, and Shadows

This theme depends on hard geometry.

```css
--border-width: 3px;
--shadow-offset: 5px;
--shadow: 5px 5px 0px var(--border);
--shadow-sm: 3px 3px 0px var(--border);
--shadow-hover: 7px 7px 0px var(--border);
--radius: 12px;
--radius-sm: 8px;
--transition: all 0.15s ease;
```

Rules:

- Cards use `3px` dark borders, `12px` radius, and `5px 5px` hard shadow.
- Buttons and controls use `8px` radius and smaller hard shadows.
- Badges use `2px` borders, `6px` radius, and `2px 2px` shadow.
- Sliders use a `12px` track height, `2px` border, and a square-ish yellow thumb.
- Avoid large pill shapes. Corners should be slightly rounded, not soft.

## Core Components

### Card

Use cards for the input panel, result panel, and error panel.

```css
.neo-card {
  background: var(--surface);
  border: 3px solid var(--border);
  border-radius: 12px;
  box-shadow: 5px 5px 0px var(--border);
  padding: 1.5rem;
}

.neo-card:hover {
  box-shadow: 7px 7px 0px var(--border);
  transform: translate(-2px, -2px);
}
```

Use hover lift sparingly. It should feel tactile, not flashy.

### Button

Buttons should be rectangular, bordered, and shadowed.

```css
.neo-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  padding: 0.75rem 1.5rem;
  font-weight: 600;
  border: 3px solid var(--border);
  border-radius: 8px;
  box-shadow: 3px 3px 0px var(--border);
}
```

Interaction:

- Hover: shadow grows, button shifts up-left by `2px`.
- Active: shadow disappears, button shifts down-right by `2px`.
- Disabled: opacity `0.6`, no transform, no shadow.

Primary button:

- Background: `--accent-red`.
- Text: white.
- Full width in the main input card.
- Padding: `1rem`.
- Font size: `1.15rem`.

Small secondary buttons:

- Background: white.
- Used for `WikiLingua Test Sample` and `Clear`.
- Padding: `0.4rem 0.8rem`.
- Font size: `0.85rem`.

### Input Textarea

The source text area should feel like a robust work surface.

```css
.neo-input {
  width: 100%;
  padding: 0.875rem 1rem;
  background: var(--surface);
  border: 3px solid var(--border);
  border-radius: 8px;
  box-shadow: 3px 3px 0px var(--border);
  outline: none;
}
```

Textarea details:

- Minimum height: `180px`.
- Maximum height: `400px`.
- Resize vertical only.
- Line-height: `1.7`.
- Font size: `0.95rem`.
- Wrap long words and long Vietnamese text safely.
- Footer shows character and word count aligned to the right.

### Segmented Decode Toggle

Use a two-option segmented control.

Options:

- `Autoregressive`
- `MTP Verified`

Style:

- Outer border: `3px solid --border`.
- Radius: `8px`.
- Shadow: `3px 3px 0px --border`.
- Each option has equal width.
- Active option background: `--accent-yellow`.
- Inactive option background: white.
- Inactive hover background: `--bg-cream`.
- Divider between options: `3px solid --border`.

The hint below the control is small muted text:

- Autoregressive: `Standard token-by-token decoding`
- MTP Verified: `Multi-Token Prediction with main-head verification`

### Slider

The max-token slider should match the border-heavy style.

Track:

- Width: full.
- Height: `12px`.
- Background: `--bg-cream`.
- Border: `2px solid --border`.
- Radius: `6px`.

Thumb:

- Size: `24px x 24px`.
- Background: `--accent-yellow`.
- Border: `3px solid --border`.
- Radius: `6px`.
- Shadow: `2px 2px 0px --border`.
- Hover: scale `1.1`, shadow `3px 3px`.

Label:

- `Max Tokens: <value>`
- Weight `700`, size `1.1rem`.

### Badges

Use badges for model status, model name, decode mode, and metrics.

Base:

```css
.neo-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.3rem 0.7rem;
  font-family: var(--font-mono);
  font-size: 0.8rem;
  font-weight: 500;
  border: 2px solid var(--border);
  border-radius: 6px;
  box-shadow: 2px 2px 0px var(--border);
  white-space: nowrap;
}
```

Badge variants:

```css
.neo-badge--green  { background: #d4f5e9; color: #0d6e4f; }
.neo-badge--red    { background: #ffd6d6; color: #b91c1c; }
.neo-badge--yellow { background: #ffe66d; color: #1a1a2e; }
.neo-badge--blue   { background: #dbeafe; color: #1e40af; }
.neo-badge--purple { background: #ede9fe; color: #6b21a8; }
.neo-badge--orange { background: #fff3cd; color: #92400e; }
```

Metric badge:

- Padding: `0.5rem 0.75rem`.
- Label uppercase, size `0.72rem`, opacity `0.8`.
- Value mono, size `0.9rem`, weight `600`.

Metric labels:

- `Latency`
- `Tokens`
- `Speed`
- `Acceptance`
- `Steps`
- `Speedup`

### Result Summary

The generated summary is shown inside a smaller cream panel inside the result card.

Style:

- Background: `--bg-cream`.
- Border: `2px solid --border`.
- Radius: `8px`.
- Padding: `1.25rem`.
- Line-height: `1.8`.
- Font size: `0.95rem`.
- Preserve line breaks with `white-space: pre-wrap`.
- Wrap long Vietnamese text safely.

### Loading State

When generating:

- Show a bordered spinner.
- Show text: `Generating summary...`
- Show four skeleton lines.

Spinner:

- Size: `1.25rem`.
- Border: `3px solid --border`.
- Top border transparent.
- Rotate with `0.8s linear infinite`.

Skeleton:

- Background: `--bg-cream-dark`.
- Border: `2px solid --border`.
- Radius: `8px`.
- Pulse opacity every `1.5s`.

### Header

Header:

- Sticky top.
- Background: white.
- Border-bottom: `3px solid --border`.
- Shadow: `0 4px 0px --border`.
- Inner max-width: `900px`.
- Padding: `1rem 1.5rem`.
- Layout: logo left, status badges right.

Logo:

- Text: `LLM` + highlighted `2Seq`.
- `LLM` is dark text.
- `2Seq` is red text on yellow background.
- Highlight has `3px` border, `6px` radius, and `4px 4px` shadow.
- Subtitle: `Text Summarization`, muted, `0.85rem`.

Status:

- `Model Ready` or `Model Loading...`.
- Ready state uses green badge and an animated green dot.
- Model name appears as a yellow badge.

### Footer

Footer is understated.

- Center aligned.
- Padding: `1.25rem`.
- Border-top: `3px solid --border`.
- Background: white.
- Font size: `0.85rem`.
- Text color: `--text-secondary`.
- GitHub link color: `--accent-blue`, weight `600`.

## App Copy

Use these labels to keep the same feel:

- Header title: `LLM2Seq`
- Subtitle: `Text Summarization`
- Status: `Model Ready`, `Model Loading...`
- Source label: `Source Text`
- Sample button: `WikiLingua Test Sample`
- Clear button: `Clear`
- Placeholder: `Paste or type the text you want to summarise...`
- Decode label: `Decode Mode`
- Decode options: `Autoregressive`, `MTP Verified`
- Slider label: `Max Tokens: <value>`
- Main button: `Summarize`
- Loading button text: `Generating...`
- Result title: `Summary`
- Metrics title: `Performance Metrics`

Keep copy short. Do not add explanatory paragraphs inside the app.

## Responsiveness

Desktop:

- Max content width `900px`.
- Input card controls use a two-column grid:
  - left: decode mode
  - right: max tokens

Tablet and mobile under `768px`:

- Header inner layout stacks vertically.
- Controls become one column.
- Cards reduce padding to `1rem`.
- Buttons reduce padding to `0.6rem 1rem`.
- Result header stacks vertically.
- Metrics stack or wrap cleanly.

Small mobile under `480px`:

- Any model-info grids or metric groups should become one column if needed.
- Keep text from overflowing buttons or badges.

## Motion Rules

Use only small tactile motion:

- Card hover: shift `-2px, -2px`.
- Button hover: shift `-2px, -2px`.
- Button active: shift `2px, 2px`.
- Slider thumb hover: scale `1.1`.
- Status dot pulse: opacity pulse.
- Loading spinner: simple rotation.

Avoid page transitions, large animations, parallax, or background motion.

## Implementation Checklist

When recreating the UI, verify:

- The first screen is the actual summarization tool, not a landing page.
- Body background is cream.
- Header and cards are white.
- Every major surface has thick dark borders.
- Shadows are hard offset shadows, not blurred shadows.
- Primary action button is red.
- Active decode mode is yellow.
- Logo highlight uses yellow fill, red text, dark border, and hard shadow.
- Summary output is inside a cream bordered panel.
- Metrics appear as colored bordered badges.
- Mobile layout has no horizontal overflow.
- No text overlaps inside buttons, badges, or cards.
- The UI still looks like a compact tool when the source text is long.

## Minimal CSS Seed

Use this seed if generating a fresh implementation:

```css
:root {
  --bg-cream: #fef3e2;
  --bg-cream-dark: #f5e6cc;
  --surface: #ffffff;
  --border: #1a1a2e;
  --text-primary: #1a1a2e;
  --text-secondary: #4a4a5e;
  --text-muted: #7a7a8e;
  --accent-red: #ff4d4d;
  --accent-red-hover: #e63939;
  --accent-green: #2ec4b6;
  --accent-yellow: #ffe66d;
  --accent-blue: #4361ee;
  --accent-purple: #9b5de5;
  --accent-orange: #ff9f1c;
  --border-width: 3px;
  --shadow: 5px 5px 0px var(--border);
  --shadow-sm: 3px 3px 0px var(--border);
  --shadow-hover: 7px 7px 0px var(--border);
  --radius: 12px;
  --radius-sm: 8px;
  --font-main: "Space Grotesk", system-ui, -apple-system, sans-serif;
  --font-mono: "JetBrains Mono", "Fira Code", monospace;
  --transition: all 0.15s ease;
}

body {
  margin: 0;
  min-height: 100vh;
  font-family: var(--font-main);
  color: var(--text-primary);
  background: var(--bg-cream);
  line-height: 1.6;
}

.neo-card {
  background: var(--surface);
  border: var(--border-width) solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 1.5rem;
}

.neo-btn {
  border: var(--border-width) solid var(--border);
  border-radius: var(--radius-sm);
  box-shadow: var(--shadow-sm);
  font-family: var(--font-main);
  font-weight: 600;
  transition: var(--transition);
}
```
