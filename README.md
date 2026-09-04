# hugoxk.github.io

Personal homepage of Ke Xu (许可) — a single static page, no build step, no framework.

## Local preview & deploy

Open `index.html` directly in a browser, or run `python3 -m http.server` and visit
`http://localhost:8000`. Deploy by pushing to `main` (GitHub Pages, with `.nojekyll`).

## How to edit

- **Text content** — everything lives in `index.html`. Every bilingual element carries
  a `data-zh="中文"` attribute next to the English text; edit both together.
- **`[VERIFY]` markers** — data that still needs confirmation. Search `[VERIFY]` in
  `index.html`.
- **`[TODO]` markers** — pending files/actions (CV PDF, gallery photos, repo links).
- **Travel map** — edit `js/countries-list.js`:
  - `[ ] Name` — not visited
  - `[X] Name` — visited
  - `[X 2019-09] Name` — visited, first visited in 2019-09
  - `[P] Name` — planned
  - `[P 2026-10] Name` — planned for 2026-10

  Map fill, hover tooltips, checklist, counters and the progress bar update
  automatically on reload.
- **Gallery** — drop photos into `files/imgs/gallery/`, then copy the commented
  `<figure>` template inside the Gallery section. Click-to-zoom (lightbox) is ready.
- **CV** — place the PDF at `files/CV_KeXu_2026.pdf`; the header button links there.
- **Version history** — keep the `<!-- Version:x.x.x -->` comment block at the top of
  `index.html` in sync when you make a release.

## Structure

| Path | Purpose |
| --- | --- |
| `index.html` | The whole page |
| `css/main.css` | All page styles |
| `css/leaflet.css` | Map library styles (self-hosted Leaflet 1.9.4) |
| `js/travel.js` | Choropleth map logic (lazy-loads Leaflet + geo data) |
| `js/countries-geo.js` | Country polygons / centroids / Chinese names — **auto-generated, do not hand-edit** |
| `js/countries-list.js` | Your visited/planned checklist — **edit this** |
| `js/i18n.js` | EN/ZH toggle (persists in `localStorage`, swaps `data-zh` elements) |
| `js/ui.js` | Nav scroll-spy, back-to-top, gallery lightbox |
| `js/leaflet.min.js` | Self-hosted Leaflet |
| `og-image.png` | Social share card (1200×630) — regenerate if your info changes |
| `favicon.svg` + PNGs | Site icons (red "K" monogram) |
| `404.html`, `robots.txt`, `.nojekyll` | Static-site housekeeping |
