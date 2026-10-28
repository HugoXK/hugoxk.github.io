# hugoxk.github.io

Personal homepage of Ke Xu (许可) — a static, bilingual academic profile with a
live travel map, a managed photo gallery, dark mode, a blog and a print-friendly CV.

## Local preview

```bash
python3 -m http.server 8000
# open http://localhost:8000
```

No build step. Deploy by pushing to `main` (GitHub Pages, `.nojekyll`).

## Content and data

| Path | Purpose |
| --- | --- |
| `index.html` | Profile: about, news, experience, research, education, honors, travel, gallery |
| `cv.html` | Print / save-as-PDF CV |
| `blog.html` + `blog/posts/<slug>.md` | Blog list and Markdown posts |
| `data/travel.json` | Live travel status and dates (written by the console, synced here) |
| `data/portfolio.json` | Gallery manifest (written by the console, synced here) |
| `portfolio/` | Gallery images and thumbnails (written by the console, synced here) |
| `js/countries-list.js` | Offline fallback for the travel map |
| `js/travel.js` | Fetches live travel data, renders the map, filters and checklist |
| `js/gallery.js` | Fetches the gallery manifest and renders the masonry grid + lightbox |
| `js/site-config.js` | Data endpoints (relative, same-origin) |
| `js/i18n.js` | EN/ZH toggle, persisted in `localStorage` |
| `js/ui.js` | Theme, scroll progress, nav spy, reveal animations, back-to-top |
| `css/main.css` | Full design system (light/dark, travel, gallery, blog, CV, print) |

## Admin console

Travel and gallery content is managed at:

**https://portal.hugoxk.online/admin** (Cloudflare Access / GitHub login)

- Check countries and set visit / planned dates (`YYYY-MM`).
- Upload, caption, reorder and delete gallery photos.
- Saves write to the server copy under `~/work/landing/{data,portfolio}` and are
  synced automatically to this repository by `~/work/sync-profile-data.js`.
- No manual `git commit` or `git push` is required for content updates.

The profile reads `data/travel.json`, `data/portfolio.json` and `portfolio/*`
from this origin, so it stays fast and does not depend on the Cloudflare Tunnel.

## Editing the site itself

- **Text** — edit `index.html`; bilingual elements carry `data-zh` (and optional
  `data-zh-placeholder`, `data-zh-title`, `data-zh-aria-label`, `data-zh-alt`).
- **Offline travel fallback** — edit `js/countries-list.js` only if the live data
  endpoint is unavailable.
- **Theme** — colors live in `:root` and `[data-theme="dark"]` at the top of
  `css/main.css`.
- **Profile image** — `profile.webp` / `profile-360.webp` with `profile.jpg` as
  the fallback. Keep the source square.
- **Cache busting** — bump `css/main.css?v=7` when changing styles.
- **Blog** — add `blog/posts/<slug>.md`, then add an entry to `js/blog-posts.js`.
  Comments use giscus (see `BLOG_GISCUS` in that file).
- **CV** — `cv.html`; use the print button to save a PDF.

## Deployment

- **Profile** — GitHub Pages from `main`.
- **Landing / Portal** — server `~/work/landing` + nginx `127.0.0.1:3090` and
  `~/work/portal.js` on `127.0.0.1:3092`, behind Cloudflare Tunnel / Access.
- **Content sync** — the console writes the server copy and pushes only changed
  data/image files to this repository through the GitHub Git Data API.
