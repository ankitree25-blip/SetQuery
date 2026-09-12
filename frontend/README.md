# SatQuery AI — Part 1: Frontend
Single file, `index.html`. No build step, no dependencies, no npm install.

## Run it

Open `index.html` directly in a browser, or serve it statically:

```
python3 -m http.server 8080
```

then visit `http://localhost:8080`.

# SatQuery AI — Part 1: Frontend
Single file, `index.html`. No build step, no dependencies, no npm install.

## Run it

Open `index.html` directly in a browser, or serve it statically:

```
python3 -m http.server 8080
```

then visit `http://localhost:8080`.

## Backend connection

Always calls the real backend at `CONFIG.BASE_URL` (`http://localhost:8000`
by default, near the top of the `<script>` block) — there's no mock/
simulated mode to flip on or off. The sidebar's connection pill (bottom
left) shows whether that backend is actually reachable right now; if the
whole app looks unresponsive, check there first before anything else.

(Earlier versions of this file — before a real backend existed — shipped a
`CONFIG.MODE = 'mock'` toggle that simulated every API call in-browser.
Once removed, that code path is gone entirely, not just switched off, so
there's nothing left to accidentally leave in the wrong position.)

## Scope

Matches `architecture.md` Section 3.1, laid out as a persistent sidebar
(Projects, each managed via a "..." menu — rename/delete — plus a "+ New
analysis" reset) alongside a main workspace: upload (with modality +
optional timestamp), query box, results with confidence + evidence overlay
(boxes / change map) + before/after toggle + execution trace, and a
client-side "download report" export. No accounts, no routing, no animation
beyond what a progress bar and a loading spinner need.

## Known limitation

Browsers can't inline-preview GeoTIFF. Anything that isn't natively renderable (basically anything except png/jpg/webp/gif/bmp) shows a dimensions-only placeholder instead of faking a preview. A real thumbnail for GeoTIFF inputs needs a server-generated preview image — not in the current Section 4 contract. Worth adding a thumbnail field/endpoint to Part 2/3 if judges need to see actual satellite imagery in the browser rather than placeholders.

## Known limitation

Browsers can't inline-preview GeoTIFF. Anything that isn't natively renderable (basically anything except png/jpg/webp/gif/bmp) shows a dimensions-only placeholder instead of faking a preview. A real thumbnail for GeoTIFF inputs needs a server-generated preview image — not in the current Section 4 contract. Worth adding a thumbnail field/endpoint to Part 2/3 if judges need to see actual satellite imagery in the browser rather than placeholders.
