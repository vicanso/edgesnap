# EdgeSnap

In-browser document / ID-card edge detection with automatic capture, written in
Rust and compiled to WebAssembly. The camera, UI and snapshot live in a thin
vanilla-JS host; all the computer vision is Rust. No model, no network — just
traditional CV (grayscale → blur → Sobel → threshold → connected components →
four-corner fit) plus simple geometric rules.

## What it does

1. **Camera** — `getUserMedia` (rear camera preferred) into a `<video>`.
2. **Edge / document detection** — each frame is downscaled and run through the
   Rust pipeline, which returns the four corners of the dominant rectangular
   object.
3. **Auto-capture** — when a card-shaped quad stays still and confident for a
   few consecutive frames, EdgeSnap snaps a cropped screenshot automatically.
   Frames are handed to the worker through a `SharedArrayBuffer`, which is why
   the dev server sends `COOP`/`COEP` headers.
4. **Recognition** — coarse classification by aspect ratio: ID-1 / CR80 cards
   (身份证, 银行卡 ≈ 1.585) vs. generic documents, with a confidence score from
   corner angles, opposite-side symmetry and aspect match.

## Architecture

```
main thread (main.js)                 Web Worker (worker.js)
  camera ─▶ <video>                      ┌───────────────────────────┐
  draw to 480px work canvas              │ Atomics.wait on header     │
  getImageData (RGBA)                    │ detect_frame(rgba,w,h)     │  ← wasm
       │  write into  ┌──────────────────┤   (src/cv.rs via lib.rs)   │
       └─────────────▶│ SharedArrayBuffer │ post {corners,type,conf}  │
  draw overlay + ◀────┤  header + pixels  └───────────────────────────┘
  auto-snapshot       └──────────────────
```

The worker is a dedicated blocking consumer: it parks on `Atomics.wait` until
the main thread publishes a frame, processes it, posts back an 11-float packed
result, and parks again. The main thread only hands off a frame when the worker
is idle, so slow frames are dropped instead of queued. The big RGBA buffer never
crosses the thread boundary via `postMessage` — it lives in the shared buffer.

If the page is **not** cross-origin isolated (e.g. served without COOP/COEP),
`main.js` falls back to running the same wasm on the main thread so the demo
still works, minus the worker/SAB benefit.

## Layout

| path                | role                                                       |
|---------------------|------------------------------------------------------------|
| `src/cv.rs`         | Pure-Rust CV pipeline. No wasm/JS deps; unit-tested natively. |
| `src/lib.rs`        | `wasm-bindgen` glue: `detect_frame(rgba, w, h) -> Float32Array`. |
| `web/index.html`    | UI (camera stage, overlay, controls, result panel).        |
| `web/main.mjs`      | Camera, draw loop, SAB transport, overlay, auto-capture.    |
| `web/worker.mjs`    | Blocking SAB consumer that runs the wasm detector.          |
| `web/pkg/`          | Generated wasm bundle (git-ignored; produced by `make web`).|
| `Makefile`          | `make web` (build) · `make test` · `make clean`.           |
| `build.sh`          | Same build as `make web`, as a standalone script.          |

## Prerequisites

- Rust (stable) with the wasm target: `rustup target add wasm32-unknown-unknown`
- [`wasm-pack`](https://rustwasm.github.io/wasm-pack/): `cargo install wasm-pack`
- A static file server for `./web` that can send COOP/COEP headers (see below).

## Build & run

```bash
make web               # compiles src/ to web/pkg/ via wasm-pack (or: ./build.sh)
# then serve ./web with any static server and open it in a browser:
#   开启摄像头 → hold an ID/card steady → auto-snapshot
```

**COOP/COEP is required for the SharedArrayBuffer path.** Whatever static server
you use to serve `./web` must send these response headers:

    Cross-Origin-Opener-Policy:   same-origin
    Cross-Origin-Embedder-Policy: require-corp

Without them the page is not cross-origin isolated, so `SharedArrayBuffer` is
unavailable and EdgeSnap falls back to running the wasm on the main thread (it
still works, just without the worker/SAB pipeline). Camera access also needs a
secure context: `localhost` is fine; on a LAN IP use HTTPS.

## Test

```bash
cargo test --lib       # native unit tests of the CV core (src/cv.rs)
```

The detector logic is plain Rust, so it is fully testable without a browser:
synthetic frames with a bright rectangle assert corner positions and the
ID-card vs. document classification.

## Packed result format

`detect_frame` returns a `Float32Array` of length 11:

| index | meaning                                        |
|-------|------------------------------------------------|
| 0     | found (1 / 0)                                  |
| 1–8   | corners TL, TR, BR, BL as x,y (input pixels)   |
| 9     | type code (0 none · 1 id_card · 2 document)    |
| 10    | confidence 0.0–1.0                             |

## Tuning

Detection thresholds live in `cv::Params` (`src/cv.rs`): edge keep-fraction,
min/max area, min component size. Auto-capture behaviour (confidence floor,
stillness tolerance, frames-to-stable, cooldown) lives at the top of
`web/main.js`.

## Limitations

This is classical CV, not a neural model. It assumes the card/document is the
dominant high-contrast rectangle in view and is not rotated much past ~45°. It
detects geometry and aspect ratio — it does **not** read text/MRZ or confirm a
document is genuinely an ID. Busy backgrounds can pull the largest edge
component away from the card; the stability gate and the tunable params reduce
false snaps. The snapshot is an axis-aligned crop of the detected quad's
bounding box (perspective un-warp would be a natural next step).
