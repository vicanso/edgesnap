#!/usr/bin/env bash
# Build the Rust CV core to wasm and drop it in web/pkg/.
set -euo pipefail
cd "$(dirname "$0")"

echo "▶ building wasm (release)…"
wasm-pack build --target web --out-dir web/pkg --no-typescript --release

# Serve the JS glue as .mjs: vicanso/static (and the spec) only accept a JS MIME
# for module scripts, and that server maps .mjs -> text/javascript but not .js.
mv -f web/pkg/edgesnap.js web/pkg/edgesnap.mjs

echo "✓ wasm written to web/pkg/ (glue as edgesnap.mjs)"
echo "▶ now serve ./web with a static server that sends COOP/COEP, then open it"
echo "  (headers: Cross-Origin-Opener-Policy: same-origin, Cross-Origin-Embedder-Policy: require-corp)"
