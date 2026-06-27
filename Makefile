# EdgeSnap build tasks.
#
# `make web` compiles the Rust CV core to a wasm bundle in web/pkg/ (wasm-pack).
# Serve ./web with any static server, BUT the SharedArrayBuffer path needs these
# response headers (without them the app falls back to single-thread wasm):
#
#   Cross-Origin-Opener-Policy:   same-origin
#   Cross-Origin-Embedder-Policy: require-corp

WASM_PACK ?= wasm-pack
OUT_DIR   := web/pkg

.PHONY: all web build test clean

all: web

## web, build: compile the wasm bundle into web/pkg/ (release)
# The glue is renamed to .mjs so static servers send a JS MIME for the module
# (vicanso/static maps .mjs -> text/javascript, but serves .js as octet-stream).
web build:
	$(WASM_PACK) build --target web --out-dir $(OUT_DIR) --no-typescript --release
	mv -f $(OUT_DIR)/edgesnap.js $(OUT_DIR)/edgesnap.mjs

## test: run the native CV unit tests
test:
	cargo test --lib

## clean: remove the generated wasm bundle
clean:
	rm -rf $(OUT_DIR)
