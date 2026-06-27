//! Browser bindings for the EdgeSnap CV core.
//!
//! All heavy lifting lives in [`cv`]; this file only marshals a flat RGBA byte
//! buffer in and a small packed `f32` array out, so there is zero per-call
//! allocation churn on the JS side and no serialization dependency.

mod cv;

use cv::{detect, Params};
use wasm_bindgen::prelude::*;

/// Length of the packed result returned by [`detect_frame`].
pub const RESULT_LEN: usize = 11;

/// Detect a document / ID card in an RGBA frame.
///
/// `rgba` must be `width * height * 4` bytes (the layout of
/// `CanvasRenderingContext2D.getImageData().data`).
///
/// Returns a packed `Float32Array` of length [`RESULT_LEN`]:
///
/// | index | meaning                                  |
/// |-------|------------------------------------------|
/// | 0     | `found` (1.0 / 0.0)                       |
/// | 1,2   | TL corner x,y (input-pixel coords)       |
/// | 3,4   | TR corner x,y                            |
/// | 5,6   | BR corner x,y                            |
/// | 7,8   | BL corner x,y                            |
/// | 9     | type code (0 none, 1 id_card, 2 document)|
/// | 10    | confidence 0.0..=1.0                      |
#[wasm_bindgen]
pub fn detect_frame(rgba: &[u8], width: usize, height: usize) -> Vec<f32> {
    let d = detect(rgba, width, height, &Params::default());

    let mut out = vec![0.0f32; RESULT_LEN];
    out[0] = if d.found { 1.0 } else { 0.0 };
    for (i, &(x, y)) in d.corners.iter().enumerate() {
        out[1 + i * 2] = x;
        out[2 + i * 2] = y;
    }
    out[9] = d.type_code as f32;
    out[10] = d.confidence;
    out
}

/// Perspective-rectify a detected card quad into an upright `out_w x out_h`
/// RGBA image. `corners` is `[x0,y0, x1,y1, x2,y2, x3,y3]` in TL,TR,BR,BL order
/// (e.g. the corners from [`detect_frame`], mapped back to full-resolution).
/// Returns `out_w * out_h * 4` RGBA bytes (empty on bad input).
#[wasm_bindgen]
pub fn warp_card(
    rgba: &[u8],
    width: usize,
    height: usize,
    corners: &[f32],
    out_w: usize,
    out_h: usize,
) -> Vec<u8> {
    if corners.len() < 8 {
        return Vec::new();
    }
    let q = [
        (corners[0], corners[1]),
        (corners[2], corners[3]),
        (corners[4], corners[5]),
        (corners[6], corners[7]),
    ];
    cv::warp_perspective(rgba, width, height, &q, out_w, out_h)
}

/// Semantic version string, handy for confirming the wasm bundle is current.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
