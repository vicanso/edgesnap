// EdgeSnap detection worker.
//
// A dedicated, blocking consumer: it parks on Atomics.wait until the main
// thread publishes a frame in the SharedArrayBuffer, runs the wasm detector,
// posts back the small packed result, then parks again. Never returns to its
// event loop after init — that is fine, it only ever needs the one frame slot.

import init, { detect_frame } from './pkg/edgesnap.mjs';

let header = null;   // Int32Array view: [0]=state, [1]=w, [2]=h, [3]=frameId
let pixels = null;   // Uint8Array view of the RGBA region

self.onmessage = async (e) => {
  if (e.data.type !== 'init') return;
  const { sab, headerLen } = e.data;
  header = new Int32Array(sab, 0, headerLen);
  pixels = new Uint8Array(sab, headerLen * 4);

  await init();
  self.postMessage({ type: 'ready' });

  // Blocking consume loop.
  for (;;) {
    // Park while state == 0 (idle). Returns when main flips it to 1.
    Atomics.wait(header, 0, 0);

    const w = Atomics.load(header, 1);
    const h = Atomics.load(header, 2);
    const frameId = Atomics.load(header, 3);

    const packed = detect_frame(pixels.subarray(0, w * h * 4), w, h);
    self.postMessage({ type: 'result', packed, frameId });

    // Mark idle so the main thread may publish the next frame.
    Atomics.store(header, 0, 0);
    Atomics.notify(header, 0, 1);
  }
};
