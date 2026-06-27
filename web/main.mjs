// EdgeSnap main thread (ID-card guide-frame UX).
//
// Owns the camera and draw loop. The CV runs in worker.mjs (frames cross via a
// SharedArrayBuffer). Detection returns the 4 real card corners (Canny -> convex
// hull -> Douglas-Peucker, in Rust/wasm); success is gated on a complete card
// quad filling the guide window, and capture perspective-rectifies it via the
// Rust `warp_card`.
//
// SAB layout:
//   Int32 header[8]:  [0]=state (0 idle / 1 frame-ready), [1]=w, [2]=h, [3]=frameId
//   Uint8 pixels[]:   RGBA, capacity SAB_PIX_CAP bytes, written at offset 0

const HEADER_LEN = 8;
const S_STATE = 0, S_W = 1, S_H = 2, S_FRAME = 3;
const PROC_MAX = 480;                       // longest processed edge, in px
const SAB_PIX_CAP = PROC_MAX * PROC_MAX * 4;

// Guide frame (the masked window the user fits the card into).
const GUIDE_W_FRAC = 0.86;                   // window width as a fraction of frame
const GUIDE_ASPECT = 1.585;                  // ID-1 / CR80 (85.6 x 54 mm)
const PORTRAIT_SIDE = 'left';                // 'right' for the real 人像面 layout

// Auto-capture tuning.
const CONF_MIN = 0.6;          // minimum detector confidence to snap
const EDGE_MARGIN_FRAC = 0.02; // corners must be this far from the frame edge (complete card)
const FILL_MIN = 0.55;         // detected quad area / guide area: lower => too small/far
const FILL_MAX = 1.7;          // upper => too close / overflowing
const COOLDOWN_MS = 2500;      // gap between two auto snaps
const HOLD_MS = 600;           // hold aligned this long (ms) before auto-capture
const OUT_W = 856, OUT_H = 540; // rectified output size (ID-1 proportions)

// --- DOM ---------------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const video = $('cam');
const overlay = $('overlay');
const octx = overlay.getContext('2d');
const shot = $('shot');
const sctx = shot.getContext('2d');
const statusEl = $('status');
const progFill = $('prog');
const isoEl = $('iso');
const startBtn = $('startBtn');
const snapBtn = $('snapBtn');
const autoChk = $('autoChk');
const dlBtn = $('dlBtn');
const shotMeta = $('shotMeta');
const flashEl = $('flash');
const pill = $('pill');
const pillText = $('pillText');
const stageEl = document.querySelector('.stage');

// Offscreen work canvases.
const proc = document.createElement('canvas');
const pctx = proc.getContext('2d', { willReadFrequently: true });
const full = document.createElement('canvas');
const fctx = full.getContext('2d', { willReadFrequently: true });

// --- state -------------------------------------------------------------------
const useWorker = (typeof SharedArrayBuffer !== 'undefined') && self.crossOriginIsolated;
let sab = null, header = null, pixels = null, worker = null;
let mainDetect = null;        // fallback: detect_frame on this thread
let warpCard = null;          // wasm warp_card, used on capture (main thread)
let engineReady = false;
let stream = null, streaming = false;
let procW = 0, procH = 0, frameId = 0, rafId = 0;

let guide = { x0: 0, y0: 0, x1: 0, y1: 0 };   // guide window in proc-pixel space
let latest = null;            // { found, corners, type, conf, inside, fill, aligned }
let alignedSince = 0, lastSnap = 0;

// --- boot --------------------------------------------------------------------
isoEl.textContent = useWorker ? 'SAB ✓ 已隔离' : 'SAB ✗ 未隔离';
isoEl.dataset.ok = useWorker ? '1' : '0';

(async function boot() {
  // Load the wasm on the main thread too: `warp_card` runs here on capture.
  const mod = await import('./pkg/edgesnap.mjs');
  await mod.default();
  warpCard = mod.warp_card;

  if (useWorker) {
    sab = new SharedArrayBuffer(HEADER_LEN * 4 + SAB_PIX_CAP);
    header = new Int32Array(sab, 0, HEADER_LEN);
    pixels = new Uint8Array(sab, HEADER_LEN * 4);
    worker = new Worker('./worker.mjs', { type: 'module' });
    worker.onmessage = onWorkerMsg;
    worker.onerror = (e) => setStatus('Worker 错误: ' + e.message, false);
    worker.postMessage({ type: 'init', sab, headerLen: HEADER_LEN });
  } else {
    setStatus('未启用跨源隔离，使用主线程模式（请配置 COOP/COEP 以启用 SAB）', false);
    mainDetect = mod.detect_frame;
    engineReady = true;
  }
})().catch((e) => setStatus('初始化失败: ' + e.message, false));

function onWorkerMsg(e) {
  const d = e.data;
  if (d.type === 'ready') {
    engineReady = true;
    if (!streaming) setStatus('引擎就绪，点击「开启摄像头」', false);
  } else if (d.type === 'result') {
    onResult(d.packed);
  }
}

// --- camera ------------------------------------------------------------------
startBtn.onclick = async () => {
  if (streaming) { stop(); return; }
  // Camera is only exposed in a secure context (HTTPS or localhost).
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    setStatus('需通过 HTTPS 或 localhost 访问才能使用摄像头', false);
    return;
  }
  startBtn.disabled = true;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
  } catch (err) {
    const msg = {
      NotAllowedError: '摄像头权限被拒绝，请在浏览器设置中允许后重试',
      NotFoundError: '未找到摄像头设备',
      NotReadableError: '摄像头被其他应用占用，请关闭后重试',
      OverconstrainedError: '没有满足要求的摄像头',
      SecurityError: '需通过 HTTPS 或 localhost 访问才能使用摄像头',
    }[err.name] || ('无法访问摄像头: ' + err.message);
    setStatus(msg, false);
    startBtn.disabled = false;
    return;
  }
  video.srcObject = stream;
  await video.play();
  sizeForVideo();
  streaming = true;
  startBtn.textContent = '停止';
  startBtn.disabled = false;
  snapBtn.disabled = false;
  loop();
};

function stop() {
  streaming = false;
  cancelAnimationFrame(rafId);
  if (stream) stream.getTracks().forEach((t) => t.stop());
  startBtn.textContent = '开启摄像头';
  snapBtn.disabled = true;
  setStatus('已停止', false);
  paintResting();
}

function sizeForVideo() {
  const vw = video.videoWidth, vh = video.videoHeight;
  const scale = PROC_MAX / Math.max(vw, vh);
  procW = Math.max(16, Math.round(vw * scale));
  procH = Math.max(16, Math.round(vh * scale));
  proc.width = procW; proc.height = procH;
  overlay.width = vw; overlay.height = vh;   // overlay drawn in capture-pixel space
  computeGuide();
}

// Centered ID-1 window, GUIDE_W_FRAC of the frame width, clamped to fit height.
function computeGuide() {
  let gw = GUIDE_W_FRAC * procW;
  let gh = gw / GUIDE_ASPECT;
  const maxH = 0.86 * procH;
  if (gh > maxH) { gh = maxH; gw = gh * GUIDE_ASPECT; }
  const cx = procW / 2, cy = procH / 2;
  guide = { x0: cx - gw / 2, y0: cy - gh / 2, x1: cx + gw / 2, y1: cy + gh / 2 };
}

// Draw the guide as a static placeholder before the camera is on.
function paintResting() {
  if (streaming) return;
  const rect = stageEl.getBoundingClientRect();
  overlay.width = Math.max(2, Math.round(rect.width));
  overlay.height = Math.max(2, Math.round(rect.height));
  const scale = PROC_MAX / Math.max(overlay.width, overlay.height);
  procW = Math.max(16, Math.round(overlay.width * scale));
  procH = Math.max(16, Math.round(overlay.height * scale));
  computeGuide();
  drawOverlay();
}

paintResting();
addEventListener('resize', paintResting);

// --- per-frame loop ----------------------------------------------------------
function loop() {
  rafId = requestAnimationFrame(loop);
  if (!streaming || video.readyState < 2) return;

  pctx.drawImage(video, 0, 0, procW, procH);
  let img;
  try {
    img = pctx.getImageData(0, 0, procW, procH);
  } catch (e) {
    setStatus('读取帧失败: ' + e.message, false);
    return;
  }

  if (useWorker) {
    if (engineReady && Atomics.load(header, S_STATE) === 0) {
      pixels.set(img.data);
      Atomics.store(header, S_W, procW);
      Atomics.store(header, S_H, procH);
      Atomics.store(header, S_FRAME, frameId++);
      Atomics.store(header, S_STATE, 1);
      Atomics.notify(header, S_STATE, 1);
    }
  } else if (mainDetect) {
    onResult(mainDetect(img.data, procW, procH));
  }

  drawOverlay();
  updateHud();
}

// --- detection result + auto capture -----------------------------------------
function onResult(packed) {
  const found = packed[0] > 0.5;
  const corners = [
    [packed[1], packed[2]], [packed[3], packed[4]],
    [packed[5], packed[6]], [packed[7], packed[8]],
  ];
  const type = packed[9] | 0;
  const conf = packed[10];

  let inside = false, fill = 0, aligned = false;
  if (found) {
    const xs = corners.map((c) => c[0]), ys = corners.map((c) => c[1]);
    const bx0 = Math.min(...xs), by0 = Math.min(...ys);
    const bx1 = Math.max(...xs), by1 = Math.max(...ys);
    const mx = EDGE_MARGIN_FRAC * procW, my = EDGE_MARGIN_FRAC * procH;
    inside = bx0 > mx && by0 > my && bx1 < procW - mx && by1 < procH - my;
    const ga = (guide.x1 - guide.x0) * (guide.y1 - guide.y0);
    fill = ga > 0 ? quadArea(corners) / ga : 0;
    aligned = type === 1 && conf >= CONF_MIN && inside && fill >= FILL_MIN && fill <= FILL_MAX;
  }
  latest = { found, corners, type, conf, inside, fill, aligned };

  const now = performance.now();
  if (aligned) {
    if (!alignedSince) alignedSince = now;
    if (autoChk.checked && now - alignedSince >= HOLD_MS && now - lastSnap > COOLDOWN_MS) {
      capture(true);
      lastSnap = now;
      alignedSince = 0;
    }
  } else {
    alignedSince = 0;
  }
}

function quadArea(c) {
  let a = 0;
  for (let i = 0; i < 4; i++) {
    const [x1, y1] = c[i], [x2, y2] = c[(i + 1) % 4];
    a += x1 * y2 - x2 * y1;
  }
  return Math.abs(a * 0.5);
}

// --- capture: perspective-rectify the detected quad --------------------------
snapBtn.onclick = () => { if (streaming) capture(false); };

function capture(auto) {
  if (!latest || !latest.found) { setStatus('未检测到证件', false); return; }
  if (!warpCard) { setStatus('引擎未就绪', false); return; }

  const vw = video.videoWidth, vh = video.videoHeight;
  full.width = vw; full.height = vh;
  fctx.drawImage(video, 0, 0, vw, vh);
  const fd = fctx.getImageData(0, 0, vw, vh);
  const src = new Uint8Array(fd.data.buffer, fd.data.byteOffset, fd.data.length);

  // Map proc-space corners back to full-resolution pixels.
  const sx = vw / procW, sy = vh / procH;
  const c = new Float32Array(8);
  for (let i = 0; i < 4; i++) {
    c[i * 2] = latest.corners[i][0] * sx;
    c[i * 2 + 1] = latest.corners[i][1] * sy;
  }

  const warped = warpCard(src, vw, vh, c, OUT_W, OUT_H);
  if (!warped || warped.length !== OUT_W * OUT_H * 4) { setStatus('校正失败', false); return; }

  shot.width = OUT_W; shot.height = OUT_H;
  sctx.putImageData(new ImageData(new Uint8ClampedArray(warped), OUT_W, OUT_H), 0, 0);
  shotMeta.innerHTML =
    `<b>证件（已透视校正）</b> · 置信度 ${(latest.conf * 100) | 0}% · ${auto ? '自动' : '手动'} · ${OUT_W}×${OUT_H}`;
  dlBtn.disabled = false;
  flash();
}

dlBtn.onclick = () => {
  if (!shot.width) return;
  shot.toBlob((b) => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(b);
    a.download = `edgesnap-${Date.now()}.png`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }, 'image/png');
};

// --- overlay: spotlight mask + brackets + portrait box + detected quad -------
function drawOverlay() {
  const W = overlay.width, H = overlay.height;
  octx.clearRect(0, 0, W, H);
  if (!procW || !procH) return;

  const sx = W / procW, sy = H / procH;
  const gx = guide.x0 * sx, gy = guide.y0 * sy;
  const gw = (guide.x1 - guide.x0) * sx, gh = (guide.y1 - guide.y0) * sy;
  const r = Math.min(gw, gh) * 0.05;
  const k = Math.min(gw, gh);

  octx.fillStyle = 'rgba(0,0,0,0.55)';
  octx.fillRect(0, 0, W, H);
  octx.save();
  octx.globalCompositeOperation = 'destination-out';
  octx.fillStyle = '#000';
  octx.beginPath();
  octx.roundRect(gx, gy, gw, gh, r);
  octx.fill();
  octx.restore();

  const aligned = !!(latest && latest.aligned);
  const detected = !!(latest && latest.found && latest.type === 1);

  octx.strokeStyle = 'rgba(255,255,255,0.14)';
  octx.lineWidth = Math.max(1, W * 0.0015);
  octx.beginPath();
  octx.roundRect(gx, gy, gw, gh, r);
  octx.stroke();

  // Portrait alignment box + silhouette.
  const pbW = gw * 0.20, pbH = gh * 0.64;
  const pbx = PORTRAIT_SIDE === 'right' ? gx + gw - pbW - gw * 0.06 : gx + gw * 0.06;
  const pby = gy + (gh - pbH) / 2;
  octx.strokeStyle = aligned ? 'rgba(0,229,160,0.85)' : 'rgba(255,255,255,0.6)';
  octx.lineWidth = Math.max(1.5, W * 0.0022);
  octx.setLineDash([Math.max(5, W * 0.012), Math.max(4, W * 0.009)]);
  octx.beginPath();
  octx.roundRect(pbx, pby, pbW, pbH, Math.min(pbW, pbH) * 0.14);
  octx.stroke();
  octx.setLineDash([]);
  drawPerson(pbx + pbW / 2, pby, pbW, pbH);

  // Rounded corner brackets.
  const L = k * 0.16, brd = k * 0.07;
  octx.lineWidth = Math.max(4, W * 0.007);
  octx.lineCap = 'round';
  octx.lineJoin = 'round';
  octx.strokeStyle = aligned ? '#00e5a0' : '#8fe6c0';
  arcBracket(gx, gy, 1, 1, L, brd);
  arcBracket(gx + gw, gy, -1, 1, L, brd);
  arcBracket(gx + gw, gy + gh, -1, -1, L, brd);
  arcBracket(gx, gy + gh, 1, -1, L, brd);

  // The real detected card quad: green aligned, yellow inside-not-aligned, red incomplete.
  if (latest && latest.found) {
    const pts = latest.corners.map(([x, y]) => [x * sx, y * sy]);
    octx.lineWidth = Math.max(2, W * 0.005);
    octx.lineCap = 'round';
    octx.lineJoin = 'round';
    octx.strokeStyle = aligned ? '#00e5a0' : (detected && latest.inside ? '#ffce4d' : '#ff6b6b');
    octx.beginPath();
    pts.forEach(([x, y], i) => (i ? octx.lineTo(x, y) : octx.moveTo(x, y)));
    octx.closePath();
    octx.stroke();
    octx.fillStyle = '#fff';
    for (const [x, y] of pts) {
      octx.beginPath();
      octx.arc(x, y, Math.max(3, W * 0.006), 0, Math.PI * 2);
      octx.fill();
    }
  }
}

function arcBracket(cx, cy, sx, sy, L, r) {
  octx.beginPath();
  octx.moveTo(cx, cy + sy * L);
  octx.arcTo(cx, cy, cx + sx * L, cy, r);
  octx.lineTo(cx + sx * L, cy);
  octx.stroke();
}

function drawPerson(cx, boxTop, bw, bh) {
  octx.fillStyle = 'rgba(255,255,255,0.45)';
  octx.beginPath();
  octx.arc(cx, boxTop + bh * 0.34, bw * 0.22, 0, Math.PI * 2);
  octx.fill();
  octx.beginPath();
  octx.ellipse(cx, boxTop + bh * 0.92, bw * 0.42, bh * 0.30, 0, Math.PI, 2 * Math.PI);
  octx.fill();
}

// --- hud ---------------------------------------------------------------------
function updateHud() {
  const l = latest;
  let msg, locked = false;
  if (!streaming) msg = '点击「开启摄像头」';
  else if (!l || !l.found) msg = '请将身份证放入框内';
  else if (l.type !== 1) msg = '检测到文档，请放入证件';
  else if (!l.inside) msg = '证件超出取景框，请整张移入';
  else if (l.fill < FILL_MIN) msg = '靠近一点，让证件填满取景框';
  else if (!l.aligned) msg = '对齐中…';
  else { msg = '对齐成功，保持稳定…'; locked = true; }
  setStatus(msg, locked);

  pill.classList.toggle('lock', locked);
  pillText.textContent = !autoChk.checked ? '自动拍照已关闭'
    : locked ? '对齐成功，正在拍照' : '对齐后自动拍照';

  const held = alignedSince ? performance.now() - alignedSince : 0;
  const p = Math.min(1, held / HOLD_MS);
  progFill.style.width = p * 100 + '%';
  progFill.dataset.ready = p >= 1 ? '1' : '0';
}

function setStatus(text, on) {
  statusEl.textContent = text;
  statusEl.dataset.on = on ? '1' : '0';
}

function flash() {
  flashEl.classList.add('on');
  setTimeout(() => flashEl.classList.remove('on'), 160);
}
