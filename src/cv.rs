//! Traditional computer-vision pipeline that finds a document / ID-card
//! quadrilateral in a single camera frame and rectifies it.
//!
//! Dependency-free and `no-wasm`: the same code compiles natively for
//! `cargo test` and to `wasm32-unknown-unknown` for the browser. The browser
//! bindings live in `lib.rs`; this module knows nothing about JS.
//!
//! Detection: RGBA -> grayscale -> blur -> Canny (non-max suppression +
//! hysteresis) -> dilate -> largest edge component -> convex hull ->
//! Douglas-Peucker to 4 corners -> validate + ID-1 aspect classification.
//!
//! Rectification: 4-point homography (Gaussian elimination) + bilinear warp.

/// A detected quadrilateral plus a coarse classification.
#[derive(Debug, Clone, Copy)]
pub struct Detection {
    pub found: bool,
    /// Corners in input-pixel coordinates, ordered TL, TR, BR, BL.
    pub corners: [(f32, f32); 4],
    /// 0 = none/unknown, 1 = id_card (ID-1 / CR80 aspect), 2 = document.
    pub type_code: u8,
    /// Heuristic confidence in `0.0..=1.0`.
    pub confidence: f32,
}

impl Detection {
    fn none() -> Self {
        Detection {
            found: false,
            corners: [(0.0, 0.0); 4],
            type_code: 0,
            confidence: 0.0,
        }
    }
}

/// Tunable pipeline parameters.
#[derive(Debug, Clone, Copy)]
pub struct Params {
    /// Fraction of *edge* pixels treated as strong in Canny (high threshold).
    pub strong_frac: f32,
    /// Minimum quad area as a fraction of the whole frame.
    pub min_area_frac: f32,
    /// Maximum quad area as a fraction of the whole frame.
    pub max_area_frac: f32,
    /// Minimum edge-pixel count for a connected component to be considered.
    pub min_component_px: usize,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            strong_frac: 0.30,
            min_area_frac: 0.10,
            max_area_frac: 0.97,
            min_component_px: 60,
        }
    }
}

// ID-1 / CR80 card (身份证, 银行卡): 85.6mm x 54mm => 1.585 aspect ratio.
const ID1_ASPECT: f32 = 1.585;
const ID1_TOL: f32 = 0.14;

/// Run the full detection pipeline over one RGBA frame.
pub fn detect(rgba: &[u8], w: usize, h: usize, p: &Params) -> Detection {
    if w < 16 || h < 16 || rgba.len() < w * h * 4 {
        return Detection::none();
    }

    let gray = to_gray(rgba, w, h);
    let blurred = box_blur3(&gray, w, h);
    let edges = canny(&blurred, w, h, p.strong_frac);
    let dilated = dilate3(&edges, w, h);

    let pts = match largest_component_points(&dilated, w, h, p.min_component_px) {
        Some(v) => v,
        None => return Detection::none(),
    };
    let hull = convex_hull(&pts);
    if hull.len() < 4 {
        return Detection::none();
    }

    let quad = quad_from_hull(&hull).unwrap_or_else(|| extreme_corners(&hull));
    let corners = order_corners(quad);

    let area_frac = quad_area(&corners) / (w * h) as f32;
    if area_frac < p.min_area_frac || area_frac > p.max_area_frac {
        return Detection::none();
    }

    let (aspect, angle_score, side_score) = quad_metrics(&corners);
    if angle_score <= 0.0 || side_score <= 0.0 {
        return Detection::none();
    }

    let aspect_err = (aspect - ID1_ASPECT).abs();
    let is_id = aspect_err <= ID1_TOL;
    let aspect_match = if is_id {
        1.0
    } else {
        (1.0 - (aspect_err - ID1_TOL)).clamp(0.0, 1.0)
    };
    let type_code = if is_id { 1 } else { 2 };
    let confidence =
        (0.45 * angle_score + 0.35 * side_score + 0.20 * aspect_match).clamp(0.0, 1.0);

    Detection {
        found: true,
        corners,
        type_code,
        confidence,
    }
}

// --- pixel stages ------------------------------------------------------------

fn to_gray(rgba: &[u8], w: usize, h: usize) -> Vec<f32> {
    let mut g = vec![0.0f32; w * h];
    for (i, px) in rgba.chunks_exact(4).take(w * h).enumerate() {
        g[i] = 0.299 * px[0] as f32 + 0.587 * px[1] as f32 + 0.114 * px[2] as f32;
    }
    g
}

/// Separable 3x3 box blur with clamped borders.
fn box_blur3(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut tmp = vec![0.0f32; w * h];
    for y in 0..h {
        let row = y * w;
        for x in 0..w {
            let x0 = x.saturating_sub(1);
            let x2 = (x + 1).min(w - 1);
            tmp[row + x] = (src[row + x0] + src[row + x] + src[row + x2]) / 3.0;
        }
    }
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        let y0 = y.saturating_sub(1);
        let y2 = (y + 1).min(h - 1);
        for x in 0..w {
            out[y * w + x] = (tmp[y0 * w + x] + tmp[y * w + x] + tmp[y2 * w + x]) / 3.0;
        }
    }
    out
}

/// Magnitude threshold that keeps the top `keep_frac` of *non-zero* gradients.
fn nonzero_percentile(mag: &[f32], keep_frac: f32) -> f32 {
    let maxv = mag.iter().copied().fold(0.0f32, f32::max);
    if maxv <= 0.0 {
        return f32::INFINITY;
    }
    const BINS: usize = 256;
    let scale = (BINS as f32 - 1.0) / maxv;
    let mut hist = [0usize; BINS];
    let mut nz = 0usize;
    for &m in mag {
        if m > 0.0 {
            hist[((m * scale) as usize).min(BINS - 1)] += 1;
            nz += 1;
        }
    }
    if nz == 0 {
        return f32::INFINITY;
    }
    let target = ((keep_frac * nz as f32) as usize).max(1);
    let mut acc = 0usize;
    let mut bin = BINS;
    while bin > 0 {
        bin -= 1;
        acc += hist[bin];
        if acc >= target {
            break;
        }
    }
    bin as f32 / scale
}

/// Canny edge detector: Sobel gradients -> non-maximum suppression ->
/// double threshold with hysteresis. Returns a thin binary edge map.
fn canny(gray: &[f32], w: usize, h: usize, strong_frac: f32) -> Vec<bool> {
    let mut gx = vec![0.0f32; w * h];
    let mut gy = vec![0.0f32; w * h];
    let mut mag = vec![0.0f32; w * h];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            let a = gray[i - w - 1];
            let b = gray[i - w];
            let c = gray[i - w + 1];
            let d = gray[i - 1];
            let f = gray[i + 1];
            let g = gray[i + w - 1];
            let hh = gray[i + w];
            let k = gray[i + w + 1];
            let sx = (c + 2.0 * f + k) - (a + 2.0 * d + g);
            let sy = (g + 2.0 * hh + k) - (a + 2.0 * b + c);
            gx[i] = sx;
            gy[i] = sy;
            mag[i] = (sx * sx + sy * sy).sqrt();
        }
    }

    let high = nonzero_percentile(&mag, strong_frac);
    let low = high * 0.4;

    // Non-maximum suppression along the gradient direction.
    let mut nms = vec![0.0f32; w * h];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            let m = mag[i];
            if m < low {
                continue;
            }
            let ang = (gy[i].atan2(gx[i]).to_degrees() + 180.0) % 180.0;
            let (m1, m2) = if ang < 22.5 || ang >= 157.5 {
                (mag[i - 1], mag[i + 1])
            } else if ang < 67.5 {
                (mag[i - w + 1], mag[i + w - 1])
            } else if ang < 112.5 {
                (mag[i - w], mag[i + w])
            } else {
                (mag[i - w - 1], mag[i + w + 1])
            };
            if m >= m1 && m >= m2 {
                nms[i] = m;
            }
        }
    }

    // Hysteresis: keep strong edges and any weak edge connected to a strong one.
    let mut state = vec![0u8; w * h]; // 1 = weak, 2 = strong
    let mut stack = Vec::new();
    for i in 0..w * h {
        if nms[i] >= high {
            state[i] = 2;
            stack.push(i);
        } else if nms[i] >= low {
            state[i] = 1;
        }
    }
    let mut out = vec![false; w * h];
    while let Some(i) = stack.pop() {
        if out[i] {
            continue;
        }
        out[i] = true;
        let x = i % w;
        let y = i / w;
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                    continue;
                }
                let ni = ny as usize * w + nx as usize;
                if !out[ni] && state[ni] >= 1 {
                    stack.push(ni);
                }
            }
        }
    }
    out
}

/// 3x3 binary dilation. Closes small gaps so a card outline forms one component.
fn dilate3(src: &[bool], w: usize, h: usize) -> Vec<bool> {
    let mut out = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            if !src[y * w + x] {
                continue;
            }
            let y0 = y.saturating_sub(1);
            let y2 = (y + 1).min(h - 1);
            let x0 = x.saturating_sub(1);
            let x2 = (x + 1).min(w - 1);
            for ny in y0..=y2 {
                for nx in x0..=x2 {
                    out[ny * w + nx] = true;
                }
            }
        }
    }
    out
}

/// Flood-fill (8-connected) every edge component, return the pixels of the one
/// with the largest bounding-box area above `min_px` pixels.
fn largest_component_points(
    edges: &[bool],
    w: usize,
    h: usize,
    min_px: usize,
) -> Option<Vec<(i32, i32)>> {
    let n = w * h;
    let mut visited = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    let mut cur: Vec<(i32, i32)> = Vec::new();
    let mut best: Option<(f32, Vec<(i32, i32)>)> = None;

    for start in 0..n {
        if !edges[start] || visited[start] {
            continue;
        }
        visited[start] = true;
        stack.clear();
        cur.clear();
        stack.push(start);
        let (mut minx, mut maxx, mut miny, mut maxy) = (w, 0usize, h, 0usize);

        while let Some(idx) = stack.pop() {
            let x = idx % w;
            let y = idx / w;
            cur.push((x as i32, y as i32));
            minx = minx.min(x);
            maxx = maxx.max(x);
            miny = miny.min(y);
            maxy = maxy.max(y);
            let x0 = x > 0;
            let x1 = x + 1 < w;
            let y0 = y > 0;
            let y1 = y + 1 < h;
            let push = |i: usize, visited: &mut [bool], stack: &mut Vec<usize>| {
                if edges[i] && !visited[i] {
                    visited[i] = true;
                    stack.push(i);
                }
            };
            if y0 {
                push(idx - w, &mut visited, &mut stack);
                if x0 {
                    push(idx - w - 1, &mut visited, &mut stack);
                }
                if x1 {
                    push(idx - w + 1, &mut visited, &mut stack);
                }
            }
            if y1 {
                push(idx + w, &mut visited, &mut stack);
                if x0 {
                    push(idx + w - 1, &mut visited, &mut stack);
                }
                if x1 {
                    push(idx + w + 1, &mut visited, &mut stack);
                }
            }
            if x0 {
                push(idx - 1, &mut visited, &mut stack);
            }
            if x1 {
                push(idx + 1, &mut visited, &mut stack);
            }
        }

        if cur.len() < min_px {
            continue;
        }
        let area = ((maxx - minx + 1) * (maxy - miny + 1)) as f32;
        match &best {
            Some((ba, _)) if *ba >= area => {}
            _ => best = Some((area, cur.clone())),
        }
    }

    best.map(|(_, p)| p)
}

// --- geometry ----------------------------------------------------------------

/// Convex hull (Andrew's monotone chain). Returns hull vertices in order,
/// collinear points removed.
fn convex_hull(pts: &[(i32, i32)]) -> Vec<(f32, f32)> {
    let mut p = pts.to_vec();
    p.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    p.dedup();
    let n = p.len();
    if n < 3 {
        return p.iter().map(|&(x, y)| (x as f32, y as f32)).collect();
    }
    let cross = |o: (i32, i32), a: (i32, i32), b: (i32, i32)| {
        (a.0 - o.0) as i64 * (b.1 - o.1) as i64 - (a.1 - o.1) as i64 * (b.0 - o.0) as i64
    };
    let mut hull: Vec<(i32, i32)> = Vec::with_capacity(2 * n);
    for &pt in &p {
        while hull.len() >= 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], pt) <= 0 {
            hull.pop();
        }
        hull.push(pt);
    }
    let lower = hull.len() + 1;
    for &pt in p.iter().rev() {
        while hull.len() >= lower && cross(hull[hull.len() - 2], hull[hull.len() - 1], pt) <= 0 {
            hull.pop();
        }
        hull.push(pt);
    }
    hull.pop();
    hull.iter().map(|&(x, y)| (x as f32, y as f32)).collect()
}

fn dist(a: (f32, f32), b: (f32, f32)) -> f32 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    (dx * dx + dy * dy).sqrt()
}

fn polyline_len(p: &[(f32, f32)]) -> f32 {
    let mut s = 0.0;
    for i in 1..p.len() {
        s += dist(p[i - 1], p[i]);
    }
    s
}

/// Perpendicular distance from `p` to the line through `a` and `b`.
fn perp_dist(p: (f32, f32), a: (f32, f32), b: (f32, f32)) -> f32 {
    let dx = b.0 - a.0;
    let dy = b.1 - a.1;
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1e-6 {
        return dist(p, a);
    }
    (dx * (a.1 - p.1) - (a.0 - p.0) * dy).abs() / len
}

/// Douglas-Peucker polyline simplification (keeps the first and last point).
fn dp(pts: &[(f32, f32)], eps: f32, out: &mut Vec<(f32, f32)>) {
    let n = pts.len();
    if n < 2 {
        if n == 1 {
            out.push(pts[0]);
        }
        return;
    }
    let (a, b) = (pts[0], pts[n - 1]);
    let mut dmax = 0.0f32;
    let mut idx = 0;
    for i in 1..n - 1 {
        let d = perp_dist(pts[i], a, b);
        if d > dmax {
            dmax = d;
            idx = i;
        }
    }
    if dmax > eps {
        let mut left = Vec::new();
        dp(&pts[..=idx], eps, &mut left);
        let mut right = Vec::new();
        dp(&pts[idx..], eps, &mut right);
        out.extend_from_slice(&left);
        out.extend_from_slice(&right[1..]); // drop the shared join vertex
    } else {
        out.push(a);
        out.push(b);
    }
}

/// Reduce a convex hull to exactly four corners via Douglas-Peucker, trying a
/// few tolerances. `None` if no tolerance yields a convex quad.
fn quad_from_hull(hull: &[(f32, f32)]) -> Option<[(f32, f32); 4]> {
    if hull.len() < 4 {
        return None;
    }
    let mut closed = hull.to_vec();
    closed.push(hull[0]);
    let perim = polyline_len(&closed);
    for &frac in &[0.02f32, 0.03, 0.05, 0.08, 0.10] {
        let mut simp = Vec::new();
        dp(&closed, frac * perim, &mut simp);
        if simp.len() >= 2 && simp[0] == *simp.last().unwrap() {
            simp.pop();
        }
        simp.dedup();
        if simp.len() == 4 && is_convex(&simp) {
            return Some([simp[0], simp[1], simp[2], simp[3]]);
        }
    }
    None
}

fn is_convex(p: &[(f32, f32)]) -> bool {
    let n = p.len();
    if n < 4 {
        return false;
    }
    let mut sign = 0i32;
    for i in 0..n {
        let a = p[i];
        let b = p[(i + 1) % n];
        let c = p[(i + 2) % n];
        let cr = (b.0 - a.0) * (c.1 - b.1) - (b.1 - a.1) * (c.0 - b.0);
        let s = if cr > 0.0 {
            1
        } else if cr < 0.0 {
            -1
        } else {
            0
        };
        if s != 0 {
            if sign == 0 {
                sign = s;
            } else if s != sign {
                return false;
            }
        }
    }
    true
}

/// The four extreme points (min/max of x±y) of a point set — a robust fallback
/// quad when polygon approximation does not land on exactly four corners.
fn extreme_corners(pts: &[(f32, f32)]) -> [(f32, f32); 4] {
    let mut tl = pts[0];
    let mut tr = pts[0];
    let mut br = pts[0];
    let mut bl = pts[0];
    let (mut mnp, mut mxp, mut mnm, mut mxm) =
        (f32::INFINITY, f32::NEG_INFINITY, f32::INFINITY, f32::NEG_INFINITY);
    for &(x, y) in pts {
        let plus = x + y;
        let minus = x - y;
        if plus < mnp {
            mnp = plus;
            tl = (x, y);
        }
        if plus > mxp {
            mxp = plus;
            br = (x, y);
        }
        if minus < mnm {
            mnm = minus;
            bl = (x, y);
        }
        if minus > mxm {
            mxm = minus;
            tr = (x, y);
        }
    }
    [tl, tr, br, bl]
}

/// Order four arbitrary corners as TL, TR, BR, BL using x±y extremes.
fn order_corners(q: [(f32, f32); 4]) -> [(f32, f32); 4] {
    let mut tl = q[0];
    let mut tr = q[0];
    let mut br = q[0];
    let mut bl = q[0];
    let (mut mnp, mut mxp, mut mnm, mut mxm) =
        (f32::INFINITY, f32::NEG_INFINITY, f32::INFINITY, f32::NEG_INFINITY);
    for &(x, y) in &q {
        let plus = x + y;
        let minus = x - y;
        if plus < mnp {
            mnp = plus;
            tl = (x, y);
        }
        if plus > mxp {
            mxp = plus;
            br = (x, y);
        }
        if minus < mnm {
            mnm = minus;
            bl = (x, y);
        }
        if minus > mxm {
            mxm = minus;
            tr = (x, y);
        }
    }
    [tl, tr, br, bl]
}

/// Shoelace area of a TL,TR,BR,BL quad.
fn quad_area(c: &[(f32, f32); 4]) -> f32 {
    let mut a = 0.0;
    for i in 0..4 {
        let (x1, y1) = c[i];
        let (x2, y2) = c[(i + 1) % 4];
        a += x1 * y2 - x2 * y1;
    }
    (a * 0.5).abs()
}

/// Returns `(aspect_ratio, angle_score, side_score)`.
fn quad_metrics(c: &[(f32, f32); 4]) -> (f32, f32, f32) {
    let top = dist(c[0], c[1]);
    let right = dist(c[1], c[2]);
    let bottom = dist(c[2], c[3]);
    let left = dist(c[3], c[0]);

    let width = (top + bottom) * 0.5;
    let height = (left + right) * 0.5;
    if width <= 1.0 || height <= 1.0 {
        return (1.0, 0.0, 0.0);
    }
    let aspect = width.max(height) / width.min(height);

    let s_h = 1.0 - (top - bottom).abs() / (top + bottom);
    let s_v = 1.0 - (left - right).abs() / (left + right);
    let side_score = (s_h * s_v).clamp(0.0, 1.0);

    let mut ang_dev = 0.0;
    for i in 0..4 {
        let prev = c[(i + 3) % 4];
        let cur = c[i];
        let next = c[(i + 1) % 4];
        let v1 = (prev.0 - cur.0, prev.1 - cur.1);
        let v2 = (next.0 - cur.0, next.1 - cur.1);
        let m1 = (v1.0 * v1.0 + v1.1 * v1.1).sqrt();
        let m2 = (v2.0 * v2.0 + v2.1 * v2.1).sqrt();
        if m1 < 1e-3 || m2 < 1e-3 {
            return (aspect, 0.0, side_score);
        }
        let cosv = ((v1.0 * v2.0 + v1.1 * v2.1) / (m1 * m2)).clamp(-1.0, 1.0);
        ang_dev += (cosv.acos().to_degrees() - 90.0).abs();
    }
    let angle_score = (1.0 - (ang_dev / 4.0) / 45.0).clamp(0.0, 1.0);

    (aspect, angle_score, side_score)
}

// --- perspective rectification ----------------------------------------------

/// Solve the NxN linear system `a x = b` by Gaussian elimination with partial
/// pivoting. Returns `None` if the matrix is singular.
fn solve_linear<const N: usize>(a: &mut [[f64; N]; N], b: &mut [f64; N]) -> Option<[f64; N]> {
    for col in 0..N {
        let mut piv = col;
        for r in (col + 1)..N {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        if a[piv][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        let d = a[col][col];
        for r in 0..N {
            if r == col {
                continue;
            }
            let f = a[r][col] / d;
            if f != 0.0 {
                for c in col..N {
                    a[r][c] -= f * a[col][c];
                }
                b[r] -= f * b[col];
            }
        }
    }
    let mut x = [0.0; N];
    for i in 0..N {
        x[i] = b[i] / a[i][i];
    }
    Some(x)
}

/// 3x3 projective transform mapping each `from[i]` to `to[i]`, returned as the 8
/// free parameters `[a,b,c,d,e,f,g,h]` (9th fixed at 1).
fn homography(from: &[(f32, f32); 4], to: &[(f32, f32); 4]) -> Option<[f64; 8]> {
    let mut m = [[0.0f64; 8]; 8];
    let mut r = [0.0f64; 8];
    for i in 0..4 {
        let (x, y) = (from[i].0 as f64, from[i].1 as f64);
        let (u, v) = (to[i].0 as f64, to[i].1 as f64);
        m[i * 2] = [x, y, 1.0, 0.0, 0.0, 0.0, -x * u, -y * u];
        r[i * 2] = u;
        m[i * 2 + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -x * v, -y * v];
        r[i * 2 + 1] = v;
    }
    solve_linear::<8>(&mut m, &mut r)
}

/// Bilinear RGBA sample at floating-point `(x, y)`, edge-clamped.
fn sample_bilinear(rgba: &[u8], w: usize, h: usize, x: f32, y: f32) -> [u8; 4] {
    let x = x.clamp(0.0, (w - 1) as f32);
    let y = y.clamp(0.0, (h - 1) as f32);
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let mut out = [0u8; 4];
    for c in 0..4 {
        let p00 = rgba[(y0 * w + x0) * 4 + c] as f32;
        let p10 = rgba[(y0 * w + x1) * 4 + c] as f32;
        let p01 = rgba[(y1 * w + x0) * 4 + c] as f32;
        let p11 = rgba[(y1 * w + x1) * 4 + c] as f32;
        let top = p00 + (p10 - p00) * fx;
        let bot = p01 + (p11 - p01) * fx;
        out[c] = (top + (bot - top) * fy).round().clamp(0.0, 255.0) as u8;
    }
    out
}

/// Perspective-rectify the quadrilateral `src_quad` (corners ordered TL,TR,BR,BL
/// in input-pixel coordinates) into an upright `out_w x out_h` RGBA image.
pub fn warp_perspective(
    rgba: &[u8],
    w: usize,
    h: usize,
    src_quad: &[(f32, f32); 4],
    out_w: usize,
    out_h: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; out_w * out_h * 4];
    if w < 2 || h < 2 || out_w == 0 || out_h == 0 || rgba.len() < w * h * 4 {
        return out;
    }
    let dst = [
        (0.0, 0.0),
        (out_w as f32 - 1.0, 0.0),
        (out_w as f32 - 1.0, out_h as f32 - 1.0),
        (0.0, out_h as f32 - 1.0),
    ];
    let hm = match homography(&dst, src_quad) {
        Some(m) => m,
        None => return out,
    };
    for oy in 0..out_h {
        for ox in 0..out_w {
            let xf = ox as f64;
            let yf = oy as f64;
            let denom = hm[6] * xf + hm[7] * yf + 1.0;
            if denom.abs() < 1e-9 {
                continue;
            }
            let sx = ((hm[0] * xf + hm[1] * yf + hm[2]) / denom) as f32;
            let sy = ((hm[3] * xf + hm[4] * yf + hm[5]) / denom) as f32;
            let px = sample_bilinear(rgba, w, h, sx, sy);
            let oi = (oy * out_w + ox) * 4;
            out[oi..oi + 4].copy_from_slice(&px);
        }
    }
    out
}

// --- tests -------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Dark frame with one bright filled rectangle.
    fn make_card(w: usize, h: usize, rx: usize, ry: usize, rw: usize, rh: usize) -> Vec<u8> {
        let mut img = vec![0u8; w * h * 4];
        for px in img.chunks_exact_mut(4) {
            px[0] = 20;
            px[1] = 20;
            px[2] = 20;
            px[3] = 255;
        }
        for y in ry..ry + rh {
            for x in rx..rx + rw {
                let i = (y * w + x) * 4;
                img[i] = 230;
                img[i + 1] = 230;
                img[i + 2] = 230;
                img[i + 3] = 255;
            }
        }
        img
    }

    #[test]
    fn detects_id_card_aspect() {
        let (w, h) = (480, 360);
        let (rw, rh) = (254, 160); // 254/160 = 1.5875, an ID-1 aspect
        let (rx, ry) = ((w - rw) / 2, (h - rh) / 2);
        let img = make_card(w, h, rx, ry, rw, rh);

        let det = detect(&img, w, h, &Params::default());
        assert!(det.found, "expected a detection");
        assert_eq!(det.type_code, 1, "expected id_card classification");
        assert!(det.confidence > 0.6, "confidence too low: {}", det.confidence);

        let tl = det.corners[0];
        assert!(
            (tl.0 - rx as f32).abs() < 8.0 && (tl.1 - ry as f32).abs() < 8.0,
            "TL corner off: {:?} (want ~({},{}))",
            tl,
            rx,
            ry
        );
        let br = det.corners[2];
        assert!(
            (br.0 - (rx + rw - 1) as f32).abs() < 8.0 && (br.1 - (ry + rh - 1) as f32).abs() < 8.0,
            "BR corner off: {:?}",
            br
        );
    }

    #[test]
    fn wide_rectangle_classified_as_document() {
        let (w, h) = (480, 360);
        let (rw, rh) = (300, 150); // aspect 2.0 -> document, not an ID card
        let (rx, ry) = ((w - rw) / 2, (h - rh) / 2);
        let img = make_card(w, h, rx, ry, rw, rh);

        let det = detect(&img, w, h, &Params::default());
        assert!(det.found);
        assert_eq!(det.type_code, 2, "expected document classification");
    }

    #[test]
    fn flat_frame_yields_no_detection() {
        let (w, h) = (320, 240);
        let img = vec![20u8; w * h * 4];
        let det = detect(&img, w, h, &Params::default());
        assert!(!det.found);
    }

    #[test]
    fn rejects_undersized_input() {
        let det = detect(&[0u8; 4], 1, 1, &Params::default());
        assert!(!det.found);
    }

    #[test]
    fn warp_rectifies_full_rect() {
        // R grows left->right, G grows top->bottom.
        let (w, h) = (100, 100);
        let mut img = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 4;
                img[i] = (x * 255 / (w - 1)) as u8;
                img[i + 1] = (y * 255 / (h - 1)) as u8;
                img[i + 3] = 255;
            }
        }
        let src = [(0.0, 0.0), (99.0, 0.0), (99.0, 99.0), (0.0, 99.0)];
        let out = warp_perspective(&img, w, h, &src, 50, 50);
        let tl = &out[0..4];
        assert!(tl[0] < 30 && tl[1] < 30, "TL should be the dark corner: {:?}", tl);
        let br = &out[(49 * 50 + 49) * 4..(49 * 50 + 49) * 4 + 4];
        assert!(br[0] > 200 && br[1] > 200, "BR should be the bright corner: {:?}", br);
    }
}
