/* cobb_decode.js — yijingru landmark ONNX çıktısını (hm,reg,wh) ana eğrinin
   uç-plak noktalarına (u1,u2,l1,l2) ve Cobb açısına çevirir.
   ctdet_decode + cobb_angle_calc (yijingru) birebir JS portu. */
(function (global) {
  var HW = 256;   // ısı haritası boyutu (1024 / down_ratio 4)
  var DR = 4;     // down_ratio
  var IN = 1024;  // model girdi boyutu (kare)

  function sigmoidInplace(a) {
    for (var i = 0; i < a.length; i++) a[i] = 1 / (1 + Math.exp(-a[i]));
    return a;
  }

  function maxpool3(heat) { // 3x3, stride1, pad1
    var out = new Float32Array(HW * HW);
    for (var y = 0; y < HW; y++) {
      for (var x = 0; x < HW; x++) {
        var m = -Infinity;
        for (var dy = -1; dy <= 1; dy++) {
          var yy = y + dy; if (yy < 0 || yy >= HW) continue;
          for (var dx = -1; dx <= 1; dx++) {
            var xx = x + dx; if (xx < 0 || xx >= HW) continue;
            var v = heat[yy * HW + xx]; if (v > m) m = v;
          }
        }
        out[y * HW + x] = m;
      }
    }
    return out;
  }

  // hm: Float32Array HW*HW (sigmoid uygulanmış), reg: 2*HW*HW, wh: 8*HW*HW
  // W,H: özgün görüntü boyutu. Döner: {u1,u2,l1,l2,major,n} veya null.
  function decodeCobb(hm, reg, wh, W, H) {
    var chan = HW * HW;
    var hmax = maxpool3(hm);
    var peaks = [];
    for (var i = 0; i < chan; i++) { if (hm[i] === hmax[i]) peaks.push([hm[i], i]); }
    peaks.sort(function (a, b) { return b[0] - a[0]; });
    var K = Math.min(17, peaks.length);
    if (K < 3) return null;

    var rows = [];
    for (var k = 0; k < K; k++) {
      var ind = peaks[k][1], sc = peaks[k][0];
      var xs = (ind % HW) + reg[0 * chan + ind];
      var ys = Math.floor(ind / HW) + reg[1 * chan + ind];
      var w0 = wh[0 * chan + ind], w1 = wh[1 * chan + ind], w2 = wh[2 * chan + ind], w3 = wh[3 * chan + ind],
          w4 = wh[4 * chan + ind], w5 = wh[5 * chan + ind], w6 = wh[6 * chan + ind], w7 = wh[7 * chan + ind];
      rows.push([xs, ys, xs - w0, ys - w1, xs - w2, ys - w3, xs - w4, ys - w5, xs - w6, ys - w7, sc]);
    }
    // ölçek: *DR sonra özgün boyuta
    for (var r = 0; r < rows.length; r++) {
      var row = rows[r];
      for (var j = 0; j < 10; j++) row[j] *= DR;
      for (var j2 = 0; j2 < 10; j2 += 2) { row[j2] = row[j2] / IN * W; row[j2 + 1] = row[j2 + 1] / IN * H; }
    }
    // y'ye göre sırala (üstten alta)
    rows.sort(function (a, b) { return a[1] - b[1]; });
    var n = rows.length;
    var leftmid = [], rightmid = [];
    for (var t = 0; t < n; t++) {
      var rr = rows[t];
      leftmid.push([(rr[2] + rr[6]) / 2, (rr[3] + rr[7]) / 2]);   // (tl+bl)/2
      rightmid.push([(rr[4] + rr[8]) / 2, (rr[5] + rr[9]) / 2]);  // (tr+br)/2
    }
    var vec = [];
    for (var v = 0; v < n; v++) vec.push([rightmid[v][0] - leftmid[v][0], rightmid[v][1] - leftmid[v][1]]);
    function ang(a, b) {
      var u = vec[a], w = vec[b];
      var d = u[0] * w[0] + u[1] * w[1];
      var mu = Math.hypot(u[0], u[1]), mw = Math.hypot(w[0], w[1]);
      if (mu < 1e-9 || mw < 1e-9) return 0;
      var cs = d / (mu * mw); if (cs < 0) cs = 0; if (cs > 1) cs = 1;
      return Math.acos(cs);
    }
    var best = -1, A = 0, B = 0;
    for (var ii = 0; ii < n; ii++) {
      var mv = -1, mj = 0;
      for (var jj = 0; jj < n; jj++) { var a = ang(ii, jj); if (a > mv) { mv = a; mj = jj; } }
      if (mv > best) { best = mv; A = ii; B = mj; }
    }
    var U = A, L = B;
    if (leftmid[A][1] > leftmid[B][1]) { U = B; L = A; }
    return {
      u1: leftmid[U], u2: rightmid[U], l1: leftmid[L], l2: rightmid[L],
      major: best * 180 / Math.PI, n: n
    };
  }

  global.sigmoidInplace = sigmoidInplace;
  global.decodeCobb = decodeCobb;
  if (typeof module !== 'undefined' && module.exports) module.exports = { decodeCobb: decodeCobb, sigmoidInplace: sigmoidInplace };
})(typeof self !== 'undefined' ? self : globalThis);
