/* Otomatik Cobb ölçümü — yijingru landmark modeli ayrı bir Web Worker içinde çalışır,
   böylece ölçüm arayüzü (ana iş parçacığı) kilitlenmez. Çıkarım yavaştır (~20-40 sn),
   bu sırada arayüzde "ölçüm yapılıyor" gösterilir. Çözümleme (nokta→uç plak) burada yapılır. */
try { importScripts('./ort.min.js'); }
catch (e) { importScripts('https://cdnjs.cloudflare.com/ajax/libs/onnxruntime-web/1.17.3/ort.min.js'); }
importScripts('./cobb_decode.js');
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
ort.env.wasm.wasmPaths = './';

let session = null, loadErr = null, loading = null;
function ensure() {
  if (session) return Promise.resolve();
  if (loading) return loading;
  loading = (async () => {
    // Model (48 MB) GitHub Pages'ten (CORS açık) çekilir; böylece Hostinger'a büyük dosya yüklenmez.
    // ort.min.js ve cobb_decode.js aynı origin'den (site ile birlikte) yüklenir.
    // NOT: Model URL'si ort'a doğrudan verilince GitHub Pages'in gzip'i yüzünden
    // parse hatası oluyor; onun yerine baytları kendimiz indirip ort'a veriyoruz.
    const MODEL_URL = 'https://nefret2121.github.io/skolyoz-cobb/cobb_lm_fp16.onnx';
    try {
      const resp = await fetch(MODEL_URL);
      if (!resp.ok) throw new Error('model indirilemedi (HTTP ' + resp.status + ')');
      const bytes = new Uint8Array(await resp.arrayBuffer());
      session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
    }
    catch (e) { loadErr = '' + ((e && e.message) || e); session = null; }
  })();
  return loading;
}

self.onmessage = async (ev) => {
  const { id, data, W, H } = ev.data;   // data: Float32Array [1,3,1024,1024] (BGR, /255-0.5)
  await ensure();
  if (!session) { self.postMessage({ id, error: loadErr || 'model yüklenemedi' }); return; }
  try {
    const t = new ort.Tensor('float32', data, [1, 3, 1024, 1024]);
    const out = await session.run({ input: t });
    const hm = out['hm'].data, reg = out['reg'].data, wh = out['wh'].data;
    sigmoidInplace(hm);
    const r = decodeCobb(hm, reg, wh, W, H);
    if (!r) { self.postMessage({ id, error: 'omur noktaları yeterince tespit edilemedi' }); return; }
    self.postMessage({ id, result: r });
  } catch (e) { self.postMessage({ id, error: '' + ((e && e.message) || e) }); }
};
