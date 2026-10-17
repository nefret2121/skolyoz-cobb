/* İmplant/metal kontrolü — çıkarım ayrı bir Web Worker içinde çalışır,
   böylece ölçüm arayüzü (ana iş parçacığı) hiç kilitlenmez.
   Model ve ort çalışma zamanı aynı klasörden (./) yüklenir. */
try { importScripts('./ort.min.js'); }
catch (e) { importScripts('https://cdnjs.cloudflare.com/ajax/libs/onnxruntime-web/1.17.3/ort.min.js'); }
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
ort.env.wasm.wasmPaths = './';

let session = null, loadErr = null, loading = null;
function ensure(){
  if (session) return Promise.resolve();
  if (loading) return loading;
  loading = (async () => {
    try { session = await ort.InferenceSession.create('./implant_int8.onnx', { executionProviders: ['wasm'] }); }
    catch (e) { loadErr = '' + ((e && e.message) || e); session = null; }
  })();
  return loading;
}
self.onmessage = async (ev) => {
  const { id, data } = ev.data;
  await ensure();
  if (!session) { self.postMessage({ id, error: loadErr || 'yüklenemedi' }); return; }
  try {
    const t = new ort.Tensor('float32', data, [1, 3, 256, 256]);
    const out = await session.run({ input: t });
    const logit = out[Object.keys(out)[0]].data[0];
    self.postMessage({ id, logit });
  } catch (e) { self.postMessage({ id, error: '' + ((e && e.message) || e) }); }
};
