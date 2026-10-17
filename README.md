# Skolyoz · Cobb Ölçüm Aracı

Omurga grafisinde (röntgen) Cobb açısını ölçmeye ve metal/implant varlığını
işaretlemeye yardımcı olan, **tamamen tarayıcıda çalışan** tek sayfalık web aracı.
Online Skolyoz Kampı (skolyozftr.com) için geliştirilmiştir.

> Araştırma/eğitim amaçlı bir **asistandır**, tıbbi tanı aracı değildir.
> Ölçüm ve karar hekime aittir. Otomatik ölçüm, ağır eğrilerde gerçeğinden
> küçük çıkabilir; her sonuç hekim tarafından kontrol edilmelidir.

## Nasıl çalışır (önemli)

- **Sunucu yok, API yok.** Her şey ziyaretçinin tarayıcısında çalışır.
- **Cobb ölçümü:** saf JavaScript (son-plak yöntemi, görüntünün özgün
  koordinatlarında). Model önerir, hekim kenara alınmış tutamaklarla düzeltir,
  ölçümü onaylar.
- **İmplant/metal kontrolü:** küçük bir sinir ağı (ResNet-18, VinDr-SpineXR
  "Surgical implant" etiketiyle eğitildi) ONNX olarak **tarayıcıda**
  (onnxruntime-web, WebAssembly) ve ayrı bir Web Worker içinde çalışır — bu
  sayede ölçüm arayüzü hiç donmaz. Dışarıya hiçbir veri gönderilmez.

Çıkarım için dış sunucuya istek **gitmez**; dolayısıyla kullanım başına
API/çıkarım ücreti yoktur.

## Dosyalar

| Dosya | Açıklama |
|---|---|
| `index.html` | Uygulamanın tamamı (arayüz + ölçüm motoru). Örnek film ve "yaklaşık ölçüm" gömülüdür. |
| `implant_worker.js` | İmplant çıkarımını arka planda çalıştıran Web Worker. |
| `implant_int8.onnx` | İmplant sınıflandırıcı (int8, ~11 MB). |
| `ort.min.js`, `ort-wasm-simd.wasm`, `ort-wasm.wasm` | onnxruntime-web 1.17.3 çalışma zamanı. |

## Yayınlama (deploy)

Statik dosyalardır; herhangi bir statik barındırmada çalışır
(GitHub Pages, Netlify, Vercel, Cloudflare Pages veya kendi sunucun).
Tüm dosyaları aynı klasöre koyup `index.html`'i aç; derleme adımı yoktur.
Sunucunun `.wasm` dosyalarını `application/wasm`, `.onnx`'u `application/octet-stream`
olarak sunması yeterlidir (çoğu barındırma bunu kendiliğinden yapar).

İlk açılışta tarayıcı ~30 MB (model + çalışma zamanı) indirir; sonrası
önbellekten gelir.

## Modeller ve lisanslar

- İmplant sınıflandırıcı: VinDr-SpineXR veri kümesi ile eğitildi. VinDr-SpineXR'ın
  kendi (PhysioNet kimlik-doğrulamalı, araştırma amaçlı) kullanım koşulları geçerlidir.
- Cobb açısı için kullanılan açık model önerileri araştırma paketinden gelir
  (örn. yijingru/Vertebra-Landmark-Detection — MIT; mazurowski/Scoliosis_project — Apache-2.0).
- **Ticari/halka açık sürüm öncesi** veri ve model lisanslarının ve klinik
  validasyon + regülasyon (TİTCK/CE) gereksinimlerinin ayrıca gözden geçirilmesi gerekir.

## Örnek görsel ve lisans

`index.html` içine gömülü örnek film, Wikimedia Commons'tan alınan açık lisanslı
bir skolyoz röntgenidir: **File:Scoliosis_recklinghausen.jpg**
(https://commons.wikimedia.org/wiki/File:Scoliosis_recklinghausen.jpg) — önden (PA),
tüm omurga, T4–T10 ~75°. Üzerindeki Cobb çizgileri örnek/temsilidir; hekim düzeltir.
Depoda hasta kimliği içeren hiçbir veri yoktur.

⚠️ Yayına almadan önce bu görselin **lisansını ve gerekli atıf metnini** Commons
sayfasının "Licensing" bölümünden teyit edip buraya (ve lisans gerektiriyorsa
sayfanın altına görünür bir künye olarak) ekleyin.
