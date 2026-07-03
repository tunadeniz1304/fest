# FTR Raporu için Gerçek Sonuç/Metrik Malzemesi

> Bu dosya, raporu YAZARKEN kullanacağın gerçek (uydurma olmayan) verileri içerir.
> Rapor şablonu: Proje Özeti(5) / Veri Seti(20) / YZ Çözümü(50) / Sınama(20) / Kaynakça(5).
> **Sen kendi cümlelerinle yazacaksın** (AI raporu eleme sebebi). Bu sadece ham malzeme.

## SİSTEM MİMARİSİ (Çözüm Mimarisi + Detayları bölümü için)

Modüler, çok-katmanlı çıkarım pipeline'ı (sıfırdan eğitim yerine entegrasyon):

```
video → frame sampling (~5fps, FPS-robust)
  → YOLOv8m (COCO) + ByteTrack tracking
       → track-level temporal voting (çoğunluk oyu)
  Katmanlar:
   - Araç: tip (bbox oran heuristiği) + renk (OpenCV HSV) + plaka (EasyOCR + 4-nokta perspektif + karakter oylama)
   - COCO nesneleri: telefon→telefonla_konusma, şişe→su_icme, laptop→bilgisayar
   - Yolcular: person tespiti + konum heuristiği + voting
   - DMS (MediaPipe Face Mesh): esneme (MAR), arkaya_bakma/etrafa_bakinma (yaw)
   - Slalom: ByteTrack trajectory lateral zig-zag (eğitimsiz)
   - teknocan: YOLO-World open-vocabulary (gömülü embedding, offline)
   - sigara_icme: YOLOv8n fine-tune (driver_smoking, CC BY 4.0)
  → tespit dedup → results.json (şartname şeması)
```

## KANITLANMIŞ TEKNİK GERÇEKLER (Sınama + Mimari için)

- **Docker imajı: 4.5GB** (imaj.tar) — 8GB limitin altında ✓
- **Offline çalışma: KANITLANDI** — `docker run --network none` ile internet kapalıyken
  gerçek 4K trafik videosu işlendi, geçerli results.json üretildi, çökme yok.
- **Şema doğruluğu: %100** — programatik validator (src/labels.py) ile her çıktı doğrulanıyor;
  ASCII-safe küçük harf etiketler, confidence_score anahtarı birebir, TR plaka regex.
- **Test kapsamı: 29 birim testi** (TDD ile yazıldı), tracking/voting/dedup/slalom/şema.
- **Robustluk:** her katman izole try/except — biri çökse sistem ayakta (E2E'de kanıtlandı:
  teknocan dill eksikken atlandı ama ana akış geçerli çıktı üretti).
- **Gerçek video testi:** Roboflow trafik videosu (4K, 21s) → araç: pickup/gri (conf 0.87),
  slalom tespiti (conf 1.0), şema hatası 0.

## PRECISION/RECALL ARTIRAN TEKNİKLER (Çözüm Detayları için — kaynaklı)

- **Track-level temporal voting:** tek-frame yanlış pozitifleri track boyunca eler (precision↑),
  track buffer kaçan kareleri köprüler (recall↑). [arXiv 2402.09241]
- **ByteTrack** (sabit kamera için ideal). [Ultralytics docs]
- **Plaka karakter-bazlı oylama:** O↔0, l↔1 OCR hatalarını çoğunlukla düzeltir.
- **4-nokta perspektif düzeltme:** eğik plakada OCR doğruluğu ~+%3. [perspective rectification çalışması]
- **Sınıf-bazlı conf eşiği:** küçük nesne düşük eşik (recall), büyük nesne yüksek (precision).
- **Ticari DMS mimarisi replikası:** Cipia/Seeing Machines iki-katmanlı yaklaşımı
  (landmark + head-pose → fizyolojik durum) açık kaynakla. [EE Times/Semicast]

## VERİ SETLERİ (Veri Seti bölümü 20p — komite veri vermiyor, açık kaynak)

| Görev | Veri seti | Lisans | Kaynak |
|---|---|---|---|
| In-cabin nesne | COCO (YOLOv8m hazır) | - | docs.ultralytics.com |
| sigara_icme | driver_smoking (312 görsel) | CC BY 4.0 | Roboflow |
| Plaka | License Plate Recognition (10125 görsel) | CC BY 4.0 | Roboflow |
| Renk | VCoR (~10500 görsel, 15 sınıf) | CC BY 4.0 | Kaggle |
| Sürücü davranışı | State Farm Distracted Driver | yarışma | Kaggle |

DMD/AUC kullanılmadı (ticari lisans uygun değil — dürüstçe belirt).

## SINAMA BÖLÜMÜ İÇİN (20p — gerçek metrikler)

- sigara_icme fine-tune (YOLOv8n, driver_smoking, CPU/5 epoch baseline):
  precision %40, recall %31, mAP50 %26 — **bu bir başlangıç; final etabında GPU + daha çok
  epoch ile artırılacak.** (Dürüstçe: baseline rakam, SOTA değil.)
- Docker offline + 4.5GB + 10dk-altı (T4'te) doğrulandı.
- [10 test videosu kalite raporu: data/output/kalite_raporu.json — batch_kalite.py çıktısı]

## DÜRÜSTLÜK NOTLARI (rapor için kritik)
- Benchmark rakamları domain-specific: "literatürde %X bildirilmiş" de, "bizim sistem %X" deme.
- FTR-vs-final ayrımını açıkça yaz: neyi şimdi yaptık, neyi final etabına bıraktık.
- 5G QoD entegrasyonu: FTR'de mimari/plan olarak anlat (kod final etabında).
