# Veri Seti & Test Videosu İndirme Talimatı

> Docker doğrulaması bittikten sonra kullanılacak. Kaggle/Roboflow setleri
> giriş gerektirir → **manuel indirme** (Claude otomatik indiremez).
> Kaynak: `deepresearch2.md` (proje kökü). Tüm Roboflow setleri için indirmeden
> önce projenin **License** alanını doğrula (çoğu CC BY 4.0).

İndirdiğin her dosyayı şu klasöre koy: `teknofest_model/data/samples/` (test) veya
`teknofest_model/datasets/<isim>/` (fine-tune). Sonra Claude'a dosya adını söyle.

---

## 1) TEST VİDEOLARI (önce bunlar — modeli ölçmek için)

### a) Araç içi sürücü videosu (DMS testi: esneme/telefon/bakma)
- **Pexels**: https://www.pexels.com/search/videos/driver/ → "Free Download" (720p yeterli)
  - Ara: `driver`, `person driving`, `driver phone`
- **Pixabay**: https://pixabay.com/videos/search/driver/
- Hedef: yüzü net görünen, 10-30 sn, MP4. → `data/samples/surucu.mp4`

### b) Plaka yakın çekim videosu (plaka OCR testi)
- **Pexels**: https://www.pexels.com/search/videos/license%20plate/
- Hedef: plaka okunaklı, araç yakın. → `data/samples/plaka.mp4`

---

## 2) FINE-TUNE VERİ SETLERİ (eksik 2 etiket: sigara_icme, emniyet_kemeri_ihlali)

### a) Roboflow "Driver Monitoring / DMS" (EN ÖNCELİKLİ)
- Roboflow Universe'de ara: **"driver monitoring seatbelt smoking"**
  - Örnek: "DRIVER MENTORING" (~8047 görsel: drinking/drowsy/seatbelt/smoking/phone)
- İndirme: Roboflow hesabı (ücretsiz) → "Download Dataset" → format **YOLOv8**
- Zip'i aç → `datasets/dms/` içine koy (data.yaml + train/ + valid/)
- Lisans: CC BY 4.0 olduğunu doğrula

### b) State Farm Distracted Driver (opsiyonel, geniş RGB)
- Kaggle: https://www.kaggle.com/c/state-farm-distracted-driver-detection/data
- Kaggle hesabı + yarışma kurallarını kabul → indir (~4GB)

---

## 3) PLAKA FINE-TUNE (opsiyonel)
- Roboflow "License Plate Recognition" (10125 görsel, CC BY 4.0):
  https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e
- TR'ye özel: Kaggle "smaildurcan/turkish-license-plate-dataset"

---

## 4) ARAÇ RENK/TİP (opsiyonel)
- Renk: Kaggle "landrykezebou/vcor-vehicle-color-recognition-dataset" (15 sınıf)
- Tip: Roboflow "vehicle-classification-v2" (21 sınıf, TR tiplerine yakın)

---

## Notlar
- DMD ve AUC veri setleri **ticari kullanım için uygun DEĞİL** → kullanma.
- Etiketler ASCII-safe küçük harf olmalı (sistem gereği).
- İndirme yapamazsan/giriş sorunu olursa Claude'a söyle, alternatif bulalım.
