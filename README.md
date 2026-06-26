# TEKNOFEST 2026 — 5G & Yapay Zeka ile Akıllı Yol Güvenliği

Yol kenarı / araç içi video akışından **araç (tip, plaka, renk)**, **sürücü davranışları**
(telefon, su içme, sigara vb.), **araç içi nesneler** ve **yolcu konumları** tespiti yapan
yapay zekâ çıkarım (inference) sistemi.

Bu repo, yarışmanın **Final Tasarım Raporu (FTR)** aşaması için geliştirilen YZ modelinin
Docker'a paketlenmiş çıkarım pipeline'ını içerir.

## Mimari

```
video.mp4
  └─ frame sampling (~5 fps, FPS-robust)
       └─ YOLOv8m (COCO, sıfır eğitim — entegrasyon yaklaşımı)
            ├─ araç (car/truck/bus) → tip heuristiği + renk (HSV) + plaka (kırp → EasyOCR → TR regex)
            ├─ cell phone → telefonla_konusma
            ├─ bottle    → su_icme
            ├─ laptop    → bilgisayar
            └─ person    → yolcular (konum heuristiği)
  └─ results.json  (şartname şemasına birebir uyumlu, ASCII-safe, küçük harf)
```

Tasarım ilkeleri:
- **Çökmez:** her adım `try/except`; tespit yoksa boş liste, geçerli şema yine yazılır.
- **Offline:** ağırlıklar imaja gömülüdür, çalışma anında internet kapalı.
- **Süre güvenli:** 9 dk wall-clock guard (10 dk timeout'tan önce eldekini yazar).
- **Şema garantili:** `src/labels.py` tek doğruluk kaynağı; çıktı programatik doğrulanır.

## Proje Yapısı

```
Dockerfile            # nvidia/cuda:12.1.0-base, T4, <8GB
main.py               # giriş noktası: video oku → çıkarım → results.json
src/
  labels.py           # şartname etiketleri + şema validator
  predict.py          # YOLO + OCR + renk pipeline
  utils.py            # renk(HSV), TR plaka regex, araç tipi heuristiği, frame sampling
weights/              # best_model.pt (YOLO) + easyocr/ (download_weights.py ile üretilir)
requirements.txt
scripts/download_weights.py   # ağırlıkları build öncesi indirir
```

## Kurulum & Çalıştırma

### 1. Ağırlıkları indir (internet açıkken, build öncesi — bir kez)
```bash
python scripts/download_weights.py
```

### 2. Docker imajını oluştur
```bash
docker build -t teknofest/yol-guvenligi:latest .
docker images teknofest/yol-guvenligi:latest   # boyut < 8GB doğrula
```

### 3. Çalıştır (değerlendirme ortamı simülasyonu — offline)
```bash
docker run --rm --gpus all --network none --shm-size=2g \
  -v "$(pwd)/data/input/video.mp4:/app/data/input/video.mp4" \
  -v "$(pwd)/data/output:/app/data/output" \
  teknofest/yol-guvenligi:latest
```

Çıktı: `data/output/results.json`

### 4. Teslim arşivi
```bash
docker save -o imaj.tar teknofest/yol-guvenligi:latest
```

## Çıktı Formatı

```json
{
  "video_id": "video.mp4",
  "arac_bilgisi": { "tip": "sedan", "plaka": "34ABC123", "renk": "beyaz", "confidence_score": 0.94 },
  "tespitler": [
    { "zaman_saniye": 14.5, "kategori": "sofor_eylemi", "etiket": "telefonla_konusma", "confidence_score": 0.89 }
  ]
}
```

## Kısıtlar (değerlendirme sunucusu)

| Özellik | Değer |
|---|---|
| GPU | NVIDIA Tesla T4 |
| vCPU / RAM / SHM | 4 / 16 GB / 2 GB |
| Base image | nvidia/cuda:12.1.0-base-ubuntu22.04 |
| Maks. imaj | 8 GB |
| Maks. çalışma | 10 dk |
| Çalışma anı internet | KAPALI |
