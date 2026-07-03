<div align="center">

# 🚗 SynchAgent — 5G & Yapay Zekâ ile Akıllı Yol Güvenliği

**TEKNOFEST 2026 · 5G ve Yapay Zekâ ile Akıllı Yol Güvenliği Yarışması · Final Tasarım Raporu (FTR)**

Araç içi / yol kenarı video akışından **araç kimliği** (tip · plaka · renk), **sürücü davranışları**
(telefon, sigara, su içme, esneme, bakınma, slalom), **araç içi nesneler** ve **yolcu konumlarını**
tespit eden; tamamen **çevrimdışı**, **çökmez** ve **süre-güvenli** bir yapay zekâ çıkarım sistemi.

<br/>

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?logo=pytorch&logoColor=white)
![Ultralytics](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?logo=yolo&logoColor=black)
![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?logo=nvidia&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-T4%20GPU-2496ED?logo=docker&logoColor=white)
![Offline](https://img.shields.io/badge/Runtime-Offline-critical)
![Takım](https://img.shields.io/badge/Takım-SynchAgent-blueviolet)

</div>

---

## 📑 İçindekiler

- [Genel Bakış](#-genel-bakış)
- [Öne Çıkan Özellikler](#-öne-çıkan-özellikler)
- [Mimari](#-mimari)
- [Tespit Katmanları](#-tespit-katmanları)
- [Çıktı Formatı](#-çıktı-formatı)
- [Proje Yapısı](#-proje-yapısı)
- [Kurulum ve Çalıştırma](#-kurulum-ve-çalıştırma)
- [Değerlendirme Ortamı Kısıtları](#-değerlendirme-ortamı-kısıtları)
- [Tasarım İlkeleri](#-tasarım-i̇lkeleri)
- [Araştırma ve Kaynaklar](#-araştırma-ve-kaynaklar)
- [Takım](#-takım)

---

## 🎯 Genel Bakış

Bu depo, yarışmanın **Final Tasarım Raporu (FTR)** aşaması için geliştirilen yapay zekâ modelinin,
Docker imajına paketlenmiş çıkarım (inference) pipeline'ını içerir. Sistem tek bir video dosyasını
girdi olarak alır ve şartname şemasına birebir uyumlu bir `results.json` üretir.

Tasarımın merkezinde üç kısıt vardır: değerlendirme sunucusunda **internet kapalıdır**, süreç en fazla
**10 dakikada** bitmelidir ve imaj **8 GB'ı aşmamalıdır**. Bu nedenle tüm model ağırlıkları imaja gömülüdür,
pipeline'a bir wall-clock koruması yerleştirilmiştir ve her modül birbirinden izole `try/except`
katmanlarına ayrılmıştır — herhangi bir alt modül çökse dahi ana akış geçerli bir çıktı yazmaya devam eder.

Yaklaşımımızın ayırt edici yanı, **tek-frame kararlar yerine track-tabanlı geçici (temporal) oylamadır**:
ByteTrack ile her nesne bir `track_id` boyunca izlenir, kararlar tüm track üzerinden çoğunluk oyuyla verilir.
Bu, anlık yanlış-pozitifleri (motion blur, OCR hatası, anlık parlama) elerken (**precision ↑**), track
tamponu sayesinde kaçan kareleri köprüler (**recall ↑**).

## ✨ Öne Çıkan Özellikler

| Özellik | Açıklama |
|---|---|
| 🔌 **Tam çevrimdışı** | Tüm ağırlıklar imaja gömülü; çalışma anında ağ erişimi gerekmez (`YOLO_OFFLINE`, `HF_HUB_OFFLINE`). |
| 🛡️ **Çökmez tasarım** | Her tespit katmanı izole `try/except`; hata halinde boş liste döner, ana pipeline etkilenmez. |
| ⏱️ **Süre güvenli** | 9 dk wall-clock guard — 10 dk timeout'tan önce eldeki sonucu yazar. |
| 🗳️ **Temporal voting** | ByteTrack + track-boyu çoğunluk oyu; tek-frame yanlış-pozitifleri eler. |
| 📐 **Şema garantili** | `src/labels.py` tek doğruluk kaynağı; çıktı programatik doğrulanır, ASCII-safe ve küçük harf. |
| 🧩 **Modüler katmanlar** | COCO tespiti, MediaPipe DMS, YOLO-World open-vocab ve fine-tune modelleri bağımsız takılıp çıkarılabilir. |
| 🚫 **Anti-cheat temiz** | Ortam / hostname / IP tespiti yok; akış deterministik, saf görsel analiz. |

## 🏗️ Mimari

```text
                          data/input/video.mp4
                                   │
                                   ▼
                ┌──────────────────────────────────────┐
                │  Kare örnekleme (~5 FPS, FPS-robust)   │
                └──────────────────────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────────┐
          ▼                        ▼                             ▼
  ┌───────────────┐     ┌────────────────────┐        ┌────────────────────┐
  │  YOLOv8 +     │     │  İzole tespit       │        │  İnovasyon          │
  │  ByteTrack    │     │  katmanları         │        │  katmanları         │
  │  (COCO)       │     │                     │        │                     │
  │               │     │  • MediaPipe DMS    │        │  • YOLO-World       │
  │  araç·telefon │     │    (esneme, bakınma)│        │    (teknocan,       │
  │  ·su·laptop   │     │  • Kaggle DMS FT    │        │     open-vocab)     │
  │  ·person      │     │    (sigara/telefon) │        │  • slalom (yörünge) │
  └───────────────┘     └────────────────────┘        └────────────────────┘
          │                        │                             │
          └────────────────────────┼─────────────────────────────┘
                                   ▼
                ┌──────────────────────────────────────┐
                │  Temporal voting + dedup + şema doğru. │
                │  (çoğunluk oyu · plaka karakter oyu ·  │
                │   eylem frame-oranı eşiği)             │
                └──────────────────────────────────────┘
                                   │
                                   ▼
                          data/output/results.json
                 (şartname şemasına birebir, ASCII-safe)
```

**Omurga (backbone):** `weights/best_model.pt` üzerinde çalışan **YOLOv8 + ByteTrack** akışı; araç,
telefon, şişe (su), laptop ve kişi tespitini COCO sınıflarından çıkarır ve her nesneye bir `track_id` atar.

**Araç kimliği:** en güvenilir araç track'inden **tip** (geometrik heuristik), **renk** (VCoR fine-tune
modeli varsa onunla, yoksa HSV heuristiği) ve **plaka** (fine-tune plaka lokalizasyonu → perspektif
düzeltme → EasyOCR → TR plaka regex doğrulaması + pozisyon-bazlı karakter oylaması) belirlenir.

## 🧠 Tespit Katmanları

Her katman **izoledir** — ağırlık/video/hata yoksa boş liste döner ve ana pipeline'ı asla etkilemez.

| Katman | Teknoloji | Ürettiği etiketler | Durum |
|---|---|---|---|
| **COCO omurga** | YOLOv8 + ByteTrack | `telefonla_konusma`, `su_icme`, `bilgisayar`, araç, `yolcular` | ✅ Aktif |
| **DMS — kafa/ağız** | MediaPipe Face Mesh + solvePnP (3D yaw) | `esneme`, `arkaya_bakma`, `etrafa_bakinma` | ✅ Aktif |
| **DMS — Kaggle FT** | YOLOv8m fine-tune (Apache 2.0, negatif-dengeli) | `sigara_icme`, `telefonla_konusma` | ✅ Aktif |
| **teknocan** | YOLO-World (open-vocabulary, eğitimsiz) | `teknocan` | ✅ Aktif |
| **slalom** | ByteTrack yörünge analizi (eğitimsiz) | `slalom` | ✅ Aktif |
| **sigara (eski)** | 0-negatif fine-tune | `sigara_icme` | ⚪ Kapalı (interior ezberi) |
| **State Farm** | Distracted-driver sınıflandırıcı | sürücü eylemi | ⚪ Kapalı (leakage / c9 bias) |
| **dms_v4 actions** | fine-tune aksiyon modeli | sürücü eylemi | ⚪ Kapalı (near-dup leakage) |

> **Neden bazı katmanlar kapalı?** Deep-research bulgularımız, bazı hazır/eğitilmiş modellerin
> [frame-level leakage](docs/research/05_leakage-free-distraction.md) ve arka-plan ezberi nedeniyle
> gerçek dashcam görüntüsünde precision'ı düşürdüğünü gösterdi. Bu modeller kod içinde tek bir bayrakla
> (`ENABLE_*`) kapatılmıştır; gerçek veride doğrulanınca geri açılabilirler. Ayrıntı için
> [araştırma notlarına](#-araştırma-ve-kaynaklar) bakın.

### DMS kafa-pozu detayı

Yan/profil kameralarda basit landmark-geometri yaw kestirimi şaşırdığı için, kafa yönü **solvePnP**
(6 nokta 3D-2D eşleme → Rodrigues → Euler) ile hesaplanır; bu, kamera açısına dayanıklıdır. Bakınma
davranışı, kesintisiz yaw serilerinin en uzunundan ve o seride görülen en büyük açıdan
(`arkaya_bakma` > `etrafa_bakinma`) türetilir — böylece arkaya↔etrafa zikzağı seriyi bölmez.

## 📤 Çıktı Formatı

```json
{
  "video_id": "video.mp4",
  "arac_bilgisi": {
    "tip": "sedan",
    "plaka": "34ABC123",
    "renk": "beyaz",
    "confidence_score": 0.94
  },
  "tespitler": [
    { "zaman_saniye": 14.5, "kategori": "sofor_eylemi", "etiket": "telefonla_konusma", "confidence_score": 0.89 },
    { "zaman_saniye": 22.0, "kategori": "nesneler",     "etiket": "teknocan",           "confidence_score": 0.71 }
  ]
}
```

**Şema kuralları** ([`src/labels.py`](src/labels.py) tek doğruluk kaynağıdır):

- Tüm etiketler **ASCII-safe** ve **küçük harf** (Türkçe karakter yasak).
- JSON anahtarları birebir: `confidence_score` (asla `score` / `guven_skoru`).
- Plaka birleşik format (`34ABC123`) ve TR plaka regex'ine uygun normalize edilir.
- Geçerli kategoriler: `sofor_eylemi`, `nesneler`, `yolcular`.

## 📂 Proje Yapısı

```text
teknofest_model/
├── Dockerfile               # nvidia/cuda:12.1.0-base, T4, imaj < 8 GB
├── main.py                  # giriş noktası: video oku → çıkarım → results.json
├── requirements.txt         # pin'li bağımlılıklar (offline build)
├── src/
│   ├── labels.py            # şartname etiketleri + şema validator (tek doğruluk kaynağı)
│   ├── predict.py           # ByteTrack + temporal voting ana pipeline
│   ├── aggregate.py         # çoğunluk oyu · plaka karakter oyu · slalom · dedup
│   ├── dms.py               # MediaPipe Face Mesh + solvePnP (esneme/bakınma)
│   ├── dms_logic.py         # MAR / yaw eşik mantığı
│   ├── dms_kaggle.py        # Kaggle DMS fine-tune (sigara/telefon)
│   ├── openvocab.py         # YOLO-World open-vocab (teknocan)
│   ├── plaka_model.py       # plaka lokalizasyon fine-tune (izole)
│   ├── renk_model.py        # VCoR renk sınıflandırıcı (izole)
│   ├── utils.py             # renk (HSV) · TR plaka regex · araç tipi · kare örnekleme
│   └── ...                  # sigara / statefarm / dms_actions (kapalı katmanlar)
├── scripts/
│   ├── download_weights.py  # ağırlıkları build öncesi indirir
│   └── ...                  # test & kalite scriptleri
├── colab/                   # A100 eğitim notebook'ları (v1 & v2)
├── docs/
│   ├── research/            # deep-research raporları (mimari kararların dayanağı)
│   └── ...                  # FTR malzemeleri, veri indirme talimatı
├── ftr/FTR.tex              # Final Tasarım Raporu (IEEE tek-sütun, XeLaTeX)
├── tests/                   # şema & pipeline testleri
└── weights/                 # ağırlıklar (git'te değil — download_weights.py / Colab üretir)
```

> **Not:** `weights/` altındaki `*.pt` / `*.pth` ağırlıkları büyük olduğu için versiyon kontrolüne
> dâhil edilmez (`.gitignore`). Build öncesi `scripts/download_weights.py` ile üretilir veya Colab
> eğitim çıktısından kopyalanır.

## 🚀 Kurulum ve Çalıştırma

### 1. Ağırlıkları indir (internet açıkken, build öncesi — bir kez)

```bash
python scripts/download_weights.py
```

### 2. Docker imajını oluştur

```bash
docker build -t teknofest/yol-guvenligi:latest .
docker images teknofest/yol-guvenligi:latest   # boyut < 8 GB doğrula
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

## 🖥️ Değerlendirme Ortamı Kısıtları

| Özellik | Değer |
|---|---|
| GPU | NVIDIA Tesla T4 |
| vCPU / RAM / SHM | 4 / 16 GB / 2 GB |
| Base image | `nvidia/cuda:12.1.0-base-ubuntu22.04` |
| Maks. imaj boyutu | 8 GB |
| Maks. çalışma süresi | 10 dk |
| Çalışma anı internet | **KAPALI** |

## 🧭 Tasarım İlkeleri

- **Çökmez:** her adım `try/except`; tespit yoksa boş liste, geçerli şema yine yazılır.
- **Offline:** ağırlıklar imaja gömülüdür, çalışma anında internet kapalıdır.
- **Süre güvenli:** 9 dk wall-clock guard — 10 dk timeout'tan önce eldekini yazar.
- **Şema garantili:** `src/labels.py` tek doğruluk kaynağı; çıktı programatik doğrulanır.
- **İzole katmanlar:** her tespit modülü bağımsız; biri çökse diğerleri ve ana akış çalışmaya devam eder.
- **Deterministik & dürüst:** ortam tespiti/anti-cheat kaçamağı yok; saf görsel analiz.

## 📚 Araştırma ve Kaynaklar

Mimari kararlarımız, [`docs/research/`](docs/research/) altındaki deep-research raporlarına dayanır:

| Doküman | Konu |
|---|---|
| [`01_cv-iyilestirme-genel.md`](docs/research/01_cv-iyilestirme-genel.md) | Genel bilgisayarlı görü iyileştirme yol haritası (ROI önceliklendirme). |
| [`02_veri-setleri-rehberi.md`](docs/research/02_veri-setleri-rehberi.md) | Açık kaynak veri setleri, lisanslar ve pratik birleşimler. |
| [`03_head-pose-gaze-dual-angle.md`](docs/research/03_head-pose-gaze-dual-angle.md) | Ön + profil kamera için dayanıklı kafa-pozu / bakış kestirimi. |
| [`04_yolov8-veri-agirliklari.md`](docs/research/04_yolov8-veri-agirliklari.md) | Sürücü davranışı için YOLOv8 veri setleri ve hazır ağırlıklar. |
| [`05_leakage-free-distraction.md`](docs/research/05_leakage-free-distraction.md) | Leakage'sız dikkat dağınıklığı tespiti (State Farm dersleri). |

## 👥 Takım

**SynchAgent** — TEKNOFEST 2026, 5G & Yapay Zekâ ile Akıllı Yol Güvenliği Yarışması.

| | |
|---|---|
| Takım Adı | SynchAgent |
| Takım ID | 998490 |
| Başvuru ID | 5141062 |

---

<div align="center">
<sub>Bu depo yalnızca yarışma çıkarım pipeline'ını içerir. Eğitim, Colab notebook'ları ve deep-research
notları ilgili klasörlerdedir.</sub>
</div>
