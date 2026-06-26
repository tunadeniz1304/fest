# Precision/Recall + İnovasyon Katmanları Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mevcut tek-frame YOLO baseline'ını, ByteTrack tracking + track-level temporal voting + sınıf-bazlı eşik + few-shot open-vocab (teknocan) katmanlarıyla yükselterek precision/recall'u artıran, jüriye inovasyon olarak sunulabilen bir çıkarım sistemi kurmak.

**Architecture:** `model.track()` (ByteTrack) ile her nesneye kalıcı `track_id` ata; tahminleri track boyunca biriktir; track bitince çoğunluk oyu (cls/tip/renk/koltuk) + karakter-bazlı plaka oylaması + eylem-frame-oranı eşiğiyle nihai karar ver. Tek-frame yanlış pozitifleri temporal redundancy ile ele → precision↑; track buffer ile kaçan frame'leri köprüle → recall↑. teknocan için YOLOE prompt-free open-vocab katmanı eklenir (opsiyonel, izole).

**Tech Stack:** Python 3.11, ultralytics (YOLOv8m + ByteTrack built-in, opsiyonel YOLOE), OpenCV (HSV renk), EasyOCR (plaka), pytest (test).

## Global Constraints

- Tesla T4 GPU; çalışma anında **internet KAPALI** — tüm ağırlıklar imaja gömülü, otomatik indirme kapalı.
- Maks **10 dk** çalışma; imaj **< 8 GB**.
- **Anti-cheat:** ortam tespiti (env/hostname/IP/dosya kontrolü ile farklı davranış) YASAK → tüm akış deterministik, path-based.
- Çıktı şeması birebir: anahtar `confidence_score` (asla `score`/`guven_skoru`); etiketler **ASCII-safe + küçük harf** (Türkçe karakter yok); plaka birleşik TR format (`34ABC123`).
- Geçerli etiketler tek doğruluk kaynağı: `src/labels.py`. Tip ∈ {sedan,suv,hatchback,pickup,minibus,panelvan,kamyon}; renk ∈ {beyaz,siyah,gri,kirmizi,mavi,sari,yesil,turuncu,kahverengi}.
- Pipeline ASLA çökmez: hata olursa boş ama geçerli şema döner.
- Çıktı yolu `/app/data/output/results.json`, girdi `/app/data/input/video.mp4`.

---

### Task 1: Track-bazlı karar fonksiyonları (saf, test edilebilir çekirdek)

Tüm temporal voting mantığını saf fonksiyonlar olarak `src/aggregate.py`'de topla. Saf = I/O yok, model yok → hızlı ve birim test edilebilir. Pipeline bunları çağıracak.

**Files:**
- Create: `src/aggregate.py`
- Test: `tests/test_aggregate.py`
- Create: `tests/__init__.py` (boş)

**Interfaces:**
- Consumes: `src.labels` (sabit listeler), `collections.Counter`
- Produces:
  - `cogunluk_oyu(samples: list[tuple[str, float]], min_count: int = 3, min_ratio: float = 0.3) -> tuple[str | None, float]` — (etiket, ortalama_conf) veya (None, 0.0)
  - `plaka_karakter_oylama(okumalar: list[tuple[str, float]]) -> tuple[str, float]` — karakter-bazlı çoğunluk; (plaka, conf) veya ("", 0.0)
  - `eylem_karari(gorulen_frame: int, track_uzunlugu: int, ort_conf: float, esik_oran: float = 0.3, min_frame: int = 2) -> tuple[bool, float]` — (raporla_mi, conf)

- [ ] **Step 1: tests/__init__.py oluştur (boş paket dosyası)**

```python
```

- [ ] **Step 2: Çoğunluk oyu testini yaz (failing)**

`tests/test_aggregate.py`:
```python
from src.aggregate import cogunluk_oyu, plaka_karakter_oylama, eylem_karari


def test_cogunluk_oyu_net_kazanan():
    samples = [("sedan", 0.9), ("sedan", 0.8), ("suv", 0.4), ("sedan", 0.85)]
    etiket, conf = cogunluk_oyu(samples)
    assert etiket == "sedan"
    assert 0.8 <= conf <= 0.9  # sadece kazanan sinifin ortalama conf'u


def test_cogunluk_oyu_yetersiz_ornek_eler():
    # min_count alti -> gurultu track'i ele (precision korumasi)
    assert cogunluk_oyu([("sedan", 0.9)], min_count=3) == (None, 0.0)


def test_cogunluk_oyu_dusuk_oran_eler():
    # cogunluk orani esigin altinda -> kararsiz, ele
    samples = [("a", 0.9), ("b", 0.9), ("c", 0.9), ("d", 0.9)]
    assert cogunluk_oyu(samples, min_count=3, min_ratio=0.5) == (None, 0.0)
```

- [ ] **Step 3: Testi çalıştır, fail gör**

Run: `python -m pytest tests/test_aggregate.py -k cogunluk -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.aggregate'`

- [ ] **Step 4: cogunluk_oyu implementasyonu yaz**

`src/aggregate.py`:
```python
# -*- coding: utf-8 -*-
"""Track-bazli temporal voting (saf fonksiyonlar)."""
from collections import Counter


def cogunluk_oyu(samples, min_count=3, min_ratio=0.3):
    """
    samples: [(etiket, conf), ...] tek track'in tum frame tahminleri.
    En cok gorulen etiketi dondurur; yeterli destek yoksa (None, 0.0).
    Conf = sadece kazanan etiketin ortalama guveni.
    """
    if not samples or len(samples) < min_count:
        return (None, 0.0)
    etiketler = [e for e, _ in samples]
    top, n = Counter(etiketler).most_common(1)[0]
    if n / len(etiketler) < min_ratio:
        return (None, 0.0)
    confs = [c for e, c in samples if e == top]
    return (top, round(sum(confs) / len(confs), 2))
```

- [ ] **Step 5: Çoğunluk oyu testini geçir**

Run: `python -m pytest tests/test_aggregate.py -k cogunluk -v`
Expected: PASS (3 test)

- [ ] **Step 6: Plaka karakter oylama testini yaz (failing)**

`tests/test_aggregate.py` sonuna ekle:
```python
def test_plaka_karakter_oylama_hatayi_duzeltir():
    # Cogunluk dogru karakteri secer: O->0 hatasini diger okumalar duzeltir
    okumalar = [("34ABC123", 0.8), ("34ABC123", 0.7), ("34ABCl23", 0.5)]
    plaka, conf = plaka_karakter_oylama(okumalar)
    assert plaka == "34ABC123"
    assert conf > 0.0


def test_plaka_karakter_oylama_bos():
    assert plaka_karakter_oylama([]) == ("", 0.0)
```

- [ ] **Step 7: Testi çalıştır, fail gör**

Run: `python -m pytest tests/test_aggregate.py -k plaka -v`
Expected: FAIL — `ImportError: cannot import name 'plaka_karakter_oylama'`

- [ ] **Step 8: plaka_karakter_oylama implementasyonu yaz**

`src/aggregate.py` sonuna ekle:
```python
def plaka_karakter_oylama(okumalar):
    """
    okumalar: [(plaka_str, conf), ...]. En sik gorulen uzunluktaki okumalari
    alip pozisyon-bazli karakter cogunlugu uretir. Tek frame OCR hatasini
    (O<->0, l<->1) cogunluk duzeltir.
    """
    okumalar = [(p, c) for p, c in okumalar if p]
    if not okumalar:
        return ("", 0.0)
    # En sik uzunlugu sec
    uzunluklar = Counter(len(p) for p, _ in okumalar)
    hedef_uzunluk = uzunluklar.most_common(1)[0][0]
    secili = [(p, c) for p, c in okumalar if len(p) == hedef_uzunluk]
    sonuc = []
    for i in range(hedef_uzunluk):
        kar = Counter(p[i] for p, _ in secili)
        sonuc.append(kar.most_common(1)[0][0])
    plaka = "".join(sonuc)
    conf = round(sum(c for _, c in secili) / len(secili), 2)
    return (plaka, conf)
```

- [ ] **Step 9: Plaka testini geçir**

Run: `python -m pytest tests/test_aggregate.py -k plaka -v`
Expected: PASS (2 test)

- [ ] **Step 10: Eylem kararı testini yaz (failing)**

`tests/test_aggregate.py` sonuna ekle:
```python
def test_eylem_karari_anlik_flash_eler():
    # 40 frame track'te eylem sadece 1 frame gorulduyse -> raporlama (precision)
    raporla, _ = eylem_karari(gorulen_frame=1, track_uzunlugu=40, ort_conf=0.9)
    assert raporla is False


def test_eylem_karari_surekli_eylemi_raporlar():
    # 40 frame'in 20'sinde gorulduyse -> raporla
    raporla, conf = eylem_karari(gorulen_frame=20, track_uzunlugu=40, ort_conf=0.85)
    assert raporla is True
    assert conf == 0.85
```

- [ ] **Step 11: Testi çalıştır, fail gör**

Run: `python -m pytest tests/test_aggregate.py -k eylem -v`
Expected: FAIL — `ImportError: cannot import name 'eylem_karari'`

- [ ] **Step 12: eylem_karari implementasyonu yaz**

`src/aggregate.py` sonuna ekle:
```python
def eylem_karari(gorulen_frame, track_uzunlugu, ort_conf, esik_oran=0.3, min_frame=2):
    """
    Bir eylemi (telefon/sigara/su) track boyunca gorulme oranina gore raporla.
    Anlik tek-frame flash'lari eler -> false positive dususu.
    """
    if track_uzunlugu <= 0 or gorulen_frame < min_frame:
        return (False, 0.0)
    if gorulen_frame / track_uzunlugu < esik_oran:
        return (False, 0.0)
    return (True, round(float(ort_conf), 2))
```

- [ ] **Step 13: Tüm aggregate testlerini geçir**

Run: `python -m pytest tests/test_aggregate.py -v`
Expected: PASS (7 test)

- [ ] **Step 14: Commit**

```bash
git add src/aggregate.py tests/test_aggregate.py tests/__init__.py
git commit -m "feat: track-bazli temporal voting cekirdek fonksiyonlari (cogunluk/plaka/eylem)"
```

---

### Task 2: Sınıf-bazlı eşik yapılandırması

Tek global `conf` yerine sınıf-bazlı eşik: küçük/zor nesneler (telefon, şişe) düşük eşik (recall↑), büyük/kolay nesneler (araç, kişi) yüksek eşik (precision↑). Kalibrasyonun pratik faydasını sıfır maliyetle verir.

**Files:**
- Modify: `src/labels.py` (sonuna eşik tablosu ekle)
- Test: `tests/test_thresholds.py`

**Interfaces:**
- Consumes: yok
- Produces:
  - `SINIF_ESIK: dict[str, float]` — COCO sınıf adı → minimum conf
  - `esik_al(coco_label: str) -> float` — tabloda yoksa `VARSAYILAN_ESIK` döner

- [ ] **Step 1: Eşik testini yaz (failing)**

`tests/test_thresholds.py`:
```python
from src.labels import esik_al, VARSAYILAN_ESIK


def test_kucuk_nesne_dusuk_esik():
    # telefon kucuk/zor -> recall icin dusuk esik
    assert esik_al("cell phone") < esik_al("car")


def test_bilinmeyen_sinif_varsayilan():
    assert esik_al("zürafa") == VARSAYILAN_ESIK
```

- [ ] **Step 2: Testi çalıştır, fail gör**

Run: `python -m pytest tests/test_thresholds.py -v`
Expected: FAIL — `ImportError: cannot import name 'esik_al'`

- [ ] **Step 3: Eşik tablosunu ekle**

`src/labels.py` sonuna ekle:
```python
# --- Sinif-bazli conf esikleri (kalibrasyon yerine pratik ayar) ---
VARSAYILAN_ESIK = 0.35
SINIF_ESIK = {
    # Kucuk/zor nesneler -> dusuk esik (recall)
    "cell phone": 0.25,
    "bottle": 0.25,
    "laptop": 0.30,
    # Buyuk/kolay nesneler -> yuksek esik (precision)
    "car": 0.45,
    "truck": 0.45,
    "bus": 0.45,
    "person": 0.40,
}


def esik_al(coco_label):
    return SINIF_ESIK.get(coco_label, VARSAYILAN_ESIK)
```

- [ ] **Step 4: Testi geçir**

Run: `python -m pytest tests/test_thresholds.py -v`
Expected: PASS (2 test)

- [ ] **Step 5: Commit**

```bash
git add src/labels.py tests/test_thresholds.py
git commit -m "feat: sinif-bazli conf esikleri (kucuk nesne recall, buyuk nesne precision)"
```

---

### Task 3: predict.py'yi tracking + voting mimarisine geçir

Tek-frame döngüsünü `model.track(stream=True)` ByteTrack akışına çevir; tahminleri track_id bazında biriktir; Task 1 fonksiyonlarıyla nihai karar üret. Eski tek-frame `run_inference` yerine track-tabanlı sürüm.

**Files:**
- Modify: `src/predict.py` (tüm `run_inference` gövdesi)
- Create: `weights/bytetrack.yaml` (tracker config — stride'a göre düşük buffer)
- Test: `tests/test_predict_smoke.py` (gerçek ağırlık olmadan import + boş-video davranışı)

**Interfaces:**
- Consumes: `src.aggregate.{cogunluk_oyu, plaka_karakter_oylama, eylem_karari}`, `src.labels.esik_al`, `src.utils.*`
- Produces: `run_inference(video_path, weights_path) -> dict` (şema değişmedi; içerik artık track-aggregated)

- [ ] **Step 1: bytetrack.yaml tracker config oluştur**

`weights/bytetrack.yaml`:
```yaml
tracker_type: bytetrack
track_high_thresh: 0.35
track_low_thresh: 0.1
new_track_thresh: 0.5
track_buffer: 12
match_thresh: 0.8
fuse_score: true
```

- [ ] **Step 2: Smoke testi yaz (failing) — açılamayan video boş şema döner**

`tests/test_predict_smoke.py`:
```python
from src.predict import run_inference
from src.labels import validate_results


def test_acilamayan_video_bos_gecerli_sema():
    # Olmayan dosya -> cokmez, gecerli bos sema doner
    sonuc = run_inference("yok_boyle_bir_video.mp4", "yok.pt")
    assert validate_results(sonuc) == []          # sema gecerli
    assert sonuc["tespitler"] == []
    assert sonuc["arac_bilgisi"]["tip"] == ""
```

- [ ] **Step 3: Testi çalıştır, fail gör**

Run: `python -m pytest tests/test_predict_smoke.py -v`
Expected: Mevcut `run_inference` import edilir ama davranış garanti değil — fail veya hata. (Eğer mevcut kod zaten geçerse, Task 3 yine de tracking'e geçirir; testi koruma olarak tut.)

- [ ] **Step 4: predict.py'yi track-tabanlı yeniden yaz**

`src/predict.py` içeriğini şununla değiştir:
```python
# -*- coding: utf-8 -*-
"""Track-tabanli cikarim pipeline: ByteTrack + temporal voting."""
import os
import time
from collections import defaultdict

import cv2

from src.labels import esik_al
from src.aggregate import cogunluk_oyu, plaka_karakter_oylama, eylem_karari
from src.utils import baskin_renk, arac_tipi_heuristik, plaka_normalize

WEIGHTS_DIR = "/app/weights"
YOLO_WEIGHTS = os.path.join(WEIGHTS_DIR, "best_model.pt")
TRACKER_CFG = os.path.join(WEIGHTS_DIR, "bytetrack.yaml")
EASYOCR_DIR = os.path.join(WEIGHTS_DIR, "easyocr")

VID_STRIDE = 5
SURE_GUARD_SN = 9 * 60
COCO_ARAC = {"car", "truck", "bus"}
COCO_KOLAY = {
    "cell phone": ("sofor_eylemi", "telefonla_konusma"),
    "bottle": ("sofor_eylemi", "su_icme"),
    "laptop": ("nesneler", "bilgisayar"),
}


def _bos_sonuc(video_id):
    return {
        "video_id": video_id,
        "arac_bilgisi": {"tip": "", "plaka": "", "renk": "", "confidence_score": 0.0},
        "tespitler": [],
    }


def _lazy_ocr():
    try:
        import easyocr, torch
        return easyocr.Reader(
            ["en"], gpu=torch.cuda.is_available(),
            model_storage_directory=EASYOCR_DIR if os.path.isdir(EASYOCR_DIR) else None,
            download_enabled=False,
        )
    except Exception as e:
        print(f"[OCR] EasyOCR yuklenemedi: {e}")
        return None


def run_inference(video_path, weights_path=YOLO_WEIGHTS):
    video_id = os.path.basename(video_path) if video_path else "video.mp4"
    # Video acilabilir mi? (cokmeden erken don)
    cap = cv2.VideoCapture(video_path)
    if not cap or not cap.isOpened():
        if cap:
            cap.release()
        print(f"[Inference] Video acilamadi: {video_path}")
        return _bos_sonuc(video_id)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    cap.release()

    try:
        from ultralytics import YOLO
        import torch
        model = YOLO(weights_path)
        names = model.names
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        reader = _lazy_ocr()
    except Exception as e:
        print(f"[Inference] Model yuklenemedi: {e}")
        return _bos_sonuc(video_id)

    # track_id -> birikim
    arac_tip = defaultdict(list)      # [(tip, conf)]
    arac_renk = defaultdict(list)     # [(renk, conf)]
    arac_plaka = defaultdict(list)    # [(plaka, conf)]
    arac_conf = defaultdict(list)     # [conf]
    eylem_say = defaultdict(lambda: defaultdict(int))   # tid -> {(kat,etk): frame}
    eylem_conf = defaultdict(lambda: defaultdict(list))
    eylem_zaman = defaultdict(dict)   # tid -> {(kat,etk): ilk_saniye}
    track_len = defaultdict(int)
    yolcu_oy = defaultdict(list)      # tid -> [(etk, conf)]
    yolcu_zaman = {}

    baslangic = time.time()
    try:
        akis = model.track(
            source=video_path, tracker=TRACKER_CFG, persist=True, stream=True,
            conf=0.20, vid_stride=VID_STRIDE, device=dev,
            half=(dev == "cuda"), verbose=False,
        )
        f_idx = 0
        for r in akis:
            if time.time() - baslangic > SURE_GUARD_SN:
                print("[Inference] Sure guard tetiklendi.")
                break
            f_idx += VID_STRIDE
            zaman_sn = round(f_idx / fps, 2)
            if r.boxes is None or r.boxes.id is None:
                continue
            ids = r.boxes.id.int().cpu().tolist()
            clss = r.boxes.cls.int().cpu().tolist()
            confs = r.boxes.conf.cpu().tolist()
            xyxy = r.boxes.xyxy.cpu().tolist()
            frame = r.orig_img
            for tid, c, cf, box in zip(ids, clss, confs, xyxy):
                coco_ad = names.get(c, str(c))
                if cf < esik_al(coco_ad):
                    continue
                track_len[tid] += 1
                x1, y1, x2, y2 = box
                if coco_ad in COCO_ARAC:
                    crop = frame[int(max(0, y1)):int(y2), int(max(0, x1)):int(x2)]
                    arac_tip[tid].append((arac_tipi_heuristik(coco_ad, (x1, y1, x2, y2)), cf))
                    arac_renk[tid].append((baskin_renk(crop), cf))
                    arac_conf[tid].append(cf)
                    if reader is not None:
                        try:
                            for it in reader.readtext(crop, detail=1, paragraph=False):
                                norm = plaka_normalize(it[1])
                                if norm:
                                    arac_plaka[tid].append((norm, float(it[2])))
                        except Exception:
                            pass
                elif coco_ad in COCO_KOLAY:
                    anahtar = COCO_KOLAY[coco_ad]
                    eylem_say[tid][anahtar] += 1
                    eylem_conf[tid][anahtar].append(cf)
                    eylem_zaman[tid].setdefault(anahtar, zaman_sn)
                elif coco_ad == "person":
                    alan = (x2 - x1) * (y2 - y1)
                    etk = "on_koltuk" if alan > 0.20 * frame_w * frame_w else (
                        "arka_koltuk_1" if (x1 + x2) / 2 < frame_w / 2 else "arka_koltuk_2")
                    yolcu_oy[tid].append((etk, cf))
                    yolcu_zaman.setdefault(tid, zaman_sn)
    except Exception as e:
        print(f"[Inference] Track akisi hatasi: {e}")

    # --- Nihai karar: en guvenilir aracdan arac_bilgisi ---
    en_iyi_tid, en_iyi_conf = None, -1.0
    for tid, cs in arac_conf.items():
        ort = sum(cs) / len(cs)
        if ort > en_iyi_conf:
            en_iyi_conf, en_iyi_tid = ort, tid
    arac_bilgisi = {"tip": "", "plaka": "", "renk": "", "confidence_score": 0.0}
    if en_iyi_tid is not None:
        tip, _ = cogunluk_oyu(arac_tip[en_iyi_tid], min_count=1, min_ratio=0.0)
        renk, _ = cogunluk_oyu(arac_renk[en_iyi_tid], min_count=1, min_ratio=0.0)
        plaka, _ = plaka_karakter_oylama(arac_plaka[en_iyi_tid])
        arac_bilgisi = {
            "tip": tip or "", "plaka": plaka or "", "renk": renk or "",
            "confidence_score": round(en_iyi_conf, 2),
        }

    # --- Tespitler: eylem-frame-orani esigi ---
    tespitler = []
    for tid, anahtarlar in eylem_say.items():
        for anahtar, sayi in anahtarlar.items():
            confs = eylem_conf[tid][anahtar]
            ort = sum(confs) / len(confs) if confs else 0.0
            raporla, conf = eylem_karari(sayi, track_len[tid], ort)
            if raporla:
                kat, etk = anahtar
                tespitler.append({
                    "zaman_saniye": eylem_zaman[tid][anahtar],
                    "kategori": kat, "etiket": etk, "confidence_score": conf,
                })
    for tid, oylar in yolcu_oy.items():
        etk, conf = cogunluk_oyu(oylar, min_count=2, min_ratio=0.4)
        if etk:
            tespitler.append({
                "zaman_saniye": yolcu_zaman[tid],
                "kategori": "yolcular", "etiket": etk, "confidence_score": conf,
            })

    tespitler.sort(key=lambda t: t["zaman_saniye"])
    return {"video_id": video_id, "arac_bilgisi": arac_bilgisi, "tespitler": tespitler}
```

- [ ] **Step 5: Smoke testini geçir**

Run: `python -m pytest tests/test_predict_smoke.py -v`
Expected: PASS — açılamayan video boş geçerli şema döner.

- [ ] **Step 6: Dockerfile'a bytetrack.yaml dahil olduğunu doğrula**

`weights/` zaten COPY ediliyor (`COPY weights/ /app/weights/`) → `bytetrack.yaml` otomatik dahil. Değişiklik gerekmez; doğrula.

- [ ] **Step 7: Commit**

```bash
git add src/predict.py weights/bytetrack.yaml tests/test_predict_smoke.py
git commit -m "feat: ByteTrack + temporal voting pipeline (precision/recall katmani)"
```

---

### Task 4: Şema doğrulama testini sağlamlaştır (regresyon kalkanı)

Voting çıktısının her zaman geçerli şema ürettiğini garanti eden bir test. Bu, sonraki değişikliklerde şema bozulmasını yakalar.

**Files:**
- Test: `tests/test_schema_contract.py`

**Interfaces:**
- Consumes: `src.labels.validate_results`, sentetik voting çıktısı
- Produces: yok (sadece test)

- [ ] **Step 1: Şema sözleşme testini yaz (failing)**

`tests/test_schema_contract.py`:
```python
from src.labels import validate_results


def test_gecerli_voting_ciktisi_sema_gecer():
    cikti = {
        "video_id": "video.mp4",
        "arac_bilgisi": {"tip": "sedan", "plaka": "34ABC123", "renk": "beyaz", "confidence_score": 0.9},
        "tespitler": [
            {"zaman_saniye": 1.0, "kategori": "sofor_eylemi", "etiket": "telefonla_konusma", "confidence_score": 0.8},
            {"zaman_saniye": 2.0, "kategori": "yolcular", "etiket": "on_koltuk", "confidence_score": 0.7},
        ],
    }
    assert validate_results(cikti) == []


def test_turkce_karakterli_etiket_yakalanir():
    cikti = {
        "video_id": "v.mp4",
        "arac_bilgisi": {"tip": "", "plaka": "", "renk": "kırmızı", "confidence_score": 0.0},
        "tespitler": [],
    }
    # "kırmızı" (Turkce i) gecersiz -> hata listesi bos olmamali
    assert validate_results(cikti) != []
```

- [ ] **Step 2: Testi çalıştır**

Run: `python -m pytest tests/test_schema_contract.py -v`
Expected: PASS (validate_results zaten Task 0'da mevcut; geçer). Eğer 2. test fail ederse `validate_results` renk kontrolünü düzelt.

- [ ] **Step 3: Tüm test paketini çalıştır**

Run: `python -m pytest tests/ -v`
Expected: PASS (tüm testler)

- [ ] **Step 4: Commit**

```bash
git add tests/test_schema_contract.py
git commit -m "test: sema sozlesme testi (voting ciktisi + Turkce karakter regresyon kalkani)"
```

---

### Task 5: teknocan için YOLOE open-vocab katmanı (İNOVASYON, izole, opsiyonel)

`teknocan` COCO'da yok. YOLOE (Ultralytics 2025) prompt-free open-vocabulary detektörü ile text-prompt ("can", "branded can") tespit dener. İZOLE: ağırlık yoksa veya hata olursa ana pipeline'ı etkilemez (boş döner). Jüriye "few-shot open-vocab inovasyonu" olarak sunulur.

**Files:**
- Create: `src/openvocab.py`
- Modify: `src/predict.py` (run_inference sonuna izole çağrı)
- Modify: `scripts/download_weights.py` (YOLOE ağırlığı indir — opsiyonel)
- Test: `tests/test_openvocab.py`

**Interfaces:**
- Consumes: ultralytics YOLOE (varsa), `weights/yoloe.pt`
- Produces:
  - `teknocan_tespit(video_path: str, vid_stride: int = 15) -> list[dict]` — tespit yoksa veya ağırlık yoksa `[]`; her tespit `{zaman_saniye, kategori:"nesneler", etiket:"teknocan", confidence_score}`

- [ ] **Step 1: İzolasyon testini yaz (failing) — ağırlık yokken boş liste, çökme yok**

`tests/test_openvocab.py`:
```python
from src.openvocab import teknocan_tespit


def test_agirlik_yoksa_bos_liste_cokmez():
    # weights/yoloe.pt yok (test ortami) -> [] doner, exception atmaz
    assert teknocan_tespit("yok.mp4") == []
```

- [ ] **Step 2: Testi çalıştır, fail gör**

Run: `python -m pytest tests/test_openvocab.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.openvocab'`

- [ ] **Step 3: openvocab.py implementasyonu yaz**

`src/openvocab.py`:
```python
# -*- coding: utf-8 -*-
"""teknocan icin YOLOE open-vocabulary tespit (izole, opsiyonel inovasyon katmani)."""
import os

YOLOE_WEIGHTS = "/app/weights/yoloe.pt"
PROMPTLAR = ["can", "beverage can", "branded can"]   # teknocan ~ markali kutu
SURE_STRIDE = 15
CONF_ESIK = 0.30


def teknocan_tespit(video_path, vid_stride=SURE_STRIDE):
    """teknocan tespitlerini dondurur. Agirlik/hata yoksa [] (ana akisi etkilemez)."""
    if not os.path.exists(YOLOE_WEIGHTS) or not os.path.exists(video_path):
        return []
    try:
        from ultralytics import YOLOE
        import torch, cv2
        model = YOLOE(YOLOE_WEIGHTS)
        model.set_classes(PROMPTLAR, model.get_text_pe(PROMPTLAR))
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        sonuc, gorulen = [], False
        for r in model.predict(source=video_path, stream=True, conf=CONF_ESIK,
                               vid_stride=vid_stride, device=dev, verbose=False):
            if gorulen:
                break
            if r.boxes is None or len(r.boxes) == 0:
                continue
            # Ilk guclu tespiti raporla (dedup: tek teknocan yeter)
            cf = float(r.boxes.conf.max())
            sonuc.append({
                "zaman_saniye": 0.0, "kategori": "nesneler",
                "etiket": "teknocan", "confidence_score": round(cf, 2),
            })
            gorulen = True
        return sonuc
    except Exception as e:
        print(f"[OpenVocab] teknocan tespiti atlandi: {e}")
        return []
```

- [ ] **Step 4: İzolasyon testini geçir**

Run: `python -m pytest tests/test_openvocab.py -v`
Expected: PASS — ağırlık yok → `[]`, çökme yok.

- [ ] **Step 5: predict.py'ye izole çağrı ekle**

`src/predict.py` içinde `tespitler.sort(...)` satırından ÖNCE ekle:
```python
    # --- INOVASYON: teknocan open-vocab (izole; hata ana akisi bozmaz) ---
    try:
        from src.openvocab import teknocan_tespit
        tespitler.extend(teknocan_tespit(video_path))
    except Exception as e:
        print(f"[Inference] teknocan katmani atlandi: {e}")
```

- [ ] **Step 6: Smoke + schema testlerini tekrar çalıştır (regresyon yok)**

Run: `python -m pytest tests/ -v`
Expected: PASS (tümü) — teknocan ağırlığı test ortamında yok, çağrı `[]` döner, şema bozulmaz.

- [ ] **Step 7: download_weights.py'ye opsiyonel YOLOE indirme ekle**

`scripts/download_weights.py` içinde `indir_easyocr()` tanımından SONRA, `__main__` bloğundan ÖNCE ekle:
```python
def indir_yoloe():
    """YOLOE prompt-free agirligini indirir (teknocan icin). Basarisizsa atlar."""
    hedef = os.path.join(WEIGHTS, "yoloe.pt")
    if os.path.exists(hedef):
        print(f"[YOLOE] Zaten var: {hedef}")
        return
    try:
        from ultralytics import YOLOE
        import shutil
        m = YOLOE("yoloe-11m-seg.pt")   # indirir
        kaynak = getattr(m, "ckpt_path", None) or "yoloe-11m-seg.pt"
        if os.path.exists(kaynak):
            shutil.copy(kaynak, hedef)
            print(f"[YOLOE] Kaydedildi: {hedef}")
    except Exception as e:
        print(f"[YOLOE] Indirilemedi (teknocan katmani opsiyonel, atlaniyor): {e}")
```
Ve `__main__` bloğunda `indir_easyocr()` çağrısından sonra `indir_yoloe()` ekle.

- [ ] **Step 8: Commit**

```bash
git add src/openvocab.py src/predict.py scripts/download_weights.py tests/test_openvocab.py
git commit -m "feat: teknocan icin YOLOE open-vocab katmani (izole, opsiyonel inovasyon)"
```

---

### Task 6: Uçtan uca yerel doğrulama (gerçek ağırlık + sentetik/örnek video)

Ağırlıkları indirip pipeline'ı gerçek bir video üzerinde koştur; çıktının şemaya uyduğunu ve sürenin makul olduğunu doğrula. Bu, raporun "Sınama" bölümü için kanıt üretir.

**Files:**
- Create: `scripts/make_test_video.py` (örnek video yoksa sentetik 10sn video üret)
- Create: `scripts/verify_e2e.py` (çalıştır + şema doğrula + süre ölç)

**Interfaces:**
- Consumes: `src.predict.run_inference`, `src.labels.validate_results`
- Produces: konsola PASS/FAIL + süre raporu; `data/output/results.json`

- [ ] **Step 1: Sentetik test videosu üretici yaz**

`scripts/make_test_video.py`:
```python
# -*- coding: utf-8 -*-
"""Ornek video yoksa sentetik 10sn test videosu uretir (pipeline cokme testi)."""
import os
import cv2
import numpy as np

OUT = "data/input/video.mp4"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
vw = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), 25, (640, 480))
for i in range(250):  # 10 sn @ 25fps
    f = np.full((480, 640, 3), 60, np.uint8)
    cv2.rectangle(f, (200 + i % 50, 200), (400 + i % 50, 350), (200, 200, 200), -1)
    vw.write(f)
vw.release()
print(f"Sentetik video yazildi: {OUT}")
```

- [ ] **Step 2: Sentetik video üret**

Run: `python scripts/make_test_video.py`
Expected: `Sentetik video yazildi: data/input/video.mp4`

- [ ] **Step 3: E2E doğrulama scripti yaz**

`scripts/verify_e2e.py`:
```python
# -*- coding: utf-8 -*-
"""Pipeline'i gercek agirlikla kostur, sema dogrula, sure olc."""
import os, sys, json, time
sys.path.insert(0, os.path.abspath("."))
from src.predict import run_inference
from src.labels import validate_results

VIDEO = "data/input/video.mp4"
WEIGHTS = "weights/best_model.pt"
os.makedirs("data/output", exist_ok=True)

t0 = time.time()
sonuc = run_inference(VIDEO, WEIGHTS)
sure = time.time() - t0

hatalar = validate_results(sonuc)
with open("data/output/results.json", "w", encoding="utf-8") as f:
    json.dump(sonuc, f, ensure_ascii=True, indent=2)

print(f"Sure: {sure:.1f}s | Tespit: {len(sonuc['tespitler'])} | Sema hatasi: {len(hatalar)}")
for h in hatalar:
    print("  HATA:", h)
sys.exit(0 if not hatalar else 1)
```

- [ ] **Step 4: Ağırlıkları indir (internet açıkken)**

Run: `python scripts/download_weights.py`
Expected: `weights/best_model.pt` ve `weights/easyocr/` oluşur. (YOLOE opsiyonel, başarısız olursa atlanır.)

- [ ] **Step 5: E2E doğrulamayı çalıştır**

Run: `python scripts/verify_e2e.py`
Expected: `Sema hatasi: 0` ve exit code 0. Süre sentetik 10sn video için birkaç saniye olmalı.

- [ ] **Step 6: Commit**

```bash
git add scripts/make_test_video.py scripts/verify_e2e.py
git commit -m "test: uctan uca yerel dogrulama scriptleri (sentetik video + sema/sure)"
```

---

### Task 7: Branch'i main'e entegre et

Feature tamamlandı, testler geçiyor → finishing-a-development-branch ile entegrasyon.

**Files:** yok (git işlemi)

- [ ] **Step 1: Tüm testleri son kez çalıştır**

Run: `python -m pytest tests/ -v`
Expected: PASS (tümü)

- [ ] **Step 2: Feature branch'i push et**

```bash
git push -u origin feat/precision-recall-iyilestirme
```

- [ ] **Step 3: main'e merge (veya PR — kullanıcı tercihi)**

```bash
git checkout main
git merge --no-ff feat/precision-recall-iyilestirme -m "merge: precision/recall + inovasyon katmanlari"
git push origin main
```

---

## Self-Review Notları

- **Spec coverage:** Tracking+voting (precision/recall), sınıf-bazlı eşik, teknocan open-vocab (inovasyon), şema garantisi, offline/8GB/10dk/anti-cheat kısıtları — hepsi task'lara bağlandı. 5G QoD entegrasyonu raporda (FTR) anlatılacak, kod final etabında → bu plan kapsamı dışı (bilinçli).
- **Placeholder yok:** Tüm kod blokları tam.
- **Tip tutarlılığı:** `cogunluk_oyu`/`plaka_karakter_oylama`/`eylem_karari` imzaları tüm task'larda aynı; `run_inference` şeması değişmedi.
- **Riskler:** YOLOE ağırlığı indirilemezse teknocan katmanı izole olarak `[]` döner (ana akış etkilenmez). Voting eşikleri (min_count/min_ratio) gerçek örnek videoda ayarlanmalı — Task 6 bunun zeminini kurar.
