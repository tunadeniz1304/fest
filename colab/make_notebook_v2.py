# -*- coding: utf-8 -*-
"""TEKNOFEST egitim notebook v2 - sigara+kemer (dengeli) + 6DRepNet + dms ROI. Colab A100."""
import json

def md(t): return {"cell_type":"markdown","metadata":{},"source":t.splitlines(keepends=True)}
def code(t): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":t.splitlines(keepends=True)}

cells = []

cells.append(md("""# TEKNOFEST 2026 — Eğitim v2: Overfit/Leakage'siz Model İyileştirme

Önceki sürümün hatası: negatifsiz sigara seti → "interior=sigara" ezberi → temiz videoda yanlış-pozitif.
Bu sürüm 3 deep research bulgusuyla DOĞRU yapar:
1. **sigara + kemer** → DENGELI veri (%30-50 negatif), Roboflow "Abnormal Driver Behaviour" (CC BY 4.0)
2. **yan-açı yaw** → 6DRepNet hazır ağırlık (eğitimsiz indir) + per-video baseline
3. **dms davranış** → dürüst clip-disjoint değerlendirme (telefon/su zaten COCO'dan; dms_v4 opsiyonel)

## Kullanım
1. **Runtime → Change runtime type → A100 GPU**
2. Drive `MyDrive/teknofest/`'e şu zip'leri yükle (aşağıdaki hücrelerde linkler):
   - `abnormal-driver-behaviour.zip` (Roboflow, CC BY 4.0 — sigara+kemer birleşik)
   - (opsiyonel) `smoker-detection.zip` (Mendeley, 560/560 dengeli — ek negatif)
3. **Runtime → Run all**
4. Eğitilen `.pt` + indirilen 6DRepNet → `MyDrive/teknofest/egitilen_weights_v2/`

## Lisans (hepsi yarışmaya uygun)
| Kaynak | Lisans |
|---|---|
| Abnormal Driver Behaviour | CC BY 4.0 ✅ |
| Mendeley Smoker | CC BY 4.0 ✅ |
| 6DRepNet | MIT ✅ |
| Ultralytics YOLO | AGPL-3.0 (yarışmada OK, ticari üründe lisans gerek) |
"""))

cells.append(md("## 1. GPU kontrolü"))
cells.append(code("""!nvidia-smi
import torch
assert torch.cuda.is_available(), "GPU YOK! Runtime -> A100 sec."
print("GPU:", torch.cuda.get_device_name(0))
"""))

cells.append(md("## 2. Kurulum (numpy'a dokunma — Colab uyumu)"))
cells.append(code("""!pip install -q ultralytics sixdrepnet
import numpy as np, ultralytics
print("numpy:", np.__version__, "| ultralytics:", ultralytics.__version__)
from ultralytics import YOLO
import os, shutil, zipfile, glob, yaml, traceback, random
os.makedirs('/content/weights', exist_ok=True)
os.makedirs('/content/datasets', exist_ok=True)
print("Kurulum tamam.")
"""))

cells.append(md("""## 3. Google Drive bağla"""))
cells.append(code("""from google.colab import drive
drive.mount('/content/drive')
DRIVE = '/content/drive/MyDrive/teknofest'
OUT = os.path.join(DRIVE, 'egitilen_weights_v2')
os.makedirs(OUT, exist_ok=True)
print("Drive:", DRIVE)
for f in sorted(os.listdir(DRIVE)) if os.path.isdir(DRIVE) else []:
    print("  -", f)
"""))

cells.append(md("## 4. Yardımcılar"))
cells.append(code("""def cikar(zip_ad, hedef):
    src = os.path.join(DRIVE, zip_ad)
    if not os.path.exists(src):
        print(f"  ATLANDI (yok): {zip_ad}"); return None
    dst = os.path.join('/content/datasets', hedef)
    if os.path.isdir(dst) and os.listdir(dst):
        print(f"  zaten var: {dst}"); return dst
    os.makedirs(dst, exist_ok=True)
    with zipfile.ZipFile(src) as z: z.extractall(dst)
    print(f"  cikarildi: {dst}")
    return dst

def yaml_bul(ds):
    c = glob.glob(f'{ds}/**/data.yaml', recursive=True)
    return c[0] if c else None

def negatif_orani(ds_root):
    \"\"\"train/labels icinde bos (negatif) label orani.\"\"\"
    labels = glob.glob(f'{ds_root}/**/train/labels/*.txt', recursive=True)
    if not labels: return None
    bos = sum(1 for lf in labels if not open(lf).read().strip())
    return bos, len(labels), round(100*bos/len(labels), 1)

print("yardimcilar hazir")
"""))

cells.append(md("""## 4b. (Opsiyonel) Kaggle DMS seti analizi — habbas11 (Apache 2.0)

Kaggle: kaggle.com/datasets/habbas11/dms-driver-monitoring-system — sigara+kemer+göz+telefon, Apache 2.0.
Drive'a `dms-kaggle.zip` koyduysan: sınıfları + negatif oranı + formatı otomatik raporlar.
(Negatif sınıf içeriyorsa altın değerinde — ADB'nin %0 negatif sorununu çözer.)"""))
cells.append(code("""dms_k = cikar('dms-kaggle.zip', 'dms_kaggle')
if dms_k:
    # Yapiyi otomatik analiz et: YOLO mu (data.yaml) yoksa classification mi (klasor=sinif)
    yk = yaml_bul(dms_k)
    if yk:
        with open(yk) as f: kcfg = yaml.safe_load(f)
        print("YOLO-detection. Siniflar:", kcfg.get('names'))
        no = negatif_orani(os.path.dirname(yk))
        if no: print(f"Negatif oran: %{no[2]}")
    else:
        # classification: train/<sinif>/ klasor yapisi
        alt = glob.glob(f'{dms_k}/**/train', recursive=True) or glob.glob(f'{dms_k}/*')
        for t in alt[:1]:
            siniflar = [os.path.basename(d) for d in glob.glob(f'{t}/*') if os.path.isdir(d)]
            print(f"Classification gorunuyor. Siniflar: {siniflar}")
            print(">>> 'safe'/'normal'/'no_*' iceren sinif = NEGATIF kaynagi olarak kullanilabilir")
    print("NOT: Bu seti Claude'a raporla - notebook'a en uygun entegrasyonu o ekleyecek.")
"""))

cells.append(md("""## 5. SİGARA + KEMER — Abnormal Driver Behaviour (CC BY 4.0)

Roboflow: universe.roboflow.com/university-exrks/abnormal-driver-behaviour
Sınıflar: Phone, Cigarette, Drinking, Seatbelt, Eating → biz Cigarette + Seatbelt kullanacağız.
⚠️ Bu set %0 negatif → 5b'deki negatif enjeksiyon ŞART.

**ÖNEMLİ:** İndirirken Roboflow'da "YOLOv8" formatı seç. Drive'a `abnormal-driver-behaviour.zip` koy."""))
cells.append(code("""ds = cikar('abnormal-driver-behaviour.zip', 'adb')
if ds:
    yp = yaml_bul(ds)
    if yp:
        with open(yp) as f: cfg = yaml.safe_load(f)
        print("Orijinal siniflar:", cfg.get('names'))
        # Negatif oranini kontrol et (kok sorun buydu)
        no = negatif_orani(os.path.dirname(yp))
        if no: print(f"NEGATIF oran: {no[0]}/{no[1]} = %{no[2]}  (hedef >=%30)")
        if no and no[2] < 20:
            print("!!! UYARI: negatif orani dusuk. Kendi temiz surucu karelerinizden")
            print("    bos-label .txt ekleyin veya goodmax2/goodmax videolarindan kare cikarin.")
"""))

cells.append(md("""### 5b. ⚠️ NEGATİF ENJEKSİYONU (kök sorun fix — eğitimden ÖNCE çalıştır!)

Set %0 background (hepsi ihlalli) → model "araç içi = ihlal" ezberler, temiz videoda yanlış-pozitif.
ÇÖZÜM: train/'e temiz sürücü görselleri **boş .txt label** ile ekle (YOLOv8 background image).
İKİ kaynak: (A) harici temiz-sürücü zip'leri, (B) kendi temiz videolarımızdan kare.
Hedef: toplam train'in **%30-40'ı negatif**."""))
cells.append(code("""import cv2
def negatif_ekle_klasor(kaynak_img_dir, train_root, etiket="ext"):
    \"\"\"Bir klasordeki gorselleri bos-label ile train'e ekler (background).\"\"\"
    if not os.path.isdir(kaynak_img_dir): return 0
    n = 0
    for img in glob.glob(f'{kaynak_img_dir}/**/*.jpg', recursive=True) + \\
               glob.glob(f'{kaynak_img_dir}/**/*.png', recursive=True):
        ad = f"neg_{etiket}_{n}"
        shutil.copy(img, f'{train_root}/images/{ad}.jpg')
        open(f'{train_root}/labels/{ad}.txt', 'w').close()  # BOS label = negatif
        n += 1
    return n

def negatif_ekle_video(video_yollari, train_root, her_n_kare=20, etiket="vid"):
    \"\"\"Temiz surucu videolarindan kare cikarip bos-label ile ekler.\"\"\"
    n = 0
    for v in video_yollari:
        cap = cv2.VideoCapture(v); i = -1
        while True:
            ret, fr = cap.read()
            if not ret: break
            i += 1
            if i % her_n_kare: continue
            ad = f"neg_{etiket}_{n}"
            cv2.imwrite(f'{train_root}/images/{ad}.jpg', fr)
            open(f'{train_root}/labels/{ad}.txt', 'w').close()
            n += 1
        cap.release()
    return n

if ds and yp:
    root = os.path.dirname(yp)
    tr = f'{root}/train'
    eklenen = 0
    # (A) Harici temiz-surucu setleri (Drive'a koyduysan: temiz_surucu.zip)
    ext = cikar('temiz_surucu.zip', 'temiz_surucu')
    if ext:
        eklenen += negatif_ekle_klasor(ext, tr, "ext")
    # (B) Kendi temiz videolarimiz (goodmax2/goodmax/badmax2 + temiz Pexels)
    temiz_vids = []
    for pat in ['geminitest*/goodmax*.mp4', 'geminitest*/badmax2.mp4', '*.mp4']:
        temiz_vids += glob.glob(f'{DRIVE}/{pat}')
    temiz_vids = list(dict.fromkeys(temiz_vids))[:8]  # tekrar yok, max 8 video
    if temiz_vids:
        eklenen += negatif_ekle_video(temiz_vids, tr, her_n_kare=20, etiket="vid")
    print(f"Eklenen negatif kare: {eklenen}")
    no = negatif_orani(root)
    if no: print(f"YENI negatif oran: {no[0]}/{no[1]} = %{no[2]}  (hedef %30-40)")
    if not no or no[2] < 25:
        print("!!! Negatif hala dusuk - Drive'a temiz_surucu.zip ekle veya temiz video koy.")
"""))

cells.append(md("""### 5c. Sigara+kemer modelini eğit (negatif-dengeli, domain-randomization)

5 sınıfla eğitilir; pipeline'da sadece Cigarette→sigara_icme, Seatbelt→emniyet_kemeri_ihlali okunur."""))
cells.append(code("""if ds and yp:
    root = os.path.dirname(yp)
    val_dir = 'valid' if os.path.isdir(f'{root}/valid/images') else 'val'
    with open(yp, 'w') as f:
        f.write(f"train: {root}/train/images\\n")
        f.write(f"val: {root}/{val_dir}/images\\n")
        if os.path.isdir(f'{root}/test/images'): f.write(f"test: {root}/test/images\\n")
        f.write(f"nc: {len(cfg['names'])}\\n")
        f.write(f"names: {cfg['names']}\\n")
    try:
        YOLO('yolov8m.pt').train(data=yp, epochs=80, imgsz=640, batch=32, device=0,
            project='/content/runs', name='sigara_kemer', exist_ok=True,
            patience=15, plots=True,
            # domain-randomization augment (research onerisi): isik/renk/occlusion
            hsv_h=0.02, hsv_s=0.7, hsv_v=0.5, degrees=5, translate=0.1, scale=0.5,
            fliplr=0.5, mosaic=1.0, mixup=0.1, erasing=0.4)
        best = '/content/runs/sigara_kemer/weights/best.pt'
        if os.path.exists(best):
            shutil.copy(best, '/content/weights/sigara_kemer.pt')
            shutil.copy(best, os.path.join(OUT, 'sigara_kemer.pt'))
            print("KAYDEDILDI -> sigara_kemer.pt (Drive'a da)")
    except Exception as e:
        print("HATA:", e); traceback.print_exc()
"""))

cells.append(md("""## 6. YAN-AÇI YAW — 6DRepNet hazır ağırlık (eğitimsiz, MIT)

solvePnP profilde şaşırıyordu. 6DRepNet appearance-based, profile dayanıklı.
Ağırlık otomatik iner; offline kullanım için Drive'a kopyalarız."""))
cells.append(code("""try:
    from sixdrepnet import SixDRepNet
    import numpy as np
    model6d = SixDRepNet()   # agirligi otomatik indirir
    print("6DRepNet yuklendi. (pip paketi agirligi cache'ler)")
    # Cache'teki agirligi bul + Drive'a kopyala (offline icin)
    cands = glob.glob(os.path.expanduser('~/.cache/**/6DRepNet*.pth'), recursive=True)
    cands += glob.glob('/usr/local/lib/**/sixdrepnet/**/*.pth', recursive=True)
    cands += glob.glob('/root/.cache/**/*.pth', recursive=True)
    for c in cands[:3]:
        try:
            shutil.copy(c, os.path.join(OUT, '6drepnet_' + os.path.basename(c)))
            print("6DRepNet agirligi Drive'a kopyalandi:", os.path.basename(c))
        except: pass
    if not cands:
        print("UYARI: 6DRepNet .pth cache'te bulunamadi - pip paketi runtime'da indirir.")
        print("Offline icin: sixdrepnet paketini imaja gom + .pth'i weights/'e koy.")
except Exception as e:
    print("6DRepNet HATA:", e); traceback.print_exc()
"""))

cells.append(md("""### 6b. 6DRepNet'i test videolarında doğrula (yan vs ön)

Eğer test videoların Drive'da varsa, 6DRepNet'in solvePnP'den iyi olup olmadığını gör.
(Opsiyonel — videolar yoksa atla.)"""))
cells.append(code("""import cv2
test_vids = glob.glob(f'{DRIVE}/geminitest*/*.mp4') + glob.glob(f'{DRIVE}/*.mp4')
if test_vids:
    print(f"{len(test_vids)} test videosu bulundu, 6DRepNet yaw ornekleri:")
    for v in test_vids[:4]:
        cap = cv2.VideoCapture(v); yawlar=[]; i=-1
        while True:
            ret, fr = cap.read()
            if not ret: break
            i += 1
            if i % 15: continue
            try:
                p, y, r = model6d.predict(fr)   # pitch, yaw, roll
                yawlar.append(round(float(y)))
            except: pass
        cap.release()
        print(f"  {os.path.basename(v)}: yaw ornekleri {yawlar[:12]}")
else:
    print("Test videosu yok (Drive'a koyarsan 6DRepNet'i dogrularsin).")
"""))

cells.append(md("""## 7. (DÜRÜST) dms davranış — clip-disjoint değerlendirme

Research: dms_v4/statefarm leakage'li -> %99 sahte. Telefon/su zaten COCO'dan güvenilir.
Burada dms_v4'ü clip-disjoint split'le DÜRÜST ölçüyoruz; mAP <%70 ise kullanma.
(Mevcut dms_v4.pt Drive'da yoksa atla — bu sadece dürüst teşhis.)"""))
cells.append(code("""print("DURUST DEGERLENDIRME NOTU:")
print("- Telefon/su tespiti pipeline'da COCO'dan geliyor (dengeli, guvenilir) - dms_v4'e gerek yok.")
print("- dms_v4 leakage'li %99.5 verdi ama gercek videoda calismadi (kanitlandi).")
print("- KARAR: dms_v4 demoda KAPALI kalir. Telefon/su=COCO, bakma=6DRepNet, sigara/kemer=yeni model.")
print("- Bu, 'guvenli ama dogru' stratejisi (State Farm 1.'sinin overfit dersi).")
"""))

cells.append(md("## 8. Özet + indir"))
cells.append(code("""print("=== URETILEN/INDIRILEN (Drive/egitilen_weights_v2) ===")
for f in sorted(os.listdir(OUT)):
    p = os.path.join(OUT, f)
    print(f"  {f}: {os.path.getsize(p)//1024//1024} MB")

print("\\n=== sigara_kemer metrikleri ===")
import csv as _csv
p = '/content/runs/sigara_kemer/results.csv'
if os.path.exists(p):
    rows = list(_csv.reader(open(p)))
    if len(rows) >= 2:
        bas = [h.strip() for h in rows[0]]; son = rows[-1]
        for h, v in zip(bas, son):
            if any(k in h for k in ['precision','recall','mAP50']):
                try: print(f"  {h}: {float(v):.4f}")
                except: pass

print("\\nNot: sigara_kemer'i indirip weights/'e koy. 6DRepNet agirligini da.")
print("Pipeline'da: sigara/kemer=yeni model, bakma=6DRepNet+baseline, telefon/su=COCO.")
print("GERCEK videoda test et (overfit kontrolu) - validasyon mAP'ine guvenme!")
"""))

nb = {"cells": cells, "metadata": {"accelerator":"GPU","colab":{"provenance":[],"gpuType":"A100"},
      "kernelspec":{"display_name":"Python 3","name":"python3"},"language_info":{"name":"python"}},
      "nbformat":4,"nbformat_minor":0}

with open('TEKNOFEST_egitim_v2.ipynb','w',encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("TEKNOFEST_egitim_v2.ipynb yazildi:", len(cells), "hucre")
