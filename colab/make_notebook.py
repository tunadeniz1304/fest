# -*- coding: utf-8 -*-
"""TEKNOFEST egitim notebook'unu (.ipynb) uretir. Colab A100 icin. Hatasiz, saglam."""
import json

def md(t): return {"cell_type":"markdown","metadata":{},"source":t.splitlines(keepends=True)}
def code(t): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":t.splitlines(keepends=True)}

cells = []

cells.append(md("""# TEKNOFEST 2026 — 5G & YZ Akıllı Yol Güvenliği — Tek Notebook Eğitim

**Colab A100 için.** Tüm modelleri tek seferde, yüksek precision/recall ile eğitir.
Her hücre bağımsız hata yakalar; biri patlasa diğerleri çalışır.

## Adımlar
1. **Runtime → Change runtime type → A100 GPU**
2. Zip'leri Google Drive `MyDrive/teknofest/` klasörüne yükle:
   - `driver smoking detecor.v1i.yolov8.zip`
   - `driver monitoring system.v4-6claases_validation_images_increased.yolov8.zip`
   - `driver monitoring.v1i.yolov8.zip`
   - `License Plate Recognition.v13i.yolov8.zip`
   - `archive (5).zip` (VCoR renk)
   - `state-farm-distracted-driver-detection.zip` (opsiyonel)
3. Hücreleri sırayla çalıştır (Runtime → Run all da olur)
4. Sonda eğitilen `.pt`'ler `MyDrive/teknofest/egitilen_weights/`'e iner

## Eğitilen modeller → proje karşılığı
| Çıktı | Veri | Pipeline'da |
|---|---|---|
| `sigara.pt` | driver_smoking | sigara_icme (src/sigara.py) |
| `dms_actions.pt` | driver_monitoring_v4 (7 sınıf) | telefon/içme/uzanma |
| `plate.pt` | License Plate (98K) | plaka tespiti (src/utils plaka) |
| `color.pt` | VCoR (15 renk) | araç rengi (HSV yerine) |
"""))

cells.append(md("## 1. GPU kontrolü"))
cells.append(code("""!nvidia-smi
import torch
assert torch.cuda.is_available(), "GPU YOK! Runtime -> Change runtime type -> A100 sec."
print("GPU:", torch.cuda.get_device_name(0))
"""))

cells.append(md("""## 2. Kurulum
ultralytics + numpy<2 (Colab pandas/numpy ikili uyumu icin sabit).
**Bu hücreden sonra Colab 'Restart session' isteyebilir — istersse Restart yapip
bu hücreden devam et (1. hücreyi tekrar çalıştırmana gerek yok).**"""))
cells.append(code("""!pip install -q "numpy<2" ultralytics==8.2.103 pyyaml
from ultralytics import YOLO
import os, shutil, zipfile, glob, yaml, traceback
os.makedirs('/content/weights', exist_ok=True)
os.makedirs('/content/datasets', exist_ok=True)
print("Kurulum tamam. ultralytics yuklendi.")
"""))

cells.append(md("## 3. Google Drive bağla"))
cells.append(code("""from google.colab import drive
drive.mount('/content/drive')
DRIVE = '/content/drive/MyDrive/teknofest'   # ZIP'LERIN OLDUGU KLASOR
if not os.path.isdir(DRIVE):
    print("!!! KLASOR YOK:", DRIVE)
    print("Drive'da MyDrive/teknofest/ olusturup zip'leri oraya yukle, ya da DRIVE yolunu duzelt.")
else:
    print("Klasor bulundu. Icindekiler:")
    for f in os.listdir(DRIVE): print("  -", f)
"""))

cells.append(md("## 4. Yardımcı fonksiyonlar (sağlam path bulma)"))
cells.append(code("""def cikar(zip_ad, hedef):
    \"\"\"Zip'i /content/datasets/<hedef> altina cikarir. Yoksa None.\"\"\"
    src = os.path.join(DRIVE, zip_ad)
    if not os.path.exists(src):
        print(f"  ATLANDI (zip yok): {zip_ad}"); return None
    dst = os.path.join('/content/datasets', hedef)
    if os.path.isdir(dst) and os.listdir(dst):
        print(f"  zaten cikarilmis: {dst}"); return dst
    os.makedirs(dst, exist_ok=True)
    with zipfile.ZipFile(src) as z: z.extractall(dst)
    print(f"  cikarildi: {dst}")
    return dst

def yaml_bul(ds_dir):
    \"\"\"data.yaml'i bulur (kokte veya alt klasorde).\"\"\"
    c = glob.glob(f'{ds_dir}/**/data.yaml', recursive=True)
    return c[0] if c else None

def yaml_duzelt(yaml_path, names=None):
    \"\"\"data.yaml'i absolute yollarla yeniden yazar. names verilmezse mevcut korunur.\"\"\"
    root = os.path.dirname(yaml_path)
    with open(yaml_path) as f: orig = yaml.safe_load(f)
    if names is None: names = orig.get('names')
    # train/valid klasor adini tespit et (valid vs val)
    val_dir = 'valid' if os.path.isdir(f'{root}/valid/images') else 'val'
    with open(yaml_path, 'w') as f:
        f.write(f"train: {root}/train/images\\n")
        f.write(f"val: {root}/{val_dir}/images\\n")
        if os.path.isdir(f'{root}/test/images'):
            f.write(f"test: {root}/test/images\\n")
        f.write(f"nc: {len(names)}\\n")
        f.write(f"names: {names}\\n")
    print(f"  data.yaml duzeltildi (nc={len(names)})")
    return yaml_path

def egit(ad, yaml_path, cikti_ad, model_tipi='yolov8m.pt', epochs=80, imgsz=640, batch=32):
    \"\"\"Tek modeli egitir, best.pt'yi /content/weights/<cikti_ad>'a kopyalar. Hata yakalar.\"\"\"
    try:
        print(f"\\n=== EGITIM: {ad} ({epochs} epoch) ===")
        model = YOLO(model_tipi)
        model.train(data=yaml_path, epochs=epochs, imgsz=imgsz, batch=batch, device=0,
                    project='/content/runs', name=ad, exist_ok=True, patience=max(10,epochs//5),
                    plots=True, verbose=True)
        best = f'/content/runs/{ad}/weights/best.pt'
        if os.path.exists(best):
            shutil.copy(best, f'/content/weights/{cikti_ad}')
            print(f"  TAMAM -> /content/weights/{cikti_ad}")
        else:
            print(f"  UYARI: best.pt yok: {best}")
    except Exception as e:
        print(f"  HATA ({ad}): {e}")
        traceback.print_exc()
print("yardimcilar hazir")
"""))

cells.append(md("""## 5. MODEL 1 — sigara_icme (YOLOv8m, 100 epoch)
CPU'da mAP %26 idi. A100 + yolov8m + 100 epoch ile çok daha yüksek beklenir."""))
cells.append(code("""ds = cikar('driver smoking detecor.v1i.yolov8.zip', 'sigara')
if ds:
    yp = yaml_bul(ds)
    if yp:
        yaml_duzelt(yp, names=['sigara'])
        egit('sigara', yp, 'sigara.pt', epochs=100, imgsz=640, batch=32)
    else: print("data.yaml bulunamadi")
"""))

cells.append(md("""## 6. MODEL 2 — sürücü davranışı (driver_monitoring_v4, 7 sınıf, 80 epoch)
Telefon/içme/uzanma vb. Orijinal sınıf isimleri korunur."""))
cells.append(code("""ds = cikar('driver monitoring system.v4-6claases_validation_images_increased.yolov8.zip', 'dms_v4')
if ds:
    yp = yaml_bul(ds)
    if yp:
        yaml_duzelt(yp)   # orijinal isimleri koru
        egit('dms_v4', yp, 'dms_actions.pt', epochs=80, imgsz=640, batch=32)
    else: print("data.yaml bulunamadi")
"""))

cells.append(md("""## 7. MODEL 3 — plaka detektörü (License Plate, 98K görsel, 30 epoch)
Çok büyük set. A100'de bile saatlerce sürebilir; oturum kopmasın diye 30 epoch.
İstersen epoch'u artır (oturum süren yeterse)."""))
cells.append(code("""ds = cikar('License Plate Recognition.v13i.yolov8.zip', 'plate')
if ds:
    yp = yaml_bul(ds)
    if yp:
        yaml_duzelt(yp, names=['plaka'])
        egit('plate', yp, 'plate.pt', epochs=30, imgsz=640, batch=32)
    else: print("data.yaml bulunamadi")
"""))

cells.append(md("""## 8. MODEL 4 — araç rengi (VCoR, 15 sınıf, SINIFLANDIRMA, 40 epoch)
YOLO classification modu. Zip kökünde train/val/test + renk klasörleri var."""))
cells.append(code("""ds = cikar('archive (5).zip', 'vcor')
if ds:
    # VCoR train/ klasoru nerede (kokte mi alt klasorde mi)
    if os.path.isdir(f'{ds}/train'):
        vcor_root = ds
    else:
        alt = [d for d in glob.glob(f'{ds}/*') if os.path.isdir(f'{d}/train')]
        vcor_root = alt[0] if alt else ds
    print("VCoR kok:", vcor_root, "| icerik:", os.listdir(vcor_root)[:5])
    try:
        model = YOLO('yolov8m-cls.pt')
        model.train(data=vcor_root, epochs=40, imgsz=224, batch=64, device=0,
                    project='/content/runs', name='color', exist_ok=True, patience=10, plots=True)
        shutil.copy('/content/runs/color/weights/best.pt', '/content/weights/color.pt')
        print("color.pt hazir")
    except Exception as e:
        import traceback; print("HATA (color):", e); traceback.print_exc()
"""))

cells.append(md("""## 9. (OPSİYONEL) MODEL 5 — State Farm sürücü davranışı (10 sınıf, sınıflandırma)
Çok büyük (22K görsel). İstersen çalıştır. State Farm klasör yapısı: imgs/train/c0..c9/"""))
cells.append(code("""ds = cikar('state-farm-distracted-driver-detection.zip', 'statefarm')
if ds:
    # imgs/train/c0.. yapisini YOLO-cls formatina getir (train/ + val/ ayrimi)
    src_train = None
    for c in glob.glob(f'{ds}/**/train', recursive=True):
        if glob.glob(f'{c}/c*'): src_train = c; break
    if src_train:
        # %85 train / %15 val split
        import random
        base = '/content/datasets/statefarm_cls'
        for split in ['train','val']:
            os.makedirs(f'{base}/{split}', exist_ok=True)
        for cls_dir in sorted(glob.glob(f'{src_train}/c*')):
            cls = os.path.basename(cls_dir)
            imgs = glob.glob(f'{cls_dir}/*.jpg')
            random.seed(42); random.shuffle(imgs)
            n_val = max(1, int(len(imgs)*0.15))
            for sp, lst in [('val', imgs[:n_val]), ('train', imgs[n_val:])]:
                d = f'{base}/{sp}/{cls}'; os.makedirs(d, exist_ok=True)
                for im in lst: shutil.copy(im, d)
        print("statefarm split hazir:", base)
        try:
            model = YOLO('yolov8m-cls.pt')
            model.train(data=base, epochs=30, imgsz=224, batch=64, device=0,
                        project='/content/runs', name='statefarm', exist_ok=True, patience=8, plots=True)
            shutil.copy('/content/runs/statefarm/weights/best.pt', '/content/weights/statefarm.pt')
            print("statefarm.pt hazir")
        except Exception as e:
            import traceback; print("HATA:", e); traceback.print_exc()
    else:
        print("train/c0.. klasoru bulunamadi")
"""))

cells.append(md("## 10. Sonuçları Drive'a kaydet + indir"))
cells.append(code("""print("=== EGITILEN AGIRLIKLAR ===")
ws = os.listdir('/content/weights')
if not ws:
    print("  (hic agirlik uretilmedi - hucreleri kontrol et)")
for f in ws:
    p = os.path.join('/content/weights', f)
    print(f"  {f}: {os.path.getsize(p)//1024//1024} MB")

# Drive'a kalici kopyala
out = os.path.join(DRIVE, 'egitilen_weights')
os.makedirs(out, exist_ok=True)
for f in ws:
    shutil.copy(os.path.join('/content/weights', f), os.path.join(out, f))
print(f"\\nDrive'a kaydedildi: {out}")
print("-> Bu klasoru indir, projedeki teknofest_model/weights/ icine koy")

# Zip + indirme (opsiyonel; takilirsa Drive'dan al)
try:
    shutil.make_archive('/content/egitilen_weights', 'zip', '/content/weights')
    from google.colab import files
    files.download('/content/egitilen_weights.zip')
except Exception as e:
    print("Otomatik indirme atlandi (Drive'dan alabilirsin):", e)
"""))

cells.append(md("""## 11. Metrik özeti (FTR raporu için — gerçek sayılar)
pandas KULLANMAZ (numpy ikili uyumsuzlugundan kacinmak icin saf csv okuma)."""))
cells.append(code("""import csv as _csv
print("=== EGITIM METRIKLERI (rapor icin) ===")
for ad in ['sigara','dms_v4','plate','color','statefarm']:
    p = f'/content/runs/{ad}/results.csv'
    if not os.path.exists(p):
        continue
    with open(p) as f:
        rows = list(_csv.reader(f))
    if len(rows) < 2:
        continue
    basliklar = [h.strip() for h in rows[0]]
    son = rows[-1]
    print(f"\\n[{ad}] (son epoch / {len(rows)-1} epoch):")
    for h, v in zip(basliklar, son):
        if any(k in h for k in ['precision','recall','mAP50','accuracy_top1']):
            try: print(f"   {h}: {float(v):.4f}")
            except: pass
print("\\nNot: Bu rakamlari FTR 'Sinama' bolumunde kullan. Domain-gap nedeniyle")
print("gercek yarisma videosunda farkli olabilir - 'literaturde/validasyonda' diye belirt.")
"""))

nb = {"cells": cells, "metadata": {"accelerator":"GPU","colab":{"provenance":[],"gpuType":"A100"},
      "kernelspec":{"display_name":"Python 3","name":"python3"},"language_info":{"name":"python"}},
      "nbformat":4,"nbformat_minor":0}

with open('TEKNOFEST_egitim.ipynb','w',encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("TEKNOFEST_egitim.ipynb yazildi:", len(cells), "hucre")
