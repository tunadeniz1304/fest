# -*- coding: utf-8 -*-
"""TEKNOFEST egitim notebook'unu (.ipynb) sifirdan uretir. Colab A100. numpy-cakismasiz."""
import json

def md(t): return {"cell_type":"markdown","metadata":{},"source":t.splitlines(keepends=True)}
def code(t): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":t.splitlines(keepends=True)}

cells = []

cells.append(md("""# TEKNOFEST 2026 — 5G & YZ Akıllı Yol Güvenliği — Eğitim Notebook'u (A100)

Tüm modelleri tek seferde eğitir. **numpy çakışması yok** (ultralytics güncel sürüm,
Colab'in numpy 2.x'ine dokunmaz).

## Kullanım
1. **Runtime → Change runtime type → A100 GPU**
2. Zip'leri Google Drive `MyDrive/teknofest/` klasörüne yükle (aşağıda liste)
3. **Runtime → Run all** (veya hücre hücre)
4. Eğitilen `.pt`'ler eğitildikçe `MyDrive/teknofest/egitilen_weights/`'e kaydedilir
   (oturum koparsa biten modeller kaybolmaz)

## Süre (A100, yaklaşık)
sigara ~10dk · dms ~30dk · renk ~15dk · statefarm ~30dk · **plaka ~1.5-2sa** → toplam ~3-3.5sa

## Eğitilen modeller
| Çıktı | Veri seti (Drive'daki zip) |
|---|---|
| sigara.pt | driver smoking detecor.v1i.yolov8.zip |
| dms_actions.pt | driver monitoring system.v4...zip (7 sınıf) |
| color.pt | archive (5).zip (VCoR, 15 renk) |
| statefarm.pt | state-farm-distracted-driver-detection.zip |
| plate.pt | License Plate Recognition.v13i.yolov8.zip (98K, en uzun) |
"""))

cells.append(md("## 1. GPU kontrolü"))
cells.append(code("""!nvidia-smi
import torch
assert torch.cuda.is_available(), "GPU YOK! Runtime -> Change runtime type -> A100 sec, sonra bu hucreyi tekrar calistir."
print("GPU:", torch.cuda.get_device_name(0))
"""))

cells.append(md("""## 2. Kurulum (numpy'a DOKUNMADAN ultralytics kur)

Colab'in numpy 2.x'i + torch'u zaten uyumlu. Eski ultralytics==8.2.103 numpy<2 ister
ve çakışır. Bu yüzden ultralytics'in **güncel** sürümünü kuruyoruz (numpy 2.x uyumlu).
**numpy'ı YÜKSELTME/DÜŞÜRME — olduğu gibi bırak.**"""))
cells.append(code("""# ultralytics guncel surum (numpy pin YOK -> Colab numpy 2.x ile uyumlu)
!pip install -q ultralytics
import numpy as np, ultralytics
print("numpy:", np.__version__, "| ultralytics:", ultralytics.__version__)
from ultralytics import YOLO
import os, shutil, zipfile, glob, yaml, traceback, random
os.makedirs('/content/weights', exist_ok=True)
os.makedirs('/content/datasets', exist_ok=True)
print("Kurulum tamam.")
"""))

cells.append(md("""## 3. Google Drive bağla

Drive'da `MyDrive/teknofest/` klasörü olsun, içinde 5 zip:
- `driver smoking detecor.v1i.yolov8.zip`
- `driver monitoring system.v4-6claases_validation_images_increased.yolov8.zip`
- `License Plate Recognition.v13i.yolov8.zip`
- `archive (5).zip`
- `state-farm-distracted-driver-detection.zip`"""))
cells.append(code("""from google.colab import drive
drive.mount('/content/drive')
DRIVE = '/content/drive/MyDrive/teknofest'
OUT = os.path.join(DRIVE, 'egitilen_weights')
os.makedirs(OUT, exist_ok=True)
if not os.path.isdir(DRIVE):
    print("!!! KLASOR YOK:", DRIVE, "- Drive'da olusturup zip'leri yukle")
else:
    print("Klasor:", DRIVE)
    for f in sorted(os.listdir(DRIVE)): print("  -", f)
"""))

cells.append(md("## 4. Yardımcı fonksiyonlar"))
cells.append(code("""def cikar(zip_ad, hedef):
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
    c = glob.glob(f'{ds_dir}/**/data.yaml', recursive=True)
    return c[0] if c else None

def yaml_duzelt(yaml_path, names=None):
    root = os.path.dirname(yaml_path)
    with open(yaml_path) as f: orig = yaml.safe_load(f)
    if names is None: names = orig.get('names')
    val_dir = 'valid' if os.path.isdir(f'{root}/valid/images') else 'val'
    with open(yaml_path, 'w') as f:
        f.write(f"train: {root}/train/images\\n")
        f.write(f"val: {root}/{val_dir}/images\\n")
        if os.path.isdir(f'{root}/test/images'): f.write(f"test: {root}/test/images\\n")
        f.write(f"nc: {len(names)}\\n")
        f.write(f"names: {names}\\n")
    print(f"  data.yaml duzeltildi (nc={len(names)})")
    return yaml_path

def kaydet(ad, cikti_ad):
    \"\"\"runs/<ad>/weights/best.pt -> /content/weights + Drive (kalici).\"\"\"
    best = f'/content/runs/{ad}/weights/best.pt'
    if os.path.exists(best):
        shutil.copy(best, f'/content/weights/{cikti_ad}')
        shutil.copy(best, os.path.join(OUT, cikti_ad))
        print(f"  KAYDEDILDI -> weights/{cikti_ad} + Drive/egitilen_weights/{cikti_ad}")
    else:
        print(f"  UYARI: best.pt yok: {best}")

def egit_detect(ad, yaml_path, cikti_ad, epochs=80, imgsz=640, batch=32):
    try:
        print(f"\\n=== DETECT EGITIM: {ad} ({epochs} epoch) ===")
        YOLO('yolov8m.pt').train(data=yaml_path, epochs=epochs, imgsz=imgsz, batch=batch,
            device=0, project='/content/runs', name=ad, exist_ok=True,
            patience=max(10,epochs//5), plots=True, verbose=True)
        kaydet(ad, cikti_ad)
    except Exception as e:
        print(f"  HATA ({ad}): {e}"); traceback.print_exc()

def egit_classify(ad, data_dir, cikti_ad, epochs=40, imgsz=224, batch=64):
    try:
        print(f"\\n=== CLASSIFY EGITIM: {ad} ({epochs} epoch) ===")
        YOLO('yolov8m-cls.pt').train(data=data_dir, epochs=epochs, imgsz=imgsz, batch=batch,
            device=0, project='/content/runs', name=ad, exist_ok=True,
            patience=max(8,epochs//5), plots=True, verbose=True)
        kaydet(ad, cikti_ad)
    except Exception as e:
        print(f"  HATA ({ad}): {e}"); traceback.print_exc()

print("yardimcilar hazir")
"""))

cells.append(md("## 5. sigara_icme (YOLOv8m detect, 100 epoch, ~10dk)"))
cells.append(code("""ds = cikar('driver smoking detecor.v1i.yolov8.zip', 'sigara')
if ds:
    yp = yaml_bul(ds)
    if yp:
        yaml_duzelt(yp, names=['sigara'])
        egit_detect('sigara', yp, 'sigara.pt', epochs=100)
"""))

cells.append(md("## 6. sürücü davranışı (driver_monitoring_v4, 7 sınıf detect, 80 epoch, ~30dk)"))
cells.append(code("""ds = cikar('driver monitoring system.v4-6claases_validation_images_increased.yolov8.zip', 'dms_v4')
if ds:
    yp = yaml_bul(ds)
    if yp:
        yaml_duzelt(yp)   # orijinal sinif isimlerini koru
        egit_detect('dms_v4', yp, 'dms_actions.pt', epochs=80)
"""))

cells.append(md("## 7. araç rengi (VCoR, 15 sınıf classify, 40 epoch, ~15dk)"))
cells.append(code("""ds = cikar('archive (5).zip', 'vcor')
if ds:
    root = ds if os.path.isdir(f'{ds}/train') else next(
        (d for d in glob.glob(f'{ds}/*') if os.path.isdir(f'{d}/train')), ds)
    print("VCoR kok:", root, "| icerik:", os.listdir(root)[:5])
    egit_classify('color', root, 'color.pt', epochs=40)
"""))

cells.append(md("## 8. State Farm sürücü davranışı (10 sınıf classify, 30 epoch, ~30dk)"))
cells.append(code("""ds = cikar('state-farm-distracted-driver-detection.zip', 'statefarm')
if ds:
    src_train = next((c for c in glob.glob(f'{ds}/**/train', recursive=True)
                      if glob.glob(f'{c}/c*')), None)
    if src_train:
        base = '/content/datasets/statefarm_cls'
        if not os.path.isdir(f'{base}/train'):
            for cls_dir in sorted(glob.glob(f'{src_train}/c*')):
                cls = os.path.basename(cls_dir)
                imgs = glob.glob(f'{cls_dir}/*.jpg'); random.seed(42); random.shuffle(imgs)
                n_val = max(1, int(len(imgs)*0.15))
                for sp, lst in [('val', imgs[:n_val]), ('train', imgs[n_val:])]:
                    d = f'{base}/{sp}/{cls}'; os.makedirs(d, exist_ok=True)
                    for im in lst: shutil.copy(im, d)
            print("statefarm split hazir")
        egit_classify('statefarm', base, 'statefarm.pt', epochs=30)
    else:
        print("train/c0.. bulunamadi")
"""))

cells.append(md("""## 9. plaka detektörü (License Plate, 98K görsel detect, 20 epoch, ~1.5-2sa)

**EN UZUN.** Oturum kopabilir; en sona bilerek koyduk (diğerleri zaten Drive'da).
Süre azsa epoch'u düşür (örn. 15)."""))
cells.append(code("""ds = cikar('License Plate Recognition.v13i.yolov8.zip', 'plate')
if ds:
    yp = yaml_bul(ds)
    if yp:
        yaml_duzelt(yp, names=['plaka'])
        egit_detect('plate', yp, 'plate.pt', epochs=20)
"""))

cells.append(md("## 10. Özet + metrikler (pandas YOK, saf csv)"))
cells.append(code("""import csv as _csv
print("=== EGITILEN AGIRLIKLAR (Drive/egitilen_weights) ===")
for f in sorted(os.listdir(OUT)):
    print(f"  {f}: {os.path.getsize(os.path.join(OUT,f))//1024//1024} MB")

print("\\n=== METRIKLER (FTR raporu icin) ===")
for ad in ['sigara','dms_v4','color','statefarm','plate']:
    p = f'/content/runs/{ad}/results.csv'
    if not os.path.exists(p): continue
    with open(p) as f: rows = list(_csv.reader(f))
    if len(rows) < 2: continue
    bas = [h.strip() for h in rows[0]]; son = rows[-1]
    print(f"\\n[{ad}] (son epoch / {len(rows)-1} epoch):")
    for h, v in zip(bas, son):
        if any(k in h for k in ['precision','recall','mAP50','accuracy_top1']):
            try: print(f"   {h}: {float(v):.4f}")
            except: pass
print("\\nNot: Bu rakamlari FTR 'Sinama' bolumunde kullan. 'validasyonda %X' diye belirt;")
print("gercek yarisma videosunda domain-gap nedeniyle farkli olabilir.")
print("\\nEgitilen weights'ler Drive/teknofest/egitilen_weights/ -> indir, projeye koy.")
"""))

nb = {"cells": cells, "metadata": {"accelerator":"GPU","colab":{"provenance":[],"gpuType":"A100"},
      "kernelspec":{"display_name":"Python 3","name":"python3"},"language_info":{"name":"python"}},
      "nbformat":4,"nbformat_minor":0}

with open('TEKNOFEST_egitim.ipynb','w',encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("TEKNOFEST_egitim.ipynb yazildi:", len(cells), "hucre")
