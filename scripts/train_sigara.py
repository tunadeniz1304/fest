# -*- coding: utf-8 -*-
"""
sigara_icme tespiti icin YOLOv8n fine-tune (driver_smoking seti, CC BY 4.0).

driver_smoking: 312 etiketli gorsel, tek sinif (sigara bbox). YOLOv8n (en kucuk,
hizli) COCO agirligindan transfer learning. Egitilen agirlik weights/sigara.pt
olarak kaydedilir, pipeline'da izole katman olarak kullanilir.

Calistir: python scripts/train_sigara.py
CPU'da yavas; az epoch + kucuk imgsz ile hizli baseline. Final tur GPU'da artirilir.
"""
import os
import shutil

# wandb/online logging'i kapat (no-tty ortamda api_key istemesini engelle)
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["YOLO_VERBOSE"] = "true"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "datasets", "driver_smoking", "data.yaml")
HEDEF = os.path.join(ROOT, "weights", "sigara.pt")

# data.yaml'daki goreli yollar bu dosyaya gore cozulsun diye absolute yaz
def _fix_data_yaml():
    ds = os.path.join(ROOT, "datasets", "driver_smoking")
    icerik = f"""train: {os.path.join(ds, 'train', 'images')}
val: {os.path.join(ds, 'valid', 'images')}
test: {os.path.join(ds, 'test', 'images')}
nc: 1
names: ['sigara']
""".replace("\\", "/")
    with open(DATA, "w", encoding="utf-8") as f:
        f.write(icerik)
    print(f"[data.yaml] absolute yollarla guncellendi")


def main():
    _fix_data_yaml()
    from ultralytics import YOLO
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Egitim] cihaz: {dev}")

    model = YOLO("yolov8n.pt")   # en kucuk, hizli; transfer learning
    model.train(
        data=DATA,
        epochs=25,
        imgsz=480,
        batch=8,
        device=dev,
        project=os.path.join(ROOT, "runs"),
        name="sigara",
        exist_ok=True,
        patience=8,        # erken durdurma
        verbose=True,
    )
    # En iyi agirligi weights/sigara.pt olarak kopyala
    best = os.path.join(ROOT, "runs", "sigara", "weights", "best.pt")
    if os.path.exists(best):
        shutil.copy(best, HEDEF)
        print(f"[Egitim] TAMAM -> {HEDEF}")
    else:
        print(f"[Egitim] HATA: best.pt bulunamadi: {best}")


if __name__ == "__main__":
    main()
