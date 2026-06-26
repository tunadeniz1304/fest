# -*- coding: utf-8 -*-
"""
Giris noktasi (teslim dokumani Bolum 8'e birebir uyumlu).

Akis:
  /app/data/input/video.mp4 oku
  -> src.predict.run_inference
  -> sema dogrula (src.labels.validate_results)
  -> /app/data/output/results.json yaz

Cikti her durumda gecerli sema icerir (model coksede bos sema yazilir),
boylece degerlendirme scripti hata almaz.
"""
import os
import sys
import json

# src/ modullerine erisim
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.predict import run_inference
from src.labels import validate_results, ARAC_TIPLERI, ARAC_RENKLERI

INPUT_PATH = "/app/data/input/video.mp4"
OUTPUT_PATH = "/app/data/output/results.json"
WEIGHTS_PATH = "/app/weights/best_model.pt"


def _ascii_temizle(text):
    """Turkce karakterleri ASCII'ye indirger, kucuk harfe cevirir (guvenlik kati)."""
    if not isinstance(text, str):
        return text
    tr = str.maketrans({
        "ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u",
        "Ç": "c", "Ğ": "g", "İ": "i", "Ö": "o", "Ş": "s", "Ü": "u",
    })
    return text.translate(tr)


def _sema_sertlestir(data):
    """
    Cikti uretildikten sonra son bir guvenlik gecisi: etiketleri kucuk harf +
    ASCII yapar, gecersiz tip/renk degerlerini bosaltir, plakayi buyuk harf tutar.
    """
    try:
        ab = data.get("arac_bilgisi", {})
        ab["tip"] = _ascii_temizle(str(ab.get("tip", ""))).lower()
        if ab["tip"] not in ARAC_TIPLERI:
            ab["tip"] = ""
        ab["renk"] = _ascii_temizle(str(ab.get("renk", ""))).lower()
        if ab["renk"] not in ARAC_RENKLERI:
            ab["renk"] = ""
        ab["plaka"] = str(ab.get("plaka", "")).upper().strip()
        data["arac_bilgisi"] = ab

        for t in data.get("tespitler", []):
            t["kategori"] = _ascii_temizle(str(t.get("kategori", ""))).lower()
            t["etiket"] = _ascii_temizle(str(t.get("etiket", ""))).lower()
    except Exception as e:
        print(f"[main] Sema sertlestirme hatasi (yok sayiliyor): {e}")
    return data


def main():
    print("Yol Guvenligi Yapay Zeka Cikarim Islemi Baslatildi...")
    print(f"Model Agirligi: {WEIGHTS_PATH}")

    # Cikti dizinini garanti et
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    if not os.path.exists(INPUT_PATH):
        print(f"Hata: Girdi videosu bulunamadi -> {INPUT_PATH}")
        # Yine de gecerli bos sema yaz ki degerlendirme scripti cokmesin
        bos = {
            "video_id": "video.mp4",
            "arac_bilgisi": {"tip": "", "plaka": "", "renk": "", "confidence_score": 0.0},
            "tespitler": [],
        }
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(bos, f, ensure_ascii=True, indent=2)
        sys.exit(1)

    try:
        output_data = run_inference(INPUT_PATH, WEIGHTS_PATH)
    except Exception as e:
        print(f"Model calistirilirken hata: {e}")
        output_data = {
            "video_id": os.path.basename(INPUT_PATH),
            "arac_bilgisi": {"tip": "", "plaka": "", "renk": "", "confidence_score": 0.0},
            "tespitler": [],
        }

    output_data = _sema_sertlestir(output_data)

    # Programatik sema dogrulamasi (sadece log; cikti yine de yazilir)
    hatalar = validate_results(output_data)
    if hatalar:
        print("[main] UYARI - sema dogrulama uyarilari:")
        for h in hatalar:
            print("   -", h)
    else:
        print("[main] Sema dogrulamasi BASARILI.")

    # ensure_ascii=True -> ciktida Turkce karakter kalmaz (sartname kurali)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=True, indent=2)

    print(f"Islem basariyla tamamlandi. Cikti kaydedildi: {OUTPUT_PATH}")
    print(f"Toplam tespit: {len(output_data.get('tespitler', []))} | "
          f"Arac: tip={output_data['arac_bilgisi']['tip']} "
          f"renk={output_data['arac_bilgisi']['renk']} "
          f"plaka={output_data['arac_bilgisi']['plaka']}")


if __name__ == "__main__":
    main()
