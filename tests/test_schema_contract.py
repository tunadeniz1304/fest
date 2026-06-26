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
        "arac_bilgisi": {"tip": "", "plaka": "", "renk": "kirmizi" + "ı", "confidence_score": 0.0},
        "tespitler": [],
    }
    # Turkce karakterli renk gecersiz -> hata listesi bos olmamali
    assert validate_results(cikti) != []
