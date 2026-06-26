from src.aggregate import (
    cogunluk_oyu, plaka_karakter_oylama, eylem_karari, tespit_dedup,
)


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


def test_plaka_karakter_oylama_hatayi_duzeltir():
    # Cogunluk dogru karakteri secer: l->1 hatasini diger okumalar duzeltir
    okumalar = [("34ABC123", 0.8), ("34ABC123", 0.7), ("34ABCl23", 0.5)]
    plaka, conf = plaka_karakter_oylama(okumalar)
    assert plaka == "34ABC123"
    assert conf > 0.0


def test_plaka_karakter_oylama_bos():
    assert plaka_karakter_oylama([]) == ("", 0.0)


def test_eylem_karari_anlik_flash_eler():
    # 40 frame track'te eylem sadece 1 frame gorulduyse -> raporlama (precision)
    raporla, _ = eylem_karari(gorulen_frame=1, track_uzunlugu=40, ort_conf=0.9)
    assert raporla is False


def test_eylem_karari_surekli_eylemi_raporlar():
    # 40 frame'in 20'sinde gorulduyse -> raporla
    raporla, conf = eylem_karari(gorulen_frame=20, track_uzunlugu=40, ort_conf=0.85)
    assert raporla is True
    assert conf == 0.85


def test_tespit_dedup_ayni_etiketi_birlestirir():
    # Ayni (kategori, etiket) birden cok track'ten geldi -> tek tespit, en yuksek conf
    tespitler = [
        {"zaman_saniye": 0.2, "kategori": "yolcular", "etiket": "arka_koltuk_1", "confidence_score": 0.93},
        {"zaman_saniye": 0.8, "kategori": "yolcular", "etiket": "arka_koltuk_1", "confidence_score": 0.59},
        {"zaman_saniye": 0.2, "kategori": "yolcular", "etiket": "arka_koltuk_2", "confidence_score": 0.92},
    ]
    sonuc = tespit_dedup(tespitler)
    # arka_koltuk_1 tek, arka_koltuk_2 tek -> 2 tespit
    assert len(sonuc) == 2
    k1 = [t for t in sonuc if t["etiket"] == "arka_koltuk_1"][0]
    assert k1["confidence_score"] == 0.93   # en yuksek conf tutuldu
    assert k1["zaman_saniye"] == 0.2        # en yuksek conf'un zamani


def test_tespit_dedup_farkli_etiket_korur():
    tespitler = [
        {"zaman_saniye": 1.0, "kategori": "sofor_eylemi", "etiket": "telefonla_konusma", "confidence_score": 0.8},
        {"zaman_saniye": 2.0, "kategori": "nesneler", "etiket": "bilgisayar", "confidence_score": 0.7},
    ]
    assert len(tespit_dedup(tespitler)) == 2
