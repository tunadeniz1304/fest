# -*- coding: utf-8 -*-
"""
Track-bazli temporal voting (saf fonksiyonlar).

Mantik: bir nesneyi tek frame yerine track boyunca dogrula. Tek-frame yanlis
pozitifleri (motion blur, anlik parlama, OCR hatasi) cevredeki frame'ler duzeltir.
  - cogunluk_oyu: cls/tip/renk/koltuk icin track-boyu majority vote.
  - plaka_karakter_oylama: pozisyon-bazli karakter cogunlugu (O<->0, l<->1 duzeltir).
  - eylem_karari: eylem frame-orani esigi (anlik flash'lari eler).

Kaynak: track-level majority voting (arXiv 2402.09241, Ultralytics #2687).
"""
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


def tespit_dedup(tespitler):
    """
    Ayni (kategori, etiket) icin birden cok track tespiti varsa en yuksek
    confidence_score'lu olani tutar. Mukerrer raporlamayi onler (precision).
    Cikti zaman_saniye'ye gore sirali.
    """
    en_iyi = {}
    for t in tespitler:
        anahtar = (t["kategori"], t["etiket"])
        mevcut = en_iyi.get(anahtar)
        if mevcut is None or t["confidence_score"] > mevcut["confidence_score"]:
            en_iyi[anahtar] = t
    return sorted(en_iyi.values(), key=lambda t: t["zaman_saniye"])


def slalom_var_mi(merkez_x_listesi, frame_genislik, min_nokta=5,
                  min_yon_degisim=3, min_genlik_oran=0.15):
    """
    Bir aracin track boyunca yatay (x) merkez pozisyonundan slalom (zig-zag) tespiti.
    Deep research: UFLD lane detection yerine ByteTrack trajectory'sinden egitimsiz
    cikarim. Yon degisimi sayisi + salinim genligi esige gore karar.

    merkez_x_listesi: track boyu arac merkez x degerleri (frame sirasiyla).
    frame_genislik: normalize icin goruntu genisligi (px).
    Doner: (slalom_var_mi: bool, confidence: float)
    """
    if not merkez_x_listesi or len(merkez_x_listesi) < min_nokta or frame_genislik <= 0:
        return (False, 0.0)
    xs = merkez_x_listesi
    # Yon degisimi (zig-zag) say: ardisik farklarin isaret degistirmesi
    yon_degisim = 0
    onceki_isaret = 0
    for i in range(1, len(xs)):
        fark = xs[i] - xs[i - 1]
        isaret = (fark > 0) - (fark < 0)   # +1, 0, -1
        if isaret != 0 and onceki_isaret != 0 and isaret != onceki_isaret:
            yon_degisim += 1
        if isaret != 0:
            onceki_isaret = isaret
    # Salinim genligi (max-min) / frame_genislik
    genlik_oran = (max(xs) - min(xs)) / float(frame_genislik)
    if yon_degisim >= min_yon_degisim and genlik_oran >= min_genlik_oran:
        # Confidence: yon degisim ve genlik ne kadar belirginse o kadar yuksek
        conf = min(1.0, 0.4 + 0.1 * yon_degisim + genlik_oran)
        return (True, round(conf, 2))
    return (False, 0.0)


def eylem_karari(gorulen_frame, track_uzunlugu, ort_conf, esik_oran=0.3, min_frame=2):
    """
    Bir eylemi (telefon/sigara/su) track boyunca gorulme oranina gore raporla.
    Anlik tek-frame flash'lari eler -> false positive dususu (precision).
    """
    if track_uzunlugu <= 0 or gorulen_frame < min_frame:
        return (False, 0.0)
    if gorulen_frame / track_uzunlugu < esik_oran:
        return (False, 0.0)
    return (True, round(float(ort_conf), 2))
