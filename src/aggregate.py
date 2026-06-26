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
