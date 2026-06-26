import numpy as np
import cv2
from src.utils import dort_nokta_duzelt, _kose_sirala


def test_kose_sirala_dogru_sira():
    # Karisik sirali 4 nokta -> [sol-ust, sag-ust, sag-alt, sol-alt]
    noktalar = np.array([[100, 100], [0, 0], [100, 0], [0, 100]], dtype="float32")
    s = _kose_sirala(noktalar)
    assert tuple(s[0]) == (0, 0)        # sol-ust (min toplam)
    assert tuple(s[2]) == (100, 100)    # sag-alt (max toplam)


def test_dort_nokta_duzelt_dikdortgen_uretir():
    # 200x100 gri goruntu, kose noktalari verilince duz dikdortgen doner
    img = np.full((150, 250, 3), 128, np.uint8)
    cv2.rectangle(img, (30, 40), (220, 130), (255, 255, 255), -1)
    kose = np.array([[30, 40], [220, 40], [220, 130], [30, 130]], dtype="float32")
    sonuc = dort_nokta_duzelt(img, kose)
    # Cikti gecerli bir goruntu, en > boy (plaka yatay)
    assert sonuc is not None
    assert sonuc.shape[0] > 0 and sonuc.shape[1] > 0
    assert sonuc.shape[1] >= sonuc.shape[0]   # genislik >= yukseklik


def test_dort_nokta_duzelt_bozuk_girdi_none():
    # Yetersiz nokta -> None (cokmeden)
    img = np.zeros((50, 50, 3), np.uint8)
    assert dort_nokta_duzelt(img, np.array([[0, 0], [1, 1]], dtype="float32")) is None
