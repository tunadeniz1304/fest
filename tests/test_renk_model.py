from src.renk_model import renk_tahmin, VCOR_ESLEME
from src.labels import ARAC_RENKLERI


def test_bos_crop_none_doner():
    assert renk_tahmin(None) is None


def test_esleme_hep_gecerli_renge_gider():
    # VCoR eslemesindeki her hedef bizim 9 renkten biri olmali
    for v in VCOR_ESLEME.values():
        assert v in ARAC_RENKLERI, f"gecersiz renk eslemesi: {v}"
