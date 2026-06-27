from src.plaka_model import plaka_crop_bul


def test_bos_crop_none_doner():
    assert plaka_crop_bul(None) is None
