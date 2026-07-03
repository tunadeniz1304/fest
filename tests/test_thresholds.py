from src.labels import esik_al, VARSAYILAN_ESIK


def test_kucuk_nesne_dusuk_esik():
    # telefon kucuk/zor -> recall icin dusuk esik
    assert esik_al("cell phone") < esik_al("car")


def test_bilinmeyen_sinif_varsayilan():
    assert esik_al("zurafa") == VARSAYILAN_ESIK
