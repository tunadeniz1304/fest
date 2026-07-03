from src.dms_kaggle import dms_kaggle_tespit, SINIF_ESLEME


def test_agirlik_yoksa_bos_liste():
    assert dms_kaggle_tespit("yok_video.mp4") == []


def test_esleme_gecerli_etiketler():
    gecerli = {"sigara_icme", "telefonla_konusma", None}
    for v in SINIF_ESLEME.values():
        assert v in gecerli
