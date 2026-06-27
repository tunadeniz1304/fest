from src.statefarm import statefarm_tespit, SF_ESLEME


def test_agirlik_yoksa_bos_liste():
    assert statefarm_tespit("yok_video.mp4") == []


def test_esleme_gecerli_etiketler():
    gecerli = {"telefonla_konusma", "su_icme", "arkaya_uzanma", None}
    for v in SF_ESLEME.values():
        assert v in gecerli
