from src.dms_actions import dms_actions_tespit, SINIF_ESLEME


def test_agirlik_yoksa_bos_liste_cokmez():
    # weights/dms_actions.pt veya video yoksa -> [], exception yok
    assert dms_actions_tespit("yok_video.mp4") == []


def test_sinif_esleme_bizim_etiketlere_uygun():
    # Eslenen etiketler ASCII-safe kucuk harf, bizim sema ile uyumlu
    gecerli = {"telefonla_konusma", "su_icme", "arkaya_uzanma"}
    for v in SINIF_ESLEME.values():
        if v is not None:
            assert v in gecerli, f"beklenmeyen etiket: {v}"
            assert v == v.lower()
