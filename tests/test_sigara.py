from src.sigara import sigara_tespit


def test_agirlik_yoksa_bos_liste_cokmez():
    # weights/sigara.pt yoksa (veya video yoksa) -> [] doner, exception atmaz
    assert sigara_tespit("yok_video.mp4") == []
