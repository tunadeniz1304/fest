from src.dms import dms_tespit


def test_video_yoksa_bos_liste_cokmez():
    # Olmayan video -> [] doner, exception atmaz (izole katman)
    assert dms_tespit("yok_video.mp4") == []
