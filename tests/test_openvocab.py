from src.openvocab import teknocan_tespit


def test_agirlik_yoksa_bos_liste_cokmez():
    # weights/yolo_world.pt yok (test ortami) -> [] doner, exception atmaz
    assert teknocan_tespit("yok.mp4") == []
