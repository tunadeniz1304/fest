from src.predict import run_inference
from src.labels import validate_results


def test_acilamayan_video_bos_gecerli_sema():
    # Olmayan dosya -> cokmez, gecerli bos sema doner
    sonuc = run_inference("yok_boyle_bir_video.mp4", "yok.pt")
    assert validate_results(sonuc) == []          # sema gecerli
    assert sonuc["tespitler"] == []
    assert sonuc["arac_bilgisi"]["tip"] == ""
