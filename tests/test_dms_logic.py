from src.dms_logic import mar_hesapla, yaw_davranisi, esneme_karari


def test_mar_hesapla_acik_agiz_yuksek():
    # Acik agiz: ust-alt dudak mesafesi buyuk -> yuksek MAR
    # landmark formati: (ust_dudak_y, alt_dudak_y, sol_kose_x, sag_kose_x)
    acik = mar_hesapla(ust=0.40, alt=0.60, sol=0.30, sag=0.50)
    kapali = mar_hesapla(ust=0.49, alt=0.51, sol=0.30, sag=0.50)
    assert acik > kapali
    assert acik > 0.5


def test_yaw_davranisi_buyuk_aci_arkaya_bakma():
    # Cok buyuk yaw (kafa tamamen yana/arkaya) -> arkaya_bakma
    assert yaw_davranisi(75.0) == "arkaya_bakma"
    assert yaw_davranisi(-75.0) == "arkaya_bakma"


def test_yaw_davranisi_orta_aci_etrafa_bakinma():
    # Orta-buyuk yaw (kalibrasyon sonrasi esik 40) -> etrafa_bakinma
    assert yaw_davranisi(50.0) == "etrafa_bakinma"


def test_yaw_davranisi_normal_kafa_hareketi_yok():
    # 35 derece artik 'normal direksiyon kafa hareketi' (esik 40) -> None
    # Yanlis-pozitif onleme: goodmax testinde her videoda bakma uyduruyordu.
    assert yaw_davranisi(35.0) is None


def test_yaw_davranisi_kucuk_aci_yok():
    # Kucuk yaw (yola bakiyor) -> davranis yok
    assert yaw_davranisi(10.0) is None


def test_esneme_karari_surekli_acik_agiz():
    # MAR esik ustu yeterince frame -> esneme dogrulandi
    assert esneme_karari(esik_ustu_frame=18, ardisik_esik=15) is True


def test_esneme_karari_anlik_konusma_eler():
    # Kisa sureli acik agiz (konusma) -> esneme degil
    assert esneme_karari(esik_ustu_frame=5, ardisik_esik=15) is False
