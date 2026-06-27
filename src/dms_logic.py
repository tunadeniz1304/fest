# -*- coding: utf-8 -*-
"""
DMS (Driver Monitoring System) saf karar mantigi.

Deep research (Cipia/Seeing Machines iki-katmanli mimari): katman-1 landmark/
head-pose, katman-2 fizyolojik durum cikarimi. Bu modul katman-2: landmark'tan
turetilen olculeri (MAR, yaw) davranis etiketine cevirir.

Saf fonksiyonlar (MediaPipe'tan bagimsiz) -> hizli birim test.
  - mar_hesapla: agiz acikligi orani (esneme icin)
  - yaw_davranisi: kafa yaw acisindan arkaya_bakma / etrafa_bakinma
  - esneme_karari: MAR esik-ustu sureklilik -> esneme (konusmayi eler)
"""

# Esikler (Gemini test kalibrasyonu sonrasi - yanlis-pozitif azaltma).
# YAW_ETRAFA 25->40: normal direksiyon kafa hareketi 'bakma' sanilmasin.
# YAW_ARKAYA 60->65: gercek arka donus icin daha emin.
MAR_ESIK = 0.55           # agiz acik kabul esigi (konusma yanlis-pozitifi azalt)
YAW_ARKAYA = 65.0         # |yaw| bu ustu -> arkaya_bakma
YAW_ETRAFA = 40.0         # |yaw| bu ustu (arkaya altinda) -> etrafa_bakinma
ESNEME_ARDISIK = 15       # MAR esik-ustu min ardisik frame -> esneme
YAW_MIN_ARDISIK = 8       # bakma icin min ARDISIK frame (anlik bakisi ele - precision)


def mar_hesapla(ust, alt, sol, sag):
    """
    Mouth Aspect Ratio = dikey acikligin yatay genislige orani.
    ust/alt: ust ve alt dudak orta noktasinin y koordinati (normalize).
    sol/sag: agiz koselerinin x koordinati (normalize).
    """
    dikey = abs(alt - ust)
    yatay = abs(sag - sol)
    if yatay <= 1e-6:
        return 0.0
    return dikey / yatay


def yaw_davranisi(yaw_aci, yaw_arkaya=YAW_ARKAYA, yaw_etrafa=YAW_ETRAFA):
    """
    Kafa yaw acisindan (derece) davranis etiketi.
    |yaw| >= yaw_arkaya -> arkaya_bakma; >= yaw_etrafa -> etrafa_bakinma; degilse None.
    """
    a = abs(yaw_aci)
    if a >= yaw_arkaya:
        return "arkaya_bakma"
    if a >= yaw_etrafa:
        return "etrafa_bakinma"
    return None


def esneme_karari(esik_ustu_frame, ardisik_esik=ESNEME_ARDISIK):
    """
    Agiz MAR esik-ustu yeterince ardisik frame goruldu mu? Konusma (kisa acilis)
    ile esnemeyi (uzun acilis) ayirir.
    """
    return esik_ustu_frame >= ardisik_esik
