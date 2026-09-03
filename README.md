# Klasifikacija i restauracija slika sa sintetički generisanim oštećenjima upotrebom CNN

Ovaj rad predstavlja celovit sistem dubokog učenja za automatsku detekciju, višestruku klasifikaciju i digitalnu rekonstrukciju slika zahvaćenih fizičko-hemijskim degradacijama. Razvijena je originalna konvoluciona arhitektura obučena od početka (*from scratch*) sa dualnim tokom obrade:
1. **Prostorna grana:** Rekurzivni mikro-blokovi sa deljenim težinama za očuvanje lokalnih geometrijskih ivica i tekstura.
2. **Spektralna grana:** Višeskalna dekompozicija frekvencijskih pojaseva (niskofrekventne i visokofrekventne komponente).
3. **Asimetrični mostovi i fuzija sa kapijom:** Dinamičko balansiranje reprezentacija na svim nivoima dubine.
4. **EdgeBranch & CCR modul:** Namenska grana za očuvanje oštrih kontura i modul za dinamičku restauraciju boja i kontrasta.

---

## Glavni rezultati

- **Klasifikacija degradacija:** 
  - Primarni test skup: **99,64%** tačnost
  - Nezavisna baza (*Vyronas*): **97,37%** tačnost
  - Evaluacija robusnosti uz kompresiju: **ROC-AUC = 0,7214** (rekalibrisana tačnost: **69,09%** preko Youden's $J$).
- **Digitalna restauracija:**
  - Umerena oštećenja ($0,1 - 0,5$): **PSNR do 36,19 dB | SSIM do 0,9795**
  - Teška oštećenja ($0,5 - 1,0$): **PSNR do 32,02 dB | SSIM do 0,9546**
  - Uporedna evaluacija ($N=160$, Bootstrap 1000 iteracija): Predloženi model postiže **30,06 dB** (+7,41 dB vs polazni ulaz, +8,61 dB vs adaptirani Microsoft BOPBL, $p < 0,001$).

---

## 🛠️ Struktura repozitorijuma
