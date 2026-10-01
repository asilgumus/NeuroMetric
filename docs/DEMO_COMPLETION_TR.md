# Gerçek MRI demo özellikleri

## Tamamlananlar

- OASIS-30: korunmuş orijinal Analyze dosyasından belgelenmiş ASL geometriyle yeni NIfTI; SynthStrip, FAST ve 12-DOF FLIRT; dokuz düzlemde teknik kayıt kontrolü.
- Kilitli contrast-cont4 SFCN tahmini: 62.344738 yaş; kronolojik yaş 65; fark -2.655262 yıl. Yeni signed Grad-CAM ve üç düzlem ayrı görüntüsü dashboard'a bağlı.
- FastSurfer v2.4.2 VINN segmentasyonu yeniden hesaplandı; 15 düzlemde teknik görsel kontrol ve tam anatomik etiket denetimi sonrası beş bilateral hacim yayınlandı.
- Hacimler (mL): hippocampus 7.504; temporal kortikal gri madde 99.570; frontal 142.800; parietal 102.996; occipital 54.759. Kortikal gruplar tüm lob hacmi değildir; beyaz madde, singulat ve insula dahil değildir.
- OASIS-2 OAS2_0001 MR1/MR2: iki gerçek MRI, 457 gün ziyaret aralığı, 87/88 yaş. Aynı ön işleme ve modelle tahminler 70.542854/73.303589. Gap değişimi +1.760735 yıl; biyolojik yaşlanma hızı olarak yorumlanamaz.
- `longitudinal.html`: gerçek veri seti nWBV değerleri ayrı, model tahminleri ayrı; uydurulmuş takvim tarihi yok. Bu örnekte belirgin düşük tahmin açıkça belirtiliyor.

## Sınırlar ve yayın

Teknik görsel kontrol klinisyen onayı veya klinik validasyon değildir. Sağlıklı beyin, Alzheimer riski, normatif yüzdelik veya bölgesel yaş hesaplanmıyor. Yerel statik demo tamamlanan ölçümleri gösterir; canlı MRI yükleme/uzak inference servisi değildir. Kamuya açık dağıtım henüz yapılmadı. OASIS kullanım koşulları ve atıfları korunmalıdır.

Kaynaklar:
- https://sites.wustl.edu/oasisbrains/home/oasis-2/
- https://doi.org/10.1162/jocn.2009.21407
- https://brainder.org/2011/08/13/converting-oasis-brains-to-nifti/
- https://4dfp.readthedocs.io/en/latest/format.html

Yeniden üretim: `tools/fetch_oasis2_demo.py`, `tools/prepare_oasis2_demo.py`, görsel kontrol sonrası `tools/record_technical_qc.py`, `tools/finish_demo_measurements.py --longitudinal --regional-reviewed`. İncelenmeyen görüntüler için kontrol bayrakları verilmemelidir.
