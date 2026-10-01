# OASIS-1 Alzheimer kohortu araştırma testi

Bu test model eğitmez, Alzheimer tanısı koymaz ve gerçek biyolojik beyin yaşı
etiketi olduğunu varsaymaz. Soru: OASIS-1'in CDR ile ayrılan demans/Alzheimer
kohortunda beyin yaşı farkı, aynı kaynaktaki yaş/cinsiyet eşleştirilmiş
kontrollerden yüksek mi? CDR demans şiddetidir; biyobelirteçle doğrulanmış
Alzheimer tanısı yerine geçmez. Kaynak, yaşlı demans grubunu klinik AD olarak
tanımlar: https://sites.wustl.edu/oasisbrains/home/oasis-1/ .

## Sabit model

`artifacts/alzheimer/model_lock.json` tamamlanmış agebalance-cont2 SFCN
checkpoint'ini SHA-256 ile sabitler. Checkpoint diskte mevcut ve yükleme testi
yapılır. Yeni eğitim bu kilidi otomatik değiştirmez. OASIS sonuçları model
seçimi, eğitimi veya kalibrasyon için kullanılamaz. Ön eğitimle kişi çakışması
henüz doğrulanmadı; sonuç bağımsız klinik doğrulama diye sunulamaz.

## Veri ve çalıştırma

DUA'yı önce okuyun: https://www.nitrc.org/frs/download.php/6348/oasis_cross-sectional.csv .
Koşullar yeniden kimliklendirmeme, OASIS'e atıf ve gerekli hibe teşekkürlerini
korumayı gerektirir. MRI'lar herkese açık siteye bu hat tarafından yüklenmez.
Aktif pilot 25 demans/Alzheimer kohortu vakası ve 25 yaş/cinsiyet eşleştirilmiş
kontrol içerir. Yalnız bu 50 kişinin ilk ham T1 img/hdr dosyaları saklanır.
Kaynak sıkıştırılmış arşiv sunduğu için seçilmeyen veriler de ağ üzerinden
geçebilir; bu yöntem ağ trafiğini 50 MRI boyutuna indirmeyi garanti etmez.
Tam arşivlerin toplamı yaklaşık 18 GB olabilir; süre ağ ve CPU'ya bağlıdır.

Koşulları kabul ettiğinizde:

```sh
.venv/bin/python tools/run_oasis_test.py --accept-dua --cases 25
```

Bu komut indirme, eşleşmiş manifest ve sabit modelle tahmini sırayla çalıştırır.
Seçim `data/oasis1/pilot_clinical.csv` içinde kayıtlıdır.

`--discs 1` yalnızca teknik pilot indirme içindir; yeterli eşleşmiş çift
bulunacağı garanti değildir. Tam kohort için 12 disk varsayılandır.
`--allow-exploratory-overlap`, ön eğitim çakışması belirsizliğini açıkça kabul
ederek yalnız keşifsel analize izin verir; veri kullanım sözleşmesinin yerine
geçmez. Girdi PNG/JPG değildir: RAW klasöründeki ilk T1 Analyze img/hdr çifti.
MR2 ve çoklu çekimler bağımsız kişi olarak sayılmaz. T88 işlenmiş dosyalar
MNI hizalamasının yerine kullanılmaz. Ham T1 bizim SynthStrip/FAST/FLIRT
SFCN hattımızdan geçirilir; işlem CPU'da dakikalar alabilir.

## Protokol ve sonuçlar

60–100 yaş, geçerli CDR 0/0.5/1/2, MR1; cinsiyet tam eşleşmesi, yaş farkı
en fazla 3 yıl, tekrar kullanılmayan kontrollerle deterministik greedy eşleşme.
Bu global optimum eşleştirme değildir. Eksik görüntüler ve başarısız işlemler
raporlanır; eksik eşin karşılığı grup karşılaştırmasına dahil edilmez.

`predictions.csv`: gerçek yaş, tahmin, yaş farkı ve kişi/pair kimliği.
`failures.json`: başarısız ön işlemler.
`qc/`: kayıt/hizalama görselleri, attention map değildir.
`report.json`: tam çiftlerde demans-kontrol yaş farkı ortalaması ve 5.000
bootstrap örneklemesiyle %95 güven aralığı. En az 5 çift teknik alt sınırdır,
yeterli bilimsel güç değildir. İstatistikler görsel QC onayına kadar geçicidir.

Alzheimer kohortunda yaş farkının pozitif olması tek başına tanı veya başarı
ölçütü değildir. Aynı kaynaktan eşleştirme çekim farkını tamamen yok etmez.
OASIS yaşlı katılımcılarında modelin eğitim yaş dağılımının seyrek uçlarına
genellenmesi ayrıca sınırlıdır. Sonuçlar iyi çıkmazsa hasta etiketlerinden
yararlanarak modeli değiştirmeyin.
