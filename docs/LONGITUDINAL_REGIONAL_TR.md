# Gerçek takip ve bölgesel ölçüm hattı

`data/oasis30_visits.csv` mevcut gerçek tahmini içerir. Kaynak:
`artifacts/alzheimer/predictions_contrast_cont4/all_predictions.csv`.
Tarama tarihi bilinmediği için boş bırakılmıştır; tahmin koşusunun tarihi tarama tarihi değildir.
Ön işleme karşılaştırılabilirliği ve görsel QC henüz onaylanmamıştır.

Takip özeti:

```sh
.venv/bin/python tools/longitudinal_measures.py --input data/oasis30_visits.csv --output artifacts/alzheimer/longitudinal_OAS1_0030.json
```

Dashboard verisini de yenilemek için aynı komuta
`--javascript-output assets/oasis30-longitudinal.js` eklenir. Dashboard bu statik
veriyi yükler; tarayıcıda model eğitimi veya MRI tahmini çalışmaz.

Yeni ziyaretler aynı kişinin gerçek tarama tarihi, kronolojik yaşı, model çıktısı,
checkpoint SHA256, ön işleme imzası ve görsel QC durumu ile eklenir.
`visual_qc=approved` yalnızca görüntü gerçekten kontrol edildikten sonra yazılır.
Tek taramada, farklı modellerde veya kontrol edilmemiş görüntülerde değişim hızı üretilmez.
İki veya daha fazla karşılaştırılabilir ziyarette ilk–son ziyaret farkı ve yıllıklaştırılmış
model farkı hesaplanır; bu ölçüm klinik olarak doğrulanmış yaşlanma hızı değildir.

Bölgesel hacim:

```sh
.venv/bin/python tools/regional_measures.py --mri native.nii.gz --segmentation native_labels.nii.gz --labels label_mapping.json --output regional.json --qc-approved
```

`label_mapping.json`: anatomik segmentasyon aracının etiket sözlüğüne göre
bölge adını etiket numaralarıyla eşleyen JSON. Örnek numaraları gerçek aracın
sözlüğü kontrol edilmeden kullanmayın. MRI ve etiket haritası aynı native
koordinatlarda olmalı, uzamsal birimler açıkça mm olmalı.
Üretilen ölçüm mL'dir; sağlıklı kontrol yüzdelikleri ve bölgesel yaş değildir.
Bir atlasın sabit şablon hacmi hastanın anatomik hacmi olarak sunulamaz.

Anatomik segmentasyon için değerlendirilen kaynak:
[SynthSeg resmî deposu](https://github.com/BBillot/SynthSeg),
[FreeSurfer SynthSeg dokümantasyonu](https://surfer.nmr.mgh.harvard.edu/fswiki/SynthSeg).
Depo indirilmesi veya birim testlerin geçmesi gerçek hasta segmentasyonunun
başarılı olduğu anlamına gelmez; gerçek çıktı ve görüntü QC ayrıca doğrulanmalıdır.

Yerel SynthSeg araştırma kurulumu: `.cache/SynthSeg`, ayrı Python 3.8 ortamı
`.cache/synthseg-env`; mevcut eğitim ortamı değiştirilmez.
İndirilen kod commit'i: `2a2aa3bbfccb83f8253a51ca8b329b9938a2646d`.
Depodaki SynthSeg 1.0 ağırlığı SHA256:
`7d2e32d298fe38dc51ea38e6a8e8fc5c665d56b63cddb409e8b199535f8b5298`.
1.0 model tek başına kortikal lob parselasyonu sağlamaz; hipokampus gibi
etiketli subkortikal yapılar ile başlayıp kortikal parselasyonu ayrıca tamamlamak gerekir.

## OAS30 deneysel sonuç

Orijinal Analyze dönüşümündeki genel affine ile SynthSeg hipokampusu bulamadı.
Resmî arşivdeki `OAS1_0030_MR1.xml`, çekimin sagittal olduğunu ve voksel
boyutlarının mm cinsinden 1 × 1 × 1,25 olduğunu doğruluyor.
Orijinal MRI değiştirilmeden ayrı ASR yön hipotezi üretildi; sağ/sol yönü
henüz doğrulanmış değil. Bu deneysel segmentasyonda bilateral hipokampus
7,939 mL, amigdala 3,091 mL, talamus 11,832 mL ve serebral korteks
505,623 mL sert etiket hacmi elde edildi. SynthSeg'in kendi CSV'si soft
posterior hacmi içerdiğinden aynı sayılar olmak zorunda değildir.

`tools/export_oasis30_candidate_regions.py` bu sonuçları yalnızca deneysel,
onaysız veri olarak `assets/oasis30-regional-candidate.js` dosyasına aktarır.
Onaylanmış bölgesel hacim export hattı bundan ayrıdır. Dashboard bu sayıları
“candidate” etiketiyle ayrı bir bölümde gösterir. Yaş veya hastalık riski değildir.
Temporal/frontal/parietal lob parselasyonu ve yön doğrulaması tamamlanmamıştır.

## Kortikal parselasyon denemesi

FastSurfer v2.4.2 (commit `7e5334356e5abc0b600c11d6e9890e386d176433`),
resmî Zenodo kaydından üç VINN checkpoint'i ile CPU üzerinde deneysel ASR
görüntüsünde çalıştırılıyor. Kaynak konfigürasyon:
`.cache/FastSurfer/FastSurferCNN/config/checkpoint_paths.yaml`.
Batch size 1, thread sayısı 2; mevcut eğitim ortamı değiştirilmedi.
Ayrı ortam `.cache/fastsurfer-env`, PyTorch için mevcut ortamın paketlerini
salt okunur şekilde kullanıyor; kendi NumPy ve görüntüleme bağımlılıkları var.
Tek kaynak değişikliği: tek elemanlı `Compose([ToTensorTest()])`, aynı
işlemi yapan `ToTensorTest()` çağrısıyla değiştirildi; böylece yalnızca Compose
için torchvision kurulması gerekmiyor. Değiştirilmiş inference.py SHA256:
`36067c70ecb7eb1195aea40b1da9f6afc2fc5bf67dd989c10f30ea8bda541e6a`.

`data/dkt_lobe_groups.json` DKT etiketlerini proje için açık, iki taraflı,
birbirinden ayrık gri madde gruplarına toplar. Bunlar beyaz maddeyi içeren
tüm lob hacimleri değildir; singulat korteks ve insula ayrı tutulur ve bu
dosyadaki lob toplamlarına katılmaz. Bu gruplama klinik norm oluşturmaz.
Gerçek parselasyon tamamlanmadan bu gruplardan hasta hacimleri üretilmez.

FastSurfer denemesi tamamlandı; 15 kesitlik overlay kontrol görseli
`artifacts/alzheimer/fastsurfer_OAS1_0030/cortical_qc.png` incelendi.
Tüm gerekli grup etiketleri çıktı içinde mevcut; deneysel bilateral sert-etiket
hacimleri: hipokampus 7,557 mL; temporal kortikal gri madde 99,239 mL;
frontal 143,460 mL; parietal 102,706 mL; oksipital 55,145 mL.
Dashboard'ın deneysel tablosu artık bu FastSurfer çıktısını gösteriyor;
önceki SynthSeg sonuçları kendi artifact dizinlerinde korunuyor.
Yön ve klinik doğrulama tamamlanmadı; hiçbir sonuç onaylanmış ölçüm olarak
sunulmuyor. İlk ham MRI girişindeki yön sorunu, mevcut brain-age çıktısı
ve Grad-CAM için ayrıca düzeltilmiş kayıt ve yeniden değerlendirme gerektiriyor.
