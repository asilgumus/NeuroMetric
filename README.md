# BrainAge — MR tabanlı beyin yaşı tahmini (araştırma prototipi)

T1 MR görüntülerinden beyin yaşını tahmin eden 3D CNN pipeline'ı ve bunun
halka açık klinik-okuma önizleme sitesi (**NeuroMetric**).
Bu bir araştırma prototipidir; klinik cihaz ya da tanı sistemi değildir.
Research prototype — MRI-based brain age prediction, not a diagnostic device.

İki bileşen:

1. **Model pipeline** — IXI + SALD + NIMH sağlıklı kontrol T1'leri ile eğitilen,
   DLBS wave-1 ile bağımsız test edilen transfer-learning CNN'leri
   (Westman *CNN1* 3D ResNet ve UK Biobank ön-eğitimli 3D SFCN ince ayarı).
2. **NeuroMetric önizleme sitesi** — gerçek model çıktıları ve Grad-CAM
   açıklamalarından hazırlanmış statik klinik-okuma dashboard'u. Ziyaretçinin
   MR'ını yüklemez, analiz çalıştırmaz (`DATA_SOURCES.md`).

## Sonuçlar

Mevcut en iyi modelin (V3 SFCN ince ayarı, `contrast-cont4` checkpoint, epoch 22)
kilitli held-out değerlendirmesi. Checkpoint yalnızca validation MAE ile seçildi;
held-out sonuçları model ayarına kullanılmadı.

| Ayrım | n | MAE | RMSE | R² | ±5 yıl |
|---|---:|---:|---:|---:|---:|
| Validation (IXI+SALD+NIMH) | 218 | 3.41 | — | — | — |
| IXI tarihsel test | 85 | 3.60 | 4.34 | 0.935 | %76,5 |
| DLBS (harici kohort) | 464 | 5.17 | 7.20 | 0.842 | %60,3 |

Önceki jenerasyon (V2 ensemble, 4 üyeli CNN1): IXI test MAE **6.49** yıl
(tek üye fine-tune: 6.76; eğitim yaşı ortalaması 14.69) — ayrıntı için
`docs/DELIVERY_TR.md`. Model kuyruğu ve deneme günlüğü `docs/MODEL.md`.

Tam katalog: **1.691 farklı kişi** (924 train / 218 validation / 85 IXI tarihsel
test / 464 DLBS harici) — `artifacts/v3/catalog/summary.json`; ayrıca
`v3_data.py` kaynak sürüm/lisans provenansı.

## Depo yapısı

```
.
├── brainage.py            # CLI: prepare / train / predict
├── v3_*.py                # V3 pipeline (prepare, merge, train, evaluate, test, predict)
├── oasis_test.py          # OASIS tarafı test / Grad-CAM üretimi
├── kaggle/                # üretilmiş Kaggle kernel kaynakları (kernel-metadata.json + run.py)
├── tools/                 # kernel ve site derleyiciler, QC araçları
├── tests/                 # pytest (pipeline + site içerik testleri)
├── systemd/               # yerel Kaggle kuyruk servisi (opsiyonel)
├── docs/                  # MODEL.md (deneme günlüğü), DELIVERY_TR.md, çalışma notları
├── index.html             # NeuroMetric ana sayfa (statik)
├── report.html            # değerlendirme dashboard'u (?case=ixi361 …)
├── longitudinal.html      # OAS2_0001 takip örneği
├── resources.html         # model kartı / yayın notları / veri politikası
├── assets/                # site görselleri, örnek rapor PDF, Grad-CAM PNG'leri
├── DATA_SOURCES.md        # veri atfı ve demo kapsamı
└── robots.txt
```

`data/` (2.6 GB Görüntüler) ve `artifacts/` (4.4 GB checkpoint/sonuç) depoya
girmez; katalog ve hazırlama komutlarıyla yeniden üretilebilir.
İstisna: `data/dkt_lobe_groups.json` (FastSurfer DKT→lobe eşleme yapılandırması,
testler ve araçlar buna bağlı).

## Kurulum ve çalıştırma

### Model

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Eğitilmiş checkpoint ile yeni bir T1 NIfTI tahmini
python brainage.py predict \
  --input /path/to/scan.nii.gz \
  --checkpoint <checkpoint.pt> \
  --age 57 --subject-id anonymous \
  --output artifacts/prediction

python -m pytest tests/ -q   # bazı testler yerel veri/artifact bekler
```

Eğitim Kaggle GPU kernel'larıyla çalışır: `kaggle/<job>/kernel-metadata.json`
+ `run.py` çiftleri Kaggle'a yüklenir; `tools/v3_queue.py` hazırlık/
eğitim/merge kuyruğunu yönetir (`docs/MODEL.md`).

### Site

```bash
python3 -m http.server 8123   # http://localhost:8123
```

Tamamen statiktir — MR yükleme yok, analiz yok, sunucu yok.

## Veri kaynakları ve lisanslar

| Kaynak | Yayın | Lisans |
|---|---|---|
| IXI | [brain-development.org](https://brain-development.org/ixi-dataset/) | CC BY-SA 3.0 |
| SALD | 2018 public release | CC BY-NC (kullanım amacını kontrol edin) |
| NIMH | [OpenNeuro ds005752](https://openneuro.org/datasets/ds005752/versions/2.0.0) | CC0 |
| DLBS | [OpenNeuro ds004856](https://openneuro.org/datasets/ds004856/versions/1.2.0) | CC0 |
| OASIS-2 | [sites.wustl.edu/oasisbrains](https://sites.wustl.edu/oasisbrains/home/oasis-2/) | kendi kullanım şartları |

Atıflar, sürümler ve SHA-256 provenansı: `DATA_SOURCES.md` ve `v3_data.py`.

## Açıklık (disclaimers)

- Beyin yaşı farkı tek başına patoloji kanıtı değildir; bölgesel farklar
  tüm-beyin farkına toplanmaz.
- Dashboard vaka kartları seçilmiş araştırma örnekleridir ve unbiased performans
  özeti değildir; dört vaka uydurma değerler içerir, IXI170/IXI361/IXI566,
  DLBS sub-3898 ve OAS2_0007 gerçek model tahminleridir.
- Site veri yüklemez veya analiz çalıştırmaz; tüm rakamlar önceden hesaplanmıştır.
- `technical_review` ve QC işaretleri teknik kontrol bilgisidir, klinik validasyon değildir.