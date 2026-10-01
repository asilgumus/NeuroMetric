# BrainAGE teslim özeti

> V2 güncellemesi: validation ile seçilmiş dört üyeli ensemble, test MAE'yi
> 6.757 yıldan 6.490 yıla düşürdü. Güncel teslim `artifacts/ensemble-v2` içindedir.

## Sonuç

Westman CNN1 3D ResNet modeli IXI T1 MRI verisi üzerinde Kaggle Tesla T4 ile
fine-tune edildi. Katılımcılar model eğitiminden önce, seed 42 ile merkez ve
yaş üçlülerine göre %70/%15/%15 ayrıldı.

| Model | Test MAE | Test RMSE | Test R² | Ortalama gap |
|---|---:|---:|---:|---:|
| Eğitim yaşı ortalaması | 14.690 | 17.033 | -0.001 | -0.547 |
| Pretrained CNN1 üye 0 | 13.046 | 16.638 | 0.045 | +12.109 |
| Fine-tune model | **6.757** | **8.414** | **0.756** | **+0.222** |

Fine-tuning test MAE'yi 6.289 yıl azalttı. Model seçimi yalnız validation
MAE ile yapıldı; seçilen checkpoint son ResNet bloğu aşamasının 25. epoch'u
ve validation MAE değeri 6.309 yıldır.

## Veri ve doğrulama

- Kullanılabilir 561 katılımcı: 392 train, 84 validation, 85 test.
- Yinelenen katılımcı yok; 561 görüntünün tamamı otomatik görüntü ve
  registration kontrollerinden geçti.
- IXI002, IXI300 ve IXI600 QC montajları elle örneklenerek hizalama açısından
  makul bulundu. Bu, tüm kohortun uzman tarafından manuel QC'si değildir.
- Resmi IXI demografi adresi HTTP 403 döndürdüğü için aynı `IXI.xls`
  dosyasının commit'e sabitlenmiş kopyası kullanıldı ve SHA-256 doğrulandı.
- Yaş değeri aynı olan tekrar satırları birleştirildi; çelişkili veya eksik
  yaş kayıtları veri hazırlama sırasında dışlandı.

## Çalışan tahmin

Ham NIfTI T1 MRI ile son test:

```json
{
  "subject_id": "IXI002-raw-final-check",
  "chronological_age": 35.800137,
  "predicted_brain_age": 36.29789733886719,
  "brain_age_gap": 0.49776033886718807,
  "qc_status": "automatic_checks_passed",
  "research_only": true
}
```

Yeni bir ham T1 NIfTI için:

```bash
.venv/bin/python brainage.py predict \
  --input /path/to/scan.nii.gz \
  --checkpoint artifacts/training-complete/training/best_model.pt \
  --age 57 \
  --subject-id anonymous \
  --output artifacts/prediction
```

Komut FSL ile rigid MNI hizalaması yapar ve `prediction.json` üretir.

## Teslim dosyaları

- Model: `artifacts/training-complete/training/best_model.pt`
- Metrikler: `artifacts/training-complete/training/metrics.json`
- Test tahminleri: `artifacts/training-complete/training/predictions.csv`
- Alt grup sonuçları: `artifacts/training-complete/training/subgroup_metrics.csv`
- Eğitim geçmişi: `artifacts/training-complete/training/history.csv`
- Değerlendirme grafiği: `artifacts/training-complete/training/test_evaluation.png`
- Araştırma raporu: `artifacts/training-complete/training/REPORT_TR.md`
- Sabit veri ayrımı ve QC: `artifacts/prepare-complete/prepared/manifest.csv`
- Ham MRI son testi: `artifacts/raw-final-inference/prediction.json`

Checkpoint SHA-256:
`a1184304fd94bb5a018962c31efe5b026e2b7f00ffeaad94de7b026ead6edbfa`

Bu çalışma araştırma prototipidir. Bölgesel beyin yaşı, klinik yüzdelik,
hastalık tanısı ve klinik öneri üretmez.
