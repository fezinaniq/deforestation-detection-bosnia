# Deforestation Detection in Bosnia Herzegovina
### Using EuroSAT Fine-tuned ResNet50 and Sentinel-2 Satellite Imagery

A machine learning pipeline for detecting forest change using transfer learning on satellite imagery, validated against Hansen Global Forest Watch data.

---

## Overview

This project applies a ResNet50 model fine-tuned on the EuroSAT dataset to detect deforestation in the Sarajevo Canton region of Bosnia Herzegovina between 2019 and 2024. The model predictions are compared against official Hansen GFW deforestation data to evaluate detection performance.

**Key Results:**
- EuroSAT validation accuracy: 97.94%
- Hansen GFW agreement: IoU = 0.005, Recall = 0.098
- Deforested area detected: ~680 km² (2,319 patches)

---

## Repository Structure

```
deforestation-detection/
│
├── 01_train_eurosat.ipynb        # Phase 1: Fine-tune ResNet50 on EuroSAT
├── 02_change_detection_cleaned.ipynb  # Phase 2-5: Change detection + GFW validation
└── README.md
```

---

## Pipeline

```
EuroSAT Dataset
    → Fine-tune ResNet50 (Phase 1)
        → Google Earth Engine — pull Sentinel-2 L1C imagery
            → Patch-based classification (64x64 patches)
                → Change detection (Forest 2019 → non-Forest 2024)
                    → Validation against Hansen GFW
```

---

## How to Run

### Phase 1 — Train the Model

1. Open `01_train_eurosat.ipynb` in Google Colab
2. Runtime → Change runtime type → T4 GPU
3. Run all cells
4. Download `resnet50_eurosat.pth` from the last cell

### Phase 2–5 — Change Detection

1. Export Sentinel-2 L1C images from Google Earth Engine:
   - Collection: `COPERNICUS/S2_HARMONIZED`
   - Region: `[17.5, 43.7, 18.8, 44.8]` (Bosnia Herzegovina)
   - Dates: June–September 2019 and June–September 2024
   - Scale: 10m
2. Open `02_change_detection_cleaned.ipynb` in Google Colab
3. Upload `resnet50_eurosat.pth`
4. Update file paths in Step 3 to your exported GeoTIFF files
5. Run all cells

---

## Data Sources

| Data | Source |
|---|---|
| Training dataset | [EuroSAT](https://github.com/phelber/EuroSAT) — Helber et al. (2019) |
| Satellite imagery | [Sentinel-2 via Google Earth Engine](https://earthengine.google.com) |
| Validation data | [Hansen Global Forest Watch](https://globalforestwatch.org) |

---

## Key Findings

Two failure modes were identified that limit detection reliability:

1. **L1C/SR preprocessing mismatch** — EuroSAT is built on Sentinel-2 L1C data. Using SR (surface reflectance) imagery causes pixel value range mismatch that degrades classification quality.

2. **Inter-annual classification instability** — The same forest patch gets classified differently between years due to subtle atmospheric and seasonal differences, not actual land change. This causes high false positive rates.

---

## Requirements

```
torch
torchvision
rasterio
numpy
matplotlib
scikit-learn
tqdm
Pillow
seaborn
```

Run on Google Colab with T4 GPU (free tier sufficient).

---

## References

- Helber et al. (2019). EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification. IEEE JSTARS.
- He et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
- Hansen et al. (2013). High-Resolution Global Maps of 21st-Century Forest Cover Change. Science.

---

## Author

**Fezin Aniq**
Independent Researcher — Kozhikode, Kerala, India
fezinaniqresearch@gmail.com
