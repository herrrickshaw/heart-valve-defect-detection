# Kaggle Datasets for Heart Disease Analysis

A comprehensive guide to datasets on Kaggle that work perfectly with your heart valve defect detection system.

---

## 🎯 Top Recommended Datasets

### 1. **CAMUS-Human Heart Data** ⭐⭐⭐⭐⭐
**Perfect for your system - this is the benchmark dataset referenced in your architecture spec!**

- **URL**: https://kaggle.com/datasets/shoybhasan/camus-human-heart-data
- **Size**: 3.83 GB
- **Downloads**: 2,634
- **Upvotes**: 20
- **Created**: 3 years ago
- **Usability Score**: 8.75/10

**What it contains:**
- 500 clinical TTE examinations from the University Hospital of Saint-Etienne
- **Leclerc annotation protocol** (same as your system's ground truth!)
- Left Ventricle (LV) and Left Atrium (LA) segmentation masks
- Training + Validation splits
- Includes difficult cases (off-center, partial wall visibility)
- GE Vivid E95 ultrasound scanner data

**Perfect for:**
- Training your U-Net segmentation model
- Validating against CAMUS benchmark (Dice 0.931 LV/ED)
- Testing bias calibration (0.5mm vertical, 0.3mm horizontal)
- Testing on heterogeneous image quality

**Dataset Properties:**
- ~50% of population has LVEF < 45% (good disease diversity)
- Wide variability in acquisition settings (generalization testing)
- Apical 4-chamber view focus

**Citation Required:**
> S. Leclerc, E. Smistad, J. Pedrosa, A. Ostvik, et al. "Deep Learning for Segmentation using an Open-Source Framework: A Reference Implementation of the U-Net in TensorFlow and Its Application to the CAMUS Dataset." arXiv preprint arXiv:2307.11408 (2023).

---

### 2. **CAMUS - Echocardiography Image Dataset**
**Alternative source - same data, different uploader**

- **URL**: https://kaggle.com/datasets/parsakh/camus-echocardiography-image-dataset
- **Size**: Varies
- **Downloads**: 348
- **Created**: 1 year ago
- **Upvotes**: 6

**What it contains:**
- Same CAMUS dataset as above
- Pre-processed image format
- Better for quick exploration without setup

---

### 3. **EchoNet-Dynamic** ⭐⭐⭐⭐
**Largest cardiac ultrasound video dataset**

- **URL**: https://kaggle.com/code/dskswu/sample-eda-echonet-dynamic (via notebook)
- **Size**: Large video dataset
- **Upvotes**: 38
- **Usability**: Excellent for video processing

**What it contains:**
- Large-scale TTE video dataset from Stanford
- Multiple cardiac views (not just apical 4-chamber)
- Ejection fraction labels
- Left ventricle segmentation
- Excellent for temporal consistency testing

**Perfect for:**
- Multi-view model validation
- Cardiac cycle temporal analysis
- Video-based hinge point tracking
- Generalization testing beyond apical 4-chamber

---

### 4. **HMC-QU Dataset** ⭐⭐⭐⭐
**Multi-center cardiac ultrasound data**

- **URL**: https://kaggle.com/datasets/aysendegerli/hmcqu-dataset
- **Size**: Multi-center collection
- **Downloads**: 3,643
- **Upvotes**: 47
- **Created**: 2 years ago

**What it contains:**
- GE Vivid machine data (same as CAMUS!)
- Multi-center acquisitions
- Diverse imaging protocols
- Excellent for per-site bias calibration testing

**Perfect for:**
- Testing per-site calibration (0.3mm, 0.5mm offset recalibration)
- Multi-site generalization
- Different equipment/protocols
- Regional deployment testing

---

### 5. **Echonet Pediatric** ⭐⭐⭐
**Pediatric cardiac ultrasound**

- **URL**: https://kaggle.com/datasets/snikhilrao/echonet-pediatric
- **Size**: Varies
- **Downloads**: 260
- **Upvotes**: 5
- **Created**: 1 year ago

**What it contains:**
- Stanford University Pediatric echocardiography data
- Different cardiac anatomy (smaller hearts)
- Various pathologies

**Perfect for:**
- Testing model robustness on non-standard anatomy
- Pediatric cardiac defect detection
- Generalization beyond adult populations

---

### 6. **Fetal Heart Four-Chamber Ultrasound Dataset** ⭐⭐⭐
**Specialized prenatal cardiac imaging**

- **URL**: https://kaggle.com/datasets/alimusarizvi/fetal-heart-four-chamber-ultrasound-image
- **Size**: Moderate
- **Downloads**: 216
- **Created**: 8 months ago

**What it contains:**
- FOCUS (Four-chamber Fetal Imaging) dataset
- Prenatal cardiac imaging
- Four-chamber view focus (similar to apical 4-chamber)
- Biometric measurements

**Perfect for:**
- Testing on specialized imaging contexts
- Four-chamber segmentation in different anatomy
- Prenatal cardiac defect screening

---

### 7. **PTB-XL ECG Dataset** ⭐⭐⭐
**Complementary electrical cardiac data**

- **URL**: https://kaggle.com/datasets/khyeh0719/ptb-xl-dataset
- **Size**: 2 GB
- **Downloads**: 14,200+
- **Upvotes**: 81
- **Created**: 5 years ago

**What it contains:**
- 21,837 ECG recordings
- Multiple cardiac pathologies
- Segmented beats and diagnoses

**Perfect for:**
- Multi-modal cardiac analysis (ultrasound + ECG)
- Cross-validation with ECG findings
- Cardiac disease correlation studies

---

### 8. **HeartCycle Dataset EDA**
**Multi-modal cardiac data**

- **URL**: https://kaggle.com/code/kavyadhyani/heartcycle-dataset-eda
- **Upvotes**: 10
- **Created**: 4 months ago

**What it contains:**
- Synchronized multi-modal cardiac data
- Ultrasound + other modalities
- Cardiac cycle annotations

**Perfect for:**
- Multi-modal fusion analysis
- Temporal synchronization testing
- Cross-modality validation

---

## 📊 Recommended Analysis Path

### Phase 1: Baseline Training (Week 1-2)
```
Start with: CAMUS-Human Heart Data
Goal: Reproduce CAMUS benchmarks
Validate: Dice > 0.931 (LV ED)
Metrics: X-error < 1.35mm, Y-error < 0.75mm
```

### Phase 2: Generalization Testing (Week 3-4)
```
Add: HMC-QU Dataset (multi-site)
Goal: Test per-site bias calibration
Validate: Recalibrate 0.3mm/0.5mm offsets
Action: Train per-site models
```

### Phase 3: Multi-View Expansion (Week 5-6)
```
Add: EchoNet-Dynamic
Goal: Beyond apical 4-chamber views
Validate: Multi-view consistency
Action: Extend to PLAX, PSAX views
```

### Phase 4: Edge Cases (Week 7-8)
```
Add: Echonet Pediatric + Fetal dataset
Goal: Non-standard anatomy handling
Validate: Graceful degradation on unusual cases
Action: Flag low-confidence + route to clinician
```

### Phase 5: Clinical Integration (Week 9+)
```
Combine: All datasets
Goal: Comprehensive clinical validation
Validate: Multi-center IRB approval
Action: Deploy with clinical review workflow
```

---

## 🔧 Dataset Preparation Guide

### For CAMUS Dataset (Primary):

```python
import numpy as np
from pathlib import Path

# Download from Kaggle
# kaggle datasets download -d shoybhasan/camus-human-heart-data

# Dataset structure:
# ├── camus_frames/
# │   ├── patient0001/
# │   │   ├── patient0001_4CH_ED_0.png
# │   │   ├── patient0001_4CH_ES_0.png
# │   │   ├── patient0001_2CH_ED_0.png
# │   │   └── patient0001_2CH_ES_0.png
# ├── camus_masks/
# │   ├── patient0001/
# │   │   └── (segmentation masks)
# └── camus_metadata.json

# Load into your analysis pipeline:
from heart_valve_analysis import CAMUSDataset, SegmentationTrainer

train_dataset = CAMUSDataset(split='train', target_shape=(256, 256))
val_dataset = CAMUSDataset(split='val', target_shape=(256, 256))

from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Train model
from heart_valve_analysis import UNetSegmenter, SegmentationTrainer
model = UNetSegmenter(in_channels=1, num_classes=3)
trainer = SegmentationTrainer(model, device='cuda')

for epoch in range(50):
    loss = trainer.train_epoch(train_loader)
    metrics = trainer.validate(val_loader)
    print(f"Epoch {epoch}: Dice={metrics['dice_lv']:.4f}")
```

### For HMC-QU Dataset (Multi-site):

```python
from dicom_integration import DICOMDataIngestionPipeline

pipeline = DICOMDataIngestionPipeline(output_dir='./hmc_qu_data')

# Ingest and anonymize
report = pipeline.ingest_from_hospital_export(
    hospital_export_dir='./hmc_qu_raw_dicom',
    institution_name='HMC_QU'
)

# Per-site bias calibration
# Test if 0.3mm, 0.5mm offsets transfer to this equipment
```

---

## 📋 Dataset Comparison Matrix

| Dataset | Size | Views | Frames | Masks | Multi-site | Pathologies | Best For |
|---------|------|-------|--------|-------|-----------|-------------|----------|
| **CAMUS** | 3.8GB | 4CH, 2CH | Yes | ✅ | Single | Mixed (50% diseased) | Baseline training |
| **EchoNet** | Large | Multiple | Video | ✅ | Yes | Diverse | Multi-view, video |
| **HMC-QU** | Varies | Mixed | Mixed | Limited | ✅ | Varied | Per-site calibration |
| **Pediatric** | Small | Multiple | Yes | Limited | No | Pediatric | Edge cases |
| **Fetal** | Small | 4CH | Yes | Limited | No | Prenatal | Prenatal screening |
| **PTB-XL** | 2GB | N/A (ECG) | N/A | N/A | Yes | Extensive | ECG correlation |

---

## ⚠️ Important Notes

### CAMUS Dataset Licensing
- **License**: Other (specified in description)
- **Citation**: Required for any publication
- **Usage**: Academic/research use
- Must cite: Leclerc et al., IEEE TMI 2019

### DICOM vs. Image Format
- **CAMUS on Kaggle**: Pre-processed PNG images
- **Original CAMUS**: DICOM format (from official source)
- Your system is designed for DICOM (with anonymization)
- Consider downloading original from: https://www.creatis.insa-lyon.fr/Challenge/camus/

### Data Privacy Considerations
- All datasets on Kaggle are de-identified
- CAMUS: Anonymized clinical data
- HMC-QU: De-identified per HIPAA
- Safe for research use with proper institutional review

---

## 🚀 Quick Start: Download & Analyze

### Step 1: Install Kaggle CLI
```bash
pip install kaggle

# Set up API token:
# 1. Go to https://www.kaggle.com/settings/account
# 2. Click "Create New API Token"
# 3. Place kaggle.json in ~/.kaggle/
```

### Step 2: Download CAMUS
```bash
kaggle datasets download -d shoybhasan/camus-human-heart-data
unzip camus-human-heart-data.zip
```

### Step 3: Run Your Analysis
```python
from heart_valve_analysis import CAMUSDataset, UNetSegmenter, SegmentationTrainer

# Load dataset
dataset = CAMUSDataset(split='val')

# Load pre-trained model or train new
model = UNetSegmenter()

# Analyze
trainer = SegmentationTrainer(model)
metrics = trainer.validate(val_loader)

print(f"Dice LV: {metrics['dice_lv']:.4f} (benchmark: 0.931)")
```

### Step 4: Validate Against Benchmarks
```python
# Check if you meet acceptance criteria
if metrics['dice_lv'] > 0.91 and metrics['x_error_median_mm'] < 1.35:
    print("✅ Meets CAMUS benchmarks!")
else:
    print("❌ Below benchmark - investigate")
```

---

## 📈 Multi-Dataset Strategy

### For Production Deployment:

1. **Start with CAMUS** (proven ground truth)
   - Validate segmentation pipeline
   - Meet acceptance criteria (Dice >0.92)

2. **Add Multi-site Data** (HMC-QU)
   - Test generalization
   - Recalibrate per-site offsets
   - Track performance degradation

3. **Expand Views** (EchoNet-Dynamic)
   - Train multi-view models
   - Test temporal consistency
   - Validate across cardiac cycle

4. **Edge Cases** (Pediatric, Fetal)
   - Test robustness on unusual anatomy
   - Implement confidence thresholds
   - Route low-confidence to clinician review

5. **Multi-modal** (ECG + PTB-XL)
   - Cross-validate with electrical findings
   - Improve decision support accuracy
   - Better AS classification

---

## 🔗 Related Kaggle Resources

### Code Notebooks Using CAMUS:
1. **055_notebook** by yihao zhu (18 upvotes)
   - Basic CAMUS EDA and segmentation
   
2. **Cardiac-sru-finalized-vivek** (13 upvotes)
   - Segmentation + metrics evaluation
   
3. **055_notebook** by Huy Phạm (7 upvotes)
   - Advanced segmentation techniques

### Competitions & Challenges:
- **EchoGemma for Report Generation** (Kaggle MedGemma Challenge)
- **RuralRadAI** (Google DeepMind Gemini Challenge)
- Look for active cardiac imaging competitions

---

## 📞 Getting Help

### Common Issues:

**Q: Dice score is below 0.92**
- Check: Are you using CAMUS test split?
- Verify: Is bias correction applied?
- Inspect: View failing case images

**Q: Different performance on HMC-QU data**
- Expected: Per-site hardware differences
- Solution: Recalibrate bias offsets per site
- Track: Keep per-site performance metrics

**Q: How to use multi-view data?**
- CAMUS: 4CH + 2CH views per patient
- Strategy: Train separate models or multi-view fusion
- Your system: Currently focused on 4CH (apical)

**Q: Medical device approval needed?**
- Requirement: IRB review for clinical use
- Use: CAMUS for research/development
- Deployment: Requires FDA/CE marking pathway

---

## ✅ Checklist: Before Clinical Deployment

- [ ] Downloaded and validated CAMUS dataset
- [ ] Segmentation model achieves Dice >0.92
- [ ] Bias calibration tested and verified
- [ ] Clinical review workflow integrated
- [ ] Multi-site testing completed (if applicable)
- [ ] Regional regulatory gating implemented
- [ ] Clinician training completed
- [ ] IRB approval obtained
- [ ] Audit trail system validated
- [ ] De-identification procedures in place

---

**Start with CAMUS** → Most complete, well-documented, matches your spec perfectly!

Good luck with your heart disease analysis system! 🏥❤️📊
