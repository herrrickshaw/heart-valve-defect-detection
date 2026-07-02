# Kaggle Cardiac Datasets — Quick Reference Card

## 🎯 TL;DR - Best Datasets to Download Right Now

### **#1 Priority: CAMUS-Human Heart Data** ⭐⭐⭐⭐⭐
```
🔗 https://kaggle.com/datasets/shoybhasan/camus-human-heart-data
📊 500 patients | 3.83 GB | 2,634 downloads
✅ Segmentation masks + Leclerc annotations
✅ Your system's benchmark dataset
✅ Apical 4-chamber + 2-chamber views
```
**Download**: `kaggle datasets download -d shoybhasan/camus-human-heart-data`

---

### **#2 Priority: HMC-QU Dataset** ⭐⭐⭐⭐
```
🔗 https://kaggle.com/datasets/aysendegerli/hmcqu-dataset
📊 Multi-center | 3,643 downloads
✅ GE Vivid machines (same as CAMUS!)
✅ Per-site calibration testing
✅ Multi-protocol data
```
**Download**: `kaggle datasets download -d aysendegerli/hmcqu-dataset`

---

### **#3 Priority: EchoNet-Dynamic** ⭐⭐⭐⭐
```
🔗 https://kaggle.com/code/dskswu/sample-eda-echonet-dynamic
📊 Large Stanford dataset | Video format
✅ Multi-view TTE data
✅ Ejection fraction labels
✅ Temporal video analysis
```

---

## 📋 Dataset Overview Table

| Name | URL | Type | Size | Key Features | For Your System |
|------|-----|------|------|--------------|-----------------|
| **CAMUS** | [Link](https://kaggle.com/datasets/shoybhasan/camus-human-heart-data) | Images + Masks | 3.8GB | Segmentation, Leclerc protocol | **Baseline training** |
| **HMC-QU** | [Link](https://kaggle.com/datasets/aysendegerli/hmcqu-dataset) | Multi-center | Varies | GE Vivid data, multi-site | **Per-site calibration** |
| **EchoNet** | [Link](https://kaggle.com/code/dskswu/sample-eda-echonet-dynamic) | Video | Large | Multiple views, EF labels | **Multi-view validation** |
| **Echonet Pediatric** | [Link](https://kaggle.com/datasets/snikhilrao/echonet-pediatric) | Images | Varies | Pediatric hearts | **Edge case testing** |
| **Fetal Heart** | [Link](https://kaggle.com/datasets/alimusarizvi/fetal-heart-four-chamber-ultrasound-image) | Images | Small | Prenatal 4CH | **Specialized anatomy** |
| **PTB-XL ECG** | [Link](https://kaggle.com/datasets/khyeh0719/ptb-xl-dataset) | ECG | 2GB | 21K+ ECG records | **Multi-modal validation** |

---

## 🚀 3-Minute Setup

### Step 1: Install Kaggle CLI
```bash
pip install kaggle
# Download kaggle.json from https://www.kaggle.com/settings/account
# Place in ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### Step 2: Download CAMUS (Priority #1)
```bash
kaggle datasets download -d shoybhasan/camus-human-heart-data -p ~/kaggle_data
cd ~/kaggle_data && unzip camus-human-heart-data.zip
```

### Step 3: Start Analysis
```python
from heart_valve_analysis import CAMUSDataset, UNetSegmenter

# Load data
dataset = CAMUSDataset(split='train')

# Train model
model = UNetSegmenter(in_channels=1, num_classes=3)
# ... train and validate
```

---

## 💾 Dataset File Sizes & Download Times

| Dataset | Size | Time (10Mbps) | Storage Needed |
|---------|------|---------------|----------------|
| CAMUS | 3.8 GB | ~50 min | 8 GB |
| HMC-QU | ~5-10 GB | ~1-2 hrs | 15 GB |
| EchoNet | ~50 GB | ~10+ hrs | 100 GB |
| Pediatric | < 1 GB | ~15 min | 3 GB |
| Fetal | < 1 GB | ~15 min | 3 GB |
| PTB-XL | 2 GB | ~30 min | 5 GB |
| **Total (all)** | **~60+ GB** | **~12 hrs** | **150+ GB** |

**Recommendation**: Start with CAMUS only (3.8 GB)

---

## 📊 What You Get from Each Dataset

### CAMUS ✅✅✅
- **Segmentation masks**: LV, LA, epicardium
- **Annotations**: Leclerc protocol (matches your spec)
- **Views**: Apical 4-chamber + 2-chamber
- **Cardiac phases**: End-diastole (ED) + End-systole (ES)
- **Patient data**: 500 patients, 50% with LVEF < 45%
- **Splits**: Train/Validation/Test
- **Benchmark**: Dice 0.931 (LV ED) - target for your model

### HMC-QU ✅✅
- **DICOM format**: De-identified
- **Multi-center**: Different institutions
- **Equipment**: GE Vivid (same as CAMUS!)
- **Use case**: Per-site bias calibration testing
- **No masks**: But useful for testing generalization

### EchoNet-Dynamic ✅✅
- **Video data**: Full cardiac cycle
- **Multiple views**: Not just apical 4-chamber
- **Labels**: Ejection fraction, segmentation
- **Large scale**: For robust validation
- **Temporal**: Test cardiac cycle consistency

---

## ✅ Quick Validation Checklist

After downloading CAMUS:

```bash
# Verify download
ls -lh camus-human-heart-data/

# Check structure
find . -type f | head -20

# Expected files:
# - camus_frames/ (patient images)
# - camus_masks/ (segmentation masks)
# - Training/Validation splits
# - ~500 patients total
```

---

## 🔍 Which Dataset to Use When

### For Training Initial Model
→ **CAMUS only** (3.8 GB)
- Has segmentation ground truth (masks)
- Published benchmark (Dice 0.931)
- Manageable size
- ~6-8 hours GPU training time

### For Testing Generalization
→ **CAMUS + HMC-QU**
- Different equipment/protocols
- Test per-site calibration
- Verify bias offset transfer
- Target: <5% performance drop

### For Multi-View Validation
→ **Add EchoNet-Dynamic**
- Test beyond apical 4-chamber
- Cardiac cycle consistency
- Different ultrasound machines
- Larger dataset for robust metrics

### For Edge Cases
→ **Add Pediatric + Fetal**
- Non-standard anatomy
- Graceful degradation testing
- Confidence threshold validation
- Low-confidence → clinician review

### For Decision Support Validation
→ **Add PTB-XL ECG data**
- Cross-validate aortic stenosis classification
- Electrical + structural correlation
- Multi-modal analysis
- Better clinical accuracy

---

## 📈 Recommended Analysis Timeline

| Week | Focus | Dataset | Goal | Metrics |
|------|-------|---------|------|---------|
| 1-2 | Baseline | CAMUS | Reproduce benchmarks | Dice > 0.931 |
| 3-4 | Generalization | CAMUS + HMC-QU | Per-site calibration | <5% drop |
| 5-6 | Multi-view | + EchoNet | Different views | Consistency |
| 7-8 | Edge cases | + Pediatric/Fetal | Robustness | Graceful fail |
| 9+ | Clinical | All datasets | IRB approval | Ready to deploy |

---

## 🎯 Use Cases by Dataset

### "I want to train a segmentation model"
✅ Use: **CAMUS**
- Has pixel-level masks
- Leclerc protocol
- 500 patients with splits

### "I want to test per-site bias"
✅ Use: **CAMUS + HMC-QU**
- Same equipment
- Different sites
- Calibration testing

### "I want real-world robustness"
✅ Use: **EchoNet-Dynamic**
- Multi-view
- Large scale
- Video data

### "I want to handle pediatric cases"
✅ Use: **Echonet Pediatric**
- Different anatomy
- Stanford data quality
- Representative samples

### "I want prenatal screening"
✅ Use: **Fetal Heart Dataset**
- Prenatal imaging
- 4-chamber view (apical analogue)
- Specialized use case

### "I want clinical decision support"
✅ Use: **PTB-XL + CAMUS**
- ECG + ultrasound correlation
- AS classification validation
- Multi-modal analysis

---

## ⚙️ System Requirements

**For CAMUS training:**
```
GPU: NVIDIA RTX 3080+ or equivalent
RAM: 16 GB minimum (32 GB recommended)
Disk: 20 GB free (for dataset + intermediate files)
Time: 6-8 hours training on GPU
```

**For all datasets:**
```
GPU: NVIDIA A100 or equivalent
RAM: 64 GB
Disk: 200 GB
Time: 2-3 weeks full pipeline
```

---

## 🔐 Data Privacy & Compliance

All Kaggle datasets are:
- ✅ De-identified (HIPAA Safe Harbor)
- ✅ Public research use
- ✅ Citation-required
- ✅ Academic/non-commercial approved

For production deployment:
- Still need IRB approval
- Clinical review workflow (your system has this!)
- Regional regulatory clearance
- Audit trail logging (implemented in your code)

---

## 📚 Additional Resources

**Official CAMUS Source:**
- Main site: https://www.creatis.insa-lyon.fr/Challenge/camus/
- Paper: Leclerc et al., IEEE TMI 2019
- Original DICOM data (if you need raw format)

**Kaggle Community:**
- Kernels/Notebooks using CAMUS
- Discussion sections for Q&A
- Upvoted solutions to learn from

**Your System Documentation:**
- README.md - Full architecture
- QUICKSTART.md - 5-minute tutorial
- heart_valve_analysis.py - Core code

---

## ✅ Next Steps

1. **Download CAMUS** (3.8 GB)
   ```bash
   kaggle datasets download -d shoybhasan/camus-human-heart-data
   ```

2. **Extract and explore**
   ```python
   from heart_valve_analysis import CAMUSDataset
   dataset = CAMUSDataset(split='val')
   # Load a sample and visualize
   ```

3. **Train baseline model**
   ```python
   from heart_valve_analysis import UNetSegmenter, SegmentationTrainer
   model = UNetSegmenter()
   trainer = SegmentationTrainer(model)
   # Start training...
   ```

4. **Validate against benchmarks**
   - Target Dice: > 0.931
   - X-error: < 1.35 mm
   - Y-error: < 0.75 mm

5. **Add more datasets** (weeks 3-4)
   - HMC-QU for multi-site
   - EchoNet for multi-view
   - etc.

---

**Ready to start?** Download CAMUS and follow the training examples in `heart_valve_analysis.py`! 🚀

Questions? See KAGGLE_DATASETS.md for detailed info on each dataset.
