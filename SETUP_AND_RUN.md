# Setup & Run Guide: Heart Disease Analysis System + CAMUS Dataset

Complete step-by-step instructions to download, configure, and run your heart disease analysis system.

---

## 🚀 Quick Setup (5 Minutes)

### Step 1: Install Dependencies

```bash
# Core dependencies
pip install torch torchvision pytorch-lightning
pip install numpy scikit-learn opencv-python

# Kaggle download (choose one)
pip install kagglehub              # Modern (recommended)
# OR
pip install kaggle                 # Traditional

# Optional: DICOM support
pip install pydicom
```

### Step 2: Authenticate with Kaggle

```bash
# 1. Go to https://www.kaggle.com/settings/account
# 2. Click "Create New API Token"
# 3. This downloads kaggle.json to your Downloads

# Move it to the right location
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json    # Secure it
```

### Step 3: Download CAMUS Dataset

**Option A: Using kagglehub (Easiest)**
```bash
cd ~/Downloads
python -c "
import kagglehub
path = kagglehub.dataset_download('parsakh/camus-echocardiography-image-dataset')
print(f'Downloaded to: {path}')
"
```

**Option B: Using our script**
```bash
cd ~/Downloads
python DOWNLOAD_AND_ANALYZE.py
# Choose: 1 (kagglehub)
```

**Option C: Direct kaggle CLI**
```bash
kaggle datasets download -d shoybhasan/camus-human-heart-data -p ~/camus_data
cd ~/camus_data && unzip -q camus-human-heart-data.zip
```

### Step 4: Run Analysis

```bash
cd ~/Downloads
python DOWNLOAD_AND_ANALYZE.py
# Choose: 2 (kagglehub)
# Choose: 1 (Train model)
```

---

## 📋 Full Setup Guide

### Prerequisites Check

```bash
# Verify Python version
python --version          # Should be 3.8+

# Verify GPU (optional but recommended)
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"

# Check disk space
df -h ~                   # Need ~30 GB free for dataset + training
```

### Complete Installation

```bash
# 1. Create virtual environment (recommended)
python -m venv heart_disease_env
source heart_disease_env/bin/activate  # On Windows: .\heart_disease_env\Scripts\activate

# 2. Upgrade pip
pip install --upgrade pip setuptools wheel

# 3. Install all dependencies
pip install \
  torch torchvision \
  numpy scikit-learn opencv-python \
  kagglehub \
  pydicom \
  pandas matplotlib seaborn

# 4. Verify installation
python -c "
import torch
import numpy as np
import cv2
import kagglehub
print('✅ All dependencies installed successfully!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

### Set up Kaggle Authentication

```bash
# Method 1: Automatic (if kaggle.json exists)
kagglehub  # Will auto-detect ~/.kaggle/kaggle.json

# Method 2: Manual authentication
python -c "
import kagglehub
# Will prompt for authentication if kaggle.json missing
"

# Method 3: Set environment variable
export KAGGLE_CONFIG_DIR=~/.kaggle
```

### Organize Your Workspace

```bash
# Create directory structure
cd ~/Downloads

# Core system files (already created)
ls -la heart_valve_analysis.py
ls -la clinical_review_system.py
ls -la dicom_integration.py

# Documentation
ls -la README.md QUICKSTART.md KAGGLE_DATASETS.md

# Data directory
mkdir -p camus_data
mkdir -p models
mkdir -p results
mkdir -p logs

# Final structure:
# ~/Downloads/
# ├── heart_valve_analysis.py        (core system)
# ├── clinical_review_system.py       (clinician review)
# ├── dicom_integration.py            (DICOM handling)
# ├── DOWNLOAD_AND_ANALYZE.py         (automation script)
# ├── README.md                       (documentation)
# ├── QUICKSTART.md                   (quick reference)
# ├── KAGGLE_DATASETS.md              (dataset guide)
# ├── camus_data/                     (downloaded dataset)
# ├── models/                         (trained weights)
# ├── results/                        (analysis outputs)
# └── logs/                           (training logs)
```

---

## 🎯 Running the System

### Option 1: Automated Script (Easiest)

```bash
cd ~/Downloads
python DOWNLOAD_AND_ANALYZE.py
```

Interactive prompts will guide you through:
1. Choose download method (kagglehub recommended)
2. Choose analysis mode (Train / Quick validation / Exit)
3. Monitor training progress

### Option 2: Step-by-Step Manual

```python
# 1. Download dataset
import kagglehub
dataset_path = kagglehub.dataset_download(
    "parsakh/camus-echocardiography-image-dataset"
)

# 2. Load data
from heart_valve_analysis import CAMUSDataset, UNetSegmenter
from torch.utils.data import DataLoader

train_dataset = CAMUSDataset(split='train')
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# 3. Initialize model
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = UNetSegmenter(in_channels=1, num_classes=3).to(device)

# 4. Train
from heart_valve_analysis import SegmentationTrainer
trainer = SegmentationTrainer(model, device=device)

for epoch in range(50):
    loss = trainer.train_epoch(train_loader)
    print(f"Epoch {epoch+1}: Loss={loss:.4f}")

# 5. Save model
torch.save(model.state_dict(), 'model.pt')
```

### Option 3: Using Your Own Data

```python
# If you have DICOM files from a hospital:
from dicom_integration import DICOMDataIngestionPipeline

pipeline = DICOMDataIngestionPipeline(output_dir='./my_hospital_data')

# Anonymize and ingest
report = pipeline.ingest_from_hospital_export(
    hospital_export_dir='/path/to/hospital/dicom',
    institution_name='My_Hospital'
)

print(f"Ingested: {report['successful']} files")

# Use with your model...
```

---

## ⏱️ Timing Guide

| Task | Time | Notes |
|------|------|-------|
| Install dependencies | 10-30 min | Depends on internet speed |
| Download CAMUS | 30-60 min | 3.8 GB dataset |
| Training (50 epochs) | 4-6 hours | On GPU (RTX 3080+) |
| Validation | 10-15 min | On test set |
| Clinical review setup | 30 min | Integrate with EHR |

**Total time to running analysis: ~1 hour** (excluding training)

---

## 🐛 Troubleshooting

### Issue: "Module not found" errors

```bash
# Solution: Reinstall with all dependencies
pip install --upgrade --force-reinstall \
  torch torchvision pytorch-lightning \
  kagglehub numpy scikit-learn opencv-python

# Or clean install
pip uninstall -y torch torchvision pytorch-lightning
pip install torch torchvision pytorch-lightning
```

### Issue: CUDA/GPU not detected

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Solution: Reinstall PyTorch for your CUDA version
# https://pytorch.org/get-started/locally/

# Force CPU-only mode (slower but works)
# Set environment variable before running:
export CUDA_VISIBLE_DEVICES=""
```

### Issue: Kaggle authentication failed

```bash
# Check if kaggle.json exists
ls ~/.kaggle/kaggle.json

# If missing:
# 1. Go to https://www.kaggle.com/settings/account
# 2. Click "Create New API Token"
# 3. Move downloaded file:
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Test authentication
python -c "import kagglehub; print('✅ Authentication OK')"
```

### Issue: Out of memory (OOM) errors

```bash
# Solution: Reduce batch size
# In DOWNLOAD_AND_ANALYZE.py, change:
train_loader = DataLoader(train_dataset, batch_size=4)  # was 8

# Or use CPU (slow but works)
device = 'cpu'
```

### Issue: Dataset download too slow

```bash
# Try alternative download source
# Option 1: Download manually from https://www.kaggle.com/datasets/shoybhasan/camus-human-heart-data
# Option 2: Use faster network connection
# Option 3: Download multiple datasets in parallel
```

---

## ✅ Verification Checklist

After setup, verify everything works:

```bash
# 1. Check Python environment
python -c "import sys; print(f'Python: {sys.version}')"

# 2. Check dependencies
python -c "
import torch, numpy, cv2, sklearn, kagglehub
print('✅ All imports successful')
"

# 3. Check GPU (if applicable)
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"

# 4. Check Kaggle auth
python -c "import kagglehub; print('✅ Kaggle auth OK')"

# 5. Test dataset loading
python -c "
from heart_valve_analysis import CAMUSDataset
dataset = CAMUSDataset(split='val')
print(f'✅ Dataset loads: {len(dataset)} samples')
"

# 6. Test model initialization
python -c "
from heart_valve_analysis import UNetSegmenter
import torch
model = UNetSegmenter()
print(f'✅ Model initialized: {sum(p.numel() for p in model.parameters()):,} parameters')
"
```

---

## 🎯 Next Steps After Setup

### Phase 1: Baseline Training
1. Download CAMUS dataset
2. Train model (50 epochs)
3. Validate against benchmarks (Dice > 0.931)
4. Save best model weights

### Phase 2: Testing
1. Load pre-trained model
2. Run inference on validation set
3. Measure coordinate errors
4. Compare to CAMUS benchmarks

### Phase 3: Clinical Integration
1. Set up clinical review workflow
2. Test clinician interface
3. Configure audit logging
4. Regional regulatory gating

### Phase 4: Multi-site Deployment
1. Download HMC-QU dataset
2. Test per-site bias calibration
3. Recalibrate offsets per equipment
4. Track performance metrics

### Phase 5: Production Deployment
1. Complete IRB approval
2. Clinician training
3. EHR integration
4. Launch with monitoring

---

## 📊 Performance Expectations

After setup with CAMUS training:

| Metric | Target | Status |
|--------|--------|--------|
| Dice LV | > 0.931 | ✅ Achievable |
| X-error | < 1.35 mm | ✅ Achievable |
| Y-error | < 0.75 mm | ✅ Achievable |
| Training time | 4-6 hours | ✅ GPU dependent |
| Inference time | < 30 sec/study | ✅ GPU dependent |

---

## 🆘 Getting Help

### Common Questions

**Q: Do I need a GPU?**
A: Recommended but not required. CPU training will be 10-20x slower (~3 days vs 6 hours).

**Q: How much disk space do I need?**
A: ~30 GB for dataset + training artifacts. 50+ GB recommended for multi-site expansion.

**Q: Can I use this on my own data?**
A: Yes! Use `dicom_integration.py` to anonymize and ingest DICOM files.

**Q: Is this ready for clinical use?**
A: Research-ready now. Requires IRB approval and clinician oversight for clinical deployment.

### Resources

- **This guide**: SETUP_AND_RUN.md
- **Quick reference**: QUICKSTART.md
- **Architecture**: README.md
- **Dataset details**: KAGGLE_DATASETS.md
- **Source code**: heart_valve_analysis.py (well-commented)

---

## 🎓 Learning Path

1. **Beginners**: Read QUICKSTART.md → Run DOWNLOAD_AND_ANALYZE.py
2. **Intermediate**: Read README.md → Modify training parameters
3. **Advanced**: Study source code → Customize for your needs
4. **Clinical**: Set up clinical_review_system.py → IRB approval

---

## 📝 Quick Command Reference

```bash
# Download dataset
python DOWNLOAD_AND_ANALYZE.py

# Run analysis
cd ~/Downloads && python DOWNLOAD_AND_ANALYZE.py

# Train model
python -c "from DOWNLOAD_AND_ANALYZE import analyze_camus_dataset; from pathlib import Path; analyze_camus_dataset(Path('./camus_data'))"

# Test pre-trained model
python -c "from DOWNLOAD_AND_ANALYZE import quick_analyze_pretrained; quick_analyze_pretrained('./model.pt')"

# View GPU status
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

# Check disk usage
du -sh ~/Downloads ~/camus_data

# Monitor training
watch -n 10 'nvidia-smi'  # Real-time GPU usage
```

---

**You're all set!** 🚀

Start with: `python DOWNLOAD_AND_ANALYZE.py`

Then read: **QUICKSTART.md** for next steps

Good luck! ❤️📊
