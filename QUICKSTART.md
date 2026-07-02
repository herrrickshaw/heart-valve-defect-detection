# Heart Disease Analysis System — Quick Start Guide

## 🎯 What This System Does

This is an **AI-assisted cardiac ultrasound (TTE) analysis platform** that:

1. **Segments** left ventricle and left atrium from TTE video using deep learning
2. **Measures** mitral valve hinge point coordinates with uncertainty quantification
3. **Flags** cardiac defects (e.g., severe aortic stenosis) for clinician review
4. **Enforces** mandatory clinician sign-off on all AI findings before they enter the patient record
5. **Maintains** immutable audit trails for compliance and regulatory review

**Key principle**: AI findings are *supplemental only*. Clinician judgment takes precedence.

---

## 📦 Installation

```bash
# Clone or download the heart disease analysis files
cd Downloads

# Install dependencies
pip install torch torchvision pytorch-lightning numpy scikit-learn opencv-python pydicom

# Optional: GPU acceleration (CUDA)
# Follow PyTorch installation guide for your CUDA version
```

## 🚀 5-Minute Getting Started

### 1. Analyze a Cardiac Study

```python
from heart_valve_analysis import UNetSegmenter, HeartVallueAnalysisPipeline, FrameMetadata
import numpy as np
import torch

# Load pre-trained model (or train your own)
model = UNetSegmenter(in_channels=1, num_classes=3)
# model.load_state_dict(torch.load('trained_weights.pt'))

# Create analysis pipeline
pipeline = HeartVallueAnalysisPipeline(segmentation_model=model, region='US', device='cuda')

# Prepare cardiac study data (30 frames of a cardiac cycle)
frames = np.random.randint(0, 255, size=(30, 256, 256), dtype=np.uint8)
metadata = FrameMetadata(
    cardiac_phase='ED',           # End-diastole
    view_type='a4c',              # Apical 4-chamber view
    frame_idx=0,
    pixel_spacing_x=0.30,         # mm per pixel
    pixel_spacing_y=0.15,
    image_quality='good'
)

# Run analysis
results = pipeline.analyze_study(frames, metadata)

# Print findings
print(f"Flagged findings: {len(results['flagged_findings'])}")
for finding in results['flagged_findings']:
    print(f"  - {finding}")
```

### 2. Route Findings to Clinician Review

```python
from clinical_review_system import ClinicalReviewWorkflow, ReviewStatus

# Create review workflow for patient study
workflow = ClinicalReviewWorkflow(patient_id='P001', study_id='S001')

# Register AI-generated findings
for i, finding in enumerate(results['flagged_findings']):
    workflow.add_ai_finding(f'FINDING_{i:03d}', {
        'finding_type': finding.finding_type,
        'severity': finding.severity,
        'measurement_value': finding.measurement_value,
        'reference_threshold': finding.reference_threshold
    })

# Clinician reviews each finding
workflow.clinician_review(
    finding_id='FINDING_000',
    clinician_id='DR_001',
    clinician_name='Dr. Smith',
    status=ReviewStatus.CLINICIAN_APPROVED,  # or REJECTED, MODIFIED
    comments="Confirmed: meets severity criteria",
    confidence_in_ai=0.92
)

# Finalize study (only approved findings in record)
finalized = workflow.finalize_study()

# Export audit trail for compliance
workflow.export_audit_trail('audit_trail.json')
```

### 3. Ingest Hospital Data

```python
from dicom_integration import DICOMDataIngestionPipeline

# Initialize ingestion pipeline
pipeline = DICOMDataIngestionPipeline(output_dir='./ingested_data')

# Anonymize and ingest DICOM files from hospital
report = pipeline.ingest_from_hospital_export(
    hospital_export_dir='/path/to/hospital/export',
    institution_name='Hospital_A'
)

print(f"Ingested: {report['successful']} files")
print(f"Failed: {report['failed']} files")

# Register validation measurements (for calibration checks)
pipeline.register_validation_measurements(
    series_id='SERIES_00000',
    manual_measurements={
        'mitral_valve_diameter_mm': 28.5,
        'lv_diameter_mm': 45.2,
        'measurement_method': 'manual_calipers'
    }
)
```

---

## 📋 File Reference

| File | Purpose | Key Classes |
|------|---------|-------------|
| **heart_valve_analysis.py** | Core segmentation & measurement | `UNetSegmenter`, `HingePointExtractor`, `AorticStenosisClassifier` |
| **clinical_review_system.py** | Clinician review workflow & audit | `ClinicalReviewWorkflow`, `ClinicianReview`, `AuditLog` |
| **dicom_integration.py** | DICOM anonymization & data ingestion | `DICOMAnonymizer`, `DICOMDataIngestionPipeline` |
| **README.md** | Full architecture & design documentation | — |
| **QUICKSTART.md** | This file | — |

---

## 🔑 Key Concepts

### The Clinical Precedence Rule
**Golden rule from architecture spec, Section 7:**
> "AI-generated findings are supplemental. Clinical judgment must always take precedence. Every flagged finding requires explicit clinician sign-off before entering the patient record."

**Implementation:**
- All findings start in `PENDING_REVIEW` status
- Clinician must explicitly approve, reject, or modify each finding
- Only `CLINICIAN_APPROVED` findings are finalized
- No auto-diagnosis or silent inference path exists

### The Two-Step Method
**Segmentation + Deterministic Feature Extraction**

Step 1: U-Net produces pixel-level segmentation masks (LV, LA, background)
Step 2: Deterministic algorithm extracts mitral valve hinge points from boundary contact line

Why not end-to-end neural network for hinge points?
- **Interpretability**: Hinge coordinates traceable to segmentation mask
- **Auditability**: Clinicians can verify the extraction logic
- **Clinical sign-off**: More confidence in automated measurements

### Systematic Bias Calibration
CAMUS validation revealed consistent directional bias:
- **Horizontal**: +0.3 mm (toward right of frame)
- **Vertical**: +0.5 mm (toward bottom of frame)

The system applies these offsets post-inference to improve accuracy.
Per-site recalibration may be needed for new hospital equipment.

### Uncertainty Quantification
All measurements report uncertainty bounds:
```
Mitral Valve Diameter: 28.5 ± 1.35 mm
(median error ± 85th percentile range from CAMUS)
```

Do not treat point estimates as definitive; use uncertainty bands for clinical decision-making.

### Regional Feature Gating
Some decision-support features are jurisdiction-specific:
- **Aortic Stenosis classifier**: FDA-cleared for US only
- **EU, ASIA, GLOBAL**: Decision-support disabled by default
- In production, enable features only in approved regions

```python
from clinical_review_system import RegionalFeatureGate

enabled, reason = RegionalFeatureGate.is_feature_enabled(
    'aortic_stenosis_classifier', region='EU'
)
# Returns: (False, "aortic_stenosis_classifier in EU: PENDING_CE_MARKING")
```

---

## 🧪 Test the Full Workflow

Run the example demonstrations:

```bash
# 1. Core analysis
python heart_valve_analysis.py

# 2. Clinical review workflow
python clinical_review_system.py

# 3. DICOM ingestion
python dicom_integration.py
```

Each script includes a `main()` or `example_*()` function demonstrating the full workflow.

---

## 📊 Expected Performance

### Segmentation Accuracy (CAMUS Benchmark)
| Metric | Target | Acceptance |
|--------|--------|-----------|
| Dice LV (ED) | 0.931 | >0.91 |
| Dice LV (ES) | 0.915 | >0.90 |
| X-error (median) | 1.35 mm | <2.0 mm |
| Y-error (median) | 0.75 mm | <1.0 mm |

If your model doesn't meet these, check:
1. Are you using CAMUS for training/validation?
2. Is bias correction applied post-inference?
3. Have you validated on the official CAMUS test split?

### Clinical Review Workflow
| Step | Expected Time |
|------|---------------|
| AI analysis of 30-frame study | <30 sec (GPU) |
| Clinician review of 5-10 findings | ~2-5 min |
| Study finalization & audit export | <10 sec |

---

## ⚠️ Critical Constraints

### 1. Clinical Precedence (Non-Negotiable)
- AI never finalizes diagnoses autonomously
- Every finding requires clinician sign-off
- Audit trail must be complete and immutable
- No shortcuts: if clinician hasn't reviewed, the finding doesn't go in the record

### 2. Data De-identification (Mandatory)
- All hospital-shared data must be PHI-stripped at ingestion
- No patient names, IDs, dates, institution names in modeling pipeline
- HIPAA Safe Harbor de-identification required before use
- De-identification audit trail maintained for compliance

### 3. Heterogeneous Training Data (Essential)
- Deliberately include poor-quality, off-center, low-contrast images
- Off-center framing bias is a known error source — learn from it
- Train for generalization, not just peak accuracy on curated data
- Document which data split targets which objective (accuracy vs. generalization)

### 4. Per-Site Calibration (Recommended)
- New hospital equipment may have different pixel spacing
- Bias calibration constants (0.3mm, 0.5mm) may not transfer 1:1
- When adding multi-site data, recalibrate bias offsets on new institution's validation set
- Track per-site performance metrics separately

### 5. Regional Feature Gating (Compliance)
- Decision-support features only enabled in approved jurisdictions
- AS classifier: US only (FDA-cleared)
- CE marking or equivalent required before European deployment
- Default to "measurement only" mode outside approved regions

---

## 🔧 Customization & Extension

### Train Your Own Model

```python
from heart_valve_analysis import UNetSegmenter, SegmentationTrainer, CAMUSDataset
from torch.utils.data import DataLoader

# Load CAMUS dataset (download from https://www.creatis.insa-lyon.fr/Challenge/camus/)
train_dataset = CAMUSDataset(split='train')
val_dataset = CAMUSDataset(split='val')

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Train
model = UNetSegmenter(in_channels=1, num_classes=3)
trainer = SegmentationTrainer(model, device='cuda', learning_rate=1e-4)

for epoch in range(50):
    loss = trainer.train_epoch(train_loader)
    metrics = trainer.validate(val_loader)
    print(f"Epoch {epoch}: Loss={loss:.4f}, Dice={metrics['dice_lv']:.4f}")

torch.save(model.state_dict(), 'trained_model.pt')
```

### Add New Findings

```python
from heart_valve_analysis import ClinicalFinding

# Define new finding type
mitral_regurgitation = ClinicalFinding(
    finding_type='mitral_regurgitation',
    severity='moderate',
    measurement_value=45.0,  # Regurgitant volume (mL)
    reference_threshold=30.0,
    requires_review=True,
    supporting_measurements={
        'regurgitant_orifice_area': 25.0,  # mm²
        'guideline': 'ACC/AHA 2017'
    }
)

# Add to workflow
workflow.add_ai_finding('MR_001', {
    'finding_type': finding.finding_type,
    'severity': finding.severity,
    'measurement_value': finding.measurement_value,
    'reference_threshold': finding.reference_threshold
})
```

### Multi-Site Deployment

```python
from clinical_review_system import RegionalFeatureGate

# Get deployment mode for each site
sites = ['Hospital_US', 'Hospital_EU', 'Hospital_ASIA']

for site in sites:
    region = site.split('_')[1]
    mode = RegionalFeatureGate.get_deployment_mode(region)
    
    print(f"{site}: {mode['deployment_mode']}")
    if not mode['as_classifier_enabled']:
        print(f"  ⚠️  AS classifier disabled: {mode['regulatory_status']}")
```

---

## 🐛 Troubleshooting

### Segmentation Accuracy Below Benchmark
1. **Check CAMUS metrics**: Did you validate on official CAMUS test split?
2. **Verify bias correction**: Is `apply_bias_correction=True` in `extract_hinge_points()`?
3. **Review training data**: Are you training on CAMUS + heterogeneous hospital data?
4. **Inspect failed cases**: Which image quality levels are causing errors?

### Clinician Review Incomplete
1. Check pending findings: `workflow.get_pending_reviews()`
2. Ensure all findings have explicit clinician decisions
3. Call `finalize_study()` only after all reviews complete

### Regional Feature Disabled
1. Check jurisdiction: Is your deployment region in the clearance list?
2. Update regional config in `RegionalFeatureGate.REGULATORY_STATUS`
3. Re-request regulatory approval if deploying in new region

### De-identification Failed
1. Install pydicom: `pip install pydicom`
2. Check hospital export format (DICOM standard?)
3. Review anonymization log: `pipeline.anonymizer.export_anonymization_log('log.json')`

---

## 📚 Next Steps

1. **Download CAMUS dataset** → https://www.creatis.insa-lyon.fr/Challenge/camus/
2. **Train baseline model** → Run training script against CAMUS
3. **Validate benchmarks** → Confirm Dice >0.92 on test split
4. **Set up clinical review** → Integrate review workflow with your EHR
5. **Plan multi-site rollout** → Per-site calibration & regulatory approval

---

## 📖 Full Documentation

See **README.md** for complete architecture, design decisions, references, and future enhancement roadmap.

Key sections:
- **Architecture Overview** → 3-layer framework
- **Data Sources & Validation** → CAMUS, hospital data, tier 3 verification
- **Known Limitations** → Bias, off-center framing, spatial resolution
- **Regulatory & Compliance** → Clinical Precedence Rule, data sharing
- **Building Sequence** → Recommended implementation order

---

## 💬 Support

Questions? Check:
1. **Architecture Spec** → Section-by-section explanation of design choices
2. **README.md** → Full system documentation
3. **Source Code** → Docstrings explain why each decision was made
4. **Example Scripts** → `main()` functions in each module show expected usage

---

**Last Updated**: June 2026  
**System Status**: Reference Implementation — Ready for IRB review and institutional deployment with clinician oversight

Good luck! 🏥❤️
