# Heart Valve Defect Detection (research reference implementation)

A reference implementation of an AI-assisted transthoracic echocardiography (TTE) workflow for heart-valve analysis,
following the "Technical Architecture Specification: An AI-Assisted Heart Valve Defect Detection and Verification
Platform" (June 2026). It provides a U-Net segmentation model definition, deterministic mitral-valve hinge-point
extraction, a rule-based aortic-stenosis flag, a DICOM de-identification/ingestion scaffold and a clinician-review
workflow with an audit trail. Outputs are Python dicts, a text review report and a JSON audit trail.

> Status: research-only reference code. Not a medical device and not validated for clinical use; any clinical use
> requires IRB approval and clinician oversight. No patient data and no trained model weights are included.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/herrrickshaw/heart-valve-defect-detection/blob/main/notebooks/colab_test.ipynb)
New here? See [How to use this repo with AI](HOW_TO_USE_WITH_AI.md).

## What it does
- Defines a VGG16-style U-Net (`UNetSegmenter`) that outputs LV / LA / background probability maps.
- Extracts mitral-valve hinge points from a segmentation mask (step 2 of the "two-step method": segment, then a
  deterministic feature step) and applies the CAMUS bias offsets (+0.5 mm vertical, +0.3 mm horizontal).
- Flags severe aortic stenosis from valve area (< 1.0 cm2), gated to the US region (`RegionalFeatureGate`).
- Runs a clinician-review workflow: every finding starts `PENDING_REVIEW`, `finalize_study()` refuses to finalise
  while any finding is unreviewed, and only `CLINICIAN_APPROVED` findings are returned. All events are logged.
- Scaffolds DICOM anonymisation (PHI attribute removal, private-tag removal, date placeholder) and ingestion reporting.

## Quick start
```bash
git clone https://github.com/herrrickshaw/heart-valve-defect-detection.git
cd heart-valve-defect-detection
python -m venv .venv && source .venv/bin/activate
pip install numpy opencv-python torch   # full pipeline; see requirements.txt for the wider list
python clinical_review_system.py                     # review workflow demo, no torch needed
python heart_valve_analysis.py                       # pipeline demo on a synthetic random frame
```
`heart_valve_analysis.py` runs an untrained model on random noise, so its output only demonstrates the plumbing.

```python
from heart_valve_analysis import FrameMetadata, HeartVallueAnalysisPipeline, UNetSegmenter

pipeline = HeartVallueAnalysisPipeline(UNetSegmenter(in_channels=1, num_classes=3), region="US", device="cpu")
meta = FrameMetadata("ED", "a4c", 0, pixel_spacing_x=0.30, pixel_spacing_y=0.15, image_quality="good")
results = pipeline.analyze_study(frames, meta)       # frames: uint8 array (n_frames, H, W)
```
The class name `HeartVallueAnalysisPipeline` (sic) is kept for backwards compatibility.

## How it works
| Stage | Module | Output |
|---|---|---|
| Segment each frame | `heart_valve_analysis.UNetSegmenter` | (3, H, W) probability map |
| Hinge points + confidence | `valve_measurements.HingePointExtractor` | `HingePointEstimate` (mm) |
| Decision support (US only) | `valve_measurements.AorticStenosisClassifier` | `ClinicalFinding` or `None` |
| Route to review | `heart_valve_analysis.HeartVallueAnalysisPipeline` | dict with `flagged_findings` |
| Review, finalise, audit | `clinical_review_system.ClinicalReviewWorkflow` | approved findings + JSON audit trail |
| Ingest and de-identify | `dicom_integration.DICOMDataIngestionPipeline` | anonymised files + ingestion report |

Design decisions: the second step is deterministic (not another network) so hinge coordinates can be traced to mask
boundaries for sign-off; AS support is region-gated because the reference tool it is modelled on is FDA-cleared for
the US only; the CAMUS bias constants may need per-site recalibration. Clinical Precedence Rule (spec section 7): no
auto-finalised diagnosis path exists. Reference values from the spec (CAMUS): LV Dice 0.931 (ED) / 0.915 (ES), median
hinge error 1.35 mm (x) / 0.75 mm (y), 85th percentile 3.15 / 1.88 mm. They are targets coded in `CAMUS_Benchmark`;
this repo does not reproduce them.

## Repository layout
| Path | Purpose |
|---|---|
| `heart_valve_analysis.py` | U-Net, trainer, inference pipeline, validator (needs torch); re-exports the names below |
| `valve_measurements.py` | Torch-free data classes, CAMUS constants, `dice`, hinge extraction, AS rule |
| `clinical_review_system.py` | Review workflow, audit log, text report, regional gate |
| `dicom_integration.py` | DICOM anonymiser and ingestion scaffold (pydicom optional) |
| `DOWNLOAD_AND_ANALYZE.py` | Helpers/demo to fetch CAMUS via kagglehub or the kaggle CLI |
| `tests/` | pytest suite; torch, cv2 and sklearn are stubbed in `tests/stubs.py` |
| `notebooks/colab_test.ipynb` | Colab smoke test |
| `*.md` guides | Older overlapping guides (START_HERE, QUICKSTART, SETUP_AND_RUN, ...); this README is canonical |

## Data sources
| Source | What is taken | Access route | Licence / caveat |
|---|---|---|---|
| CAMUS challenge | ~500 apical 4-chamber TTE sequences, Leclerc annotation protocol | https://www.creatis.insa-lyon.fr/Challenge/camus/ (or Kaggle mirrors named in `DOWNLOAD_AND_ANALYZE.py`) | Download separately; check the dataset terms. Never committed here |
| Hospital DICOM exports | De-identified cine-loops | Supplied by partners | Must be de-identified at ingestion; none in this repo |

## Configuration
No environment variables are read. The Kaggle download helpers use your own `~/.kaggle/kaggle.json`; never commit it
(it is gitignored, as are `*.pt`/`*.pth` weights and DICOM folders).

## Testing
```bash
pip install pytest ruff numpy
ruff check . && pytest -q       # 30 tests, synthetic arrays only
```

## Limitations and known issues
- `ECHODataset` and `CAMUSDataset` are stubs (`__getitem__` returns `None`), so training on real data does not work yet.
- With the AS classifier enabled (region `US`), the pipeline feeds fixed placeholder values (area 0.8 cm2, 4.5 m/s), so
  every US study gets a "severe aortic stenosis" flag. It is a demo, not a measurement.
- Hinge-point "confidence" compares a mask with itself, so it is effectively always 1.0 for non-empty masks.
- `DICOMAnonymizer` falls back to a simulated report when pydicom is missing and ingestion still counts the file as
  successful; series metadata in ingestion is hard-coded. The date shift is a fixed placeholder. Do not use for real PHI.
- Ingestion's `*.dcm` glob visits top-level files twice; `_infer_view_type` matches "parasternal" before "short", so PSAX is never inferred.
- `SegmentationTrainer` applies `CrossEntropyLoss` to softmax outputs and expects different batch shapes in train vs validate.
- Spec performance claims (latency, review time) are not verified by code here and have been dropped.

## Licence
MIT (see `LICENSE`).
