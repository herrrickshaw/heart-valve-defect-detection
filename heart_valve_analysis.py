"""
Heart Valve Defect Detection and Analysis Platform
Based on: Technical Architecture Specification
Implements AI-assisted TTE (Transthoracic Echocardiography) analysis
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.metrics import dice, jaccard_score
import cv2
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# 1. DATA STRUCTURES & CONFIGURATION
# ============================================================================

@dataclass
class FrameMetadata:
    """Metadata for cardiac imaging frame"""
    cardiac_phase: str  # 'ED' (end-diastole) or 'ES' (end-systole)
    view_type: str  # 'a4c' (apical 4-chamber), 'PLAX', etc.
    frame_idx: int
    pixel_spacing_x: float  # mm per pixel
    pixel_spacing_y: float  # mm per pixel
    image_quality: str  # 'good', 'acceptable', 'poor'


@dataclass
class HingePointEstimate:
    """Mitral valve hinge point coordinates and confidence"""
    anterior_hinge: Tuple[float, float]  # (x, y) in mm
    posterior_hinge: Tuple[float, float]  # (x, y) in mm
    x_uncertainty_mm: float = 1.35  # Median x-error from CAMUS
    y_uncertainty_mm: float = 0.75  # Median y-error from CAMUS
    confidence_score: float = 0.0  # 0-1

    @property
    def mv_diameter_mm(self) -> float:
        """Compute mitral valve diameter from hinge points"""
        dx = self.posterior_hinge[0] - self.anterior_hinge[0]
        dy = self.posterior_hinge[1] - self.anterior_hinge[1]
        return np.sqrt(dx**2 + dy**2)


@dataclass
class ClinicalFinding:
    """Flagged clinical finding for clinician review"""
    finding_type: str  # 'severe_aortic_stenosis', 'mitral_regurgitation', etc.
    severity: str  # 'mild', 'moderate', 'severe'
    measurement_value: float
    reference_threshold: float
    requires_review: bool = True
    supporting_measurements: Dict = None

    def __repr__(self) -> str:
        return (f"{self.finding_type.upper()}: {self.severity} "
                f"({self.measurement_value:.2f} vs. threshold {self.reference_threshold:.2f})")


class CAMUS_Benchmark:
    """Reference benchmarks from CAMUS dataset validation"""
    DICE_LV_ED = 0.931
    DICE_LV_ES = 0.915
    DICE_LA_ED = 0.923

    X_ERROR_MEDIAN_MM = 1.35
    Y_ERROR_MEDIAN_MM = 0.75
    X_ERROR_P85_MM = 3.15
    Y_ERROR_P85_MM = 1.88

    # Systematic bias (to be applied post-inference)
    BIAS_VERTICAL_MM = 0.5
    BIAS_HORIZONTAL_MM = 0.3


# ============================================================================
# 2. U-NET SEGMENTATION ARCHITECTURE (VGG16 Backbone)
# ============================================================================

class VGG16DoubleConv(nn.Module):
    """Double convolution block with VGG16 design principles"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNetSegmenter(nn.Module):
    """
    U-Net with VGG16 encoder backbone for cardiac chamber segmentation.

    Outputs:
    - Channel 0: Left Ventricle (LV) segmentation
    - Channel 1: Left Atrium (LA) segmentation
    - Channel 2: Background

    Per Leclerc protocol: LV contour delineation terminates at MV leaflet hinge points.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 3):
        super().__init__()

        # Encoder (VGG16-inspired)
        self.enc1 = VGG16DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = VGG16DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = VGG16DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = VGG16DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = VGG16DoubleConv(512, 1024)

        # Decoder (upsampling path)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = VGG16DoubleConv(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = VGG16DoubleConv(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = VGG16DoubleConv(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = VGG16DoubleConv(128, 64)

        # Output segmentation
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.out_softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        x = self.pool1(enc1)

        enc2 = self.enc2(x)
        x = self.pool2(enc2)

        enc3 = self.enc3(x)
        x = self.pool3(enc3)

        enc4 = self.enc4(x)
        x = self.pool4(enc4)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.upconv4(x)
        x = torch.cat([x, enc4], dim=1)
        x = self.dec4(x)

        x = self.upconv3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)

        x = self.upconv2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)

        x = self.upconv1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)

        # Output
        x = self.out_conv(x)
        return self.out_softmax(x)


# ============================================================================
# 3. FEATURE EXTRACTION: Hinge Point Detection (Step 2 of Two-Step Method)
# ============================================================================

class HingePointExtractor:
    """
    Deterministic feature-based extraction of mitral valve hinge points.

    Step 2 of the Two-Step Method:
    1. Identify the contact line where LV and LA segments meet
    2. Extract the endpoints as anterior (aMVL) and posterior (pMVL) hinge points
    """

    @staticmethod
    def extract_hinge_points(
        segmentation_mask: np.ndarray,
        metadata: FrameMetadata,
        apply_bias_correction: bool = True
    ) -> HingePointEstimate:
        """
        Extract MV hinge points from segmentation mask.

        Args:
            segmentation_mask: Shape (H, W, 3) with channels [LV, LA, BG]
            metadata: Frame metadata including pixel spacing
            apply_bias_correction: Apply systematic bias offsets per CAMUS calibration

        Returns:
            HingePointEstimate with hinge coordinates in mm
        """
        lv_mask = segmentation_mask[:, :, 0] > 0.5
        la_mask = segmentation_mask[:, :, 1] > 0.5

        # Find contact line: pixels on LV boundary adjacent to LA
        lv_boundary = cv2.Canny(lv_mask.astype(np.uint8) * 255, 100, 200)
        contact_line = lv_boundary & la_mask

        # Extract hinge point coordinates (endpoints of contact line)
        y_coords, x_coords = np.where(contact_line)

        if len(x_coords) < 2:
            # Fallback: use LV boundary extremes
            lv_y, lv_x = np.where(lv_mask)
            anterior = (np.min(lv_x), np.mean(lv_y))
            posterior = (np.max(lv_x), np.mean(lv_y))
        else:
            # Find two endpoints of contact line
            points = np.column_stack([x_coords, y_coords])
            # Use PCA or extrema to find endpoints
            extreme_idx = np.argsort(np.linalg.norm(
                points - points.mean(axis=0), axis=1
            ))[-2:]
            anterior = tuple(points[extreme_idx[0]])
            posterior = tuple(points[extreme_idx[1]])

        # Convert pixel coordinates to millimeters
        anterior_mm = (
            anterior[0] * metadata.pixel_spacing_x,
            anterior[1] * metadata.pixel_spacing_y
        )
        posterior_mm = (
            posterior[0] * metadata.pixel_spacing_x,
            posterior[1] * metadata.pixel_spacing_y
        )

        # Apply systematic bias correction per CAMUS calibration
        if apply_bias_correction:
            anterior_mm = (
                anterior_mm[0] + CAMUS_Benchmark.BIAS_HORIZONTAL_MM,
                anterior_mm[1] + CAMUS_Benchmark.BIAS_VERTICAL_MM
            )
            posterior_mm = (
                posterior_mm[0] + CAMUS_Benchmark.BIAS_HORIZONTAL_MM,
                posterior_mm[1] + CAMUS_Benchmark.BIAS_VERTICAL_MM
            )

        # Compute confidence based on segmentation quality
        lv_dice = dice(lv_mask, segmentation_mask[:, :, 0] > 0.5)
        la_dice = dice(la_mask, segmentation_mask[:, :, 1] > 0.5)
        confidence = (lv_dice + la_dice) / 2

        return HingePointEstimate(
            anterior_hinge=anterior_mm,
            posterior_hinge=posterior_mm,
            confidence_score=float(confidence)
        )


# ============================================================================
# 4. CLINICAL DECISION SUPPORT: Aortic Stenosis Classification
# ============================================================================

class AorticStenosisClassifier:
    """
    Decision-support layer for Aortic Stenosis (AS) severity classification.

    Modeled on EchoSolv™ AS reference. Takes structural/functional measurements
    and outputs severity flags for clinician review.

    REGULATORY CONSTRAINT: FDA-cleared for US only. Must be geofenced in production.
    """

    # Clinical thresholds per ACC/AHA guidelines
    AORTIC_VALVE_AREA_THRESHOLDS = {
        'mild': (1.5, float('inf')),      # cm²
        'moderate': (1.0, 1.5),            # cm²
        'severe': (0.0, 1.0),              # cm²
    }

    PEAK_VELOCITY_THRESHOLDS = {
        'mild': (0.0, 2.5),                # m/s
        'moderate': (2.5, 4.0),            # m/s
        'severe': (4.0, float('inf')),     # m/s
    }

    @staticmethod
    def classify_aortic_stenosis(
        aortic_valve_area_cm2: float,
        peak_velocity_ms: float
    ) -> Optional[ClinicalFinding]:
        """
        Classify aortic stenosis severity.

        Args:
            aortic_valve_area_cm2: Aortic valve area in cm²
            peak_velocity_ms: Peak aortic velocity in m/s

        Returns:
            ClinicalFinding if severe AS detected, else None
        """
        # Determine severity from valve area (primary criterion)
        severity = None
        if aortic_valve_area_cm2 < 1.0:
            severity = 'severe'
        elif aortic_valve_area_cm2 < 1.5:
            severity = 'moderate'
        else:
            severity = 'mild'

        # Flag only if severe (requires clinician review)
        if severity == 'severe':
            return ClinicalFinding(
                finding_type='severe_aortic_stenosis',
                severity=severity,
                measurement_value=aortic_valve_area_cm2,
                reference_threshold=1.0,
                requires_review=True,
                supporting_measurements={
                    'peak_velocity_ms': peak_velocity_ms,
                    'guideline': 'ACC/AHA 2014'
                }
            )

        return None


# ============================================================================
# 5. DATA LOADING & PREPROCESSING
# ============================================================================

class ECHODataset(Dataset):
    """
    Cardiac ultrasound dataset loader with de-identification.

    Expects:
    - DICOM cine-loop files (anonymized, PHI-stripped)
    - View-tagged metadata only (no narrative notes)
    - Frame rate and pixel spacing per study
    """

    def __init__(
        self,
        dicom_dir: Path,
        transform=None,
        target_shape: Tuple[int, int] = (256, 256)
    ):
        self.dicom_dir = Path(dicom_dir)
        self.transform = transform
        self.target_shape = target_shape
        self.frames = []
        self._load_frames()

    def _load_frames(self):
        """Load and catalog anonymized DICOM frames"""
        # In production: use pydicom to load actual DICOM files
        # For demo: simulate dataset structure
        logger.info(f"Loading frames from {self.dicom_dir}")
        # Frame loading logic here

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        # Return (image, segmentation_mask, metadata)
        pass


class CAMUSDataset(Dataset):
    """
    CAMUS benchmark dataset for training and validation.

    Reference dataset: ~500 2D apical-4-chamber TTE sequences.
    Annotation: Leclerc protocol (LV contour terminates at MV hinge).
    """

    def __init__(self, split: str = 'train', target_shape: Tuple[int, int] = (256, 256)):
        self.split = split
        self.target_shape = target_shape
        self.samples = []

    def download(self):
        """Download CAMUS from public repository"""
        # https://www.creatis.insa-lyon.fr/Challenge/camus/
        logger.info("CAMUS download instructions at https://www.creatis.insa-lyon.fr/Challenge/camus/")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pass


# ============================================================================
# 6. TRAINING PIPELINE
# ============================================================================

class SegmentationTrainer:
    """
    Training harness for U-Net segmentation model.

    Validation metrics: Dice coefficient (LV/LA), coordinate error (x/y, mm).
    """

    def __init__(
        self,
        model: UNetSegmenter,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 1e-4
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.history = {'train_loss': [], 'val_dice': []}

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0

        for images, masks in train_loader:
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Forward pass
            logits = self.model(images)
            loss = self.criterion(logits, masks)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        self.history['train_loss'].append(epoch_loss)
        logger.info(f"Epoch train loss: {epoch_loss:.4f}")
        return epoch_loss

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate against CAMUS benchmark"""
        self.model.eval()
        lv_dice_scores = []
        la_dice_scores = []
        x_errors = []
        y_errors = []

        with torch.no_grad():
            for images, masks, metadata in val_loader:
                images = images.to(self.device)
                logits = self.model(images)
                preds = torch.softmax(logits, dim=1)

                # Compute Dice for LV (channel 0) and LA (channel 1)
                lv_pred = preds[:, 0, :, :].cpu().numpy()
                la_pred = preds[:, 1, :, :].cpu().numpy()
                lv_true = masks[:, 0, :, :].cpu().numpy()
                la_true = masks[:, 1, :, :].cpu().numpy()

                lv_dice_scores.extend([
                    dice(lv_pred[i], lv_true[i])
                    for i in range(len(lv_pred))
                ])
                la_dice_scores.extend([
                    dice(la_pred[i], la_true[i])
                    for i in range(len(la_pred))
                ])

        metrics = {
            'dice_lv': np.median(lv_dice_scores),
            'dice_la': np.median(la_dice_scores),
            'benchmark_lv_ed': CAMUS_Benchmark.DICE_LV_ED,
            'benchmark_lv_es': CAMUS_Benchmark.DICE_LV_ES,
        }
        self.history['val_dice'].append(metrics['dice_lv'])
        logger.info(f"Validation Dice LV: {metrics['dice_lv']:.4f} (benchmark: {metrics['benchmark_lv_ed']:.4f})")

        return metrics


# ============================================================================
# 7. INFERENCE PIPELINE WITH CLINICAL REVIEW
# ============================================================================

class HeartVallueAnalysisPipeline:
    """
    End-to-end inference pipeline implementing the Clinical Precedence Rule.

    CRITICAL CONSTRAINT: Every AI-generated finding requires explicit clinician
    sign-off before entering the patient record. No auto-finalized diagnosis path.
    """

    def __init__(
        self,
        segmentation_model: UNetSegmenter,
        region: str = 'US',  # 'US', 'EU', 'ASIA', etc. (controls feature gating)
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.segmentation_model = segmentation_model.to(device)
        self.device = device
        self.region = region
        self.hinge_extractor = HingePointExtractor()

        # Feature gating per regulatory scope
        self.as_classifier_enabled = (region == 'US')  # FDA-cleared for US only
        if not self.as_classifier_enabled:
            logger.warning(f"AS decision-support disabled for region={region}")

    def analyze_study(
        self,
        frame_sequence: np.ndarray,  # Shape: (num_frames, height, width)
        metadata: FrameMetadata
    ) -> Dict:
        """
        Analyze a cardiac study and flag findings for clinician review.

        Args:
            frame_sequence: Stack of cardiac cycle frames
            metadata: Frame acquisition metadata

        Returns:
            Dictionary with segmentation, measurements, and flagged findings
        """
        results = {
            'frames_analyzed': len(frame_sequence),
            'metadata': metadata,
            'hinge_point_estimates': [],
            'flagged_findings': [],
            'requires_clinician_review': False
        }

        # Segment each frame
        with torch.no_grad():
            for frame_idx, frame in enumerate(frame_sequence):
                # Preprocess: normalize to [0, 1]
                frame_norm = frame.astype(np.float32) / 255.0
                frame_tensor = torch.from_numpy(frame_norm[np.newaxis, np.newaxis, :, :]).to(self.device)

                # Segment
                segmentation = self.segmentation_model(frame_tensor)
                seg_mask = segmentation[0].cpu().numpy()

                # Extract hinge points (Step 2 of Two-Step Method)
                hinge_estimate = self.hinge_extractor.extract_hinge_points(
                    seg_mask, metadata, apply_bias_correction=True
                )
                results['hinge_point_estimates'].append(hinge_estimate)

                # Log measurement
                logger.info(
                    f"Frame {frame_idx} ({metadata.cardiac_phase}): "
                    f"MV diameter = {hinge_estimate.mv_diameter_mm:.2f} mm "
                    f"(confidence: {hinge_estimate.confidence_score:.2f})"
                )

                # Check for measurement confidence below threshold
                if hinge_estimate.confidence_score < 0.85:
                    results['flagged_findings'].append(
                        ClinicalFinding(
                            finding_type='low_confidence_segmentation',
                            severity='moderate',
                            measurement_value=hinge_estimate.confidence_score,
                            reference_threshold=0.85,
                            supporting_measurements={
                                'frame_idx': frame_idx,
                                'image_quality': metadata.image_quality
                            }
                        )
                    )

        # Decision support layer (AS classification, if enabled)
        if self.as_classifier_enabled and len(results['hinge_point_estimates']) > 0:
            # Estimate aortic valve area from measurements (placeholder)
            mv_diameters = [h.mv_diameter_mm for h in results['hinge_point_estimates']]
            avg_mv_diameter = np.mean(mv_diameters)

            # Placeholder: in production, compute actual aortic valve area from echo
            aortic_valve_area_cm2 = 0.8  # Simulated severe AS
            peak_velocity_ms = 4.5

            as_finding = AorticStenosisClassifier.classify_aortic_stenosis(
                aortic_valve_area_cm2, peak_velocity_ms
            )
            if as_finding:
                results['flagged_findings'].append(as_finding)

        # Set review flag if any findings present
        results['requires_clinician_review'] = len(results['flagged_findings']) > 0

        return results


# ============================================================================
# 8. VALIDATION & METRICS
# ============================================================================

class SegmentationValidator:
    """
    Comprehensive validation harness comparing against CAMUS benchmarks.
    """

    @staticmethod
    def compute_metrics(
        predictions: np.ndarray,  # (B, H, W, 3)
        ground_truth: np.ndarray,  # (B, H, W, 3)
        pixel_spacing_x: float,
        pixel_spacing_y: float
    ) -> Dict[str, float]:
        """
        Compute Dice, coordinate error, and bias metrics.

        Args:
            predictions: Model segmentation outputs
            ground_truth: Annotated reference masks
            pixel_spacing_x, pixel_spacing_y: Resolution in mm/pixel

        Returns:
            Metrics dict with Dice, coordinate errors, bias, etc.
        """
        batch_size = predictions.shape[0]

        dice_lv = []
        dice_la = []
        x_errors = []
        y_errors = []

        for i in range(batch_size):
            # Dice coefficient
            lv_pred = predictions[i, :, :, 0] > 0.5
            la_pred = predictions[i, :, :, 1] > 0.5
            lv_true = ground_truth[i, :, :, 0] > 0.5
            la_true = ground_truth[i, :, :, 1] > 0.5

            dice_lv.append(dice(lv_pred, lv_true))
            dice_la.append(dice(la_pred, la_true))

            # Coordinate error (mm)
            pred_hinge = HingePointExtractor.extract_hinge_points(
                predictions[i],
                FrameMetadata(
                    cardiac_phase='unknown',
                    view_type='a4c',
                    frame_idx=i,
                    pixel_spacing_x=pixel_spacing_x,
                    pixel_spacing_y=pixel_spacing_y,
                    image_quality='unknown'
                )
            )
            true_hinge = HingePointExtractor.extract_hinge_points(
                ground_truth[i],
                FrameMetadata(
                    cardiac_phase='unknown',
                    view_type='a4c',
                    frame_idx=i,
                    pixel_spacing_x=pixel_spacing_x,
                    pixel_spacing_y=pixel_spacing_y,
                    image_quality='unknown'
                )
            )

            x_err = abs(pred_hinge.anterior_hinge[0] - true_hinge.anterior_hinge[0])
            y_err = abs(pred_hinge.anterior_hinge[1] - true_hinge.anterior_hinge[1])
            x_errors.append(x_err)
            y_errors.append(y_err)

        return {
            'dice_lv_median': np.median(dice_lv),
            'dice_la_median': np.median(dice_la),
            'dice_lv_p15_p85': (np.percentile(dice_lv, 15), np.percentile(dice_lv, 85)),
            'x_error_median_mm': np.median(x_errors),
            'y_error_median_mm': np.median(y_errors),
            'x_error_p85_mm': np.percentile(x_errors, 85),
            'y_error_p85_mm': np.percentile(y_errors, 85),
            'benchmark_lv_ed': CAMUS_Benchmark.DICE_LV_ED,
            'benchmark_lv_es': CAMUS_Benchmark.DICE_LV_ES,
        }


# ============================================================================
# 9. EXAMPLE USAGE
# ============================================================================

def main():
    """Demonstration of the pipeline"""
    logger.info("Initializing Heart Valve Analysis Pipeline")

    # 1. Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    segmentation_model = UNetSegmenter(in_channels=1, num_classes=3)

    # 2. Create analysis pipeline
    pipeline = HeartVallueAnalysisPipeline(
        segmentation_model=segmentation_model,
        region='US',  # FDA-cleared region
        device=device
    )

    # 3. Simulate analysis of a cardiac study
    # In production: load actual anonymized DICOM
    synthetic_frame = np.random.randint(0, 255, size=(256, 256), dtype=np.uint8)
    frame_sequence = np.stack([synthetic_frame] * 30)  # 30-frame cardiac cycle

    metadata = FrameMetadata(
        cardiac_phase='ED',
        view_type='a4c',
        frame_idx=0,
        pixel_spacing_x=0.3,  # mm/pixel (0.3mm in x)
        pixel_spacing_y=0.15,  # mm/pixel (0.15mm in y)
        image_quality='good'
    )

    # 4. Run analysis
    results = pipeline.analyze_study(frame_sequence, metadata)

    # 5. Report findings (with Clinical Precedence Rule enforced)
    logger.info(f"\n=== ANALYSIS COMPLETE ===")
    logger.info(f"Frames analyzed: {results['frames_analyzed']}")
    logger.info(f"Requires clinician review: {results['requires_clinician_review']}")

    if results['flagged_findings']:
        logger.warning(f"\nFLAGGED FINDINGS (Require Clinician Sign-Off):")
        for i, finding in enumerate(results['flagged_findings'], 1):
            logger.warning(f"  {i}. {finding}")

    if results['hinge_point_estimates']:
        logger.info(f"\nMITRAL VALVE MEASUREMENTS (First frame):")
        h = results['hinge_point_estimates'][0]
        logger.info(f"  Anterior hinge: {h.anterior_hinge}")
        logger.info(f"  Posterior hinge: {h.posterior_hinge}")
        logger.info(f"  MV Diameter: {h.mv_diameter_mm:.2f} mm (±{h.x_uncertainty_mm:.2f} mm)")
        logger.info(f"  Confidence: {h.confidence_score:.2f}")


if __name__ == '__main__':
    main()
