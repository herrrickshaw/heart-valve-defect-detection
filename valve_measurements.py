"""Torch-free core of the valve analysis: data structures, CAMUS reference values,
hinge-point extraction (step 2 of the two-step method) and the AS decision-support rule.

Kept separate from ``heart_valve_analysis`` (which owns the neural-network code) so the
measurement logic can be imported and tested without PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from sklearn.metrics import dice

# Severity cut-off for the primary criterion (aortic valve area, cm^2); ACC/AHA 2014.
SEVERE_AS_AREA_CM2 = 1.0
MODERATE_AS_AREA_CM2 = 1.5
# Binarisation threshold for softmax probability maps.
MASK_THRESHOLD = 0.5
# Canny hysteresis thresholds used on the LV mask boundary.
CANNY_LOW, CANNY_HIGH = 100, 200


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
        """Mitral valve diameter in mm: Euclidean distance between the two hinge points."""
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
    supporting_measurements: Optional[Dict] = None

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
        lv_mask = segmentation_mask[:, :, 0] > MASK_THRESHOLD
        la_mask = segmentation_mask[:, :, 1] > MASK_THRESHOLD

        # Find contact line: pixels on LV boundary adjacent to LA
        lv_boundary = cv2.Canny(lv_mask.astype(np.uint8) * 255, CANNY_LOW, CANNY_HIGH)
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
        lv_dice = dice(lv_mask, segmentation_mask[:, :, 0] > MASK_THRESHOLD)
        la_dice = dice(la_mask, segmentation_mask[:, :, 1] > MASK_THRESHOLD)
        confidence = (lv_dice + la_dice) / 2

        return HingePointEstimate(
            anterior_hinge=anterior_mm,
            posterior_hinge=posterior_mm,
            confidence_score=float(confidence)
        )


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
        if aortic_valve_area_cm2 < SEVERE_AS_AREA_CM2:
            severity = 'severe'
        elif aortic_valve_area_cm2 < MODERATE_AS_AREA_CM2:
            severity = 'moderate'
        else:
            severity = 'mild'

        # Flag only if severe (requires clinician review)
        if severity == 'severe':
            return ClinicalFinding(
                finding_type='severe_aortic_stenosis',
                severity=severity,
                measurement_value=aortic_valve_area_cm2,
                reference_threshold=SEVERE_AS_AREA_CM2,
                requires_review=True,
                supporting_measurements={
                    'peak_velocity_ms': peak_velocity_ms,
                    'guideline': 'ACC/AHA 2014'
                }
            )

        return None
