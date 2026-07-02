"""
Clinical Review & Reporting System
Implements the Clinical Precedence Rule: AI findings require clinician sign-off
before entering the patient record.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ReviewStatus(Enum):
    """Workflow state for AI finding review"""
    PENDING_REVIEW = 'pending_review'
    CLINICIAN_APPROVED = 'clinician_approved'
    CLINICIAN_REJECTED = 'clinician_rejected'
    CLINICIAN_MODIFIED = 'clinician_modified'
    ARCHIVED = 'archived'


@dataclass
class ClinicianReview:
    """Clinician's decision on an AI-generated finding"""
    finding_id: str
    clinician_id: str
    clinician_name: str
    review_timestamp: datetime
    status: ReviewStatus
    comments: str = ""
    modified_severity: Optional[str] = None
    confidence_in_ai: float = 0.0  # 0-1 scale on clinician's confidence in AI

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['review_timestamp'] = self.review_timestamp.isoformat()
        d['status'] = self.status.value
        return d


@dataclass
class AuditLog:
    """Immutable audit trail for compliance & regulatory review"""
    event_type: str  # 'finding_generated', 'review_started', 'review_completed', etc.
    timestamp: datetime
    actor: str  # 'AI', 'clinician_name', 'system'
    details: Dict

    def to_dict(self) -> Dict:
        return {
            'event_type': self.event_type,
            'timestamp': self.timestamp.isoformat(),
            'actor': self.actor,
            'details': self.details
        }


class ClinicalReviewWorkflow:
    """
    Implements the Clinical Precedence Rule workflow.

    CRITICAL CONSTRAINT (from architecture spec, Section 7):
    "Every flagged finding requires explicit clinician sign-off before
    entering the patient record. No auto-finalized diagnosis path should exist."
    """

    def __init__(self, patient_id: str, study_id: str):
        self.patient_id = patient_id
        self.study_id = study_id
        self.findings: Dict[str, Dict] = {}  # finding_id -> finding data
        self.reviews: Dict[str, ClinicianReview] = {}  # finding_id -> review
        self.audit_trail: List[AuditLog] = []
        self.created_at = datetime.now()

    def add_ai_finding(self, finding_id: str, finding_data: Dict) -> None:
        """
        Register an AI-generated finding (PENDING_REVIEW).

        This finding WILL NOT be finalized until clinician sign-off.
        """
        self.findings[finding_id] = {
            **finding_data,
            'status': ReviewStatus.PENDING_REVIEW.value,
            'ai_generated_timestamp': datetime.now().isoformat()
        }

        self.audit_trail.append(AuditLog(
            event_type='finding_generated',
            timestamp=datetime.now(),
            actor='AI',
            details={
                'finding_id': finding_id,
                'finding_type': finding_data.get('finding_type'),
                'severity': finding_data.get('severity')
            }
        ))

        logger.info(f"[{self.study_id}] AI finding flagged: {finding_id} - {finding_data.get('finding_type')}")

    def clinician_review(
        self,
        finding_id: str,
        clinician_id: str,
        clinician_name: str,
        status: ReviewStatus,
        comments: str = "",
        modified_severity: Optional[str] = None,
        confidence_in_ai: float = 0.5
    ) -> ClinicianReview:
        """
        Clinician explicitly reviews and approves/rejects/modifies AI finding.

        This is the gate: findings do not finalize without this step.
        """
        if finding_id not in self.findings:
            raise ValueError(f"Finding {finding_id} not found in workflow")

        review = ClinicianReview(
            finding_id=finding_id,
            clinician_id=clinician_id,
            clinician_name=clinician_name,
            review_timestamp=datetime.now(),
            status=status,
            comments=comments,
            modified_severity=modified_severity,
            confidence_in_ai=confidence_in_ai
        )

        self.reviews[finding_id] = review

        # Update finding status in parallel
        self.findings[finding_id]['status'] = status.value
        if modified_severity:
            self.findings[finding_id]['clinician_modified_severity'] = modified_severity

        self.audit_trail.append(AuditLog(
            event_type='review_completed',
            timestamp=datetime.now(),
            actor=clinician_name,
            details={
                'finding_id': finding_id,
                'review_status': status.value,
                'clinician_confidence': confidence_in_ai,
                'modified': bool(modified_severity)
            }
        ))

        logger.info(
            f"[{self.study_id}] Clinician review: {finding_id} - {status.value} "
            f"(confidence in AI: {confidence_in_ai:.2f})"
        )

        return review

    def finalize_study(self) -> Dict:
        """
        Finalize study: only findings with clinician approval enter patient record.

        Returns EMPTY if any findings lack clinician review (study is incomplete).
        """
        pending = [
            fid for fid, f in self.findings.items()
            if f['status'] == ReviewStatus.PENDING_REVIEW.value
        ]

        if pending:
            raise ValueError(
                f"Study {self.study_id} has {len(pending)} unreviewed findings. "
                f"Clinician sign-off required before finalization. Pending: {pending}"
            )

        # Only approved findings enter the patient record
        approved_findings = {
            fid: f for fid, f in self.findings.items()
            if f['status'] == ReviewStatus.CLINICIAN_APPROVED.value
        }

        self.audit_trail.append(AuditLog(
            event_type='study_finalized',
            timestamp=datetime.now(),
            actor='system',
            details={
                'total_findings': len(self.findings),
                'approved_findings': len(approved_findings),
                'rejected_findings': len([
                    f for f in self.findings.values()
                    if f['status'] == ReviewStatus.CLINICIAN_REJECTED.value
                ])
            }
        ))

        logger.info(
            f"[{self.study_id}] Study finalized: "
            f"{len(approved_findings)} approved, "
            f"{len([f for f in self.findings.values() if f['status'] == ReviewStatus.CLINICIAN_REJECTED.value])} rejected"
        )

        return {
            'patient_id': self.patient_id,
            'study_id': self.study_id,
            'finalized_timestamp': datetime.now().isoformat(),
            'approved_findings': approved_findings,
            'audit_trail': [a.to_dict() for a in self.audit_trail]
        }

    def get_pending_reviews(self) -> List[Dict]:
        """Get list of findings awaiting clinician review"""
        return [
            {'finding_id': fid, **f}
            for fid, f in self.findings.items()
            if f['status'] == ReviewStatus.PENDING_REVIEW.value
        ]

    def export_audit_trail(self, filepath: str) -> None:
        """Export immutable audit trail for compliance audit"""
        audit_data = {
            'patient_id': self.patient_id,
            'study_id': self.study_id,
            'exported_at': datetime.now().isoformat(),
            'audit_trail': [a.to_dict() for a in self.audit_trail],
            'reviews': {
                fid: r.to_dict() for fid, r in self.reviews.items()
            }
        }
        with open(filepath, 'w') as f:
            json.dump(audit_data, f, indent=2)
        logger.info(f"Audit trail exported to {filepath}")


class ClinicalDecisionSupportReport:
    """
    Generates structured clinical reports for clinician review.

    Per architecture spec Section 7: "AI-generated findings are supplemental.
    Clinical judgment must always take precedence."
    """

    @staticmethod
    def generate_review_report(
        patient_id: str,
        study_id: str,
        findings: List[Dict],
        metadata: Dict,
        include_measurements: bool = True
    ) -> str:
        """
        Generate human-readable report for clinician review.

        Format: structured text emphasizing that all findings require
        clinician judgment and sign-off.
        """
        lines = [
            "=" * 70,
            "CARDIAC ULTRASOUND ANALYSIS REPORT",
            "AI-Assisted Decision Support (Clinical Review Required)",
            "=" * 70,
            f"Patient ID: {patient_id}",
            f"Study ID: {study_id}",
            f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "IMPORTANT: CLINICAL PRECEDENCE RULE",
            "-" * 70,
            "This report contains AI-generated findings and measurements.",
            "AI findings are SUPPLEMENTAL ONLY and do not constitute a diagnosis.",
            "Clinical judgment must take precedence in patient management.",
            "All flagged findings require explicit clinician review and sign-off.",
            "No findings enter the patient record without clinician approval.",
            "",
        ]

        if metadata:
            lines.extend([
                "ACQUISITION METADATA:",
                "-" * 70,
                f"View Type: {metadata.get('view_type', 'N/A')}",
                f"Cardiac Phase: {metadata.get('cardiac_phase', 'N/A')}",
                f"Image Quality: {metadata.get('image_quality', 'N/A')}",
                f"Frame Rate: {metadata.get('frame_rate', 'N/A')} fps",
                ""
            ])

        if findings:
            lines.extend([
                "FLAGGED FINDINGS REQUIRING REVIEW:",
                "-" * 70,
            ])
            for i, finding in enumerate(findings, 1):
                lines.extend([
                    f"{i}. {finding.get('finding_type', 'Unknown').upper()}",
                    f"   Severity: {finding.get('severity', 'N/A')}",
                    f"   Measurement: {finding.get('measurement_value', 'N/A')}",
                    f"   Threshold: {finding.get('reference_threshold', 'N/A')}",
                    f"   Status: PENDING CLINICIAN REVIEW",
                    ""
                ])
        else:
            lines.extend([
                "No findings flagged.",
                ""
            ])

        if include_measurements:
            lines.extend([
                "QUANTITATIVE MEASUREMENTS:",
                "-" * 70,
                "Note: All measurements carry uncertainty bounds per CAMUS validation.",
                "Do not interpret point estimates as definitive values.",
                ""
            ])

        lines.extend([
            "NEXT STEPS:",
            "-" * 70,
            "1. Clinician reviews this report and all flagged findings",
            "2. Clinician approves, rejects, or modifies each finding",
            "3. Only approved findings are finalized in patient record",
            "4. Audit trail is generated and retained for compliance review",
            "=" * 70,
        ])

        return "\n".join(lines)


class RegionalFeatureGate:
    """
    Implements regional regulatory gating per Section 7 of architecture spec.

    Some decision-support components (e.g., EchoSolv™ AS) are FDA-cleared only
    for specific regions (US). This gate enforces that constraint.
    """

    REGULATORY_STATUS = {
        'aortic_stenosis_classifier': {
            'US': {'status': 'FDA_CLEARED', 'gates_enabled': True},
            'EU': {'status': 'PENDING_CE_MARKING', 'gates_enabled': False},
            'ASIA': {'status': 'REGIONAL_APPROVAL_REQUIRED', 'gates_enabled': False},
            'GLOBAL': {'status': 'MEASUREMENT_ONLY', 'gates_enabled': False},
        }
    }

    @staticmethod
    def is_feature_enabled(feature_name: str, region: str) -> Tuple[bool, str]:
        """
        Check if a feature is enabled in a given region.

        Returns: (enabled: bool, reason: str)
        """
        if feature_name not in RegionalFeatureGate.REGULATORY_STATUS:
            return False, f"Unknown feature: {feature_name}"

        region_config = RegionalFeatureGate.REGULATORY_STATUS[feature_name].get(
            region, RegionalFeatureGate.REGULATORY_STATUS[feature_name].get('GLOBAL')
        )

        enabled = region_config['gates_enabled']
        status = region_config['status']

        return enabled, f"{feature_name} in {region}: {status}"

    @staticmethod
    def get_deployment_mode(region: str) -> Dict:
        """Get permitted deployment mode for a region"""
        as_enabled, as_reason = RegionalFeatureGate.is_feature_enabled(
            'aortic_stenosis_classifier', region
        )

        return {
            'region': region,
            'as_classifier_enabled': as_enabled,
            'deployment_mode': 'segmentation_measurement_and_decision_support' if as_enabled else 'segmentation_measurement_only',
            'regulatory_status': as_reason,
            'notes': "Decision-support features disabled outside cleared jurisdictions" if not as_enabled else ""
        }


# ============================================================================
# EXAMPLE WORKFLOW
# ============================================================================

def example_clinical_review_workflow():
    """Demonstrates the Clinical Precedence Rule enforcement"""

    logger.basicConfig(level=logging.INFO)
    logger.info("Starting Clinical Review Workflow Example\n")

    # 1. Initialize workflow for a patient study
    workflow = ClinicalReviewWorkflow(
        patient_id='PATIENT_001_DEIDENTIFIED',
        study_id='STUDY_20260629_001'
    )

    # 2. Simulate AI analysis generating findings
    findings_from_ai = [
        {
            'finding_type': 'severe_aortic_stenosis',
            'severity': 'severe',
            'measurement_value': 0.85,
            'reference_threshold': 1.0,
        },
        {
            'finding_type': 'low_confidence_segmentation',
            'severity': 'moderate',
            'measurement_value': 0.82,
            'reference_threshold': 0.85,
        }
    ]

    for i, finding in enumerate(findings_from_ai):
        workflow.add_ai_finding(f'FINDING_{i:03d}', finding)

    # 3. Generate clinical review report
    metadata = {
        'view_type': 'apical_4_chamber',
        'cardiac_phase': 'ED',
        'image_quality': 'good',
        'frame_rate': 60
    }

    report = ClinicalDecisionSupportReport.generate_review_report(
        patient_id='PATIENT_001',
        study_id='STUDY_20260629_001',
        findings=findings_from_ai,
        metadata=metadata
    )
    print(report)

    # 4. Simulate clinician reviewing the findings
    logger.info("\n[CLINICIAN REVIEW PHASE]")
    logger.info("Clinician Dr. Smith reviews findings...\n")

    # Clinician approves the severe AS finding
    workflow.clinician_review(
        finding_id='FINDING_000',
        clinician_id='DR_001',
        clinician_name='Dr. Smith',
        status=ReviewStatus.CLINICIAN_APPROVED,
        comments="Confirmed: severe AS criteria met. Recommend cardiology referral.",
        confidence_in_ai=0.92
    )

    # Clinician rejects the low-confidence segmentation (not clinically relevant)
    workflow.clinician_review(
        finding_id='FINDING_001',
        clinician_id='DR_001',
        clinician_name='Dr. Smith',
        status=ReviewStatus.CLINICIAN_REJECTED,
        comments="Segmentation quality adequate for clinical purposes.",
        confidence_in_ai=0.78
    )

    # 5. Finalize study (only approved findings enter record)
    logger.info("[STUDY FINALIZATION]")
    finalized_data = workflow.finalize_study()

    logger.info(f"\nFinal approved findings:")
    for fid, finding in finalized_data['approved_findings'].items():
        logger.info(f"  {fid}: {finding['finding_type']} ({finding['severity']})")

    # 6. Export audit trail for compliance
    workflow.export_audit_trail('/tmp/audit_trail.json')

    # 7. Check regional feature gating
    logger.info("\n[REGIONAL REGULATORY GATING]")
    for region in ['US', 'EU', 'ASIA']:
        mode = RegionalFeatureGate.get_deployment_mode(region)
        logger.info(f"Region {region}: {mode['deployment_mode']}")


if __name__ == '__main__':
    example_clinical_review_workflow()
