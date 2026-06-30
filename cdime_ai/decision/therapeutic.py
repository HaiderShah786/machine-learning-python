"""Therapeutic / follow-up decision support.

Combines the calibrated diagnosis, predictive confidence, epistemic uncertainty
(MC-Dropout) and explanatory evidence into a structured, evidence-based
recommendation. This is a *decision-support* layer (proposal Section 10: the
system never makes autonomous diagnoses) — every recommendation is conditioned
on confidence and flags low-certainty cases for human review.

The rules below are guideline-aligned placeholders; they are intentionally
conservative and should be reviewed by clinicians before any real use.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict


# Evidence-based follow-up suggestions per pathology (decision-support only).
_RECOMMENDATIONS: Dict[str, str] = {
    "Atelectasis": "Encourage incentive spirometry / chest physiotherapy; "
                   "correlate with clinical status; consider follow-up imaging.",
    "Cardiomegaly": "Recommend echocardiography and BNP; review cardiac history "
                    "and consider cardiology referral.",
    "Consolidation": "Consider sputum culture and empiric antibiotics if "
                     "infection suspected; follow-up radiograph in 4-6 weeks.",
    "Edema": "Assess fluid status; consider diuretics and echocardiography; "
             "monitor renal function and electrolytes.",
    "Pleural Effusion": "Consider diagnostic thoracentesis if clinically "
                        "significant; ultrasound to characterise the effusion.",
}


@dataclass
class DecisionReport:
    diagnoses: List[Dict] = field(default_factory=list)  # per positive label
    overall_confidence: float = 0.0
    needs_review: bool = False
    recommendations: List[str] = field(default_factory=list)
    notes: str = ""

    def to_text(self) -> str:
        lines = ["=== CDIME-AI Decision Support Report ==="]
        if not self.diagnoses:
            lines.append("No pathology exceeds the decision threshold "
                         "(study likely normal). Correlate clinically.")
        for d in self.diagnoses:
            lines.append(
                f"- {d['label']}: p={d['probability']:.2f} "
                f"(±{d['uncertainty']:.2f}), confidence={d['confidence']}")
        lines.append(f"Overall confidence: {self.overall_confidence:.2f}")
        if self.recommendations:
            lines.append("Suggested follow-up:")
            lines += [f"  * {r}" for r in self.recommendations]
        if self.needs_review:
            lines.append("** LOW CERTAINTY — flagged for radiologist review. **")
        if self.notes:
            lines.append(self.notes)
        return "\n".join(lines)


def make_decision(labels: List[str], probs, uncertainties,
                  evidence: List[str] | None = None,
                  threshold: float = 0.5, high_unc: float = 0.15) -> DecisionReport:
    """Build a structured decision report for a single study.

    Parameters
    ----------
    probs : per-label calibrated probabilities (length C)
    uncertainties : per-label MC-Dropout std (length C)
    evidence : optional list of textual/visual evidence snippets
    """
    report = DecisionReport()
    confidences = []
    for i, label in enumerate(labels):
        p = float(probs[i])
        u = float(uncertainties[i]) if uncertainties is not None else 0.0
        if p >= threshold:
            conf = "high" if (p >= 0.7 and u < high_unc) else "moderate" if p >= 0.55 else "low"
            confidences.append(p * (1 - min(u / high_unc, 1.0)))
            report.diagnoses.append({
                "label": label, "probability": p, "uncertainty": u, "confidence": conf,
            })
            report.recommendations.append(_RECOMMENDATIONS.get(label, "Clinical correlation advised."))

    report.overall_confidence = float(sum(confidences) / len(confidences)) if confidences else 0.0
    # Flag for human review if any positive finding is uncertain or borderline.
    report.needs_review = any(
        d["confidence"] != "high" for d in report.diagnoses
    ) or (len(report.diagnoses) == 0)
    if evidence:
        report.notes = "Evidence: " + "; ".join(evidence)
    return report
