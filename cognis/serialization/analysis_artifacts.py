from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from cognis.analysis.features import AnalysisResult


ANALYSIS_ARTIFACT_ROLES = ("input", "output", "reference")


def _normalize_role(role: str) -> str:
    normalized = role.strip().lower()
    if normalized not in ANALYSIS_ARTIFACT_ROLES:
        allowed = ", ".join(ANALYSIS_ARTIFACT_ROLES)
        raise ValueError(f"artifact role must be one of: {allowed}")
    return normalized


def analysis_artifact_filename(stem: str, role: str) -> str:
    return f"{stem}.analysis.{_normalize_role(role)}.json"


def analysis_artifact_path(root: str | Path, stem: str, role: str) -> Path:
    return Path(root) / analysis_artifact_filename(stem, role)


def build_analysis_artifact(
    analysis: AnalysisResult,
    *,
    role: str,
    artifact_stem: str,
    source_label: str | None = None,
    source_path: str | None = None,
) -> AnalysisResult:
    normalized_role = _normalize_role(role)
    return replace(
        analysis,
        identity=replace(
            analysis.identity,
            role=normalized_role,
            source_name=source_label or artifact_stem or normalized_role,
            source_path=source_path,
        ),
    )
