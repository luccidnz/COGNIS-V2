from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

from cognis.serialization.analysis_artifacts import (
    analysis_artifact_path,
    build_analysis_artifact,
)
from cognis.reports.qc import render_report_markdown
from cognis.serialization.recipe import serialize_recipe

if TYPE_CHECKING:
    from cognis.engine import RenderResult


def _normalize(payload: Any) -> Any:
    if hasattr(payload, "to_dict"):
        return payload.to_dict()
    if is_dataclass(payload):
        return asdict(payload)
    return payload


def serialize_analysis(analysis: Any) -> str:
    return json.dumps(_normalize(analysis), indent=2, sort_keys=True) + "\n"


def serialize_report(report: Any) -> str:
    return json.dumps(_normalize(report), indent=2, sort_keys=True) + "\n"


def serialize_decision_history(decision_history: Any) -> str:
    return json.dumps(_normalize(decision_history), indent=2, sort_keys=True) + "\n"


def serialize_analysis_artifact(artifact: Any) -> str:
    return json.dumps(_normalize(artifact), indent=2, sort_keys=True) + "\n"


def _analysis_payload(value: Any) -> Any:
    return getattr(value, "analysis", value)


def write_render_artifacts(
    render_result: RenderResult,
    output_path: str,
    artifacts_dir: str | None = None,
    *,
    write_recipe: bool = True,
    write_analysis: bool = True,
    reference_analysis: Any | None = None,
    write_report: bool = True,
    write_markdown_report: bool = False,
) -> dict[str, str]:
    output = Path(output_path)
    artifact_root = Path(artifacts_dir) if artifacts_dir else output.parent
    artifact_root.mkdir(parents=True, exist_ok=True)
    stem = output.stem

    written: dict[str, str] = {}

    if write_recipe:
        path = artifact_root / f"{stem}.recipe.json"
        path.write_text(serialize_recipe(render_result.recipe), encoding="utf-8")
        written["recipe"] = str(path)

    if write_analysis:
        input_artifact = build_analysis_artifact(
            render_result.input_analysis,
            role="input",
            artifact_stem=stem,
            source_label="input",
        )
        output_artifact = build_analysis_artifact(
            render_result.output_analysis,
            role="output",
            artifact_stem=stem,
            source_label="output",
            source_path=str(output),
        )
        input_path = analysis_artifact_path(artifact_root, stem, "input")
        output_analysis_path = analysis_artifact_path(artifact_root, stem, "output")
        input_path.write_text(serialize_analysis_artifact(input_artifact), encoding="utf-8")
        output_analysis_path.write_text(serialize_analysis_artifact(output_artifact), encoding="utf-8")
        written["analysis_input"] = str(input_path)
        written["analysis_output"] = str(output_analysis_path)

        if reference_analysis is None:
            reference_analysis = getattr(render_result, "reference_analysis", None)

        if reference_analysis is not None:
            reference_source_path = None
            if isinstance(getattr(render_result, "recipe", None), dict):
                config = render_result.recipe.get("config", {})
                reference_source_path = config.get("reference_path")

            reference_artifact = build_analysis_artifact(
                _analysis_payload(reference_analysis),
                role="reference",
                artifact_stem=stem,
                source_label="reference",
                source_path=reference_source_path,
            )
            reference_path = analysis_artifact_path(artifact_root, stem, "reference")
            reference_path.write_text(serialize_analysis_artifact(reference_artifact), encoding="utf-8")
            written["analysis_reference"] = str(reference_path)

    if write_report:
        report_path = artifact_root / f"{stem}.report.json"
        report_path.write_text(serialize_report(render_result.report), encoding="utf-8")
        written["report"] = str(report_path)

    decision_history = getattr(render_result, "decision_history", None)
    if decision_history is not None:
        decision_history_path = artifact_root / f"{stem}.decision_history.json"
        decision_history_path.write_text(serialize_decision_history(decision_history), encoding="utf-8")
        written["decision_history"] = str(decision_history_path)

    if write_markdown_report:
        markdown_path = artifact_root / f"{stem}.report.md"
        markdown_path.write_text(render_report_markdown(render_result.report), encoding="utf-8")
        written["report_markdown"] = str(markdown_path)

    return written
