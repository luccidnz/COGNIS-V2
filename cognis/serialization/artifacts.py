from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from cognis.engine import RenderResult
from cognis.reports.qc import render_report_markdown
from cognis.serialization.recipe import serialize_recipe


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


def write_render_artifacts(
    render_result: RenderResult,
    output_path: str,
    artifacts_dir: str | None = None,
    *,
    write_recipe: bool = True,
    write_analysis: bool = True,
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
        input_path = artifact_root / f"{stem}.analysis.input.json"
        output_analysis_path = artifact_root / f"{stem}.analysis.output.json"
        input_path.write_text(serialize_analysis(render_result.input_analysis), encoding="utf-8")
        output_analysis_path.write_text(serialize_analysis(render_result.output_analysis), encoding="utf-8")
        written["analysis_input"] = str(input_path)
        written["analysis_output"] = str(output_analysis_path)

    if write_report:
        report_path = artifact_root / f"{stem}.report.json"
        report_path.write_text(serialize_report(render_result.report), encoding="utf-8")
        written["report"] = str(report_path)

    if write_markdown_report:
        markdown_path = artifact_root / f"{stem}.report.md"
        markdown_path.write_text(render_report_markdown(render_result.report), encoding="utf-8")
        written["report_markdown"] = str(markdown_path)

    return written
