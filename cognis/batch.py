from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cognis.config import CeilingMode, MasteringConfig, MasteringMode
from cognis.engine import Engine, RenderResult
from cognis.io.audio import load_audio, save_audio
from cognis.serialization.artifacts import write_render_artifacts
from cognis.session_compare import compare_session_files


BATCH_MANIFEST_SCHEMA_VERSION = "cognis_batch_manifest_v1"
BATCH_SESSION_SCHEMA_VERSION = "cognis_batch_session_v1"
DOGFOOD_CORPUS_SCHEMA_VERSION = "cognis_dogfood_corpus_v1"

RUN_STATE_SUCCESS = "success"
RUN_STATE_FAILED = "failed"

DEFAULT_RUN_OPTIONS: dict[str, Any] = {
    "target_loudness": -14.0,
    "ceiling_db": -1.0,
    "ceiling_mode": "TRUE_PEAK",
    "oversampling": 4,
    "bass_preservation": 1.0,
    "stereo_width": 1.0,
    "dynamics_preservation": 1.0,
    "brightness": 0.0,
    "fir_backend": "AUTO",
}
RUN_OPTION_KEYS = frozenset(DEFAULT_RUN_OPTIONS)


@dataclass(frozen=True)
class BatchReference:
    id: str
    path: str


@dataclass(frozen=True)
class BatchRunPlan:
    run_id: str
    track_id: str
    input_path: str
    mode: str
    reference: BatchReference | None
    options: dict[str, Any]
    tags: tuple[str, ...]
    notes: str | None


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _slug(value: str, *, fallback: str = "item") -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    normalized = re.sub(r"_+", "_", normalized)
    return normalized[:72] or fallback


def _short_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:8]


def _as_list(value: Any, *, default: list[Any] | None = None) -> list[Any]:
    if value is None:
        return list(default or [])
    if isinstance(value, list):
        return value
    return [value]


def _merge_options(*parts: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_RUN_OPTIONS)
    for part in parts:
        if part:
            merged.update(part)
    return merged


def _option_values(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    values = {key: payload[key] for key in RUN_OPTION_KEYS if key in payload}
    nested = payload.get("options")
    if isinstance(nested, dict):
        values.update(nested)
    return values


def _parse_reference(value: Any, index: int) -> BatchReference:
    if isinstance(value, str):
        path = value
        ref_id = Path(value).stem or f"reference{index:02d}"
    elif isinstance(value, dict):
        path = value.get("path")
        if not path:
            raise ValueError("reference entries must include 'path'")
        ref_id = value.get("id") or Path(path).stem or f"reference{index:02d}"
    else:
        raise ValueError("reference entries must be strings or objects")
    return BatchReference(id=_slug(str(ref_id), fallback=f"reference{index:02d}"), path=str(path))


def _resolve_manifest_path(base_dir: Path, value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def _asset_base_dir(manifest: dict[str, Any], manifest_dir: str | Path) -> Path:
    base_dir = Path(manifest_dir)
    corpus = manifest.get("corpus")
    asset_root = corpus.get("asset_root") if isinstance(corpus, dict) else None
    if not asset_root:
        return base_dir
    root = Path(str(asset_root))
    return root if root.is_absolute() else (base_dir / root).resolve()


def load_batch_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("batch manifest must be a JSON object")
    schema_version = payload.get("schema_version", BATCH_MANIFEST_SCHEMA_VERSION)
    if schema_version != BATCH_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"unsupported batch manifest schema_version: {schema_version}")
    if not payload.get("tracks") and not payload.get("runs"):
        raise ValueError("batch manifest must define at least one track or explicit run")
    return payload


def expand_manifest(manifest: dict[str, Any], *, manifest_dir: str | Path = ".") -> list[BatchRunPlan]:
    base_dir = _asset_base_dir(manifest, manifest_dir)
    defaults = manifest.get("defaults", {})
    default_modes = [str(mode).upper() for mode in _as_list(defaults.get("modes") or defaults.get("mode"), default=["STREAMING_SAFE"])]
    plans: list[BatchRunPlan] = []

    for track_index, track in enumerate(manifest.get("tracks", []), start=1):
        if not isinstance(track, dict):
            raise ValueError("track entries must be objects")
        input_path = track.get("path")
        if not input_path:
            raise ValueError("track entries must include 'path'")
        track_id = _slug(str(track.get("id") or Path(input_path).stem or f"track{track_index:02d}"), fallback=f"track{track_index:02d}")
        modes = [str(mode).upper() for mode in _as_list(track.get("modes"), default=default_modes)]
        references = [_parse_reference(ref, i) for i, ref in enumerate(_as_list(track.get("references")), start=1)]
        reference_choices: list[BatchReference | None] = references or [None]
        if track.get("include_unreferenced") and references:
            reference_choices = [None] + reference_choices
        options = _merge_options(_option_values(defaults), _option_values(track), track.get("overrides"))
        tags = tuple(str(item) for item in _as_list(defaults.get("tags")) + _as_list(track.get("tags")))

        for mode in modes:
            for reference in reference_choices:
                identity = {
                    "track": track_id,
                    "input": _resolve_manifest_path(base_dir, str(input_path)),
                    "mode": mode,
                    "reference": _resolve_manifest_path(base_dir, reference.path) if reference else None,
                    "options": options,
                }
                parts = [track_id, "mode", _slug(mode, fallback="mode")]
                if reference is not None:
                    parts.extend(["ref", reference.id])
                run_id = "_".join(parts)
                plans.append(
                    BatchRunPlan(
                        run_id=run_id,
                        track_id=track_id,
                        input_path=identity["input"],
                        mode=mode,
                        reference=BatchReference(reference.id, _resolve_manifest_path(base_dir, reference.path)) if reference else None,
                        options=options,
                        tags=tags,
                        notes=track.get("notes"),
                    )
                )

    for run_index, run in enumerate(manifest.get("runs", []), start=1):
        if not isinstance(run, dict):
            raise ValueError("explicit run entries must be objects")
        input_path = run.get("input") or run.get("path")
        if not input_path:
            raise ValueError("explicit run entries must include 'input' or 'path'")
        track_id = _slug(str(run.get("track_id") or Path(input_path).stem or f"track{run_index:02d}"), fallback=f"track{run_index:02d}")
        mode = str(run.get("mode") or defaults.get("mode") or "STREAMING_SAFE").upper()
        reference = _parse_reference(run["reference"], 1) if run.get("reference") else None
        options = _merge_options(_option_values(defaults), _option_values(run), run.get("overrides"))
        identity = {
            "track": track_id,
            "input": _resolve_manifest_path(base_dir, str(input_path)),
            "mode": mode,
            "reference": _resolve_manifest_path(base_dir, reference.path) if reference else None,
            "options": options,
        }
        run_id = _slug(str(run.get("id") or f"{track_id}_mode_{mode}_{_short_hash(identity)}"), fallback=f"run{run_index:02d}")
        plans.append(
            BatchRunPlan(
                run_id=run_id,
                track_id=track_id,
                input_path=identity["input"],
                mode=mode,
                reference=BatchReference(reference.id, identity["reference"]) if reference else None,
                options=options,
                tags=tuple(str(item) for item in _as_list(defaults.get("tags")) + _as_list(run.get("tags"))),
                notes=run.get("notes"),
            )
        )

    return _dedupe_run_ids(plans)


def _dedupe_run_ids(plans: list[BatchRunPlan]) -> list[BatchRunPlan]:
    seen: dict[str, int] = {}
    deduped: list[BatchRunPlan] = []
    for plan in plans:
        count = seen.get(plan.run_id, 0) + 1
        seen[plan.run_id] = count
        run_id = plan.run_id if count == 1 else f"{plan.run_id}_{count:02d}"
        deduped.append(
            BatchRunPlan(
                run_id=run_id,
                track_id=plan.track_id,
                input_path=plan.input_path,
                mode=plan.mode,
                reference=plan.reference,
                options=plan.options,
                tags=plan.tags,
                notes=plan.notes,
            )
        )
    return deduped


def _config_from_plan(plan: BatchRunPlan) -> MasteringConfig:
    options = plan.options
    try:
        mode = MasteringMode(plan.mode)
    except ValueError as exc:
        raise ValueError(f"invalid mode for run {plan.run_id}: {plan.mode}") from exc
    try:
        ceiling_mode = CeilingMode(str(options.get("ceiling_mode", "TRUE_PEAK")).upper())
    except ValueError as exc:
        raise ValueError(f"invalid ceiling_mode for run {plan.run_id}: {options.get('ceiling_mode')}") from exc
    return MasteringConfig(
        mode=mode,
        target_loudness=float(options["target_loudness"]),
        ceiling_mode=ceiling_mode,
        ceiling_db=float(options["ceiling_db"]),
        oversampling=int(options["oversampling"]),
        bass_preservation=float(options["bass_preservation"]),
        stereo_width=float(options["stereo_width"]),
        dynamics_preservation=float(options["dynamics_preservation"]),
        brightness=float(options["brightness"]),
        reference_path=plan.reference.path if plan.reference else None,
        fir_backend=str(options.get("fir_backend", "AUTO")).upper(),
    )


def _path_map(root: Path, paths: dict[str, str]) -> dict[str, str]:
    mapped: dict[str, str] = {}
    for key, value in paths.items():
        path = Path(value)
        try:
            mapped[key] = path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            mapped[key] = str(path)
    return dict(sorted(mapped.items()))


def summarize_successful_run(plan: BatchRunPlan, result: RenderResult, *, run_dir: Path, output_path: Path, written: dict[str, str], session_root: Path) -> dict[str, Any]:
    report = result.report
    warning_codes = [finding.code for finding in report.findings if finding.severity == "warning"]
    fail_codes = [finding.code for finding in report.findings if finding.severity == "fail"]
    reference_summary = _reference_summary(report.reference_assessment)
    decision_history_summary = _decision_history_summary(report.decision_history_summary)
    return {
        "run_id": plan.run_id,
        "track_id": plan.track_id,
        "state": RUN_STATE_SUCCESS,
        "mode": plan.mode,
        "input_path": plan.input_path,
        "reference_path": plan.reference.path if plan.reference else None,
        "tags": list(plan.tags),
        "notes": plan.notes,
        "output": {
            "run_dir": run_dir.resolve().relative_to(session_root.resolve()).as_posix(),
            "audio": output_path.resolve().relative_to(session_root.resolve()).as_posix(),
            "artifacts": _path_map(session_root, written),
        },
        "qc": {
            "overall_status": report.overall_status,
            "warning_count": len(warning_codes),
            "fail_count": len(fail_codes),
            "warning_codes": warning_codes,
            "fail_codes": fail_codes,
        },
        "metrics": {
            "integrated_lufs": report.achieved.integrated_lufs,
            "true_peak_dbfs": report.achieved.true_peak_dbfs,
            "target_loudness_lufs": report.requested.target_loudness_lufs,
            "loudness_delta_lu": report.delta.loudness_delta_lu,
            "abs_loudness_delta_lu": abs(report.delta.loudness_delta_lu),
            "true_peak_margin_db": report.delta.true_peak_margin_db,
            "limiter_stress_estimate": report.achieved.limiter_stress_estimate,
            "codec_risk_estimate": report.achieved.codec_risk_estimate,
            "clipping_risk_estimate": report.achieved.clipping_risk_estimate,
        },
        "reference": reference_summary,
        "decision_history": decision_history_summary,
        "remaining_issues": _remaining_issues(report),
    }


def _reference_summary(assessment: Any | None) -> dict[str, Any]:
    if assessment is None:
        return {
            "available": False,
            "status": "unavailable",
            "reason": "No reference assessment was produced for this run.",
        }
    comparisons = []
    residuals = []
    movements = []
    for comparison in assessment.comparisons:
        residual = abs(comparison.output_delta_to_reference)
        normalized_residual = residual / comparison.tolerance if comparison.tolerance else residual
        comparisons.append(
            {
                "metric": comparison.metric,
                "status": comparison.status,
                "output_delta_to_reference": comparison.output_delta_to_reference,
                "movement_toward_reference": comparison.movement_toward_reference,
                "normalized_residual": normalized_residual,
                "attribution": "measured",
            }
        )
        residuals.append(normalized_residual)
        movements.append(comparison.movement_toward_reference)
    return {
        "available": True,
        "status": assessment.outcome,
        "average_normalized_residual": sum(residuals) / len(residuals) if residuals else None,
        "total_movement_toward_reference": sum(movements),
        "matched_metric_count": sum(item["status"] == "matched" for item in comparisons),
        "comparison_count": len(comparisons),
        "comparisons": comparisons,
    }


def _decision_history_summary(summary: Any | None) -> dict[str, Any]:
    if summary is None:
        return {
            "available": False,
            "status": "unavailable",
            "reason": "No decision-history summary was produced for this run.",
        }
    return {
        "available": bool(summary.available),
        "status": "available" if summary.available else "unavailable",
        "selection_basis": summary.selection_basis,
        "candidate_count": summary.candidate_count,
        "winner_candidate_index": summary.winner_candidate_index,
        "runner_up_candidate_index": summary.runner_up_candidate_index,
        "score_margin_to_runner_up": summary.score_margin_to_runner_up,
        "limitations": list(summary.limitations),
    }


def _remaining_issues(report: Any) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    for finding in report.findings:
        if finding.severity in {"warning", "fail"}:
            issues.append({"severity": finding.severity, "code": finding.code, "message": finding.message})
    if report.reference_assessment is not None:
        for finding in report.reference_assessment.findings:
            if finding.severity in {"warning", "fail"}:
                issues.append({"severity": finding.severity, "code": finding.code, "message": finding.message})
    return issues


def summarize_failed_run(plan: BatchRunPlan, *, run_dir: Path, session_root: Path, error: BaseException) -> dict[str, Any]:
    return {
        "run_id": plan.run_id,
        "track_id": plan.track_id,
        "state": RUN_STATE_FAILED,
        "mode": plan.mode,
        "input_path": plan.input_path,
        "reference_path": plan.reference.path if plan.reference else None,
        "tags": list(plan.tags),
        "notes": plan.notes,
        "output": {"run_dir": run_dir.resolve().relative_to(session_root.resolve()).as_posix(), "audio": None, "artifacts": {}},
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
        "qc": {"overall_status": "unavailable", "warning_count": 0, "fail_count": 0, "warning_codes": [], "fail_codes": []},
        "metrics": {},
        "reference": {"available": False, "status": "unavailable", "reason": "Run failed before reference assessment could be produced."},
        "decision_history": {"available": False, "status": "unavailable", "reason": "Run failed before decision history could be produced."},
        "remaining_issues": [{"severity": "fail", "code": "RUN_EXCEPTION", "message": str(error)}],
    }


def build_session_artifact(manifest: dict[str, Any], run_summaries: list[dict[str, Any]], *, session_root: Path, manifest_path: Path | None = None) -> dict[str, Any]:
    aggregate = build_aggregate(run_summaries)
    session_block = manifest.get("session") if isinstance(manifest.get("session"), dict) else {}
    session_id = manifest.get("session_id") or session_block.get("id") or session_block.get("slug") or _slug(manifest_path.stem if manifest_path else "batch_session", fallback="batch_session")
    return {
        "schema_version": BATCH_SESSION_SCHEMA_VERSION,
        "manifest_schema_version": manifest.get("schema_version", BATCH_MANIFEST_SCHEMA_VERSION),
        "corpus": _corpus_summary(manifest.get("corpus")),
        "session_id": session_id,
        "manifest_path": str(manifest_path.resolve()) if manifest_path else None,
        "session_root": str(session_root.resolve()),
        "run_count": len(run_summaries),
        "runs": run_summaries,
        "aggregate": aggregate,
        "shortlist": build_shortlist(run_summaries),
    }


def _corpus_summary(corpus: Any) -> dict[str, Any] | None:
    if not isinstance(corpus, dict):
        return None
    return {
        "schema_version": corpus.get("schema_version", DOGFOOD_CORPUS_SCHEMA_VERSION),
        "id": corpus.get("id"),
        "name": corpus.get("name"),
        "version": corpus.get("version"),
        "description": corpus.get("description"),
        "asset_policy": corpus.get("asset_policy", "external_or_local"),
        "asset_root": corpus.get("asset_root"),
        "tags": list(corpus.get("tags", [])) if isinstance(corpus.get("tags", []), list) else [],
    }


def build_aggregate(run_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [run for run in run_summaries if run["state"] == RUN_STATE_SUCCESS]
    failed = [run for run in run_summaries if run["state"] == RUN_STATE_FAILED]
    status_counts: dict[str, int] = {}
    for run in successful:
        status = run["qc"]["overall_status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    reference_runs = [run for run in successful if run["reference"].get("available")]
    return {
        "successful_run_count": len(successful),
        "failed_run_count": len(failed),
        "qc_status_counts": dict(sorted(status_counts.items())),
        "reference_assessed_run_count": len(reference_runs),
        "decision_history_available_count": sum(1 for run in successful if run["decision_history"].get("available")),
        "warning_count": sum(run["qc"]["warning_count"] for run in successful),
        "fail_finding_count": sum(run["qc"]["fail_count"] for run in successful),
        "run_exception_count": len(failed),
    }


def build_shortlist(run_summaries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    successful = [run for run in run_summaries if run["state"] == RUN_STATE_SUCCESS]
    passed = [run for run in successful if run["qc"]["overall_status"] == "pass"]
    safe_order = sorted(successful, key=lambda run: (_status_rank(run), run["qc"]["fail_count"], run["qc"]["warning_count"], _true_peak_margin_bucket(run), run["metrics"].get("limiter_stress_estimate", 999.0), run["metrics"].get("codec_risk_estimate", 999.0), run["run_id"]))
    target_order = sorted(successful, key=lambda run: (run["metrics"].get("abs_loudness_delta_lu", 999.0), _status_rank(run), run["run_id"]))
    reference_order = sorted(
        [run for run in successful if run["reference"].get("available")],
        key=lambda run: (
            run["reference"].get("average_normalized_residual") if run["reference"].get("average_normalized_residual") is not None else 999.0,
            _status_rank(run),
            run["run_id"],
        ),
    )
    manual_review = sorted(
        [run for run in run_summaries if run["state"] == RUN_STATE_FAILED or run["qc"].get("overall_status") in {"warning", "fail"} or run.get("remaining_issues")],
        key=lambda run: (0 if run["state"] == RUN_STATE_FAILED else _status_rank(run), -len(run.get("remaining_issues", [])), run["run_id"]),
    )
    listen_first = sorted(successful, key=lambda run: (_status_rank(run), run["metrics"].get("abs_loudness_delta_lu", 999.0), _reference_sort_value(run), run["run_id"]))
    return {
        "safest_measured": [_shortlist_entry(run, "safest measured") for run in (passed or safe_order)[:5]],
        "closest_to_target": [_shortlist_entry(run, "closest measured loudness target") for run in target_order[:5]],
        "closest_to_reference": [_shortlist_entry(run, "closest measured reference residual") for run in reference_order[:5]],
        "listen_first_objective_order": [_shortlist_entry(run, "objective review order") for run in listen_first[:8]],
        "manual_review_recommended": [_shortlist_entry(run, "manual review recommended") for run in manual_review[:8]],
    }


def _status_rank(run: dict[str, Any]) -> int:
    if run["state"] != RUN_STATE_SUCCESS:
        return 3
    return {"pass": 0, "informational": 0, "warning": 1, "fail": 2}.get(run["qc"]["overall_status"], 3)


def _reference_sort_value(run: dict[str, Any]) -> float:
    if not run["reference"].get("available"):
        return 999.0
    value = run["reference"].get("average_normalized_residual")
    return float(value) if value is not None else 999.0


def _true_peak_margin_bucket(run: dict[str, Any]) -> int:
    margin = run.get("metrics", {}).get("true_peak_margin_db")
    if margin is None:
        return 3
    if margin >= 0.3:
        return 0
    if margin >= 0.0:
        return 1
    return 2


def _shortlist_entry(run: dict[str, Any], basis: str) -> dict[str, Any]:
    metrics = run.get("metrics", {})
    return {
        "run_id": run["run_id"],
        "basis": basis,
        "state": run["state"],
        "qc_status": run["qc"]["overall_status"],
        "loudness_delta_lu": metrics.get("loudness_delta_lu"),
        "true_peak_margin_db": metrics.get("true_peak_margin_db"),
        "reference_status": run["reference"].get("status"),
        "reference_average_normalized_residual": run["reference"].get("average_normalized_residual"),
        "issue_count": len(run.get("remaining_issues", [])),
    }


def render_session_markdown(session: dict[str, Any]) -> str:
    aggregate = session["aggregate"]
    lines = [
        "# COGNIS Batch Evaluation Session",
        "",
        f"- Schema: `{session['schema_version']}`",
        f"- Session: `{session['session_id']}`",
        f"- Runs: `{session['run_count']}`",
        f"- Successful runs: `{aggregate['successful_run_count']}`",
        f"- Failed runs: `{aggregate['failed_run_count']}`",
        f"- Reference-assessed runs: `{aggregate['reference_assessed_run_count']}`",
        "",
        "## Aggregate QC",
        "",
    ]
    for status, count in aggregate["qc_status_counts"].items():
        lines.append(f"- `{status}`: `{count}`")
    if not aggregate["qc_status_counts"]:
        lines.append("- No successful QC summaries were available.")

    lines.extend(["", "## Objective Shortlist", ""])
    labels = {
        "safest_measured": "Safest measured",
        "closest_to_target": "Closest to loudness target",
        "closest_to_reference": "Closest to reference",
        "listen_first_objective_order": "Listen first: objective review order",
        "manual_review_recommended": "Manual review recommended",
    }
    for key, label in labels.items():
        lines.extend([f"### {label}", ""])
        entries = session["shortlist"][key]
        if not entries:
            lines.append("- Unavailable for this session.")
        for entry in entries:
            lines.append(
                "- "
                f"`{entry['run_id']}` "
                f"({entry['basis']}; qc `{entry['qc_status']}`; "
                f"loudness delta `{_fmt(entry['loudness_delta_lu'])}` LU; "
                f"reference `{entry['reference_status']}`; "
                f"issues `{entry['issue_count']}`)"
            )
        lines.append("")

    lines.extend(["## Runs", ""])
    for run in session["runs"]:
        if run["state"] == RUN_STATE_FAILED:
            lines.append(f"- `{run['run_id']}`: `failed` run exception `{run['error']['type']}` - {run['error']['message']}")
            continue
        metrics = run["metrics"]
        lines.append(
            "- "
            f"`{run['run_id']}`: qc `{run['qc']['overall_status']}`, "
            f"loudness `{metrics['integrated_lufs']:.2f} LUFS`, "
            f"delta `{metrics['loudness_delta_lu']:+.2f} LU`, "
            f"true peak `{metrics['true_peak_dbfs']:.2f} dBFS`, "
            f"reference `{run['reference']['status']}`."
        )
    lines.append("")
    lines.extend(
        [
            "## Interpretation Boundary",
            "",
            "- This report ranks measured safety, target closeness, reference residuals, and review priority only.",
            "- It makes no subjective listening-quality claim.",
            "- Per-run QC reports remain the source of truth for release safety.",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "unavailable"
    return f"{float(value):+.2f}"


def write_session_outputs(session: dict[str, Any], session_root: Path) -> dict[str, str]:
    session_root.mkdir(parents=True, exist_ok=True)
    session_json = session_root / "session.json"
    session_md = session_root / "session.md"
    session_json.write_text(_stable_json(session), encoding="utf-8")
    session_md.write_text(render_session_markdown(session), encoding="utf-8")
    return {"session_json": str(session_json), "session_markdown": str(session_md)}


def run_batch_manifest(manifest_path: str | Path, *, output_root: str | Path | None = None, fail_fast: bool = False) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = load_batch_manifest(manifest_path)
    session_block = manifest.get("session") if isinstance(manifest.get("session"), dict) else {}
    session_root = Path(output_root or manifest.get("output_root") or session_block.get("output_root") or manifest_path.with_suffix(""))
    if not session_root.is_absolute():
        session_root = (manifest_path.parent / session_root).resolve()
    plans = expand_manifest(manifest, manifest_dir=manifest_path.parent)
    run_summaries: list[dict[str, Any]] = []
    engine = Engine()
    for plan in plans:
        run_dir = session_root / "runs" / plan.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        output_path = run_dir / "output.wav"
        try:
            result = execute_run(plan, engine=engine, output_path=output_path, run_dir=run_dir)
            written = write_render_artifacts(
                result,
                str(output_path),
                artifacts_dir=str(run_dir),
                write_recipe=True,
                write_analysis=True,
                reference_analysis=result.reference_analysis,
                write_report=True,
                write_markdown_report=True,
            )
            run_summaries.append(summarize_successful_run(plan, result, run_dir=run_dir, output_path=output_path, written=written, session_root=session_root))
        except Exception as exc:
            run_summaries.append(summarize_failed_run(plan, run_dir=run_dir, session_root=session_root, error=exc))
            if fail_fast:
                break

    session = build_session_artifact(manifest, run_summaries, session_root=session_root, manifest_path=manifest_path)
    write_session_outputs(session, session_root)
    return session


def execute_run(plan: BatchRunPlan, *, engine: Engine, output_path: Path, run_dir: Path) -> RenderResult:
    config = _config_from_plan(plan)
    audio, sr = load_audio(plan.input_path)
    reference_audio = None
    reference_sr = None
    if plan.reference is not None:
        reference_audio, reference_sr = load_audio(plan.reference.path)
    result = engine.render(audio, sr, config, reference_audio=reference_audio, reference_sr=reference_sr)
    save_audio(str(output_path), result.audio, sr)
    return result


def main(argv: list[str] | None = None) -> None:
    argv = list(argv or [])
    if argv and argv[0] == "compare":
        compare_main(argv[1:])
        return

    parser = argparse.ArgumentParser(description="COGNIS Batch Evaluation + Dogfood Mastering Lab")
    parser.add_argument("manifest", help="Path to a JSON batch manifest")
    parser.add_argument("--output-root", default=None, help="Session output directory")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failed run")
    args = parser.parse_args(argv)

    session = run_batch_manifest(args.manifest, output_root=args.output_root, fail_fast=args.fail_fast)
    print(f"Batch session complete: {session['session_root']}")
    print(f"Runs: {session['run_count']} success={session['aggregate']['successful_run_count']} failed={session['aggregate']['failed_run_count']}")
    print(f"Session JSON: {Path(session['session_root']) / 'session.json'}")
    print(f"Session Markdown: {Path(session['session_root']) / 'session.md'}")


def compare_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare two COGNIS batch session artifacts")
    parser.add_argument("baseline_session", help="Baseline session.json")
    parser.add_argument("candidate_session", help="Candidate session.json")
    parser.add_argument("--output-root", required=True, help="Comparison output directory")
    parser.add_argument("--no-linked-sessions", action="store_true", help="Do not copy source session JSON files into linked_sessions/")
    parser.add_argument("--fail-on-regression", action="store_true", help="Exit non-zero when objective regressions are present")
    args = parser.parse_args(argv)

    comparison = compare_session_files(
        args.baseline_session,
        args.candidate_session,
        output_root=args.output_root,
        copy_sessions=not args.no_linked_sessions,
    )
    print(f"Comparison complete: {Path(args.output_root).resolve()}")
    aggregate = comparison["aggregate"]
    print(
        "Runs: "
        f"improved={aggregate['improved_run_count']} "
        f"regressed={aggregate['regressed_run_count']} "
        f"unchanged={aggregate['unchanged_run_count']} "
        f"inconclusive={aggregate['inconclusive_run_count']}"
    )
    print(f"Comparison JSON: {Path(args.output_root) / 'comparison.json'}")
    print(f"Comparison Markdown: {Path(args.output_root) / 'comparison.md'}")
    if args.fail_on_regression and aggregate["regressed_run_count"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
