from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


BATCH_SESSION_SCHEMA_VERSION = "cognis_batch_session_v1"
SESSION_COMPARISON_SCHEMA_VERSION = "cognis_session_comparison_v1"

OUTCOME_IMPROVED = "improved"
OUTCOME_REGRESSED = "regressed"
OUTCOME_UNCHANGED = "unchanged"
OUTCOME_INCONCLUSIVE = "inconclusive"

EVIDENCE_EXACT = "exact"
EVIDENCE_INFERRED = "inferred"
EVIDENCE_UNAVAILABLE = "unavailable"

_MEANINGFUL_DELTA = 0.01

_STATUS_RANK = {
    "pass": 0,
    "informational": 0,
    "warning": 1,
    "fail": 2,
    "unavailable": 3,
}

_REFERENCE_STATUS_RANK = {
    "matched": 0,
    "partial": 1,
    "constrained": 2,
    "deviated": 3,
    "unavailable": 4,
}


def load_session_artifact(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("session artifact must be a JSON object")
    schema_version = payload.get("schema_version")
    if schema_version != BATCH_SESSION_SCHEMA_VERSION:
        raise ValueError(f"unsupported session schema_version: {schema_version}")
    return payload


def compare_session_artifacts(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    baseline_path: str | Path | None = None,
    candidate_path: str | Path | None = None,
) -> dict[str, Any]:
    _validate_session(baseline, "baseline")
    _validate_session(candidate, "candidate")

    baseline_runs = _runs_by_id(baseline)
    candidate_runs = _runs_by_id(candidate)
    baseline_shortlist_ranks = _shortlist_ranks(baseline.get("shortlist", {}))
    candidate_shortlist_ranks = _shortlist_ranks(candidate.get("shortlist", {}))

    run_ids = sorted(set(baseline_runs) | set(candidate_runs))
    run_comparisons = [
        _compare_run(
            run_id,
            baseline_runs.get(run_id),
            candidate_runs.get(run_id),
            baseline_shortlist_ranks,
            candidate_shortlist_ranks,
        )
        for run_id in run_ids
    ]
    aggregate = _build_aggregate(run_comparisons)
    review_priority = _build_review_priority(run_comparisons)
    return {
        "schema_version": SESSION_COMPARISON_SCHEMA_VERSION,
        "artifact_type": "batch_session_comparison",
        "baseline": _session_identity(baseline, baseline_path),
        "candidate": _session_identity(candidate, candidate_path),
        "aggregate": aggregate,
        "summary": _summary_alias(aggregate),
        "review_priority": review_priority,
        "runs": run_comparisons,
        "interpretation_boundary": {
            "evidence_levels": {
                EVIDENCE_EXACT: "Directly compared fields present in both session artifacts.",
                EVIDENCE_INFERRED: "Rule-based interpretation from measured artifact fields.",
                EVIDENCE_UNAVAILABLE: "Required artifact fields or runs were missing.",
            },
            "claim_boundary": (
                "This comparison reports objective artifact deltas only: QC status, measured loudness target "
                "closeness, true-peak margin, reference residuals, warning/fail counts, and shortlist movement. "
                "It does not claim subjective listening quality."
            ),
        },
    }


def compare_session_files(
    baseline_path: str | Path,
    candidate_path: str | Path,
    *,
    output_root: str | Path | None = None,
    copy_sessions: bool = True,
) -> dict[str, Any]:
    baseline = load_session_artifact(baseline_path)
    candidate = load_session_artifact(candidate_path)
    comparison = compare_session_artifacts(
        baseline,
        candidate,
        baseline_path=baseline_path,
        candidate_path=candidate_path,
    )
    if output_root is not None:
        write_comparison_outputs(
            comparison,
            output_root,
            baseline_path=baseline_path if copy_sessions else None,
            candidate_path=candidate_path if copy_sessions else None,
        )
    return comparison


def write_comparison_outputs(
    comparison: dict[str, Any],
    output_root: str | Path,
    *,
    baseline_path: str | Path | None = None,
    candidate_path: str | Path | None = None,
) -> dict[str, str]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    comparison_json = root / "comparison.json"
    comparison_md = root / "comparison.md"
    comparison_json.write_text(_stable_json(comparison), encoding="utf-8")
    comparison_md.write_text(render_comparison_markdown(comparison), encoding="utf-8")
    written = {"comparison_json": str(comparison_json), "comparison_markdown": str(comparison_md)}
    if baseline_path is not None or candidate_path is not None:
        linked = root / "linked_sessions"
        linked.mkdir(parents=True, exist_ok=True)
        if baseline_path is not None:
            target = linked / "baseline_session.json"
            shutil.copyfile(baseline_path, target)
            written["baseline_session"] = str(target)
        if candidate_path is not None:
            target = linked / "candidate_session.json"
            shutil.copyfile(candidate_path, target)
            written["candidate_session"] = str(target)
    return written


def render_comparison_markdown(comparison: dict[str, Any]) -> str:
    aggregate = comparison["aggregate"]
    lines = [
        "# COGNIS Session Comparison",
        "",
        f"- Schema: `{comparison['schema_version']}`",
        f"- Baseline session: `{comparison['baseline']['session_id']}`",
        f"- Candidate session: `{comparison['candidate']['session_id']}`",
        f"- Comparable runs: `{aggregate['comparable_run_count']}`",
        f"- Improved: `{aggregate['improved_run_count']}`",
        f"- Regressed: `{aggregate['regressed_run_count']}`",
        f"- Unchanged: `{aggregate['unchanged_run_count']}`",
        f"- Inconclusive: `{aggregate['inconclusive_run_count']}`",
        "",
        "## Aggregate Findings",
        "",
    ]
    for key, value in aggregate["category_counts"].items():
        lines.append(f"- `{key}`: `{value}`")
    if not aggregate["category_counts"]:
        lines.append("- No comparable measured changes were available.")

    lines.extend(["", "## Inspect First", ""])
    if not comparison["review_priority"]:
        lines.append("- No runs require prioritized inspection from the available artifacts.")
    for item in comparison["review_priority"]:
        lines.append(f"- `{item['run_id']}`: `{item['outcome']}`; {item['reason']}")

    lines.extend(["", "## Run Comparisons", ""])
    for run in comparison["runs"]:
        lines.append(f"### `{run['run_id']}`")
        lines.append("")
        lines.append(f"- Outcome: `{run['outcome']}`")
        lines.append(f"- Evidence: `{run['evidence_level']}`")
        lines.append(f"- Presence: `{run['presence']}`")
        if run["summary"]:
            lines.append(f"- Summary: {run['summary']}")
        if not run["changes"]:
            lines.append("- Changes: unavailable.")
        for change in run["changes"]:
            lines.append(
                "- "
                f"`{change['category']}` `{change['direction']}` "
                f"({change['evidence_level']}): {change['summary']}"
            )
        lines.append("")

    lines.extend(
        [
            "## Interpretation Boundary",
            "",
            "- This artifact compares generated session artifacts only.",
            "- Exact evidence means both compared fields were present in the input artifacts.",
            "- Inferred evidence means the category is a deterministic rule-based interpretation of exact measurements.",
            "- Unavailable evidence means missing runs or fields prevented a measured comparison.",
            "- It makes no subjective listening-quality claim.",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_session(session: dict[str, Any], label: str) -> None:
    schema_version = session.get("schema_version")
    if schema_version != BATCH_SESSION_SCHEMA_VERSION:
        raise ValueError(f"{label} session has unsupported schema_version: {schema_version}")
    if not isinstance(session.get("runs"), list):
        raise ValueError(f"{label} session must contain a runs list")


def _runs_by_id(session: dict[str, Any]) -> dict[str, dict[str, Any]]:
    runs: dict[str, dict[str, Any]] = {}
    for run in session.get("runs", []):
        if not isinstance(run, dict) or not run.get("run_id"):
            continue
        runs[str(run["run_id"])] = run
    return runs


def _session_identity(session: dict[str, Any], path: str | Path | None) -> dict[str, Any]:
    return {
        "schema_version": session.get("schema_version"),
        "session_id": session.get("session_id"),
        "session_root": session.get("session_root"),
        "session_path": str(Path(path)) if path is not None else None,
        "run_count": session.get("run_count"),
        "aggregate": session.get("aggregate", {}),
    }


def _compare_run(
    run_id: str,
    baseline: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
    baseline_shortlist_ranks: dict[str, dict[str, int]],
    candidate_shortlist_ranks: dict[str, dict[str, int]],
) -> dict[str, Any]:
    if baseline is None:
        return _presence_only_run(
            run_id,
            presence="baseline_missing",
            summary="Run exists only in the candidate session; no baseline artifact is available for measured comparison.",
        )
    if candidate is None:
        return _presence_only_run(
            run_id,
            presence="candidate_missing",
            summary="Run exists only in the baseline session; candidate data is missing.",
        )

    changes: list[dict[str, Any]] = []
    changes.extend(_compare_state_and_qc(baseline, candidate))
    changes.extend(_compare_metric_closeness(baseline, candidate))
    changes.extend(_compare_reference(baseline, candidate))
    changes.extend(_compare_shortlist_movements(run_id, baseline_shortlist_ranks, candidate_shortlist_ranks))

    outcome = _run_outcome(changes)
    summary = _run_summary(outcome, changes)
    return {
        "run_id": run_id,
        "presence": "both",
        "outcome": outcome,
        "evidence_level": _run_evidence_level(changes),
        "summary": summary,
        "baseline": _run_snapshot(baseline),
        "candidate": _run_snapshot(candidate),
        "changes": changes,
    }


def _presence_only_run(run_id: str, *, presence: str, summary: str) -> dict[str, Any]:
    outcome = OUTCOME_REGRESSED if presence == "candidate_missing" else OUTCOME_INCONCLUSIVE
    direction = "regressed" if presence == "candidate_missing" else "inconclusive"
    category = "missing_in_candidate" if presence == "candidate_missing" else "missing_in_baseline"
    return {
        "run_id": run_id,
        "presence": presence,
        "outcome": outcome,
        "evidence_level": EVIDENCE_UNAVAILABLE,
        "summary": summary,
        "baseline": None if presence == "baseline_missing" else {},
        "candidate": None if presence == "candidate_missing" else {},
        "changes": [
            {
                "category": category,
                "direction": direction,
                "evidence_level": EVIDENCE_UNAVAILABLE,
                "baseline_value": None,
                "candidate_value": None,
                "summary": summary,
            }
        ],
    }


def _run_snapshot(run: dict[str, Any]) -> dict[str, Any]:
    qc = run.get("qc", {})
    metrics = run.get("metrics", {})
    reference = run.get("reference", {})
    return {
        "state": run.get("state"),
        "qc_status": qc.get("overall_status"),
        "warning_count": qc.get("warning_count"),
        "fail_count": qc.get("fail_count"),
        "loudness_delta_lu": metrics.get("loudness_delta_lu"),
        "abs_loudness_delta_lu": metrics.get("abs_loudness_delta_lu"),
        "true_peak_margin_db": metrics.get("true_peak_margin_db"),
        "reference_available": reference.get("available"),
        "reference_status": reference.get("status"),
        "reference_average_normalized_residual": reference.get("average_normalized_residual"),
        "reference_matched_metric_count": reference.get("matched_metric_count"),
        "reference_comparison_count": reference.get("comparison_count"),
    }


def _compare_state_and_qc(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    baseline_state = baseline.get("state")
    candidate_state = candidate.get("state")
    if baseline_state != candidate_state:
        if baseline_state != "success" and candidate_state == "success":
            direction = "improved"
            category = "run_recovered"
            summary = "Candidate run completed where baseline did not."
        elif baseline_state == "success" and candidate_state != "success":
            direction = "regressed"
            category = "run_failed"
            summary = "Candidate run failed where baseline completed."
        else:
            direction = "inconclusive"
            category = "run_state_changed"
            summary = "Run state changed, but the available states do not define an objective improvement."
        changes.append(_change(category, direction, baseline_state, candidate_state, summary, EVIDENCE_EXACT))
        if baseline_state != "success" or candidate_state != "success":
            return changes

    baseline_status = baseline.get("qc", {}).get("overall_status")
    candidate_status = candidate.get("qc", {}).get("overall_status")
    if baseline_status is None or candidate_status is None:
        changes.append(
            _change(
                "inconclusive_missing_data",
                "inconclusive",
                baseline_status,
                candidate_status,
                "QC status comparison is unavailable because one or both artifacts omit qc.overall_status.",
                EVIDENCE_UNAVAILABLE,
            )
        )
    elif _rank(candidate_status, _STATUS_RANK) < _rank(baseline_status, _STATUS_RANK):
        changes.append(_change("qc_status_improved", "improved", baseline_status, candidate_status, "Candidate QC status is objectively less severe.", EVIDENCE_EXACT))
    elif _rank(candidate_status, _STATUS_RANK) > _rank(baseline_status, _STATUS_RANK):
        changes.append(
            _change(
                "regression_in_qc_status",
                "regressed",
                baseline_status,
                candidate_status,
                "Candidate QC status is objectively more severe.",
                EVIDENCE_EXACT,
            )
        )
    else:
        changes.append(_change("qc_status_unchanged", "unchanged", baseline_status, candidate_status, "QC status did not change.", EVIDENCE_EXACT))

    for count_key, improved_category, regressed_category in (
        ("warning_count", "less_warning_heavy", "more_warning_heavy"),
        ("fail_count", "fewer_fail_findings", "more_fail_findings"),
    ):
        baseline_count = baseline.get("qc", {}).get(count_key)
        candidate_count = candidate.get("qc", {}).get(count_key)
        if baseline_count is None or candidate_count is None:
            changes.append(
                _change(
                    "inconclusive_missing_data",
                    "inconclusive",
                    baseline_count,
                    candidate_count,
                    f"{count_key} comparison is unavailable because one or both artifacts omit it.",
                    EVIDENCE_UNAVAILABLE,
                )
            )
        elif candidate_count < baseline_count:
            changes.append(_change(improved_category, "improved", baseline_count, candidate_count, f"Candidate {count_key} decreased.", EVIDENCE_EXACT))
        elif candidate_count > baseline_count:
            changes.append(_change(regressed_category, "regressed", baseline_count, candidate_count, f"Candidate {count_key} increased.", EVIDENCE_EXACT))
        else:
            changes.append(_change(f"{count_key}_unchanged", "unchanged", baseline_count, candidate_count, f"{count_key} did not change.", EVIDENCE_EXACT))
    return changes


def _compare_metric_closeness(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    baseline_metrics = baseline.get("metrics", {})
    candidate_metrics = candidate.get("metrics", {})
    changes = [
        _compare_smaller_is_better(
            baseline_metrics.get("abs_loudness_delta_lu"),
            candidate_metrics.get("abs_loudness_delta_lu"),
            improved_category="closer_to_target",
            regressed_category="farther_from_target",
            unchanged_category="target_closeness_unchanged",
            missing_summary="Loudness target closeness is unavailable because one or both artifacts omit abs_loudness_delta_lu.",
            unit="LU",
        ),
        _compare_larger_is_safer(
            baseline_metrics.get("true_peak_margin_db"),
            candidate_metrics.get("true_peak_margin_db"),
        ),
    ]
    return changes


def _compare_reference(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    baseline_reference = baseline.get("reference", {})
    candidate_reference = candidate.get("reference", {})
    baseline_available = bool(baseline_reference.get("available"))
    candidate_available = bool(candidate_reference.get("available"))
    if not baseline_available and not candidate_available:
        return [
            _change(
                "reference_not_assessed",
                "unchanged",
                baseline_reference.get("status"),
                candidate_reference.get("status"),
                "Neither run has reference assessment artifacts; reference-specific comparison is unavailable by design.",
                EVIDENCE_UNAVAILABLE,
            )
        ]
    if not baseline_available or not candidate_available:
        return [
            _change(
                "reference_comparison_unavailable",
                "inconclusive",
                baseline_reference.get("status"),
                candidate_reference.get("status"),
                "Reference comparison is unavailable because one or both runs lack reference assessment artifacts.",
                EVIDENCE_UNAVAILABLE,
            )
        ]

    changes: list[dict[str, Any]] = []
    baseline_status = baseline_reference.get("status")
    candidate_status = candidate_reference.get("status")
    if _rank(candidate_status, _REFERENCE_STATUS_RANK) < _rank(baseline_status, _REFERENCE_STATUS_RANK):
        changes.append(_change("reference_status_improved", "improved", baseline_status, candidate_status, "Candidate reference outcome is less severe.", EVIDENCE_EXACT))
    elif _rank(candidate_status, _REFERENCE_STATUS_RANK) > _rank(baseline_status, _REFERENCE_STATUS_RANK):
        changes.append(_change("reference_status_regressed", "regressed", baseline_status, candidate_status, "Candidate reference outcome is more severe.", EVIDENCE_EXACT))
    else:
        changes.append(_change("reference_status_unchanged", "unchanged", baseline_status, candidate_status, "Reference outcome did not change.", EVIDENCE_EXACT))

    changes.append(
        _compare_smaller_is_better(
            baseline_reference.get("average_normalized_residual"),
            candidate_reference.get("average_normalized_residual"),
            improved_category="closer_to_reference",
            regressed_category="farther_from_reference",
            unchanged_category="reference_residual_unchanged",
            missing_summary="Reference residual comparison is unavailable because one or both artifacts omit average_normalized_residual.",
            unit="normalized residual",
        )
    )
    baseline_matched = baseline_reference.get("matched_metric_count")
    candidate_matched = candidate_reference.get("matched_metric_count")
    if baseline_matched is None or candidate_matched is None:
        changes.append(
            _change(
                "reference_matched_metric_count_unavailable",
                "inconclusive",
                baseline_matched,
                candidate_matched,
                "Reference matched metric count comparison is unavailable.",
                EVIDENCE_UNAVAILABLE,
            )
        )
    elif candidate_matched > baseline_matched:
        changes.append(_change("more_reference_metrics_matched", "improved", baseline_matched, candidate_matched, "Candidate matched more reference metrics.", EVIDENCE_EXACT))
    elif candidate_matched < baseline_matched:
        changes.append(_change("fewer_reference_metrics_matched", "regressed", baseline_matched, candidate_matched, "Candidate matched fewer reference metrics.", EVIDENCE_EXACT))
    else:
        changes.append(_change("reference_matched_metric_count_unchanged", "unchanged", baseline_matched, candidate_matched, "Reference matched metric count did not change.", EVIDENCE_EXACT))
    return changes


def _compare_shortlist_movements(
    run_id: str,
    baseline_ranks: dict[str, dict[str, int]],
    candidate_ranks: dict[str, dict[str, int]],
) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for shortlist_name in sorted(set(baseline_ranks) | set(candidate_ranks)):
        baseline_rank = baseline_ranks.get(shortlist_name, {}).get(run_id)
        candidate_rank = candidate_ranks.get(shortlist_name, {}).get(run_id)
        if baseline_rank is None and candidate_rank is None:
            continue
        if baseline_rank is None or candidate_rank is None:
            changes.append(
                _change(
                    "shortlist_membership_changed",
                    "inconclusive",
                    baseline_rank,
                    candidate_rank,
                    f"Run shortlist membership changed for {shortlist_name}; ranking direction is not a standalone improvement claim.",
                    EVIDENCE_EXACT,
                    shortlist=shortlist_name,
                )
            )
        elif candidate_rank < baseline_rank:
            changes.append(
                _change(
                    "shortlist_rank_moved_up",
                    "inconclusive",
                    baseline_rank,
                    candidate_rank,
                    f"Run moved up in {shortlist_name} objective ordering; this is a review-order signal, not a standalone improvement claim.",
                    EVIDENCE_INFERRED,
                    shortlist=shortlist_name,
                )
            )
        elif candidate_rank > baseline_rank:
            changes.append(
                _change(
                    "shortlist_rank_moved_down",
                    "inconclusive",
                    baseline_rank,
                    candidate_rank,
                    f"Run moved down in {shortlist_name} objective ordering; this is a review-order signal, not a standalone regression claim.",
                    EVIDENCE_INFERRED,
                    shortlist=shortlist_name,
                )
            )
        else:
            changes.append(
                _change(
                    "shortlist_rank_unchanged",
                    "unchanged",
                    baseline_rank,
                    candidate_rank,
                    f"Run kept the same rank in {shortlist_name}.",
                    EVIDENCE_EXACT,
                    shortlist=shortlist_name,
                )
            )
    return changes


def _compare_smaller_is_better(
    baseline_value: Any,
    candidate_value: Any,
    *,
    improved_category: str,
    regressed_category: str,
    unchanged_category: str,
    missing_summary: str,
    unit: str,
) -> dict[str, Any]:
    if baseline_value is None or candidate_value is None:
        return _change("inconclusive_missing_data", "inconclusive", baseline_value, candidate_value, missing_summary, EVIDENCE_UNAVAILABLE)
    baseline_float = float(baseline_value)
    candidate_float = float(candidate_value)
    delta = candidate_float - baseline_float
    if delta < -_MEANINGFUL_DELTA:
        return _change(improved_category, "improved", baseline_float, candidate_float, f"Candidate moved closer by {abs(delta):.3f} {unit}.", EVIDENCE_EXACT, delta=delta)
    if delta > _MEANINGFUL_DELTA:
        return _change(regressed_category, "regressed", baseline_float, candidate_float, f"Candidate moved farther by {delta:.3f} {unit}.", EVIDENCE_EXACT, delta=delta)
    return _change(unchanged_category, "unchanged", baseline_float, candidate_float, f"Change is within the deterministic comparison threshold of {_MEANINGFUL_DELTA:.2f} {unit}.", EVIDENCE_EXACT, delta=delta)


def _compare_larger_is_safer(baseline_value: Any, candidate_value: Any) -> dict[str, Any]:
    if baseline_value is None or candidate_value is None:
        return _change(
            "inconclusive_missing_data",
            "inconclusive",
            baseline_value,
            candidate_value,
            "True-peak margin comparison is unavailable because one or both artifacts omit true_peak_margin_db.",
            EVIDENCE_UNAVAILABLE,
        )
    baseline_float = float(baseline_value)
    candidate_float = float(candidate_value)
    delta = candidate_float - baseline_float
    if delta > _MEANINGFUL_DELTA:
        return _change("safer_true_peak_margin", "improved", baseline_float, candidate_float, f"Candidate has {delta:.3f} dB more true-peak margin.", EVIDENCE_EXACT, delta=delta)
    if delta < -_MEANINGFUL_DELTA:
        return _change("reduced_true_peak_margin", "regressed", baseline_float, candidate_float, f"Candidate has {abs(delta):.3f} dB less true-peak margin.", EVIDENCE_EXACT, delta=delta)
    return _change("true_peak_margin_unchanged", "unchanged", baseline_float, candidate_float, f"Change is within the deterministic comparison threshold of {_MEANINGFUL_DELTA:.2f} dB.", EVIDENCE_EXACT, delta=delta)


def _change(
    category: str,
    direction: str,
    baseline_value: Any,
    candidate_value: Any,
    summary: str,
    evidence_level: str,
    **extra: Any,
) -> dict[str, Any]:
    payload = {
        "category": category,
        "direction": direction,
        "evidence_level": evidence_level,
        "baseline_value": baseline_value,
        "candidate_value": candidate_value,
        "summary": summary,
    }
    payload.update(extra)
    return payload


def _run_outcome(changes: list[dict[str, Any]]) -> str:
    if not changes:
        return OUTCOME_INCONCLUSIVE
    if any(change["category"] == "inconclusive_missing_data" for change in changes):
        return OUTCOME_INCONCLUSIVE
    if any(change["direction"] == "inconclusive" for change in changes) and not any(change["direction"] in {"improved", "regressed"} for change in changes):
        return OUTCOME_INCONCLUSIVE
    improved = sum(1 for change in changes if change["direction"] == "improved")
    regressed = sum(1 for change in changes if change["direction"] == "regressed")
    if regressed and not improved:
        return OUTCOME_REGRESSED
    if improved and not regressed:
        return OUTCOME_IMPROVED
    if improved and regressed:
        return OUTCOME_INCONCLUSIVE
    if any(change["direction"] == "inconclusive" for change in changes):
        return OUTCOME_INCONCLUSIVE
    return OUTCOME_UNCHANGED


def _run_evidence_level(changes: list[dict[str, Any]]) -> str:
    if not changes or any(change["evidence_level"] == EVIDENCE_UNAVAILABLE for change in changes):
        return EVIDENCE_UNAVAILABLE
    if any(change["evidence_level"] == EVIDENCE_INFERRED for change in changes):
        return EVIDENCE_INFERRED
    return EVIDENCE_EXACT


def _run_summary(outcome: str, changes: list[dict[str, Any]]) -> str:
    meaningful = [change["category"] for change in changes if change["direction"] in {"improved", "regressed", "inconclusive"}]
    if not meaningful:
        return "No objective change exceeded comparison thresholds."
    return f"{outcome}: " + ", ".join(meaningful[:4])


def _build_aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    category_counts: dict[str, int] = {}
    direction_counts: dict[str, int] = {}
    for run in runs:
        for change in run["changes"]:
            category = change["category"]
            category_counts[category] = category_counts.get(category, 0) + 1
            direction = change["direction"]
            direction_counts[direction] = direction_counts.get(direction, 0) + 1
    return {
        "run_count": len(runs),
        "comparable_run_count": sum(1 for run in runs if run["presence"] == "both"),
        "improved_run_count": sum(1 for run in runs if run["outcome"] == OUTCOME_IMPROVED),
        "regressed_run_count": sum(1 for run in runs if run["outcome"] == OUTCOME_REGRESSED),
        "unchanged_run_count": sum(1 for run in runs if run["outcome"] == OUTCOME_UNCHANGED),
        "inconclusive_run_count": sum(1 for run in runs if run["outcome"] == OUTCOME_INCONCLUSIVE),
        "missing_baseline_run_count": sum(1 for run in runs if run["presence"] == "baseline_missing"),
        "missing_candidate_run_count": sum(1 for run in runs if run["presence"] == "candidate_missing"),
        "category_counts": dict(sorted(category_counts.items())),
        "direction_counts": dict(sorted(direction_counts.items())),
    }


def _summary_alias(aggregate: dict[str, Any]) -> dict[str, Any]:
    return {
        "matched_run_count": aggregate["comparable_run_count"],
        "added_run_count": aggregate["missing_baseline_run_count"],
        "removed_run_count": aggregate["missing_candidate_run_count"],
        "improved_count": aggregate["improved_run_count"],
        "regressed_count": aggregate["regressed_run_count"],
        "unchanged_count": aggregate["unchanged_run_count"],
        "inconclusive_count": aggregate["inconclusive_run_count"],
        "classification_counts": {
            OUTCOME_IMPROVED: aggregate["improved_run_count"],
            OUTCOME_REGRESSED: aggregate["regressed_run_count"],
            OUTCOME_UNCHANGED: aggregate["unchanged_run_count"],
            OUTCOME_INCONCLUSIVE: aggregate["inconclusive_run_count"],
        },
    }


def _build_review_priority(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(runs, key=lambda run: (_priority_rank(run), run["run_id"]))
    priority = []
    for run in ranked:
        rank = _priority_rank(run)
        if rank >= 4:
            continue
        reason = _priority_reason(run)
        priority.append({"run_id": run["run_id"], "outcome": run["outcome"], "reason": reason})
    return priority


def _priority_rank(run: dict[str, Any]) -> int:
    categories = {change["category"] for change in run["changes"]}
    if "run_failed" in categories or "regression_in_qc_status" in categories or "more_fail_findings" in categories:
        return 0
    if run["outcome"] == OUTCOME_REGRESSED:
        return 1
    if run["presence"] != "both" or run["outcome"] == OUTCOME_INCONCLUSIVE:
        return 2
    if run["outcome"] == OUTCOME_IMPROVED:
        return 3
    return 4


def _priority_reason(run: dict[str, Any]) -> str:
    for direction in ("regressed", "inconclusive", "improved"):
        for change in run["changes"]:
            if change["direction"] == direction:
                return change["summary"]
    return run["summary"] or "Objective artifact comparison changed."


def _shortlist_ranks(shortlist: dict[str, Any]) -> dict[str, dict[str, int]]:
    ranks: dict[str, dict[str, int]] = {}
    if not isinstance(shortlist, dict):
        return ranks
    for name, entries in shortlist.items():
        if not isinstance(entries, list):
            continue
        ranks[str(name)] = {}
        for index, entry in enumerate(entries, start=1):
            if isinstance(entry, dict) and entry.get("run_id"):
                ranks[str(name)][str(entry["run_id"])] = index
    return ranks


def _rank(value: Any, ranks: dict[str, int]) -> int:
    return ranks.get(str(value), max(ranks.values()) + 1)


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"
