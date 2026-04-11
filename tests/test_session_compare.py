from __future__ import annotations

import json
from pathlib import Path

import pytest

from cognis import batch
from cognis import session_compare


def _run(
    run_id: str,
    *,
    state: str = "success",
    qc: str = "pass",
    loudness_delta: float | None = 0.1,
    true_peak_margin: float | None = 0.4,
    warnings: int = 0,
    fails: int = 0,
    reference: bool = False,
    reference_status: str = "partial",
    reference_residual: float | None = 0.5,
    matched_metrics: int = 1,
) -> dict:
    if state != "success":
        return {
            "run_id": run_id,
            "track_id": run_id,
            "state": state,
            "qc": {"overall_status": "unavailable", "warning_count": 0, "fail_count": 0},
            "metrics": {},
            "reference": {"available": False, "status": "unavailable"},
            "remaining_issues": [{"severity": "fail", "code": "RUN_EXCEPTION", "message": "failed"}],
        }
    reference_block = {"available": False, "status": "unavailable"}
    if reference:
        reference_block = {
            "available": True,
            "status": reference_status,
            "average_normalized_residual": reference_residual,
            "matched_metric_count": matched_metrics,
            "comparison_count": 2,
        }
    return {
        "run_id": run_id,
        "track_id": run_id,
        "state": "success",
        "qc": {"overall_status": qc, "warning_count": warnings, "fail_count": fails},
        "metrics": {
            "loudness_delta_lu": loudness_delta,
            "abs_loudness_delta_lu": abs(loudness_delta) if loudness_delta is not None else None,
            "true_peak_margin_db": true_peak_margin,
        },
        "reference": reference_block,
        "remaining_issues": [{"severity": "warning", "code": "W", "message": "warning"} for _ in range(warnings)]
        + [{"severity": "fail", "code": "F", "message": "fail"} for _ in range(fails)],
    }


def _session(session_id: str, runs: list[dict]) -> dict:
    return {
        "schema_version": batch.BATCH_SESSION_SCHEMA_VERSION,
        "session_id": session_id,
        "session_root": f"build/{session_id}",
        "run_count": len(runs),
        "runs": runs,
        "aggregate": {
            "successful_run_count": sum(run["state"] == "success" for run in runs),
            "failed_run_count": sum(run["state"] != "success" for run in runs),
            "warning_count": sum(run.get("qc", {}).get("warning_count", 0) for run in runs),
            "fail_finding_count": sum(run.get("qc", {}).get("fail_count", 0) for run in runs),
            "reference_assessed_run_count": sum(run.get("reference", {}).get("available", False) for run in runs),
        },
        "shortlist": {
            "listen_first_objective_order": [{"run_id": run["run_id"]} for run in runs],
            "manual_review_recommended": [{"run_id": run["run_id"]} for run in runs if run.get("remaining_issues")],
        },
    }


def test_comparison_artifact_generation_covers_improvement_regression_and_reference():
    baseline = _session(
        "baseline",
        [
            _run("closer", loudness_delta=0.4, true_peak_margin=0.2, warnings=1),
            _run("qc_regressed", qc="pass", fails=0),
            _run("ref", reference=True, reference_status="partial", reference_residual=0.8, matched_metrics=1),
        ],
    )
    candidate = _session(
        "candidate",
        [
            _run("closer", loudness_delta=0.1, true_peak_margin=0.5, warnings=0),
            _run("qc_regressed", qc="fail", fails=1),
            _run("ref", reference=True, reference_status="matched", reference_residual=0.2, matched_metrics=2),
        ],
    )

    comparison = session_compare.compare_session_artifacts(baseline, candidate)

    by_id = {run["run_id"]: run for run in comparison["runs"]}
    assert comparison["schema_version"] == session_compare.SESSION_COMPARISON_SCHEMA_VERSION
    assert by_id["closer"]["outcome"] == "improved"
    assert by_id["qc_regressed"]["outcome"] == "regressed"
    assert by_id["ref"]["outcome"] == "improved"
    assert any(change["category"] == "closer_to_reference" for change in by_id["ref"]["changes"])
    assert comparison["summary"]["improved_count"] == 2
    assert comparison["summary"]["regressed_count"] == 1


def test_comparison_ordering_is_deterministic():
    baseline = _session("baseline", [_run("zeta"), _run("alpha")])
    candidate = _session("candidate", [_run("alpha"), _run("zeta")])

    comparison = session_compare.compare_session_artifacts(baseline, candidate)

    assert [run["run_id"] for run in comparison["runs"]] == ["alpha", "zeta"]


def test_missing_runs_are_reported_honestly():
    baseline = _session("baseline", [_run("removed"), _run("kept")])
    candidate = _session("candidate", [_run("added"), _run("kept")])

    comparison = session_compare.compare_session_artifacts(baseline, candidate)
    by_id = {run["run_id"]: run for run in comparison["runs"]}

    assert by_id["removed"]["presence"] == "candidate_missing"
    assert by_id["removed"]["outcome"] == "regressed"
    assert by_id["added"]["presence"] == "baseline_missing"
    assert by_id["added"]["outcome"] == "inconclusive"
    assert comparison["summary"]["removed_run_count"] == 1
    assert comparison["summary"]["added_run_count"] == 1


def test_inconclusive_when_reference_data_disappears_without_false_improvement():
    baseline = _session("baseline", [_run("ref", reference=True, reference_residual=0.5)])
    candidate = _session("candidate", [_run("ref", reference=False)])

    comparison = session_compare.compare_session_artifacts(baseline, candidate)
    run = comparison["runs"][0]

    assert run["outcome"] == "inconclusive"
    assert any(change["category"] == "reference_comparison_unavailable" for change in run["changes"])
    assert comparison["summary"]["improved_count"] == 0


def test_comparison_outputs_markdown_and_linked_sessions(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(json.dumps(_session("baseline", [_run("a", loudness_delta=0.5)])), encoding="utf-8")
    candidate_path.write_text(json.dumps(_session("candidate", [_run("a", loudness_delta=0.1)])), encoding="utf-8")

    comparison = session_compare.compare_session_files(
        baseline_path,
        candidate_path,
        output_root=tmp_path / "comparison",
        copy_sessions=True,
    )

    markdown = (tmp_path / "comparison" / "comparison.md").read_text(encoding="utf-8")
    payload = (tmp_path / "comparison" / "comparison.json").read_text(encoding="utf-8")
    assert comparison["summary"]["improved_count"] == 1
    assert (tmp_path / "comparison" / "linked_sessions" / "baseline_session.json").exists()
    assert (tmp_path / "comparison" / "linked_sessions" / "candidate_session.json").exists()
    forbidden = ("better sounding", "best sounding", "sounds better", "favorite", "perceptual superiority")
    for phrase in forbidden:
        assert phrase not in markdown.lower()
        assert phrase not in payload.lower()
    assert "no subjective listening-quality claim" in markdown.lower()


def test_batch_compare_cli_writes_artifacts(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    output_root = tmp_path / "out"
    baseline_path.write_text(json.dumps(_session("baseline", [_run("a", loudness_delta=0.5)])), encoding="utf-8")
    candidate_path.write_text(json.dumps(_session("candidate", [_run("a", loudness_delta=0.1)])), encoding="utf-8")

    batch.compare_main([str(baseline_path), str(candidate_path), "--output-root", str(output_root)])

    assert (output_root / "comparison.json").exists()
    assert (output_root / "comparison.md").exists()


def test_batch_compare_cli_can_fail_on_regression(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    output_root = tmp_path / "out"
    baseline_path.write_text(json.dumps(_session("baseline", [_run("a", qc="pass")])), encoding="utf-8")
    candidate_path.write_text(json.dumps(_session("candidate", [_run("a", qc="fail", fails=1)])), encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        batch.compare_main([str(baseline_path), str(candidate_path), "--output-root", str(output_root), "--fail-on-regression"])

    assert exc.value.code == 2
