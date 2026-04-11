from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cognis import batch


def _finding(code: str, severity: str = "informational"):
    return SimpleNamespace(code=code, severity=severity, message=f"{code} message")


def _comparison(metric: str, status: str = "improved", delta: float = 0.25, tolerance: float = 0.5):
    return SimpleNamespace(
        metric=metric,
        status=status,
        output_delta_to_reference=delta,
        movement_toward_reference=0.5,
        tolerance=tolerance,
    )


def _report(*, status: str = "pass", reference: bool = False, decision_history: bool = False):
    findings = [_finding("TARGET_LOUDNESS_HIT")]
    if status == "warning":
        findings.append(_finding("TRUE_PEAK_TOO_CLOSE_TO_CEILING", "warning"))
    if status == "fail":
        findings.append(_finding("TRUE_PEAK_ABOVE_CEILING", "fail"))

    reference_assessment = None
    if reference:
        reference_assessment = SimpleNamespace(
            outcome="partial",
            comparisons=(
                _comparison("integrated_lufs", delta=0.2, tolerance=0.5),
                _comparison("spectral_tilt_db_per_decade", status="matched", delta=0.1, tolerance=0.75),
            ),
            findings=(_finding("REFERENCE_TONAL_DEVIATION", "warning"),),
        )

    decision_history_summary = None
    if decision_history:
        decision_history_summary = SimpleNamespace(
            available=True,
            selection_basis="exact_bounded_grid_search",
            candidate_count=4,
            winner_candidate_index=1,
            runner_up_candidate_index=2,
            score_margin_to_runner_up=0.25,
            limitations=(),
        )

    return SimpleNamespace(
        overall_status=status,
        findings=tuple(findings),
        achieved=SimpleNamespace(
            integrated_lufs=-14.1,
            true_peak_dbfs=-1.4,
            limiter_stress_estimate=0.2,
            codec_risk_estimate=0.1,
            clipping_risk_estimate=0.0,
        ),
        requested=SimpleNamespace(target_loudness_lufs=-14.0),
        delta=SimpleNamespace(loudness_delta_lu=-0.1, true_peak_margin_db=0.4),
        reference_assessment=reference_assessment,
        decision_history_summary=decision_history_summary,
    )


def _result(*, status: str = "pass", reference: bool = False, decision_history: bool = False):
    return SimpleNamespace(
        report=_report(status=status, reference=reference, decision_history=decision_history),
        reference_analysis=SimpleNamespace() if reference else None,
    )


def test_manifest_expansion_supports_multi_track_modes_and_references(tmp_path):
    manifest = {
        "schema_version": batch.BATCH_MANIFEST_SCHEMA_VERSION,
        "defaults": {
            "modes": ["STREAMING_SAFE", "PRESERVE_DYNAMICS"],
            "options": {"target_loudness": -15.0},
        },
        "tracks": [
            {
                "id": "Track One",
                "path": "audio/track-one.wav",
                "references": [{"id": "Main Ref", "path": "refs/main-ref.wav"}],
                "include_unreferenced": True,
                "tags": ["dogfood"],
            }
        ],
    }

    plans = batch.expand_manifest(manifest, manifest_dir=tmp_path)

    assert [plan.run_id for plan in plans] == [
        "track_one_mode_streaming_safe",
        "track_one_mode_streaming_safe_ref_main_ref",
        "track_one_mode_preserve_dynamics",
        "track_one_mode_preserve_dynamics_ref_main_ref",
    ]
    assert all(Path(plan.input_path).is_absolute() for plan in plans)
    assert plans[0].reference is None
    assert plans[1].reference is not None
    assert plans[0].options["target_loudness"] == -15.0
    assert plans[0].tags == ("dogfood",)


def test_session_artifact_summarizes_reference_and_non_reference_runs(tmp_path):
    plan_standard = batch.BatchRunPlan(
        run_id="standard",
        track_id="track",
        input_path=str(tmp_path / "input.wav"),
        mode="STREAMING_SAFE",
        reference=None,
        options={},
        tags=(),
        notes=None,
    )
    plan_reference = batch.BatchRunPlan(
        run_id="reference",
        track_id="track",
        input_path=str(tmp_path / "input.wav"),
        mode="REFERENCE_MATCH",
        reference=batch.BatchReference("ref", str(tmp_path / "ref.wav")),
        options={},
        tags=("ref",),
        notes="reference pass",
    )

    standard = batch.summarize_successful_run(
        plan_standard,
        _result(status="pass"),
        run_dir=tmp_path / "runs" / "standard",
        output_path=tmp_path / "runs" / "standard" / "output.wav",
        written={"report": str(tmp_path / "runs" / "standard" / "output.report.json")},
        session_root=tmp_path,
    )
    reference = batch.summarize_successful_run(
        plan_reference,
        _result(status="warning", reference=True, decision_history=True),
        run_dir=tmp_path / "runs" / "reference",
        output_path=tmp_path / "runs" / "reference" / "output.wav",
        written={"report": str(tmp_path / "runs" / "reference" / "output.report.json")},
        session_root=tmp_path,
    )
    session = batch.build_session_artifact(
        {"schema_version": batch.BATCH_MANIFEST_SCHEMA_VERSION, "session_id": "dogfood"},
        [standard, reference],
        session_root=tmp_path,
    )

    assert session["schema_version"] == batch.BATCH_SESSION_SCHEMA_VERSION
    assert session["aggregate"]["successful_run_count"] == 2
    assert session["aggregate"]["reference_assessed_run_count"] == 1
    assert standard["reference"]["status"] == "unavailable"
    assert reference["reference"]["status"] == "partial"
    assert reference["decision_history"]["status"] == "available"
    assert standard["decision_history"]["status"] == "unavailable"


def test_session_artifact_preserves_corpus_metadata_and_asset_root(tmp_path):
    manifest = {
        "schema_version": batch.BATCH_MANIFEST_SCHEMA_VERSION,
        "corpus": {
            "schema_version": batch.DOGFOOD_CORPUS_SCHEMA_VERSION,
            "id": "core_dogfood",
            "name": "Core Dogfood",
            "version": 1,
            "asset_policy": "external_or_local",
            "asset_root": "local-corpora/core",
            "tags": ["dogfood", "regression"],
        },
        "runs": [{"id": "track", "input": "premasters/track.wav", "mode": "STREAMING_SAFE"}],
    }

    plans = batch.expand_manifest(manifest, manifest_dir=tmp_path)
    session = batch.build_session_artifact(manifest, [], session_root=tmp_path, manifest_path=tmp_path / "manifest.json")

    assert plans[0].input_path == str((tmp_path / "local-corpora" / "core" / "premasters" / "track.wav").resolve())
    assert session["corpus"]["schema_version"] == batch.DOGFOOD_CORPUS_SCHEMA_VERSION
    assert session["corpus"]["id"] == "core_dogfood"
    assert session["corpus"]["tags"] == ["dogfood", "regression"]


def test_markdown_and_session_json_do_not_claim_best_sounding(tmp_path):
    plan = batch.BatchRunPlan("safe", "track", str(tmp_path / "input.wav"), "STREAMING_SAFE", None, {}, (), None)
    summary = batch.summarize_successful_run(
        plan,
        _result(status="pass"),
        run_dir=tmp_path / "runs" / "safe",
        output_path=tmp_path / "runs" / "safe" / "output.wav",
        written={},
        session_root=tmp_path,
    )
    session = batch.build_session_artifact(
        {"schema_version": batch.BATCH_MANIFEST_SCHEMA_VERSION, "session_id": "dogfood"},
        [summary],
        session_root=tmp_path,
    )
    markdown = batch.render_session_markdown(session)
    payload = json.dumps(session, sort_keys=True)

    forbidden = ("best sounding", "best-sounding", "sounds best", "better sounding", "winner sounds", "favorite", "vibe")
    for phrase in forbidden:
        assert phrase not in markdown.lower()
        assert phrase not in payload.lower()
    assert "no subjective listening-quality claim" in markdown.lower()
    assert "per-run qc reports remain the source of truth" in markdown.lower()


def test_run_batch_manifest_continues_after_one_run_exception(monkeypatch, tmp_path):
    manifest_path = tmp_path / "manifest.json"
    output_root = tmp_path / "session"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": batch.BATCH_MANIFEST_SCHEMA_VERSION,
                "runs": [
                    {"id": "ok", "input": "ok.wav", "mode": "STREAMING_SAFE"},
                    {"id": "bad", "input": "bad.wav", "mode": "STREAMING_SAFE"},
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_execute_run(plan, *, engine, output_path, run_dir):
        if plan.run_id == "bad":
            raise RuntimeError("controlled failure")
        return _result(status="pass")

    def fake_write_render_artifacts(result, output_path, artifacts_dir, **kwargs):
        return {"report": str(Path(artifacts_dir) / "output.report.json")}

    monkeypatch.setattr(batch, "execute_run", fake_execute_run)
    monkeypatch.setattr(batch, "write_render_artifacts", fake_write_render_artifacts)

    session = batch.run_batch_manifest(manifest_path, output_root=output_root)

    assert session["aggregate"]["successful_run_count"] == 1
    assert session["aggregate"]["failed_run_count"] == 1
    assert [run["state"] for run in session["runs"]] == [batch.RUN_STATE_SUCCESS, batch.RUN_STATE_FAILED]
    assert session["runs"][1]["error"]["type"] == "RuntimeError"
    assert (output_root / "session.json").exists()
    assert (output_root / "session.md").exists()


def test_load_batch_manifest_rejects_unknown_schema(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"schema_version": "future_schema", "runs": [{"input": "x.wav"}]}), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported batch manifest"):
        batch.load_batch_manifest(manifest_path)
