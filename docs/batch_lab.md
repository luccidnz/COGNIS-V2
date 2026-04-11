# COGNIS Batch Evaluation + Dogfood Mastering Lab

The Batch Lab runs many deterministic mastering renders from one JSON manifest and writes a session-level evaluation pack. It is intended for engine dogfooding, real-material QA, future corpus work, and preference-collection setup.

It does not replace per-run QC. Each render report remains the source of truth for that output. The batch report only aggregates measured results.

## Command

```bash
python -m cognis.cli batch dogfood.manifest.json --output-root build/dogfood/session-01
```

Equivalent module entry point:

```bash
python -m cognis.batch dogfood.manifest.json --output-root build/dogfood/session-01
```

Outputs:

```text
build/dogfood/session-01/
  session.json
  session.md
  runs/
    track01_mode_streaming_safe/
      output.wav
      output.recipe.json
      output.analysis.input.json
      output.analysis.output.json
      output.report.json
      output.report.md
    track01_mode_reference_match_ref_main/
      output.wav
      output.recipe.json
      output.analysis.input.json
      output.analysis.output.json
      output.analysis.reference.json
      output.decision_history.json
      output.report.json
      output.report.md
```

## Manifest Schema

Schema version: `cognis_batch_manifest_v1`

Minimal explicit-run manifest:

```json
{
  "schema_version": "cognis_batch_manifest_v1",
  "session_id": "dogfood-session-01",
  "defaults": {
    "options": {
      "target_loudness": -14.0,
      "ceiling_db": -1.0,
      "ceiling_mode": "TRUE_PEAK"
    }
  },
  "runs": [
    {
      "id": "song_a_streaming",
      "input": "premasters/song-a.wav",
      "mode": "STREAMING_SAFE"
    },
    {
      "id": "song_a_reference",
      "input": "premasters/song-a.wav",
      "reference": "references/main.wav",
      "mode": "REFERENCE_MATCH",
      "overrides": {
        "dynamics_preservation": 0.8,
        "brightness": 0.1
      },
      "tags": ["reference"],
      "notes": "Reference-aware dogfood pass."
    }
  ]
}
```

Track-expansion manifest:

```json
{
  "schema_version": "cognis_batch_manifest_v1",
  "session_id": "ep-dogfood",
  "defaults": {
    "modes": ["STREAMING_SAFE", "PRESERVE_DYNAMICS"],
    "options": {
      "target_loudness": -14.0,
      "ceiling_db": -1.0,
      "ceiling_mode": "TRUE_PEAK",
      "fir_backend": "AUTO"
    }
  },
  "tracks": [
    {
      "id": "track01",
      "path": "premasters/track01.wav",
      "references": [{"id": "main", "path": "references/track01-ref.wav"}],
      "include_unreferenced": true,
      "tags": ["dogfood"]
    }
  ]
}
```

Supported options map directly to `MasteringConfig`:

- `target_loudness`
- `ceiling_db`
- `ceiling_mode`
- `oversampling`
- `bass_preservation`
- `stereo_width`
- `dynamics_preservation`
- `brightness`
- `fir_backend`

Paths are resolved relative to the manifest file unless absolute.

## Session Artifact

Schema version: `cognis_batch_session_v1`

The machine-readable session artifact is written to `session.json`. It contains:

- manifest and session schema versions
- stable `session_id`
- run list in manifest order
- per-run identities, tags, notes, input/reference paths
- output audio and artifact locations
- per-run QC summaries
- target and peak metrics
- reference summaries when available
- decision-history availability when available
- aggregate counts
- objective shortlist sections
- explicit failed-run records

Representative run summary:

```json
{
  "run_id": "song_a_reference",
  "state": "success",
  "mode": "REFERENCE_MATCH",
  "qc": {
    "overall_status": "warning",
    "warning_count": 1,
    "fail_count": 0
  },
  "metrics": {
    "integrated_lufs": -14.1,
    "loudness_delta_lu": -0.1,
    "true_peak_dbfs": -1.4,
    "true_peak_margin_db": 0.4
  },
  "reference": {
    "available": true,
    "status": "partial",
    "average_normalized_residual": 0.42
  },
  "decision_history": {
    "available": true,
    "status": "available",
    "selection_basis": "exact_bounded_grid_search"
  }
}
```

Failed runs stay in the session artifact:

```json
{
  "run_id": "bad_input",
  "state": "failed",
  "error": {
    "type": "RuntimeError",
    "message": "controlled failure"
  },
  "reference": {
    "available": false,
    "status": "unavailable",
    "reason": "Run failed before reference assessment could be produced."
  }
}
```

## Shortlist Semantics

The batch shortlist is objective and transparent. It includes:

- `safest_measured`
- `closest_to_target`
- `closest_to_reference`
- `listen_first_objective_order`
- `manual_review_recommended`

These are measured review aids, not listening-quality claims. Reference closeness is unavailable when no reference assessment exists. Decision-history conclusions are unavailable when the run did not produce decision history.

The markdown report states this boundary explicitly:

- measured safety, target closeness, reference residuals, and review priority only
- no subjective listening-quality claim
- per-run QC reports remain the source of truth

## Mixed Run States

The batch runner continues after a run exception by default. Use `--fail-fast` to stop at the first failed run.

Run states currently used:

- `success`: audio and per-run artifacts were written
- `failed`: the run raised before a complete per-run artifact set could be produced

QC `fail` is not the same as an orchestration failure. A render can complete successfully and still have `qc.overall_status == "fail"` in its measured report.

## Compatibility

The single-render CLI remains unchanged:

```bash
python -m cognis.cli input.wav output.wav --mode STREAMING_SAFE --target_loudness -14.0 --ceiling_db -1.0
```

The batch layer calls the same `Engine.render()` and `write_render_artifacts()` path used by single renders. It does not alter DSP behavior.
