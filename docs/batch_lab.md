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

Compare two completed sessions:

```bash
python -m cognis.cli batch compare build/dogfood/session-01/session.json build/dogfood/session-02/session.json --output-root build/dogfood/comparisons/session-01-vs-session-02
```

Equivalent module entry point:

```bash
python -m cognis.batch compare build/dogfood/session-01/session.json build/dogfood/session-02/session.json --output-root build/dogfood/comparisons/session-01-vs-session-02
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

Curated dogfood corpora use the same executable batch manifest format with an optional top-level corpus block:

```json
{
  "schema_version": "cognis_batch_manifest_v1",
  "corpus": {
    "schema_version": "cognis_dogfood_corpus_v1",
    "id": "core_dogfood",
    "name": "Core Dogfood Corpus",
    "version": 1,
    "asset_policy": "external_or_local",
    "asset_root": "../../local-corpora/core-dogfood",
    "tags": ["dogfood", "regression", "corpus:core"]
  }
}
```

Audio is not expected to live in git. Put real premaster/reference files under ignored local folders, an external drive, a mounted dataset, or CI-provided paths. If `corpus.asset_root` is set, relative track and reference paths resolve from that root; otherwise they resolve relative to the manifest file.

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

If `corpus.asset_root` is present, relative audio and reference paths are resolved from that root instead.

Recommended flat tag namespaces:

- `genre:pop`, `genre:rock`, `genre:hiphop`, `genre:electronic`, `genre:acoustic`, `genre:spoken-word`
- `delivery:streaming`, `delivery:club`, `delivery:broadcast`, `delivery:archive`
- `reference-based`, `non-reference`
- `stress:limiter`, `stress:true-peak`, `stress:low-end`, `stress:stereo-width`, `stress:mono`, `stress:tonal-balance`, `stress:dynamics`, `stress:codec-risk`
- `corpus:core`, `corpus:nightly`, `corpus:release-gate`

## Session Artifact

Schema version: `cognis_batch_session_v1`

The machine-readable session artifact is written to `session.json`. It contains:

- manifest and session schema versions
- optional corpus metadata copied from the manifest
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

## Session Comparison

Schema version: `cognis_session_comparison_v1`

The comparison layer reads two generated `session.json` artifacts and writes:

```text
comparison_root/
  comparison.json
  comparison.md
  linked_sessions/
    baseline_session.json
    candidate_session.json
```

The comparison is artifact-first. It does not open audio and does not require per-run artifacts beyond the session summaries.

At minimum it compares:

- run presence and absence
- run success/failure state
- QC status severity
- warning and fail counts
- absolute loudness target delta
- true-peak margin
- reference outcome and normalized residual when both runs have reference assessment
- objective shortlist membership/rank movement as review-order evidence only

Output classifications are:

- `improved`: objective measured signals improved without competing objective regressions
- `regressed`: objective measured signals got worse, or a baseline run is missing from the candidate
- `unchanged`: comparable measured signals stayed within deterministic thresholds
- `inconclusive`: artifacts are missing, mixed, or not comparable enough for a clean rule-based result

The comparison report helps prioritize listening and inspection:

- regressions first
- inconclusive or missing data next
- objective improvements after that
- unchanged runs last

It does not claim that a run is better sounding. Reference closeness means measured residuals moved closer under the reference assessment schema, not a subjective preference verdict.

Use `--fail-on-regression` when you want comparison to return exit code `2` if objective regressions are present:

```bash
python -m cognis.cli batch compare baseline/session.json candidate/session.json --output-root build/dogfood/comparisons/baseline-vs-candidate --fail-on-regression
```

## Compatibility

The single-render CLI remains unchanged:

```bash
python -m cognis.cli input.wav output.wav --mode STREAMING_SAFE --target_loudness -14.0 --ceiling_db -1.0
```

The batch layer calls the same `Engine.render()` and `write_render_artifacts()` path used by single renders. It does not alter DSP behavior.
