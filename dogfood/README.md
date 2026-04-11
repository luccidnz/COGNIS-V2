# COGNIS Dogfood Corpora

This folder stores curated dogfood manifest definitions, not audio.

Audio corpora should live in ignored local folders, external drives, mounted datasets, or CI-provided paths. The executable format remains the existing `cognis_batch_manifest_v1` batch manifest with an optional `corpus` metadata block.

## Corpus Metadata

Use this optional block at the top of a batch manifest:

```json
{
  "corpus": {
    "schema_version": "cognis_dogfood_corpus_v1",
    "id": "core_dogfood",
    "name": "Core Dogfood Corpus",
    "version": 1,
    "asset_policy": "external_or_local",
    "asset_root": "../../local-corpora/core-dogfood",
    "tags": ["dogfood", "regression", "mixed-material"]
  }
}
```

If `corpus.asset_root` is set, relative track and reference paths resolve from that root. Otherwise, paths resolve relative to the manifest file as before.

## Tag Convention

Tags are flat deterministic strings:

- `genre:pop`, `genre:rock`, `genre:hiphop`, `genre:electronic`, `genre:acoustic`, `genre:spoken-word`
- `delivery:streaming`, `delivery:club`, `delivery:broadcast`, `delivery:archive`
- `reference-based`, `non-reference`
- `stress:limiter`, `stress:true-peak`, `stress:low-end`, `stress:stereo-width`, `stress:mono`, `stress:tonal-balance`, `stress:dynamics`, `stress:codec-risk`
- `corpus:core`, `corpus:nightly`, `corpus:release-gate`

Do not use these manifests to claim subjective quality. They exist to make repeated measured evaluation practical across engine versions.
