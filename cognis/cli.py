from __future__ import annotations

import argparse

from cognis.config import CeilingMode, MasteringConfig, MasteringMode
from cognis.engine import Engine
from cognis.io.audio import load_audio, save_audio
from cognis.serialization.artifacts import write_render_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="COGNIS Mastering Engine CLI")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("output", help="Output audio file")
    parser.add_argument("--reference", type=str, default=None, help="Optional reference audio file")
    parser.add_argument("--mode", type=str, default="STREAMING_SAFE", help="Mastering mode")
    parser.add_argument("--target_loudness", type=float, default=-14.0, help="Target loudness (LUFS)")
    parser.add_argument("--ceiling_db", type=float, default=-1.0, help="Ceiling (dBFS)")
    parser.add_argument("--ceiling_mode", type=str, default="TRUE_PEAK", help="Ceiling mode")
    parser.add_argument("--artifacts-dir", type=str, default=None, help="Directory for JSON / markdown artifacts")
    parser.add_argument("--no-artifacts", action="store_true", help="Skip writing non-audio artifacts")
    parser.add_argument("--write-markdown-report", action="store_true", help="Also write a markdown report summary")
    args = parser.parse_args()

    try:
        mode = MasteringMode(args.mode.upper())
    except ValueError:
        print(f"Invalid mode: {args.mode}. Using STREAMING_SAFE.")
        mode = MasteringMode.STREAMING_SAFE

    try:
        ceiling_mode = CeilingMode(args.ceiling_mode.upper())
    except ValueError:
        print(f"Invalid ceiling mode: {args.ceiling_mode}. Using TRUE_PEAK.")
        ceiling_mode = CeilingMode.TRUE_PEAK

    config = MasteringConfig(
        mode=mode,
        target_loudness=args.target_loudness,
        ceiling_mode=ceiling_mode,
        ceiling_db=args.ceiling_db,
        oversampling=4,
        bass_preservation=1.0,
        stereo_width=1.0,
        dynamics_preservation=1.0,
        brightness=0.0,
        reference_path=args.reference,
        fir_backend="AUTO",
    )

    print(f"Loading {args.input}...")
    audio, sr = load_audio(args.input)

    reference_audio = None
    reference_sr = None
    if args.reference:
        print(f"Loading reference {args.reference}...")
        reference_audio, reference_sr = load_audio(args.reference)

    print("Running COGNIS engine...")
    engine = Engine()
    result = engine.render(audio, sr, config, reference_audio=reference_audio, reference_sr=reference_sr)

    print(f"Saving {args.output}...")
    save_audio(args.output, result.audio, sr)

    written = {}
    if not args.no_artifacts:
        written = write_render_artifacts(
            result,
            args.output,
            artifacts_dir=args.artifacts_dir,
            write_recipe=True,
            write_analysis=True,
            reference_analysis=result.reference_analysis,
            write_report=True,
            write_markdown_report=args.write_markdown_report,
        )

    report = result.report
    print("\n--- QC Summary ---")
    print(f"Overall Status:      {report.overall_status}")
    print(f"Target Loudness:     {report.requested.target_loudness_lufs:.2f} LUFS")
    print(f"Active Target:       {result.targets.target_loudness:.2f} LUFS")
    print(f"Achieved Loudness:   {report.achieved.integrated_lufs:.2f} LUFS")
    print(f"Loudness Delta:      {report.delta.loudness_delta_lu:+.2f} LU")
    print(f"True Peak:           {report.achieved.true_peak_dbfs:.2f} dBFS")
    print(f"Ceiling Margin:      {report.delta.true_peak_margin_db:.2f} dB")
    print(f"Warnings / Fails:    {sum(f.severity == 'warning' for f in report.findings)} / {sum(f.severity == 'fail' for f in report.findings)}")
    if report.reference_assessment is not None:
        print(f"Reference Outcome:   {report.reference_assessment.outcome}")
    print("------------------\n")

    if written:
        print("Artifacts:")
        for name, path in written.items():
            print(f"  {name}: {path}")
        print("")

    print("Done.")


if __name__ == "__main__":
    main()
