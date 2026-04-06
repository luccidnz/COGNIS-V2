import argparse
import json
from cognis.config import MasteringConfig, MasteringMode, CeilingMode
from cognis.engine import Engine
from cognis.io.audio import load_audio, save_audio

def main():
    parser = argparse.ArgumentParser(description="COGNIS Mastering Engine CLI")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("output", help="Output audio file")
    parser.add_argument("--mode", type=str, default="STREAMING_SAFE", help="Mastering mode")
    parser.add_argument("--target_loudness", type=float, default=-14.0, help="Target loudness (LUFS)")
    parser.add_argument("--ceiling_db", type=float, default=-1.0, help="Ceiling (dBFS)")
    
    args = parser.parse_args()
    
    try:
        mode = MasteringMode(args.mode.upper())
    except ValueError:
        print(f"Invalid mode: {args.mode}. Using STREAMING_SAFE.")
        mode = MasteringMode.STREAMING_SAFE
        
    config = MasteringConfig(
        mode=mode,
        target_loudness=args.target_loudness,
        ceiling_mode=CeilingMode.TRUE_PEAK,
        ceiling_db=args.ceiling_db,
        oversampling=4,
        bass_preservation=1.0,
        stereo_width=1.0,
        dynamics_preservation=1.0,
        brightness=0.0,
        fir_backend="AUTO"
    )
    
    print(f"Loading {args.input}...")
    audio, sr = load_audio(args.input)
    
    print("Running COGNIS engine...")
    engine = Engine()
    mastered, report, recipe = engine.process(audio, sr, config)
    
    print(f"Saving {args.output}...")
    save_audio(args.output, mastered, sr)
    
    recipe_path = args.output + ".recipe.json"
    print(f"Saving recipe to {recipe_path}...")
    from cognis.serialization.recipe import serialize_recipe
    with open(recipe_path, "w") as f:
        f.write(serialize_recipe(recipe))
    
    print("\n--- QC Report ---")
    print(f"Integrated Loudness: {report.integrated_loudness:.2f} LUFS")
    print(f"True Peak:           {report.true_peak:.2f} dBFS")
    print(f"Sample Peak:         {report.sample_peak:.2f} dBFS")
    print(f"Spectral Tilt:       {report.spectral_tilt:.4f}")
    print(f"Phase Correlation:   {report.phase_correlation:.2f}")
    print("-----------------\n")
    print("Done.")

if __name__ == "__main__":
    main()
