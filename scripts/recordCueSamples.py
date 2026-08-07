import argparse
import sys
import time
from datetime import datetime
from pathlib import Path


projectRoot = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(projectRoot))

from scripts.recordLabelingSession import recordOneTake  # noqa: E402
from src.cvPipeline.cueDataset import (  # noqa: E402
    canonicalCueName,
    cueFilename,
    cueRecordingTarget,
    recordableCueTypes,
)


defaultRecordingDir = projectRoot / "data" / "cueTraining" / "recordings"


def parseOptions():
    parser = argparse.ArgumentParser(
        description="Record multiple cue-named webcam takes for TalonCV classifier training."
    )
    parser.add_argument(
        "--cue",
        action="append",
        help="Cue to record. Repeat this option for different cues; omit it for an interactive prompt.",
    )
    parser.add_argument("--takes-per-cue", type=int, default=3, help="Number of takes to record for each cue.")
    parser.add_argument("--seconds", type=float, default=8.0, help="Length of each take.")
    parser.add_argument("--fps", type=float, default=30.0, help="Output video FPS.")
    parser.add_argument("--countdown", type=int, default=3, help="Countdown before each take.")
    parser.add_argument("--output-dir", type=Path, default=defaultRecordingDir)
    parser.add_argument("--list-cues", action="store_true", help="Print valid cue names and exit.")
    args = parser.parse_args()

    if args.list_cues:
        print("cue\tminimum_independent_takes\ttarget_independent_takes\ttier")
        for cue in recordableCueTypes:
            target = cueRecordingTarget(cue)
            print(f"{cue}\t{target['minimumTakes']}\t{target['targetTakes']}\t{target['tier']}")
        print(
            "\nCollect takes across at least 3 sessions, vary distance/lighting/clothing/background, "
            "and reserve entire sessions for validation/test splits. --takes-per-cue controls one batch, not the final target."
        )
        raise SystemExit(0)
    if args.takes_per_cue <= 0 or args.seconds <= 0 or args.fps <= 0 or args.countdown < 0:
        parser.error("takes, seconds, and fps must be positive; countdown cannot be negative.")
    return args


def resolveCues(values):
    if not values:
        print("Enter comma-separated cue names. Use 'baseline' for a normal no-cue reference take.")
        print("Run with --list-cues to see every supported name.")
        values = [value.strip() for value in input("Cues: ").split(",") if value.strip()]

    cues = []
    for value in values:
        cue = canonicalCueName(value)
        if cue is None:
            raise SystemExit(f"Unknown cue '{value}'. Run with --list-cues to see valid names.")
        if cue not in cues:
            cues.append(cue)
    if not cues:
        raise SystemExit("No cues selected.")
    return cues


def countdown(seconds, cue, takeNumber, takeCount):
    print(f"\nGet ready: {cue} (take {takeNumber}/{takeCount})")
    for remaining in range(seconds, 0, -1):
        print(f"  {remaining}...")
        time.sleep(1)


def recordCueSamples():
    args = parseOptions()
    cues = resolveCues(args.cue)
    outputDir = args.output_dir if args.output_dir.is_absolute() else projectRoot / args.output_dir
    outputDir.mkdir(parents=True, exist_ok=True)
    saved = []

    for cue in cues:
        print(f"\nCue: {cue}. Express this cue throughout each short take when practical.")
        for takeNumber in range(1, args.takes_per_cue + 1):
            countdown(args.countdown, cue, takeNumber, args.takes_per_cue)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            outputPath = outputDir / cueFilename(cue, takeNumber, timestamp)
            saved.append(
                recordOneTake(
                    args.seconds,
                    args.fps,
                    outputPath=outputPath,
                    windowTitle=f"{cue} - take {takeNumber}/{args.takes_per_cue} - q stops early",
                )
            )

    print(f"\nSaved {len(saved)} cue recording(s) to {outputDir}")
    print("Next: python scripts/extractCueTrainingData.py")


if __name__ == "__main__":
    recordCueSamples()
