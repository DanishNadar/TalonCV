"""Export the configured local YOLO face checkpoint for browser-local ONNX inference."""

import argparse
from pathlib import Path


project_root = Path(__file__).resolve().parents[1]


def parse_options():
    parser = argparse.ArgumentParser(description="Export TalonCV's local YOLOv11 face checkpoint to web/public/models.")
    parser.add_argument("--checkpoint", type=Path, default=project_root / "models" / "yolo11n-face" / "model.pt")
    parser.add_argument("--output", type=Path, default=project_root / "web" / "public" / "models" / "yolo11n-face.onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def main():
    args = parse_options()
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else project_root / args.checkpoint
    output = args.output if args.output.is_absolute() else project_root / args.output
    if not checkpoint.is_file():
        raise SystemExit(f"Missing local YOLO face checkpoint: {checkpoint}. Run scripts/setupLocalModels.ps1 first.")
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("Install requirements.txt to export the ONNX face model.") from error
    model = YOLO(str(checkpoint))
    exported = Path(model.export(format="onnx", imgsz=args.imgsz, simplify=True, dynamic=False, opset=17))
    output.parent.mkdir(parents=True, exist_ok=True)
    exported.replace(output)
    print(f"Exported browser-local face model: {output} ({output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
