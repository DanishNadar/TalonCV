from pathlib import Path

from src.localModels.config import getModelSection, modelSetupCommands, validateModelFiles

class YoloFaceDetector:
    def __init__(self, model, model_source, image_size=640, confidence_floor=0.05, device="cpu"):
        self.model = model
        self.model_source = model_source
        self.image_size = image_size
        self.confidence_floor = confidence_floor
        self.device = device

    def detect(self, frame):
        height, width = frame.shape[:2]
        results = self.model.predict(
            frame,
            imgsz=self.image_size,
            conf=self.confidence_floor,
            device=self.device,
            verbose=False,
        )

        if not results:
            return {}

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return {}

        names = getattr(result, "names", {}) or getattr(self.model, "names", {}) or {}
        candidates = []

        for box in boxes:
            class_index = int(box.cls[0]) if getattr(box, "cls", None) is not None else None
            class_name = str(names.get(class_index, "")).lower() if class_index is not None else ""

            if class_name and "face" not in class_name:
                continue

            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            box_width = max(x2 - x1, 0.0)
            box_height = max(y2 - y1, 0.0)
            area = box_width * box_height
            confidence = float(box.conf[0]) if getattr(box, "conf", None) is not None else 0.0
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            center_distance = abs(center_x - width / 2) / max(width, 1) + abs(center_y - height / 2) / max(height, 1)
            candidates.append((confidence, area, -center_distance, x1, y1, x2, y2))

        if not candidates:
            return {}

        primary_confidence, _, _, x1, y1, x2, y2 = max(candidates)
        face_width = max(x2 - x1, 0.0) / max(width, 1)
        face_height = max(y2 - y1, 0.0) / max(height, 1)
        face_center_x = ((x1 + x2) / 2) / max(width, 1)
        face_center_y = ((y1 + y2) / 2) / max(height, 1)
        face_edge_margin = min(
            x1 / max(width, 1),
            y1 / max(height, 1),
            (width - x2) / max(width, 1),
            (height - y2) / max(height, 1),
        )

        return {
            "faceDetected": True,
            "faceCount": len(candidates),
            "faceDetectionSource": "YOLOv11",
            "faceDetectionModel": self.model_source,
            "faceDetectionConfidence": round(primary_confidence, 5),
            "faceCenterX": round(face_center_x, 5),
            "faceCenterY": round(face_center_y, 5),
            "faceWidth": round(face_width, 5),
            "faceHeight": round(face_height, 5),
            "faceAreaProxy": round(face_width * face_height, 5),
            "faceEdgeMarginProxy": round(face_edge_margin, 5),
        }


def resolve_yolo_face_model():
    section = getModelSection("faceDetection")
    modelPath = Path(section["resolvedPath"])
    readiness = validateModelFiles("faceDetection", section)
    if not readiness["requiredFilesPresent"]:
        raise RuntimeError(
            f"The local YOLOv11 face checkpoint is missing at {modelPath}. "
            f"Run: {modelSetupCommands['faceDetection']}. Then update config/models.json if the downloaded .pt filename differs."
        )
    return str(modelPath), str(modelPath)


def loadYoloFaceDetector():
    try:
        from ultralytics import YOLO
    except Exception as error:
        raise RuntimeError(
            "Ultralytics is required for YOLOv11 face detection. Install requirements.txt, "
            "then run analysis again."
        ) from error

    section = getModelSection("faceDetection")
    model_path, model_source = resolve_yolo_face_model()
    model = YOLO(model_path, task="detect")
    return YoloFaceDetector(
        model,
        model_source,
        image_size=max(320, min(int(section.get("imageSize", 640)), 1280)),
        confidence_floor=max(0.01, min(float(section.get("confidenceFloor", 0.05)), 0.95)),
        device=str(section.get("device", "cpu")),
    )


def yoloDiagnostics(loadModel=False):
    section = getModelSection("faceDetection")
    readiness = validateModelFiles("faceDetection", section)
    loaded = False
    loadError = None
    if loadModel and readiness["requiredFilesPresent"]:
        try:
            detector = loadYoloFaceDetector()
            loaded = detector.model is not None
        except Exception as error:
            loadError = str(error)
    return {
        **readiness,
        "loaded": loaded,
        "loadError": loadError,
        "localFilesOnly": True,
        "runtimeDownloadsDisabled": True,
    }
