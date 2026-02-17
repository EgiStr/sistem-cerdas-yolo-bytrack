"""
YOLOv8 Helmet Detection Model Wrapper

Wraps Ultralytics YOLOv8 for helmet/no-helmet/motorbike detection.
Returns supervision.Detections for seamless tracker integration.
"""

import numpy as np
import supervision as sv
from ultralytics import YOLO
from loguru import logger
from config.settings import settings


# Class mapping for the helmet detection model
# Must match model output: {0: 'DRIVER_HELMET', 1: 'DRIVER_NO_HELMET', 2: 'MOTORCYCLE'}
CLASS_NAMES = {
    0: "DRIVER_HELMET",
    1: "DRIVER_NO_HELMET",
    2: "MOTORCYCLE",
}


class HelmetDetector:
    """YOLOv8 inference wrapper for helmet detection."""

    def __init__(
        self,
        model_path: str | None = None,
        confidence: float | None = None,
        iou_threshold: float | None = None,
        device: str | None = None,
        imgsz: int | None = None,
    ):
        self.model_path = model_path or settings.model_path
        self.confidence = confidence or settings.model_confidence
        self.iou_threshold = iou_threshold or settings.model_iou_threshold
        self.device = device or settings.model_device
        self.imgsz = imgsz or settings.model_imgsz

        logger.info(f"Loading YOLOv8 model from: {self.model_path}")
        logger.info(f"Device: {self.device} | Confidence: {self.confidence} | IoU: {self.iou_threshold}")

        self.model = YOLO(self.model_path)
        self.class_names = CLASS_NAMES

        # Warm up the model
        self._warmup()

    def _warmup(self):
        """Run a dummy inference to warm up the model."""
        try:
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            self.model.predict(
                dummy,
                conf=self.confidence,
                iou=self.iou_threshold,
                device=self.device,
                imgsz=self.imgsz,
                verbose=False,
                half=self.device != "cpu",
            )
            logger.info("Model warmup complete")
        except Exception as e:
            logger.warning(f"Warmup failed (will try CPU fallback): {e}")
            self.device = "cpu"
            logger.info("Switched to CPU inference")

    def predict(self, frame: np.ndarray) -> sv.Detections:
        """
        Run inference on a single frame.

        Args:
            frame: BGR image as numpy array (H, W, 3)

        Returns:
            supervision.Detections with class_id, confidence, xyxy, and class names in data
        """
        results = self.model.predict(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False,
            half=self.device != "cpu",
        )

        result = results[0]

        if len(result.boxes) == 0:
            return sv.Detections.empty()

        detections = sv.Detections.from_ultralytics(result)

        # Add class names to metadata
        class_names_arr = np.array([
            self.class_names.get(cid, f"class_{cid}")
            for cid in detections.class_id
        ])
        detections.data["class_name"] = class_names_arr

        return detections

    def get_class_name(self, class_id: int) -> str:
        """Get human-readable class name from class ID."""
        return self.class_names.get(class_id, f"unknown_{class_id}")
