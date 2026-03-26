from pathlib import Path
from rfdetr import RFDETRNano
import cv2

from .base import BaseDetector
from pipeline_context import Detection, FrameContext


class RFDETRDetector(BaseDetector):
    def __init__(self, model_path, confidence_threshold=0.5):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold

        if not self.model_path.exists():
            raise FileNotFoundError(f"RF-DETR checkpoint not found: {self.model_path}")

        self.model = RFDETRNano(pretrain_weights=str(self.model_path))
        self.model.optimize_for_inference()

    def detect(self, ctx: FrameContext) -> FrameContext:
        rgb_frame = cv2.cvtColor(ctx.frame, cv2.COLOR_BGR2RGB)
        predictions = self.model.predict(rgb_frame, threshold=self.confidence_threshold)
        
        detections = []
        for bbox, confidence, class_id in zip(
            predictions.xyxy,
            predictions.confidence,
            predictions.class_id,
        ):
            class_id = int(class_id)
            
            label = self.model.class_names.get(class_id, "drone")
            x1, y1, x2, y2 = (int(value) for value in bbox)
            detections.append(
                Detection(
                    label=label,
                    confidence=float(confidence),
                    bbox=(x1, y1, x2, y2),
                    class_id=class_id,
                )
            )

        ctx.detections = detections
        return ctx