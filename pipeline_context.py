
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

# Anpassad för att matcha Supervisions Detection-API
@dataclass
class Detection:
    xyxy: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int | None = None
    tracker_id: int | None = None
    label: str | None = None

@dataclass
class FrameContext:
    frame: Any
    timestamp: float
    detections: list[Detection] = field(default_factory=list)

@dataclass
class PipelineContext:
    is_static: bool
    should_continue: bool
    frame_context: FrameContext