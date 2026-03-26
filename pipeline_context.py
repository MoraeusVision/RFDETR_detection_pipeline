from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FrameContext:
    frame: Any
    timestamp: float

@dataclass
class PipelineContext:
    is_static: bool
    should_continue: bool
    frame_context: FrameContext