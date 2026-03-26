from abc import ABC, abstractmethod

from pipeline_context import FrameContext


class BaseDetector(ABC):
    @abstractmethod
    def detect(self, ctx: FrameContext) -> FrameContext:
        pass