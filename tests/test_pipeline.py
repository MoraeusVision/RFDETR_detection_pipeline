from unittest.mock import MagicMock

from pipeline import DetectionPipeline
from pipeline_context import Detection, FrameContext, PipelineContext


class TestDetectionPipeline:
    def test_process_frame_runs_tracker_for_non_static_sources(self):
        source = MagicMock()
        model = MagicMock()
        tracker = MagicMock()

        source_detections = [
            Detection(label="drone", confidence=0.9, bbox=(1, 2, 3, 4), class_id=1)
        ]
        detected_context = FrameContext(frame="frame", timestamp=1.0)
        detected_context.detections = source_detections
        tracked_detections = [
            Detection(
                label="drone",
                confidence=0.9,
                bbox=(1, 2, 3, 4),
                class_id=1,
                track_id=5,
            )
        ]
        model.detect.return_value = detected_context
        tracker.update.return_value = tracked_detections

        pipeline = DetectionPipeline(source=source, model=model, tracker=tracker)
        ctx = PipelineContext(
            is_static=False,
            should_continue=True,
            frame_context=FrameContext(frame="frame", timestamp=1.0),
        )

        result = pipeline.process_frame(ctx)

        tracker.update.assert_called_once_with(source_detections, detected_context.frame)
        assert result.frame_context.detections == tracked_detections

    def test_process_frame_skips_tracker_for_static_sources(self):
        source = MagicMock()
        model = MagicMock()
        tracker = MagicMock()

        detected_context = FrameContext(frame="frame", timestamp=1.0)
        detected_context.detections = [
            Detection(label="drone", confidence=0.9, bbox=(1, 2, 3, 4), class_id=1)
        ]
        model.detect.return_value = detected_context

        pipeline = DetectionPipeline(source=source, model=model, tracker=tracker)
        ctx = PipelineContext(
            is_static=True,
            should_continue=True,
            frame_context=FrameContext(frame="frame", timestamp=1.0),
        )

        result = pipeline.process_frame(ctx)

        tracker.update.assert_not_called()
        assert result.frame_context.detections == detected_context.detections