from pipeline_context import Detection, FrameContext


class TestDetection:
    def test_detection_stores_expected_fields(self):
        detection = Detection(
            label="drone",
            confidence=0.93,
            bbox=(10, 20, 30, 40),
            class_id=1,
        )

        assert detection.label == "drone"
        assert detection.confidence == 0.93
        assert detection.bbox == (10, 20, 30, 40)
        assert detection.class_id == 1


class TestFrameContext:
    def test_frame_context_uses_independent_default_containers(self):
        first = FrameContext(frame="frame-1", timestamp=1.0)
        second = FrameContext(frame="frame-2", timestamp=2.0)

        first.detections.append(
            Detection(label="drone", confidence=0.9, bbox=(1, 2, 3, 4))
        )

        assert second.detections == []
