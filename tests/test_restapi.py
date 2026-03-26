from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from fastapi import FastAPI

import restapi
from pipeline_context import Detection, FrameContext


def make_test_image_bytes() -> bytes:
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok is True
    return encoded.tobytes()


class DummyRequest:
    def __init__(self, content_type: str, body: bytes, detector: MagicMock):
        self.headers = {"content-type": content_type}
        self._body = body
        self.app = SimpleNamespace(state=SimpleNamespace(detector=detector))

    async def body(self):
        return self._body


class TestRestApi:
    @pytest.mark.anyio
    async def test_lifespan_loads_detector_from_config(self):
        app = FastAPI()
        detector = MagicMock()

        with patch("restapi.read_from_config") as mock_read_from_config, patch(
            "restapi.DetectorFactory.create"
        ) as mock_create:
            mock_read_from_config.return_value = {
                "detector": "rfdetr",
                "model_path": "model.pth",
                "conf": 0.7,
            }
            mock_create.return_value = detector

            async with restapi.lifespan(app):
                assert app.state.detector is detector

            mock_read_from_config.assert_called_once_with(restapi.CONFIG_PATH)
            mock_create.assert_called_once_with(
                detector_name="rfdetr",
                model_path="model.pth",
                confidence_threshold=0.7,
            )

    def test_root_returns_hello_world(self):
        assert restapi.root() == {"message": "Hello World"}

    @pytest.mark.anyio
    async def test_upload_image_rejects_non_image_content_type(self):
        request = DummyRequest(
            content_type="application/octet-stream",
            body=b"not-an-image",
            detector=MagicMock(),
        )

        with pytest.raises(restapi.HTTPException, match="Request body must be an image") as exc_info:
            await restapi.upload_image(request)

        assert exc_info.value.status_code == 400

    @pytest.mark.anyio
    async def test_upload_image_rejects_empty_body(self):
        request = DummyRequest(
            content_type="image/jpeg",
            body=b"",
            detector=MagicMock(),
        )

        with pytest.raises(restapi.HTTPException, match="Image body is empty") as exc_info:
            await restapi.upload_image(request)

        assert exc_info.value.status_code == 400

    @patch("restapi.cv2.imdecode")
    @pytest.mark.anyio
    async def test_upload_image_rejects_undecodable_image(self, mock_imdecode):
        mock_imdecode.return_value = None
        request = DummyRequest(
            content_type="image/jpeg",
            body=b"broken-image-bytes",
            detector=MagicMock(),
        )

        with pytest.raises(restapi.HTTPException, match="Could not decode image") as exc_info:
            await restapi.upload_image(request)

        assert exc_info.value.status_code == 400

    @pytest.mark.anyio
    async def test_upload_image_returns_detections(self):
        detector = MagicMock()
        frame_context = FrameContext(frame=np.zeros((8, 8, 3), dtype=np.uint8), timestamp=0)
        frame_context.detections = [
            Detection(
                label="drone",
                confidence=0.91,
                bbox=(1, 2, 3, 4),
                class_id=7,
            )
        ]
        detector.detect.return_value = frame_context
        request = DummyRequest(
            content_type="image/jpeg",
            body=make_test_image_bytes(),
            detector=detector,
        )

        response = await restapi.upload_image(request)

        assert response == {
            "detections": [
                {
                    "label": "drone",
                    "confidence": 0.91,
                    "bbox": [1, 2, 3, 4],
                }
            ]
        }
        detector.detect.assert_called_once()

    @patch("restapi.uvicorn.run")
    def test_main_runs_uvicorn_with_restapi_app(self, mock_run):
        restapi.main()

        mock_run.assert_called_once_with(
            "restapi:app",
            host="127.0.0.1",
            port=8000,
            reload=True,
        )