from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import run


class TestRun:
    @patch("run.TrackerFactory.create")
    @patch("run.DetectionPipeline")
    @patch("run.read_from_config")
    @patch("run.CleanupManager")
    @patch("run.EventManager")
    @patch("run.DetectorFactory.create")
    @patch("run.SourceFactory.create")
    @patch("run.parse_arguments")
    def test_main_without_visualizer_does_not_register_none_observer(
        self,
        mock_parse_arguments,
        mock_source_create,
        mock_detector_create,
        mock_event_manager_class,
        mock_cleanup_manager_class,
        mock_read_from_config,
        mock_pipeline_class,
        mock_tracker_create,
    ):
        mock_parse_arguments.return_value = SimpleNamespace(config="project/config.json")
        mock_read_from_config.return_value = {
            "source": "video.mp4",
            "show": False,
            "detector": "rfdetr",
            "model_path": "model.pth",
        }

        mock_source = MagicMock()
        mock_source.is_static = False
        mock_source_create.return_value = mock_source
        mock_model = MagicMock()
        mock_detector_create.return_value = mock_model

        mock_event_manager = MagicMock()
        mock_event_manager_class.return_value = mock_event_manager

        mock_cleanup = MagicMock()
        mock_cleanup_manager_class.return_value = mock_cleanup

        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline

        run.main()

        mock_read_from_config.assert_called_once_with("project/config.json")
        mock_source_create.assert_called_once_with(source_path="video.mp4")
        mock_detector_create.assert_called_once_with(
            detector_name="rfdetr",
            model_path="model.pth",
            confidence_threshold=0.5,
        )
        mock_tracker_create.assert_not_called()
        mock_event_manager.register.assert_not_called()
        mock_pipeline_class.assert_called_once_with(
            source=mock_source,
            model=mock_model,
            tracker=None,
            event_manager=mock_event_manager,
        )
        mock_pipeline.run.assert_called_once_with()
        mock_cleanup.add.assert_called_once_with(mock_source.cleanup)
        mock_cleanup.run.assert_called_once()

    @patch("run.TrackerFactory.create")
    @patch("run.logging.exception")
    @patch("run.DetectionPipeline")
    @patch("run.read_from_config")
    @patch("run.CleanupManager")
    @patch("run.EventManager")
    @patch("run.DetectorFactory.create")
    @patch("run.SourceFactory.create")
    @patch("run.parse_arguments")
    def test_main_runs_cleanup_when_pipeline_raises(
        self,
        mock_parse_arguments,
        mock_source_create,
        mock_detector_create,
        mock_event_manager_class,
        mock_cleanup_manager_class,
        mock_read_from_config,
        mock_pipeline_class,
        mock_logging_exception,
        mock_tracker_create,
    ):
        mock_parse_arguments.return_value = SimpleNamespace(config="project/config.json")
        mock_read_from_config.return_value = {
            "source": "image.jpg",
            "show": False,
            "detector": "rfdetr",
            "model_path": "model.pth",
        }

        mock_source = MagicMock()
        mock_source.is_static = True
        mock_source_create.return_value = mock_source
        mock_detector_create.return_value = MagicMock()

        mock_event_manager = MagicMock()
        mock_event_manager_class.return_value = mock_event_manager

        mock_cleanup = MagicMock()
        mock_cleanup_manager_class.return_value = mock_cleanup

        mock_pipeline = MagicMock()
        mock_pipeline.run.side_effect = RuntimeError("boom")
        mock_pipeline_class.return_value = mock_pipeline

        run.main()

        mock_tracker_create.assert_not_called()
        mock_event_manager.register.assert_not_called()
        mock_pipeline.run.assert_called_once_with()
        mock_logging_exception.assert_called_once_with("Unhandled error in main loop")
        mock_cleanup.run.assert_called_once()

    @patch("run.TrackerFactory.create")
    @patch("run.DetectionPipeline")
    @patch("run.read_from_config")
    @patch("run.CleanupManager")
    @patch("run.EventManager")
    @patch("run.DetectorFactory.create")
    @patch("run.SourceFactory.create")
    @patch("run.parse_arguments")
    def test_main_creates_tracker_for_non_static_sources(
        self,
        mock_parse_arguments,
        mock_source_create,
        mock_detector_create,
        mock_event_manager_class,
        mock_cleanup_manager_class,
        mock_read_from_config,
        mock_pipeline_class,
        mock_tracker_create,
    ):
        mock_parse_arguments.return_value = SimpleNamespace(config="project/config.json")
        mock_read_from_config.return_value = {
            "source": "video.mp4",
            "show": False,
            "detector": "rfdetr",
            "tracker": "bytetrack",
            "model_path": "model.pth",
        }

        mock_source = MagicMock()
        mock_source.is_static = False
        mock_source_create.return_value = mock_source
        mock_model = MagicMock()
        mock_detector_create.return_value = mock_model

        mock_event_manager = MagicMock()
        mock_event_manager_class.return_value = mock_event_manager

        mock_cleanup = MagicMock()
        mock_cleanup_manager_class.return_value = mock_cleanup

        mock_tracker = MagicMock()
        mock_tracker_create.return_value = mock_tracker

        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline

        run.main()

        mock_tracker_create.assert_called_once_with(
            tracker_name="bytetrack",
        )
        mock_pipeline_class.assert_called_once_with(
            source=mock_source,
            model=mock_model,
            tracker=mock_tracker,
            event_manager=mock_event_manager,
        )