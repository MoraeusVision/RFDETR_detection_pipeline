import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import cv2
from visualization import Visualizer


class TestVisualizer:
    @patch('cv2.namedWindow')
    @patch('cv2.resizeWindow')
    def test_init(self, mock_resize, mock_named):
        """Test Visualizer initialization."""
        visualizer = Visualizer(window_name="Test", width=640, height=480)
        
        assert visualizer.window_name == "Test"
        assert visualizer.width == 640
        assert visualizer.height == 480
        assert visualizer.paused == False
        assert visualizer.last_frame is None
        
        mock_named.assert_called_once_with("Test", cv2.WINDOW_NORMAL)
        mock_resize.assert_called_once_with("Test", 640, 480)

    @patch('cv2.namedWindow')
    @patch('cv2.resizeWindow')
    def test_init_defaults(self, mock_resize, mock_named):
        """Test Visualizer with default parameters."""
        visualizer = Visualizer()
        
        assert visualizer.window_name == "Detection"
        assert visualizer.width == 800
        assert visualizer.height == 600

    @patch('cv2.namedWindow')
    @patch('cv2.resizeWindow')
    @patch('visualization.Visualizer.show')
    def test_handle_event_on_inference_result(self, mock_show, mock_resize, mock_named):
        """Test handle_event calls show for on_inference_result event."""
        visualizer = Visualizer()
        mock_data = MagicMock()
        mock_data.frame_context.frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_data.frame_context.detections = [
            MagicMock(bbox=(1, 2, 3, 4), label="drone", confidence=0.95, track_id=7),
            MagicMock(bbox=(5, 6, 7, 8), label="bird", confidence=0.72, track_id=None),
        ]
        mock_data.is_static = False
        mock_show.return_value = True
        
        visualizer.handle_event("on_inference_result", mock_data)
        
        mock_show.assert_called_once_with(
            frame=mock_data.frame_context.frame,
            boxes=[(1, 2, 3, 4), (5, 6, 7, 8)],
            labels=["#7 drone 0.95", "bird 0.72"],
            is_static=False,
        )
        assert mock_data.should_continue is True

    @patch('cv2.namedWindow')
    @patch('cv2.resizeWindow')
    def test_format_detection_label_includes_track_id(self, mock_resize, mock_named):
        visualizer = Visualizer()

        detection = MagicMock(label="drone", confidence=0.95, track_id=42)

        assert visualizer._format_detection_label(detection) == "#42 drone 0.95"

    @patch('cv2.namedWindow')
    @patch('cv2.resizeWindow')
    def test_handle_event_other_event(self, mock_resize, mock_named):
        """Test handle_event does nothing for other events."""
        visualizer = Visualizer()
        
        # Should not raise error
        visualizer.handle_event("other_event", None)

    @patch('cv2.namedWindow')
    @patch('cv2.resizeWindow')
    @patch('cv2.imshow')
    @patch('cv2.waitKey')
    @patch('logging.info')
    def test_show_normal_frame(self, mock_logging, mock_waitkey, mock_imshow, mock_resize, mock_named):
        """Test show with normal frame (not paused, not image)."""
        visualizer = Visualizer()
        frame = np.ones((100, 100, 3), dtype=np.uint8)
        
        mock_waitkey.return_value = 0  # No key pressed
        
        result = visualizer.show(frame)
        
        assert result == True
        assert np.array_equal(visualizer.last_frame, frame)
        mock_imshow.assert_called_once()
        mock_waitkey.assert_called_once_with(1)
        mock_logging.assert_not_called()

    @patch('cv2.namedWindow')
    @patch('cv2.resizeWindow')
    @patch('cv2.imshow')
    @patch('cv2.waitKey')
    def test_show_with_boxes_does_not_mutate_input_frame(self, mock_waitkey, mock_imshow, mock_resize, mock_named):
        """Test show draws on a copy instead of mutating the input frame."""
        visualizer = Visualizer()
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        original = frame.copy()

        mock_waitkey.return_value = 0

        visualizer.show(frame, boxes=[[1, 1, 5, 5]])

        assert np.array_equal(frame, original)
        mock_imshow.assert_called_once()

    @patch('cv2.namedWindow')
    @patch('cv2.resizeWindow')
    @patch('cv2.imshow')
    @patch('cv2.waitKey')
    @patch('logging.info')
    def test_show_paused_frame(self, mock_logging, mock_waitkey, mock_imshow, mock_resize, mock_named):
        """Test show when paused, reuses last_frame."""
        visualizer = Visualizer()
        visualizer.paused = True
        visualizer.last_frame = np.ones((100, 100, 3), dtype=np.uint8)
        new_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        mock_waitkey.return_value = 0
        
        result = visualizer.show(new_frame)
        
        assert result == True
        # Should show last_frame, not new_frame
        displayed_frame = mock_imshow.call_args[0][1]
        assert np.array_equal(displayed_frame, visualizer.last_frame)
        mock_waitkey.assert_called_once_with(0)  # Wait indefinitely when paused

    @patch('cv2.namedWindow')
    @patch('cv2.resizeWindow')
    @patch('cv2.imshow')
    @patch('cv2.waitKey')
    @patch('logging.info')
    def test_show_static_mode_exits_on_q(self, mock_logging, mock_waitkey, mock_imshow, mock_resize, mock_named):
        """Test show in static mode blocks until q is pressed."""
        visualizer = Visualizer()
        frame = np.ones((100, 100, 3), dtype=np.uint8)
        
        mock_waitkey.return_value = ord('q')
        
        result = visualizer.show(frame, is_static=True)
        
        assert result == False
        mock_waitkey.assert_called_once_with(0)  # Wait indefinitely for images

    @patch('cv2.namedWindow')
    @patch('cv2.resizeWindow')
    @patch('cv2.imshow')
    @patch('cv2.waitKey')
    @patch('cv2.putText')
    @patch('cv2.rectangle')
    def test_show_with_boxes(self, mock_rectangle, mock_puttext, mock_waitkey, mock_imshow, mock_resize, mock_named):
        """Test show draws bounding boxes and labels."""
        visualizer = Visualizer()
        frame = np.ones((100, 100, 3), dtype=np.uint8)
        boxes = [[10, 10, 50, 50], [20, 20, 60, 60]]
        labels = ["drone 0.95", "bird 0.72"]
        
        mock_waitkey.return_value = 0
        
        visualizer.show(frame, boxes=boxes, labels=labels)
        
        calls = mock_rectangle.call_args_list
        assert len(calls) == 2
        
        # Check first box
        assert calls[0][0][1] == (10, 10)
        assert calls[0][0][2] == (50, 50)
        assert calls[0][0][3] == (0, 255, 0)
        assert calls[0][0][4] == 2
        
        # Check second box
        assert calls[1][0][1] == (20, 20)
        assert calls[1][0][2] == (60, 60)
        assert calls[1][0][3] == (0, 255, 0)
        assert calls[1][0][4] == 2

        text_calls = mock_puttext.call_args_list
        assert len(text_calls) == 2
        assert text_calls[0][0][1] == "drone 0.95"
        assert text_calls[1][0][1] == "bird 0.72"

    @patch('cv2.namedWindow')
    @patch('cv2.resizeWindow')
    @patch('cv2.imshow')
    @patch('cv2.waitKey')
    @patch('logging.info')
    def test_show_toggle_pause(self, mock_logging, mock_waitkey, mock_imshow, mock_resize, mock_named):
        """Test spacebar toggles pause."""
        visualizer = Visualizer()
        frame = np.ones((100, 100, 3), dtype=np.uint8)
        
        mock_waitkey.return_value = ord(' ')  # Space pressed
        
        result = visualizer.show(frame)
        
        assert result == True  # Continue showing
        assert visualizer.paused == True
        mock_logging.assert_called_once_with("Playback %s", "paused")

    @patch('cv2.namedWindow')
    @patch('cv2.resizeWindow')
    @patch('cv2.imshow')
    @patch('cv2.waitKey')
    @patch('logging.info')
    def test_show_space_resumes_when_already_paused(self, mock_logging, mock_waitkey, mock_imshow, mock_resize, mock_named):
        """Test spacebar resumes playback from paused state."""
        visualizer = Visualizer()
        visualizer.paused = True
        visualizer.last_frame = np.ones((100, 100, 3), dtype=np.uint8)

        mock_waitkey.return_value = ord(' ')

        result = visualizer.show(np.zeros((100, 100, 3), dtype=np.uint8))

        assert result == True
        assert visualizer.paused == False
        mock_imshow.assert_called_once()
        mock_waitkey.assert_called_once_with(0)
        mock_logging.assert_called_once_with("Playback %s", "resumed")

    @patch('cv2.namedWindow')
    @patch('cv2.resizeWindow')
    @patch('cv2.imshow')
    @patch('cv2.waitKey')
    @patch('logging.info')
    def test_show_exit_on_q(self, mock_logging, mock_waitkey, mock_imshow, mock_resize, mock_named):
        """Test q key exits."""
        visualizer = Visualizer()
        frame = np.ones((100, 100, 3), dtype=np.uint8)
        
        mock_waitkey.return_value = ord('q')
        
        result = visualizer.show(frame)
        
        assert result == False  # Exit

    @patch('cv2.namedWindow')
    @patch('cv2.resizeWindow')
    @patch('cv2.destroyAllWindows')
    @patch('logging.info')
    def test_cleanup(self, mock_logging, mock_destroy, mock_resize, mock_named):
        """Test cleanup destroys windows."""
        visualizer = Visualizer()
        
        visualizer.cleanup()
        
        mock_destroy.assert_called_once()
        mock_logging.assert_called_once_with("Closing windows..")