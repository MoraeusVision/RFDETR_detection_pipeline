import pytest
from unittest.mock import patch, MagicMock, mock_open
import tempfile
import os
from pathlib import Path
import numpy as np
import cv2
from source_factory import (
    ImageSource, VideoSource, StreamSource, USBCameraSource, 
    SourceFactory, BaseSource, MAX_SIZE_MB
)


class TestImageSource:
    def test_is_static(self):
        """Test image sources are marked as static."""
        source = ImageSource("/path/to/image.jpg")

        assert source.is_static is True

    @patch('cv2.imread')
    def test_get_frame_success(self, mock_imread):
        """Test get_frame returns image successfully."""
        mock_frame = np.ones((100, 100, 3), dtype=np.uint8)
        mock_imread.return_value = mock_frame
        
        source = ImageSource("/path/to/image.jpg")
        frame = source.get_frame()
        
        assert np.array_equal(frame, mock_frame)
        mock_imread.assert_called_once_with("/path/to/image.jpg")

    @patch('cv2.imread')
    def test_get_frame_file_not_found(self, mock_imread):
        """Test get_frame raises error when file doesn't exist."""
        mock_imread.return_value = None
        
        source = ImageSource("/path/to/nonexistent.jpg")
        
        with pytest.raises(FileNotFoundError, match="Could not read image"):
            source.get_frame()

    def test_cleanup(self):
        """Test cleanup does nothing."""
        source = ImageSource("/path/to/image.jpg")
        # Should not raise error
        source.cleanup()


class TestVideoSource:
    @patch('cv2.VideoCapture')
    def test_is_static(self, mock_capture_class):
        """Test video sources are not static."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_capture_class.return_value = mock_cap

        source = VideoSource("/path/to/video.mp4")

        assert source.is_static is False

    @patch('cv2.VideoCapture')
    def test_init_success(self, mock_capture_class):
        """Test VideoSource initialization with valid video."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_capture_class.return_value = mock_cap
        
        source = VideoSource("/path/to/video.mp4")
        
        assert source.path == "/path/to/video.mp4"
        assert source.cap == mock_cap
        mock_capture_class.assert_called_once_with("/path/to/video.mp4")

    @patch('cv2.VideoCapture')
    def test_init_failure(self, mock_capture_class):
        """Test VideoSource initialization fails with invalid video."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_capture_class.return_value = mock_cap
        
        with pytest.raises(ValueError, match="Could not open video"):
            VideoSource("/path/to/invalid.mp4")

    @patch('cv2.VideoCapture')
    def test_get_frame_success(self, mock_capture_class):
        """Test get_frame returns frame."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.ones((100, 100, 3), dtype=np.uint8))
        mock_capture_class.return_value = mock_cap
        
        source = VideoSource("/path/to/video.mp4")
        frame = source.get_frame()
        
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        mock_cap.read.assert_called_once()

    @patch('cv2.VideoCapture')
    def test_get_frame_end_of_video(self, mock_capture_class):
        """Test get_frame returns None at end of video."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        mock_capture_class.return_value = mock_cap
        
        source = VideoSource("/path/to/video.mp4")
        frame = source.get_frame()
        
        assert frame is None
        mock_cap.release.assert_called_once()

    @patch('cv2.VideoCapture')
    @patch('logging.info')
    def test_cleanup(self, mock_logging, mock_capture_class):
        """Test cleanup releases video."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_capture_class.return_value = mock_cap
        
        source = VideoSource("/path/to/video.mp4")
        source.cleanup()
        
        mock_cap.release.assert_called_once()
        mock_logging.assert_called_once_with("Closing video..")


class TestStreamSource:
    @patch('cv2.VideoCapture')
    def test_is_static(self, mock_capture_class):
        """Test stream sources are not static."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_capture_class.return_value = mock_cap

        source = StreamSource("rtsp://example.com/stream")

        assert source.is_static is False

    @patch('cv2.VideoCapture')
    def test_init_rtsp_stream(self, mock_capture_class):
        """Test StreamSource with RTSP URL."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_capture_class.return_value = mock_cap
        
        source = StreamSource("rtsp://example.com/stream")
        
        assert source.url == "rtsp://example.com/stream"
        assert source.is_youtube == False

    @patch('cv2.VideoCapture')
    def test_init_http_stream(self, mock_capture_class):
        """Test StreamSource with HTTP URL."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_capture_class.return_value = mock_cap
        
        source = StreamSource("http://example.com/stream.m3u8")
        
        assert source.is_youtube == False
        mock_capture_class.assert_called_once_with("http://example.com/stream.m3u8")

    @patch('cv2.VideoCapture')
    def test_init_invalid_stream(self, mock_capture_class):
        """Test StreamSource fails with invalid URL."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_capture_class.return_value = mock_cap
        
        with pytest.raises(ValueError, match="Could not open stream"):
            StreamSource("rtsp://invalid.com/stream")

    @patch('cv2.VideoCapture')
    def test_cleanup(self, mock_capture_class):
        """Test cleanup releases stream."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_capture_class.return_value = mock_cap
        
        source = StreamSource("rtsp://example.com/stream")
        source.cleanup()
        
        mock_cap.release.assert_called_once()

    def test_cleanup_removes_downloaded_youtube_artifacts(self, tmp_path):
        """Test cleanup removes downloaded YouTube files and temp directory."""
        download_dir = tmp_path / "youtube-download"
        download_dir.mkdir()
        downloaded_file = download_dir / "video.mp4"
        downloaded_file.write_bytes(b"video")

        source = object.__new__(StreamSource)
        source.cap = MagicMock()
        source.cap.isOpened.return_value = True
        source.downloaded_path = str(downloaded_file)
        source._temp_dir = download_dir

        source.cleanup()

        source.cap.release.assert_called_once()
        assert not downloaded_file.exists()
        assert not download_dir.exists()


class TestUSBCameraSource:
    @patch('cv2.VideoCapture')
    def test_is_static(self, mock_capture_class):
        """Test USB camera sources are not static."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_capture_class.return_value = mock_cap

        source = USBCameraSource(device_index=0)

        assert source.is_static is False

    @patch('cv2.VideoCapture')
    def test_init_success(self, mock_capture_class):
        """Test USBCameraSource initialization with valid device."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_capture_class.return_value = mock_cap
        
        source = USBCameraSource(device_index=0, width=640, height=480)
        
        mock_capture_class.assert_called_once_with(0)
        mock_cap.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH, 640)
        mock_cap.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    @patch('cv2.VideoCapture')
    def test_init_failure(self, mock_capture_class):
        """Test USBCameraSource fails with invalid device."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_capture_class.return_value = mock_cap
        
        with pytest.raises(ValueError, match="Could not open USB camera"):
            USBCameraSource(device_index=99)

    @patch('cv2.VideoCapture')
    def test_get_frame(self, mock_capture_class):
        """Test get_frame from USB camera."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_frame = np.ones((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, mock_frame)
        mock_capture_class.return_value = mock_cap
        
        source = USBCameraSource(device_index=0)
        frame = source.get_frame()
        
        assert np.array_equal(frame, mock_frame)

    @patch('cv2.VideoCapture')
    @patch('logging.info')
    def test_cleanup(self, mock_logging, mock_capture_class):
        """Test cleanup releases camera."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_capture_class.return_value = mock_cap
        
        source = USBCameraSource(device_index=0)
        source.cleanup()
        
        mock_cap.release.assert_called_once()
        assert mock_logging.call_count == 2
        mock_logging.assert_any_call("Starting up camera..")
        mock_logging.assert_any_call("Closing USB camera..")


class TestSourceFactory:
    @patch('cv2.VideoCapture')
    def test_create_rtsp_stream(self, mock_capture_class):
        """Test create returns StreamSource for RTSP URL."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_capture_class.return_value = mock_cap
        
        source = SourceFactory.create("rtsp://example.com/stream")
        
        assert isinstance(source, StreamSource)

    @patch('cv2.VideoCapture')
    def test_create_http_stream(self, mock_capture_class):
        """Test create returns StreamSource for HTTP URL."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_capture_class.return_value = mock_cap
        
        source = SourceFactory.create("http://example.com/stream.m3u8")
        
        assert isinstance(source, StreamSource)

    @patch('cv2.VideoCapture')
    def test_create_https_stream(self, mock_capture_class):
        """Test create returns StreamSource for HTTPS URL."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_capture_class.return_value = mock_cap
        
        source = SourceFactory.create("https://example.com/stream")
        
        assert isinstance(source, StreamSource)

    @patch('os.path.isfile')
    @patch('cv2.imread')
    def test_create_image_jpg(self, mock_imread, mock_isfile):
        """Test create returns ImageSource for JPG file."""
        mock_isfile.return_value = True
        mock_imread.return_value = np.ones((100, 100, 3), dtype=np.uint8)
        
        source = SourceFactory.create("/path/to/image.jpg")
        
        assert isinstance(source, ImageSource)

    @patch('os.path.isfile')
    @patch('cv2.imread')
    def test_create_image_png(self, mock_imread, mock_isfile):
        """Test create returns ImageSource for PNG file."""
        mock_isfile.return_value = True
        mock_imread.return_value = np.ones((100, 100, 3), dtype=np.uint8)
        
        source = SourceFactory.create("/path/to/image.png")
        
        assert isinstance(source, ImageSource)

    @patch('os.path.isfile')
    @patch('cv2.imread')
    def test_create_image_bmp(self, mock_imread, mock_isfile):
        """Test create returns ImageSource for BMP file."""
        mock_isfile.return_value = True
        mock_imread.return_value = np.ones((100, 100, 3), dtype=np.uint8)
        
        source = SourceFactory.create("/path/to/image.bmp")
        
        assert isinstance(source, ImageSource)

    @patch('os.path.isfile')
    def test_create_image_uppercase_extension(self, mock_isfile):
        """Test create handles uppercase image extensions."""
        mock_isfile.return_value = True

        source = SourceFactory.create("/path/to/image.JPEG")

        assert isinstance(source, ImageSource)

    @patch('os.path.isfile')
    @patch('cv2.VideoCapture')
    def test_create_video_mp4(self, mock_capture_class, mock_isfile):
        """Test create returns VideoSource for MP4 file."""
        mock_isfile.return_value = True
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_capture_class.return_value = mock_cap
        
        source = SourceFactory.create("/path/to/video.mp4")
        
        assert isinstance(source, VideoSource)

    @patch('os.path.isfile')
    @patch('cv2.VideoCapture')
    def test_create_video_avi(self, mock_capture_class, mock_isfile):
        """Test create returns VideoSource for AVI file."""
        mock_isfile.return_value = True
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_capture_class.return_value = mock_cap
        
        source = SourceFactory.create("/path/to/video.avi")
        
        assert isinstance(source, VideoSource)

    @patch('os.path.isfile')
    @patch('cv2.VideoCapture')
    def test_create_video_mov(self, mock_capture_class, mock_isfile):
        """Test create returns VideoSource for MOV file."""
        mock_isfile.return_value = True
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_capture_class.return_value = mock_cap
        
        source = SourceFactory.create("/path/to/video.mov")
        
        assert isinstance(source, VideoSource)

    @patch('os.path.isfile')
    @patch('cv2.VideoCapture')
    def test_create_video_mkv(self, mock_capture_class, mock_isfile):
        """Test create returns VideoSource for MKV file."""
        mock_isfile.return_value = True
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_capture_class.return_value = mock_cap
        
        source = SourceFactory.create("/path/to/video.mkv")
        
        assert isinstance(source, VideoSource)

    @patch('os.path.isfile')
    @patch('cv2.VideoCapture')
    def test_create_video_uppercase_extension(self, mock_capture_class, mock_isfile):
        """Test create handles uppercase video extensions."""
        mock_isfile.return_value = True
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_capture_class.return_value = mock_cap

        source = SourceFactory.create("/path/to/video.MP4")

        assert isinstance(source, VideoSource)

    @patch('os.path.isfile')
    def test_create_unsupported_file_type(self, mock_isfile):
        """Test create raises error for unsupported file type."""
        mock_isfile.return_value = True
        
        with pytest.raises(ValueError, match="Unsupported file type"):
            SourceFactory.create("/path/to/file.txt")

    @patch('os.path.isfile')
    @patch('cv2.VideoCapture')
    def test_create_usb_camera(self, mock_capture_class, mock_isfile):
        """Test create returns USBCameraSource for digit string."""
        mock_isfile.return_value = False
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_capture_class.return_value = mock_cap
        
        source = SourceFactory.create("0")
        
        assert isinstance(source, USBCameraSource)

    @patch('os.path.isfile')
    @patch('cv2.VideoCapture')
    def test_create_usb_camera_device_1(self, mock_capture_class, mock_isfile):
        """Test create with device index 1."""
        mock_isfile.return_value = False
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_capture_class.return_value = mock_cap
        
        source = SourceFactory.create("1")
        
        assert isinstance(source, USBCameraSource)
        mock_capture_class.assert_called_once_with(1)

    @patch('os.path.isfile')
    def test_create_invalid_source(self, mock_isfile):
        """Test create raises error for invalid source."""
        mock_isfile.return_value = False
        
        with pytest.raises(ValueError, match="Invalid source"):
            SourceFactory.create("invalid_source")

    def test_base_source_is_abstract(self):
        """Test BaseSource cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseSource()
