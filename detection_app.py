import cv2
import logging
from typing import List, Any
from inference.core.interfaces.camera.entities import VideoFrame
from inference import InferencePipeline
from rfdetr import RFDETRNano
import supervision as sv
from utils import get_device

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

VIDEO_PATH = "example_media/drones/vid_drone2.mp4"
MODEL_PATH = "models/checkpoint_best_ema.pth"

class DroneApp:
    def __init__(self, weights_path: str, video_path: str):
        # Model
        device = get_device()
        self.model = RFDETRNano(pretrain_weights=weights_path, device=device)
        self.model.optimize_for_inference()

        # Supervision tools
        self.tracker = sv.ByteTrack()
        self.label_annotator = sv.LabelAnnotator()
        self.box_annotator = sv.BoxAnnotator()

        # State
        self.paused = False
        self.last_frame = None

        # Pipeline
        self.pipeline = InferencePipeline.init_with_custom_logic(
            video_reference=video_path,
            on_video_frame=self.infer,
            on_prediction=self.on_prediction,
        )

    def infer(self, video_frames: List[VideoFrame]) -> List[Any]:
        predictions = self.model.predict([v.image for v in video_frames])
        return [predictions]

    def on_prediction(self, prediction, video_frame):
        self.visualization(prediction=prediction, video_frame=video_frame)
        self.another_sink()

    def another_sink(self):
        pass

    def visualization(self, prediction, video_frame: VideoFrame):
        if not self.paused:
            # Tracking
            tracked_detections = self.tracker.update_with_detections(prediction)

            # Labels
            labels = [
                f"Drone {conf:.2f} ID:{int(track_id)}"
                for track_id, conf in zip(
                    tracked_detections.tracker_id,
                    tracked_detections.confidence
                )
            ]

            # Annotate
            annotated_image = video_frame.image.copy()
            annotated_image = self.box_annotator.annotate(
                scene=annotated_image,
                detections=tracked_detections
            )

            annotated_image = self.label_annotator.annotate(
                annotated_image,
                detections=tracked_detections,
                labels=labels
            )

            self.last_frame = annotated_image

        # Show
        cv2.imshow("Predictions", self.last_frame)

        # Keyboard control
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            self.pipeline.terminate()
            exit()

        elif key == ord(" "):
            self.paused = not self.paused
            print("Paused" if self.paused else "Resumed")

    def run(self):
        self.pipeline.start()
        self.pipeline.join()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = DroneApp(
        weights_path=MODEL_PATH,
        video_path=VIDEO_PATH
    )
    app.run()