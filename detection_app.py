import cv2
import logging
from typing import List, Any
from inference.core.interfaces.camera.entities import VideoFrame
from inference import InferencePipeline
from rfdetr import RFDETRNano
import supervision as sv
from utils import get_device
import argparse

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

VIDEO_PATH = "example_media/drones/vid_drone2.mp4"
MODEL_PATH = "models/checkpoint_best_ema.pth"
OUTPUT_VIDEO_PATH = "output/saved_video.mp4"

class DetectionApp:
    def __init__(self, weights_path: str, video_path: str, show: bool, save: bool, output_path: str):
        self.show = show
        self.save = save
        self.output_path = output_path

        # Model
        self.model = RFDETRNano(pretrain_weights=weights_path, device=get_device())
        self.model.optimize_for_inference()

        # Supervision tools
        self.tracker = sv.ByteTrack()
        if self.show:
            self.label_annotator = sv.LabelAnnotator()
            self.box_annotator = sv.BoxAnnotator()

        # Video info - used for saving
        if self.save:
            self.video_info = sv.VideoInfo.from_video_path(video_path)
        
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
        self.process_predicted_frame(prediction=prediction, video_frame=video_frame)

        if self.show:
            self.visualization()
        if self.save:
            self.save_video()

    def process_predicted_frame(self, prediction, video_frame):
        tracked_detections = self.tracker.update_with_detections(prediction)

        labels = [
            f"Drone {conf:.2f} ID:{int(track_id)}"
            for track_id, conf in zip(
                tracked_detections.tracker_id,
                tracked_detections.confidence
            )
        ]

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

    def save_video(self):
        if self.last_frame is not None:
            self.sink.write_frame(self.last_frame)

    def visualization(self):
        cv2.imshow("Predictions", self.last_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            self.pipeline.terminate()
            exit()

        elif key == ord(" "):
            self.paused = not self.paused
            print("Paused" if self.paused else "Resumed")

    def run(self):
        if self.save:
            with sv.VideoSink(self.output_path, self.video_info) as sink:
                self.sink = sink
                self.pipeline.start()
                self.pipeline.join()
        else:
            self.pipeline.start()
            self.pipeline.join()

        cv2.destroyAllWindows()



def parse_args():
    parser = argparse.ArgumentParser(description="Detection App")

    parser.add_argument("--video", type=str, default=VIDEO_PATH)
    parser.add_argument("--weights", type=str, default=MODEL_PATH)
    parser.add_argument("--output", type=str, default=OUTPUT_VIDEO_PATH)

    parser.add_argument("--show", action="store_true", default=True)
    parser.add_argument("--save", action="store_true", default=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    app = DetectionApp(
        weights_path=args.weights,
        video_path=args.video,
        show=args.show,
        save=args.save,
        output_path=args.output
    )
    app.run()