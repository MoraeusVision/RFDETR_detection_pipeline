import supervision as sv
import cv2


class Visualizer():
    def __init__(self):
        self.box_annotator = sv.BoxAnnotator()

    def handle_event(self, event, data):
        if event == "on_inference_result":
            frame = data.frame_context.frame
            
            frame = self.box_annotator.annotate(scene=frame, detections=data.frame_context.detections)
            
            sv.plot_image(frame, (12, 12))

    def cleanup(self):
        pass