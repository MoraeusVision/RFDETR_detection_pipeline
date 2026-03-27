import supervision as sv
import typing

class DetectionPipeline():
    def __init__(self, source=None, model=None, event_manager=None, tracker=None):
        self.source = source
        self.model = model
        self.event_manager = event_manager
        self.tracker = tracker
        
    def run(self, ctx):
        while True:
            frame = self.source.get_frame()
            if frame is None:
                break

            ctx.frame_context.frame = frame
            ctx.frame_context.detections = self.model.predict(frame)
            
            self.notify("on_inference_result", ctx)

    def notify(self, event_name, data=None):
        if self.event_manager is not None:
            self.event_manager.notify(event_name, data)