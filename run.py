
import logging
from source_factory import SourceFactory
from events import EventManager
from visualization import Visualizer
from utils import CleanupManager, SaveManager
from config_loader import read_from_config
from pipeline import DetectionPipeline
from rfdetr import RFDETRNano
import supervision as sv

from pipeline_context import PipelineContext, FrameContext

CONFIG_PATH = "project/config.json"
MODEL_PATH = "project/model_content/checkpoint_best_ema.pth"

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = read_from_config(CONFIG_PATH)
 
    source = SourceFactory.create(source_path=config["source"])  # Returns the source depending on media
    
    model = RFDETRNano(pretrain_weights=MODEL_PATH)
    if config["tracker"]:
        tracker = sv.ByteTrack()

    visualizer = Visualizer() if config["show"] else None
    saver = SaveManager() if config["save"] else None

    event_manager = EventManager()
    if visualizer:
        event_manager.register("on_inference_result", visualizer)
    if saver:
        event_manager.register("on_inference_result", saver)
    
    # Cleanup manager collects cleanup methods and run them in the end
    cleanup = CleanupManager()
    if source:
        cleanup.add(source.cleanup)
    if visualizer:
        cleanup.add(visualizer.cleanup)
    if saver and not source.is_static:
        cleanup.add(saver.save_video)


    pipeline = DetectionPipeline(
        source=source,
        model=model,
        event_manager=event_manager,
        tracker=tracker
    )

    # Skapa initialt context med FrameContext
    ctx = PipelineContext(
        is_static=source.is_static,
        should_continue=True,
        frame_context=FrameContext(frame=None, timestamp=0)
    )

    try:
        pipeline.run(ctx)
    except Exception:
        logging.exception("Unhandled error in main loop")
    finally:
        # Ensure resources are cleaned up even on error
        cleanup.run()


if __name__ == "__main__":
    main()