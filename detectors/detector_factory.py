from .rfdetr_detector import RFDETRDetector


class DetectorFactory():
    @staticmethod
    def create(detector_name, model_path, confidence_threshold=0.5):
        detector_name = detector_name.lower()

        if detector_name == "rfdetr":
            return RFDETRDetector(model_path, confidence_threshold=confidence_threshold)
        else:
            raise ValueError(f"Unsupported detector: {detector_name}")