from .byte_tracker import ByteTrackTracker


class TrackerFactory:
	@staticmethod
	def create(tracker_name):
		if tracker_name is None:
			return None

		tracker_name = tracker_name.lower()
		# Keep tracker selection isolated from runtime wiring.
		if tracker_name == "bytetrack":
			return ByteTrackTracker()

		raise ValueError(f"Unsupported tracker: {tracker_name}")
