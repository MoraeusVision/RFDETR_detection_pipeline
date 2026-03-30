import torch
import logging


def get_device():
    """Return the best available device: cuda, mps, or cpu. Logs the selected device."""
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logging.info(f"Using device: {device}")
    return device
