import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run detection inference pipeline")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file"
    )
    
    return parser.parse_args()