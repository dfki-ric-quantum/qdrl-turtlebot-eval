import argparse

from qeval.config.loader import load_config
from qeval.utils import create_results_dir, set_gpu_memory_growing

set_gpu_memory_growing()

from qeval.runner import run_training


def parse_args():
    parser = argparse.ArgumentParser(description="Train qturtle env")
    parser.add_argument("-n", type=int, required=True, dest="number")
    parser.add_argument("-d", type=str, required=True, dest="dir")
    parser.add_argument("-c", type=str, required=True, dest="config")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    create_results_dir(args.dir)

    run_training(config, args)
