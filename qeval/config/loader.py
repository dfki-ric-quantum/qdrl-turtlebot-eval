import importlib


class ConfWrapper:
    def __init__(self, **config):
        self.__dict__.update(config)


def load_config(config_name):
    try:
        config = importlib.import_module("qeval.config." + config_name)
    except ImportError:
        raise ValueError(f"No such config {config_name}")

    return ConfWrapper(**config.CONFIG)
