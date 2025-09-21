import os
import re

import pandas as pd
import yaml
from .git_utils import GIT_ROOT


def load_yaml_safely(path):
    with open(path, "r") as f:
        content = f.read()

    # Strip !!python/tuple to just parse as a list
    content = re.sub(r'!!python/tuple', '', content)

    return yaml.safe_load(content)


class LightningLogs:
    def __init__(self, path: os.PathLike):
        self.path = path

        if not os.path.exists(self.path):
            raise RuntimeError(f"{self.path} does not exist.")

        hparams_path = os.path.join(self.path, 'hparams.yaml')
        metrics_path = os.path.join(self.path, 'metrics.csv')

        if not os.path.exists(hparams_path) or not os.path.exists(metrics_path):
            raise RuntimeError("Results are incomplete")

        self._params = None
        self._metrics = None


    def load(self) -> None:
        hparams_path = os.path.join(self.path, 'hparams.yaml')
        metrics_path = os.path.join(self.path, 'metrics.csv')

        self._params = load_yaml_safely(hparams_path)

        self._metrics = pd.read_csv(metrics_path)

    @property
    def params(self) -> dict:
        return self._params

    @property
    def metrics(self) -> pd.DataFrame:
        return self._metrics


LightningLogs_Root = os.path.join(GIT_ROOT, 'lightning_logs')


def get_all_logs() -> dict:
    all_logs = {}
    for file in os.listdir(LightningLogs_Root):
        log_path = os.path.join(LightningLogs_Root, file)
        log = LightningLogs(log_path)
        all_logs[file] = log
    return all_logs


class AllLightningLogs:
    _instance = None  # Class variable to hold the single instance

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):  # Prevent re-initialization
            self.initialized = True
        
        self._logs = get_all_logs()
        self._loaded = {log: False for log in self._logs.keys()}

    def __getitem__(self, logfile):
        if not logfile in self._logs:
            raise RuntimeError(f"{logfile} does not exist")
        
        if not self._loaded[logfile]:
            self._loaded[logfile] = True
            self._logs[logfile].load()
        
        return self._logs[logfile]

    @property
    def experiments(self) -> list:
        return list(self._logs.keys())
