"""Small dependency-free TSV logger using the supported AbstractSRLogger hook."""

from pathlib import Path
import json
from pysr.logger_specs import AbstractLoggerSpec
from pysr.julia_import import jl


class SnapshotLoggerSpec(AbstractLoggerSpec):
    def __init__(self, path, interval=2.0):
        self.path = str(path)
        self.interval = float(interval)

    def create_logger(self):
        if not bool(jl.seval("isdefined(Main, :AcousticSnapshotLogger)")):
            jl.seval(
                "include(" + json.dumps(str(Path(__file__).with_name("search_logger.jl"))) + ")"
            )
        return jl.make_acoustic_logger(self.path, self.interval)

    def write_hparams(self, logger, hparams):
        pass

    def close(self, logger):
        pass
