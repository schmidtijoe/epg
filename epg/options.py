import logging

from simple_parsing import ArgumentParser, field
import dataclasses as dc
from typing import List

logModule = logging.getLogger(__name__)


@dc.dataclass
class FileConfig:
    configFile: str = field(default="", alias=["-c"])
    outputPath: str = field(default="", alias=["-o"])
    visualize: bool = True

    sequenceType: str = field(default="semc", alias=["-s"], choices=["semc", "pulseq"])

    def __post_init__(self):
        d_available_types = {
            "semc": 1,
            "pulseq": 2
        }
        if d_available_types.get(self.sequenceType) is None:
            err = f"sequenceType must be one of {[key for key in d_available_types.keys()]}"
            logModule.error(err)
            raise ValueError(err)


@dc.dataclass
class SeqConfig:
    ESP: float = 10.0   # [ms]
    ETL: int = 7
    T1: float = 1.5             # [s]
    T2: float = 35.0            # [ms]
    excitationFA: float = 90.0
    excitationRfPhase: float = 90.0         # [°]
    excitationGradRephase: float = 20.0     # [mT/m]
    refocusingFA: List = dc.field(default_factory=lambda: [140.0])  # °
    refocusingRfPhase: List = dc.field(default_factory=lambda: [140.0])  # °
    refocusingGradCrusher: float = 20.0     # [mT/m]
    refocusingGradSliceSpoiling: float = 30.0   # [mT/m]


@dc.dataclass
class Config:
    config: FileConfig = FileConfig()
    seqParams: SeqConfig = SeqConfig()


def create_cmd_parser():
    """
        Build the parser for arguments
        Parse the input arguments.
        """
    parser = ArgumentParser(prog='epg')
    parser.add_arguments(FileConfig, dest="config")
    parser.add_arguments(SeqConfig, dest="params")
    args = parser.parse_args()

    return parser, args


