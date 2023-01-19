import logging
from simple_parsing import ArgumentParser, field, Serializable
import dataclasses as dc
from typing import List
import pathlib as pl
import json

logModule = logging.getLogger(__name__)


@dc.dataclass
class FileConfig(Serializable):
    configFile: str = field(default="", alias=["-c"])
    outputPath: str = field(default="", alias=["-o"])
    visualize: bool = True
    debug: bool = False

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
class SeqConfig(Serializable):
    ESP: float = 10.0                           # [ms]
    ETL: int = 7
    T1: float = 1500                            # [ms]
    T2: float = 35.0                            # [ms]

    excitationFA: float = 90.0                  # [°]
    excitationRfPhase: float = 90.0             # [°]
    excitationGradRephase: float = 20.0         # [mT/m]
    excitationRephaseOffset: float = 0.0        # [°]

    refocusingFA: List = dc.field(default_factory=lambda: [120.0])          # [°]
    refocusingRfPhase: List = dc.field(default_factory=lambda: [0.0])       # [°]
    refocusingGradCrusher: float = 20.0         # [mT/m]
    refocusingGradSliceSpoiling: float = 30.0   # [mT/m]

    sliceThickness: float = 0.7                 # [mm]


@dc.dataclass
class Config:
    config: FileConfig = FileConfig()
    params: SeqConfig = SeqConfig()

    @classmethod
    def load(cls, path):
        Conf = cls()
        path = pl.Path(path).absolute()
        if not path.is_file():
            raise AttributeError(f"{path} not a file")
        if path.suffix == ".json":
            with open(path, "r") as j_file:
                load_dict = json.load(j_file)
            Conf.config = FileConfig.from_dict(load_dict["config"])
            Conf.params = SeqConfig.from_dict(load_dict["params"])
        else:
            raise ValueError(f"{path} file ending not recognized!")
        return Conf

    @classmethod
    def from_cmd_args(cls, prog_args: ArgumentParser.parse_args):
        Conf = cls(config=prog_args.config, params=prog_args.params)
        if prog_args.config.configFile:
            Conf = cls.load(prog_args.config.configFile)
            Conf._check_non_default(prog_args)
        return Conf

    def _check_non_default(self, prog_args: ArgumentParser.parse_args):
        # check for non default param input:
        # This is for the case that we load a config file and provide additional cmd line inputs to overwrite
        # the config from the file. This approach fails if the additional info is default value
        d_default_config = FileConfig().to_dict()
        d_default_params = SeqConfig().to_dict()

        d_nd_config = prog_args.config
        d_nd_params = prog_args.params

        for key in d_default_config.keys():
            if d_default_config.get(key) != d_nd_config.get(key):
                self.config.__setattr__(key, d_nd_config.get(key))
        for key in d_default_params.keys():
            if d_default_params.get(key) != d_nd_params.get(key):
                self.params.__setattr__(key, d_nd_params.get(key))


def create_cmd_parser() -> (ArgumentParser, ArgumentParser.parse_args):
    """
        Build the parser for arguments
        Parse the input arguments.
        """
    parser = ArgumentParser(prog='epg')
    parser.add_arguments(FileConfig, dest="config")
    parser.add_arguments(SeqConfig, dest="params")
    args = parser.parse_args()

    return parser, args


