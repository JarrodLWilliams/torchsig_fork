from dataclasses import dataclass
from torchsig.datasets import conf

@dataclass
class PulseShapingTrainConfig(conf.Sig53Config):
    name: str = 'clean_pulse_train'
    level: int = 0
    seed: int = 246813579
    eb_no: bool = False
    num_samples: int = 1e5

@dataclass
class PulseShapingValConfig(PulseShapingTrainConfig):
    name: str = 'clean_pulse_val'
    seed: int = 135792468
    num_samples: int = 2e4

@dataclass
class PulseShapingTestConfig(PulseShapingTrainConfig):
    name: str = 'clean_pulse_test'
    seed: int = 124695783
    num_samples: int = 1e4

@dataclass
class PulseShapingTrainQAConfig(PulseShapingTrainConfig):
    num_samples = 100

@dataclass
class PulseShapingValQAConfig(PulseShapingValConfig):
    num_samples = 100
