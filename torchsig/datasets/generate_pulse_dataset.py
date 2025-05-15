from torchsig.utils.writer import DatasetCreator, DatasetLoader
from torchsig.datasets.modulations import ModulationsDataset
from typing import List
import click
import os
import numpy as np
from torchsig.datasets import pulse_shaping_confs as conf

def generate_on_the_fly():
    """
    Returns train and val datasets to be used to generate data on the fly.

    Returns:
        Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]: train_data, val_data 
    """
    train_config = conf.PulseShapingTrainConfig
    val_config = conf.PulseShapingValConfig
    train_data = ModulationsDataset(
        level=train_config.level,
        num_samples=train_config.num_samples,
        num_iq_samples=train_config.num_iq_samples,
        use_class_idx=train_config.use_class_idx,
        include_snr=train_config.include_snr,
        eb_no=train_config.eb_no,
        classes=['8fsk', '8gfsk']
    )

    val_data = ModulationsDataset(
        level=val_config.level,
        num_samples=val_config.num_samples,
        num_iq_samples=val_config.num_iq_samples,
        use_class_idx=val_config.use_class_idx,
        include_snr=val_config.include_snr,
        eb_no=val_config.eb_no,
        classes=['8fsk', '8gfsk']
    )
    
    return train_data, val_data
    
if __name__ == "__main__":
    generate_on_the_fly()
