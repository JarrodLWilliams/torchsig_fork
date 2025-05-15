from torchsig.utils.writer import DatasetCreator, DatasetLoader
from torchsig.datasets.modulations import ModulationsDataset
from torchsig.datasets import conf
from typing import List
import click
import os
import numpy as np
from dataclasses import dataclass

@dataclass
class PulseShapingTrainConfig(conf.Sig53Config):
    name: str = 'clean_pulse_train'
    level: int = 0
    seed: int = 246813579
    eb_no: bool = False
    num_samples: int = 1e4

@dataclass
class PulseShapingValConfig(PulseShapingTrainConfig):
    name: str = 'clean_pulse_val'
    seed: int = 135792468
    num_samples: int = 5e3

@dataclass
class PulseShapingTrainQAConfig(PulseShapingTrainConfig):
    num_samples = 100

@dataclass
class PulseShapingValQAConfig(PulseShapingValConfig):
    num_samples = 100

def collate_fn(batch):
    return tuple(zip(*batch))


def generate(path: str, configs: List[conf.Sig53Config], num_workers: int, num_samples_override: int):
    for config in configs:
        num_samples = config.num_samples if num_samples_override <=0 else num_samples_override
        batch_size = int(np.min((config.num_samples // num_workers, 32)))
        print(f'batch_size -> {batch_size} num_samples -> {num_samples}, config -> {config}')
        ds = ModulationsDataset(
            level=config.level,
            num_samples=num_samples,
            num_iq_samples=config.num_iq_samples,
            use_class_idx=config.use_class_idx,
            include_snr=config.include_snr,
            eb_no=config.eb_no,
            classes=['8fsk', '8gfsk']
        )
        dataset_loader = DatasetLoader(ds, seed=12345678, collate_fn=collate_fn, num_workers=num_workers, batch_size=batch_size)
        creator = DatasetCreator(ds, seed=12345678, path="{}".format(os.path.join(path, config.name)), loader=dataset_loader, num_workers=num_workers)
        creator.create()


@click.command()
@click.option("--root", default="pulse_shaping_data", help="Path to generate pulse shaping data.")
@click.option("--pulse-shapes", "pulse_shapes", default=True, help="Generate all versions of sig53 dataset.")
@click.option("--qa", default=False, help="Generate pulse shaping QA dataset.")
@click.option("--num-samples", default=-1, help="Override for number of dataset samples.")
@click.option("--num-workers", "num_workers", default=os.cpu_count() // 2, help="Define number of workers for both DatasetLoader and DatasetCreator")
def main(root: str, qa: bool, pulse_shapes: bool, num_workers: int, num_samples):
    if not os.path.isdir(root):
        os.mkdir(root)

    configs = [
        PulseShapingTrainConfig,
        PulseShapingValConfig,
    ]

    qa_configs = [
        PulseShapingTrainQAConfig,
        PulseShapingValQAConfig
    ]

    if pulse_shapes:
        generate(root, configs, num_workers, num_samples)
        return
    
    elif qa:
        generate(root, qa_configs, num_workers, num_samples)
        return
    
    else:
        print('No data generated (--pulse-shapes and --qa set to False)')


def generate_on_the_fly():
    """
    Returns train and val datasets to be used to generate data on the fly.

    Returns:
        Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]: train_data, val_data 
    """
    train_config = PulseShapingTrainConfig
    val_config = PulseShapingValConfig
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
    main()
