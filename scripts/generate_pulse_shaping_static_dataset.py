from torchsig.utils.writer import DatasetCreator, DatasetLoader
from torchsig.datasets.modulations import ModulationsDataset
from typing import List
import click
import os
import numpy as np
from torchsig.datasets import pulse_shaping_confs as conf


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
        conf.PulseShapingTrainConfig,
        conf.PulseShapingValConfig,
    ]

    qa_configs = [
        conf.PulseShapingTrainQAConfig,
        conf.PulseShapingValQAConfig
    ]

    if pulse_shapes:
        generate(root, configs, num_workers, num_samples)
        return
    
    elif qa:
        generate(root, qa_configs, num_workers, num_samples)
        return
    
    else:
        print('No data generated (--pulse-shapes and --qa set to False)')