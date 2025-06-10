from typing import Callable, List, Optional, OrderedDict

import numpy as np
from torch.utils.data import ConcatDataset

from torchsig.datasets.synthetic import DigitalModulationDataset
from torchsig.datasets.pulse_shapes_synthetic import FSKDigitalModulationDataset
from torchsig.transforms import (
    Compose,
    IQImbalance,
    Normalize,
    RandomApply,
    RandomFrequencyShift,
    RandomPhaseShift,
    RandomResample,
    RandomTimeShift,
    RayleighFadingChannel,
    TargetSNR,
    ComplexTo2D
)
from torchsig.transforms.target_transforms import (
    DescToClassIndex,
    DescToClassIndexSNR,
    DescToClassName,
    DescToClassNameSNR,
)


class FSKPulseShapesDataset(ConcatDataset):
    """ModulationsDataset serves as a standard dataset for many RF machine
    learning tasks in the modulation recognition/classification domain.

    Args:
        level (:obj:`str` int):
            * level 0 represents perfect modulations as if they synthesized by a transmitter
            * level 1 represents impairments related to a cabled environment in which a receiver is uncalibrated and unsynchronized
            * level 2 represents impairments related to over-the-air transmission in which the receiver is a non-cooperative receiver

        transform (:class:`torchsig.transforms.Transform`):
            Any additional transforms to append onto the existing transforms of the dataset -- usually to change the
            complex floating point output into another format.

    Impairments:
        * Phase shift (uniform between -pi and pi):
            Phase shifts occur as a result of the carrier wave (complex sinusoid) arriving at different times at the
            receiver.

        * Time shift (uniform between +- half a sample):
            More easily understood as a delay shift or a sample time offset, this is when a signal arrives
            at an ADC at different times. Plus or minus half a sample represents the maximum deviation from a sample
            point of view.

        * Frequency shift (uniform between -.16 and .16 relative to the sample rate):
            Frequency shifts come from Doppler shifts (movement of Tx or Rx) and from local-oscillator offsets.
            The maximum reasonable deviation for a signal of bandwidth fs/2 would be -.25 -- otherwise the kind of
            frequency shift applied here will cause the signal to "wrap-around" in the frequency domain.

        * IQ Imbalance (amplitude, phase, and DC offset):
            It's not clear what these values should be as typical receivers are calibrated to compensate for this.
            However, there are still scenarios in which it is difficult to properly calibrate this.

        * Rayleigh Fading (uniform between 2 and 20 taps with tapered power delay profile):
            This is typical in a multi-path fading environment

        * Additive White Gaussian Noise (uniform between -2 and 18 dB in an Eb/N0 sense):
            AWGN comes from a variety of effects. We include only down to -2 dB because less than this makes samples
            essentially useless for training.

        * Random Resample (uniform between .75 and 1.5):
            This effectively makes the largest bandwidth fs*2/3 and the smallest around fs/3 assuming the original
            signal was at fs/2. This represents a situation in which the bandwidth of the original signal is not
            known and may be the output of a channelizer with some fixed bandwidth.

        * Length of 4096
            For higher order modulations, it may be necessary to have more samples to see everyone symbol
            with high probability

        * Signal bandwidths at ~fs/2
            This is a good compromise between critically samples signals and vastly oversampled signals. Vastly
            oversampled signals will require models to take in perhaps millions of IQ samples to get good classification
            performance on higher order modulations. Critically sampled signals can be easily corrupted by even the
            more simple channel impairments.

    """

    pulse_shaping_default_classes: List[str] = [
        "2fsk",
        "2gfsk",
        "4fsk",
        "4gfsk",
        "8fsk",
        "8gfsk",
        "16fsk",
        "16gfsk",
    ]

    def __init__(
        self,
        classes: Optional[List[str]] = None,
        use_class_idx: bool = False,
        level: int = 0,
        num_iq_samples: int = 2048,
        iq_samples_per_symbol: int = 2,
        num_samples: int = 4500,
        include_snr: bool = False,
        eb_no: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        pulse_shaping_filter_dict: Optional[Callable] = None,
        random_pulse_shaping_override: bool = False,
        user_const_map: Optional[OrderedDict] = None,
        **kwargs,
    ) -> None:
        print('Warning: Custom Pulse Shaping is currently only implemented for FSK.')
        classes = self.pulse_shaping_default_classes if classes is None else classes
        # Set the target transform based on input options if none provided
        if not target_transform:
            if use_class_idx:
                if include_snr:
                    target_transform = DescToClassIndexSNR(class_list=classes)
                else:
                    target_transform = DescToClassIndex(class_list=classes)
            else:
                if include_snr:
                    target_transform = DescToClassNameSNR()
                else:
                    target_transform = DescToClassName()
        num_samples_per_class = int(num_samples / len(classes))
        self.class_dict = dict(zip(classes, range(len(classes))))
        self.include_snr = include_snr
        

        # Extract class info
        digital_classes = []
        custom_classes = []
        for class_name in classes:
            if class_name not in self.pulse_shaping_default_classes:
                if class_name in user_const_map.keys():
                    custom_classes.append(class_name)
                else:
                    raise ValueError(f'No matching constellation for (custom) class {class_name}')
            else:
                digital_classes.append(class_name)
        num_digital = len(digital_classes)
        num_custom = len(custom_classes)

        if level == 0:
            random_pulse_shaping = False
            internal_transforms = Compose(
                [
                    TargetSNR((100, 100), eb_no=eb_no),
                    Normalize(norm=np.inf),
                ]
            )
        elif level == 1:
            random_pulse_shaping = True
            internal_transforms = Compose(
                [
                    RandomPhaseShift((-1, 1)),
                    RandomTimeShift((-0.5, 0.5)),
                    RandomFrequencyShift((-0.16, 0.16)),
                    IQImbalance((-3, 3), (-np.pi * 1.0 / 180.0, np.pi * 1.0 / 180.0), (-0.1, 0.1)),
                    RandomResample((0.75, 1.5), num_iq_samples=num_iq_samples),
                    TargetSNR((80, 80), eb_no=eb_no),
                    Normalize(norm=np.inf),
                ]
            )
        elif level == 2:
            random_pulse_shaping = True
            internal_transforms = Compose(
                [
                    RandomApply(RandomPhaseShift((-1, 1)), 0.9),
                    RandomApply(RandomTimeShift((-32, 32)), 0.9),
                    RandomApply(RandomFrequencyShift((-0.16, 0.16)), 0.7),
                    RandomApply(RayleighFadingChannel((0.05, 0.5), power_delay_profile=(1.0, 0.5, 0.1)), 0.5),
                    RandomApply(IQImbalance((-3, 3), (-np.pi * 1.0 / 180.0, np.pi * 1.0 / 180.0), (-0.1, 0.1)), 0.9),
                    RandomApply(RandomResample((0.75, 1.5), num_iq_samples=num_iq_samples), 0.5),
                    TargetSNR((-2, 30), eb_no=eb_no),
                    Normalize(norm=np.inf),
                ]
            )
        elif level == -1:
            print(f'Applying custom transforms: {transform} with random pulse shaping set to {random_pulse_shaping_override}')
            random_pulse_shaping = random_pulse_shaping_override
            internal_transforms = Compose(
                transform,
                Normalize(norm=np.inf)
            )
        else:
            raise ValueError("Level is unrecognized. Should be -1, 0, 1 or 2.")

        if level > -1:
            if transform is not None:
                internal_transforms = Compose(
                    [
                        internal_transforms,
                        transform,
                    ]
                )

        if num_digital > 0:
            digital_dataset = FSKDigitalModulationDataset(
                modulations=digital_classes,  # effectively uses all modulations
                num_iq_samples=num_iq_samples,
                num_samples_per_class=num_samples_per_class,
                iq_samples_per_symbol=iq_samples_per_symbol,
                random_data=True,
                random_pulse_shaping=random_pulse_shaping,
                transform=internal_transforms,
                target_transform=target_transform,
                pulse_shaping_filter_dict=pulse_shaping_filter_dict,
            )
        if num_custom > 0:
            custom_dataset = FSKDigitalModulationDataset(
                modulations=custom_classes,  # custom classes
                num_iq_samples=num_iq_samples,
                num_samples_per_class=num_samples_per_class,
                iq_samples_per_symbol=iq_samples_per_symbol,
                random_data=True,
                random_pulse_shaping=random_pulse_shaping,
                transform=internal_transforms,
                target_transform=target_transform,
                pulse_shaping_filter_dict=pulse_shaping_filter_dict,
                user_const_map=None,
            )

        if num_digital > 0 and num_custom > 0:
            super().__init__([digital_dataset, custom_dataset], **kwargs)
            # Torch's ConcatDataset should create this.

        elif num_digital > 0:
            super().__init__([digital_dataset], **kwargs)
        elif num_custom > 0:
            super().__init__([custom_dataset], **kwargs)
        else:
            raise ValueError("Input classes must contain at least 1 valid class")


    def __getitem__(self, item):
        return super().__getitem__(item)

if __name__ == '__main__':
    from torchsig.transforms import Spectrogram
    from matplotlib import pyplot as plt
    transform = Compose([TargetSNR((100, 100)), Normalize(norm=np.inf), Spectrogram(nperseg=512, noverlap=256)])
                         
                        #  Spectrogram(nperseg=512, noverlap=256)])
    data = FSKPulseShapesDataset(classes=['4fsk'],
                           use_class_idx=True,
                           num_iq_samples=4096,
                           iq_samples_per_symbol=2,
                           pulse_shaping_filter_dict={'rrc':[0.1,1.0]},
                           num_samples=10,
                           transform=transform,
                           target_transform=lambda x:[y['excess_bandwidth'] for y in x])
    print(data[0])

    fig, ax = plt.subplots(5,2, layout='constrained')
    ax = ax.ravel()

    # d1, l1 = data[0]
    # breakpoint()
    # d1 = d1[int(0.45*d1.shape[0]):int(0.55*d1.shape[0])]
    # plt.imshow(d1)
    # plt.show()
    from matplotlib.colors import LogNorm
    for i in range(10):
        spec, label = data[i]
        # spec = spec[int(0.45*spec.shape[0]):int(0.55*spec.shape[0])]
        ax[i].imshow(spec, norm='log')
        ax[i].set_aspect(1.0/ax[i].get_data_ratio(), adjustable='box')
        ax[i].set_title(str(label[0]))
    plt.show()

    # for i in range(10):
    #     spec, label = data[i]
    #     fft_norm = np.abs(np.fft.fft(spec))
    #     # MIN_NORM = 0
    #     # filtered_norm = fft_norm[fft_norm > MIN_NORM]
    #     ax[i].plot(fft_norm)
    #     ax[i].set_title(str(label[0]))
    # plt.show()

    # print(data[0], data[90])
    # data_mixed = FSKPulseShapesDataset(classes=['2fsk'],
    #                        use_class_idx=True,
    #                        num_iq_samples=4096,
    #                        pulse_shaping_filter_dict={'rrc': [0.1], 'Gaussian': [0.35], 'Rectangular': [0.5]},
    #                        num_samples=100,
    #                        target_transform=lambda x:[y['pulse_shaping_filter_name'] for y in x])
    # print(list(data_mixed[i] for i in range(0,99,33)))
 