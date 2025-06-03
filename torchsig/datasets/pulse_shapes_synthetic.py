from torchsig.utils.types import SignalData, SignalMetadata, Signal, ModulatedRFMetadata
from torchsig.utils.types import (
    create_signal_metadata,
    create_rf_metadata,
    create_modulated_rf_metadata,
)
from torchsig.utils.dsp import convolve, gaussian_taps, low_pass, rrc_taps, irrational_rate_resampler
from torchsig.transforms.functional import FloatParameter, IntParameter
from torchsig.utils.dataset import SignalDataset
from torchsig.utils.dsp import estimate_filter_length
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from torch.utils.data import ConcatDataset
from scipy import signal as sp
from collections import OrderedDict
import numpy as np
import itertools
import pickle
from torchsig.transforms.filters import *

import math

def remove_corners(const):
    spacing = 2.0 / (np.sqrt(len(const)) - 1)
    cutoff = spacing * (np.sqrt(len(const)) / 6 - 0.5)
    return [
        p
        for p in const
        if np.abs(np.real(p)) < 1.0 - cutoff or np.abs(np.imag(p)) < 1.0 - cutoff
    ]


default_const_map = OrderedDict(
    {
        "ook": np.add(*map(np.ravel, np.meshgrid(np.linspace(0, 1, 2), 0j))),
        "bpsk": np.add(*map(np.ravel, np.meshgrid(np.linspace(-1, 1, 2), 0j))),
        "4pam": np.add(*map(np.ravel, np.meshgrid(np.linspace(0, 1, 4), 0j))),
        "4ask": np.add(*map(np.ravel, np.meshgrid(np.linspace(-1, 1, 4), 0j))),
        "qpsk": np.add(
            *map(
                np.ravel, np.meshgrid(np.linspace(-1, 1, 2), 1j * np.linspace(-1, 1, 2))
            )
        ),
        "8pam": np.add(*map(np.ravel, np.meshgrid(np.linspace(0, 1, 8), 0j))),
        "8ask": np.add(*map(np.ravel, np.meshgrid(np.linspace(-1, 1, 8), 0j))),
        "8psk": np.exp(2j * np.pi * np.linspace(0, 7, 8) / 8.0),
        "16qam": np.add(
            *map(
                np.ravel, np.meshgrid(np.linspace(-1, 1, 4), 1j * np.linspace(-1, 1, 4))
            )
        ),
        "16pam": np.add(*map(np.ravel, np.meshgrid(np.linspace(0, 1, 16), 0j))),
        "16ask": np.add(*map(np.ravel, np.meshgrid(np.linspace(-1, 1, 16), 0j))),
        "16psk": np.exp(2j * np.pi * np.linspace(0, 15, 16) / 16.0),
        "32qam": np.add(
            *map(
                np.ravel, np.meshgrid(np.linspace(-1, 1, 4), 1j * np.linspace(-1, 1, 8))
            )
        ),
        "32qam_cross": remove_corners(
            np.add(
                *map(
                    np.ravel,
                    np.meshgrid(np.linspace(-1, 1, 6), 1j * np.linspace(-1, 1, 6)),
                )
            )
        ),
        "32pam": np.add(*map(np.ravel, np.meshgrid(np.linspace(0, 1, 32), 0j))),
        "32ask": np.add(*map(np.ravel, np.meshgrid(np.linspace(-1, 1, 32), 0j))),
        "32psk": np.exp(2j * np.pi * np.linspace(0, 31, 32) / 32.0),
        "64qam": np.add(
            *map(
                np.ravel, np.meshgrid(np.linspace(-1, 1, 8), 1j * np.linspace(-1, 1, 8))
            )
        ),
        "64pam": np.add(*map(np.ravel, np.meshgrid(np.linspace(0, 1, 64), 0j))),
        "64ask": np.add(*map(np.ravel, np.meshgrid(np.linspace(-1, 1, 64), 0j))),
        "64psk": np.exp(2j * np.pi * np.linspace(0, 63, 64) / 64.0),
        "128qam_cross": remove_corners(
            np.add(
                *map(
                    np.ravel,
                    np.meshgrid(np.linspace(-1, 1, 12), 1j * np.linspace(-1, 1, 12)),
                )
            )
        ),
        "256qam": np.add(
            *map(
                np.ravel,
                np.meshgrid(np.linspace(-1, 1, 16), 1j * np.linspace(-1, 1, 16)),
            )
        ),
        "512qam_cross": remove_corners(
            np.add(
                *map(
                    np.ravel,
                    np.meshgrid(np.linspace(-1, 1, 24), 1j * np.linspace(-1, 1, 24)),
                )
            )
        ),
        "1024qam": np.add(
            *map(
                np.ravel,
                np.meshgrid(np.linspace(-1, 1, 32), 1j * np.linspace(-1, 1, 32)),
            )
        ),
    }
)

# This is probably redundant.
freq_map = OrderedDict(
    {
        "2fsk": np.linspace(-1 + (1 / 2), 1 - (1 / 2), 2, endpoint=True),
        "2gfsk": np.linspace(-1 + (1 / 2), 1 - (1 / 2), 2, endpoint=True),
        "2msk": np.linspace(-1 + (1 / 2), 1 - (1 / 2), 2, endpoint=True),
        "2gmsk": np.linspace(-1 + (1 / 2), 1 - (1 / 2), 2, endpoint=True),
        "4fsk": np.linspace(-1 + (1 / 4), 1 - (1 / 4), 4, endpoint=True),
        "4gfsk": np.linspace(-1 + (1 / 4), 1 - (1 / 4), 4, endpoint=True),
        "4msk": np.linspace(-1 + (1 / 4), 1 - (1 / 4), 4, endpoint=True),
        "4gmsk": np.linspace(-1 + (1 / 4), 1 - (1 / 4), 4, endpoint=True),
        "8fsk": np.linspace(-1 + (1 / 8), 1 - (1 / 8), 8, endpoint=True),
        "8gfsk": np.linspace(-1 + (1 / 8), 1 - (1 / 8), 8, endpoint=True),
        "8msk": np.linspace(-1 + (1 / 8), 1 - (1 / 8), 8, endpoint=True),
        "8gmsk": np.linspace(-1 + (1 / 8), 1 - (1 / 8), 8, endpoint=True),
        "16fsk": np.linspace(-1 + (1 / 16), 1 - (1 / 16), 16, endpoint=True),
        "16gfsk": np.linspace(-1 + (1 / 16), 1 - (1 / 16), 16, endpoint=True),
        "16msk": np.linspace(-1 + (1 / 16), 1 - (1 / 16), 16, endpoint=True),
        "16gmsk": np.linspace(-1 + (1 / 16), 1 - (1 / 16), 16, endpoint=True),
    }
)


class FSKDigitalModulationDataset(ConcatDataset):
    """Digital Modulation Dataset

    Args:
        modulations (:obj:`list` or :obj:`tuple`):
            Sequence of strings representing the constellations that should be included.

        num_iq_samples (:obj:`int`):
            number of samples to read from each file in the database

        num_samples_per_class (:obj:`int`):
            number of samples to be kept for each class

        iq_samples_per_symbol (:obj:`Optional[int]`):
            number of IQ samples per symbol

        random_data (:obj:`bool`):
            whether the modulated binary utils should be random each time, or seeded by index

        random_pulse_shaping (:obj:`bool`):
            boolean to enable/disable randomized pulse shaping

        user_const_map (:obj:`Optional[OrderedDict]`):
            optional user-defined constellation map, defaults to Sig53 modulations

    """

    def __init__(
        self,
        modulations: Optional[Union[List, Tuple]] = ("bpsk", "2gfsk"),
        num_iq_samples: int = 100,
        num_samples_per_class: int = 100,
        iq_samples_per_symbol: Optional[int] = None,
        random_data: bool = False,
        random_pulse_shaping: bool = False,
        user_const_map: Optional[OrderedDict] = None,
        pulse_shaping_filter_dict: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        
        if pulse_shaping_filter_dict is None:
            raise ValueError('No pulse shapes given.')
        
        filters = list(pulse_shaping_filter_dict.keys())
        const_map = user_const_map if user_const_map else default_const_map
        modulations = (
            list(const_map.keys()) + list(freq_map.keys())
            if modulations is None
            else modulations
        )
        freqs = [m for m in map(str.lower, modulations) if m in freq_map.keys()]

        if len(freqs) > 1:
            raise NotImplementedError(f'Pulse Shaping Dataset generation only implemented for a fixed FSK modulation scheme. Received multiple frequency modulation schemes: {freqs}')

        # parse filters


        # FSK signals with the Gaussian pulse shaping filter are handled differently than without
        fsks = []
        gfsks = []
        
        for freq_mod in freqs:
            if "g" in freq_mod:
                gfsks.append(freq_mod)
            else:
                fsks.append(freq_mod)

        filter_datasets = []
        # if mod scheme is FSK type (and not GFSK):
        
        # calculate number of filter classes

        num_filter_classes = len(list(itertools.chain(*pulse_shaping_filter_dict.values())))
        num_samples_per_filter = math.floor(num_samples_per_class/num_filter_classes)
       
        if len(fsks) > 0 and len(gfsks) == 0:

            
            for filter in filters:
                filter_params = pulse_shaping_filter_dict[filter]
                
                match filter:
                    case 'rrc':
                        rrc_filter = RRCPulseShapeFilter(bandwidth_mode='fixed')
                        
                        for fparam in filter_params:
                            rrc_filter.bandwidth = fparam
                            filter_datasets.append(
                                FSKDataset(
                                modulations=fsks,
                                num_iq_samples=num_iq_samples,
                                num_samples_per_class=num_samples_per_filter,
                                # num_samples_per_class= num_samples_per_class,
                                iq_samples_per_symbol=8,
                                pulse_shaping_filter=rrc_filter,
                                **kwargs,
                                )
                            )
                    case 'gaussian':
                        gaussian_filter = GaussianPulseShapeFilter(bandwidth_mode='fixed', bandwidth=filter_params[0])
                        filter_datasets.append(FSKDataset(
                                modulations=fsks,
                                num_iq_samples=num_iq_samples,
                                # num_samples_per_class=num_samples_per_class,
                                num_samples_per_class=num_samples_per_filter,
                                iq_samples_per_symbol=8,
                                pulse_shaping_filter=gaussian_filter,
                                **kwargs,
                                )
                        )
                    case 'rectangular':
                        rect_filter = RectangularPulseShapeFilter(bandwidth_mode='fixed', bandwidth=filter_params[0])
                        filter_datasets.append(FSKDataset(
                                modulations=fsks,
                                num_iq_samples=num_iq_samples,
                                # num_samples_per_class=num_samples_per_class,
                                num_samples_per_class=num_samples_per_filter,
                                iq_samples_per_symbol=8,
                                pulse_shaping_filter=rect_filter,
                                **kwargs,
                                )
                        )
        # for gfsk, gaussian pulse shape implemented by default, so no filter needs to be
        # passed.
        gfsks_dataset = FSKDataset(
            modulations=gfsks,
            num_iq_samples=num_iq_samples,
            num_samples_per_class=num_samples_per_class,
            iq_samples_per_symbol=8,
            random_data=random_data,
            random_pulse_shaping=random_pulse_shaping,
            pulse_shaping_filter=None, 
            **kwargs,
        )
        filter_datasets.append(gfsks_dataset)

        super(FSKDigitalModulationDataset, self).__init__(filter_datasets)

class SyntheticDataset(SignalDataset):
    def __init__(self, **kwargs) -> None:
        super(SyntheticDataset, self).__init__(**kwargs)
        self.index: List[Tuple[Any, ...]] = []

    def __getitem__(self, index: int) -> Tuple[Union[SignalData, np.ndarray], Any]:
        signal_meta = self.index[index][-1]
        signal_data = SignalData(samples=self._generate_samples(self.index[index]))
        signal = Signal(data=signal_data, metadata=signal_meta)

        if self.transform:
            signal = self.transform(signal)

        target = signal["metadata"]

        if self.target_transform:
            target = self.target_transform(signal["metadata"])

        return signal["data"]["samples"], target

    def __len__(self) -> int:
        return len(self.index)

    def _generate_samples(self, item: Tuple) -> np.ndarray:
        raise NotImplementedError




class FSKDataset(SyntheticDataset):
    """FSK Dataset

    Args:
        modulations (:obj:`list` or :obj:`tuple`):
            Sequence of strings representing the modulations that should be included

        num_iq_samples (:obj:`int`):
            number of samples to read from each file in the database

        num_samples_per_class (:obj:`int`):
            number of samples to be kept for each class

        iq_samples_per_symbol (:obj:`int`):
            number of IQ samples per symbol

        random_data (:obj:`bool`):
            whether the modulated binary utils should be random each time, or seeded by index

        center_freq (:obj:`float`):
            center frequency of the signal, will be upconverted internally

        bandwidth (:obj:`float`):
            bandwidth of the signal, will be resampled internally

        transform (:obj:`Callable`, optional):
            A function/transform that takes in an IQ vector and returns a transformed version.

    """

    def __init__(
        self,
        modulations: Optional[Union[List, Tuple]] = ("2fsk", "2gmsk"),
        num_iq_samples: int = 100,
        num_samples_per_class: int = 100,
        iq_samples_per_symbol: int = 2,
        random_data: bool = False,
        random_pulse_shaping: bool = False,
        center_freq: float = 0,
        bandwidth: float = 0.5,
        pulse_shaping_filter: Optional[Callable] = None,
        **kwargs,
    ):
        super(FSKDataset, self).__init__(**kwargs)
        self.modulations = list(freq_map.keys()) if modulations is None else modulations
        self.num_iq_samples = num_iq_samples
        self.num_samples_per_class = num_samples_per_class
        self.iq_samples_per_symbol = iq_samples_per_symbol
        self.random_data = random_data
        self.random_pulse_shaping = random_pulse_shaping
        self.index = []
        self.pulse_shaping_filter = pulse_shaping_filter

        for freq_idx, freq_name in enumerate(map(str.lower, self.modulations)):
            if 'g' in freq_name:
                pulse_shape_filter_name = 'gaussian'
            else:
                pulse_shape_filter_name = self.pulse_shaping_filter.name
            for idx in range(self.num_samples_per_class):
                # modulation index scales the bandwidth of the signal, and
                # iq_samples_per_symbol is used as an oversampling rate in
                # FSKDataset class, therefore the signal bandwidth can be
                # approximated by mod_idx/iq_samples_per_symbol.
                #mod_idx = self._mod_index(freq_name)
                #bandwidth_cutoff = mod_idx / self.iq_samples_per_symbol
                #bandwidth = np.random.uniform(bandwidth_cutoff, 0.5 - bandwidth_cutoff) #if self.random_pulse_shaping else 0.0
                meta = ModulatedRFMetadata(
                    sample_rate=0.0,
                    num_samples=float(self.num_iq_samples),
                    complex=True,
                    lower_freq=center_freq-(bandwidth/2),
                    upper_freq=center_freq+(bandwidth/2),
                    center_freq=center_freq,
                    bandwidth=bandwidth,
                    start=0.0,
                    stop=1.0,
                    duration=1.0,
                    snr=0.0,
                    bits_per_symbol=np.log2(len(freq_map[freq_name])),
                    samples_per_symbol=float(iq_samples_per_symbol),
                    class_name=freq_name,
                    class_index=freq_idx,
                    excess_bandwidth=0,
                    pulse_shaping_filter_name = pulse_shape_filter_name
                )

                if pulse_shape_filter_name == 'rrc':
                    meta['excess_bandwidth'] = self.pulse_shaping_filter.bandwidth
                # print(freq_name, freq_idx, self.num_samples_per_class, idx, meta)
                # breakpoint()
                self.index.append((freq_name, freq_idx * self.num_samples_per_class + idx, [meta])
                )

    def _generate_samples(self, item: Tuple) -> np.ndarray:
        const_name = item[0]
        index = item[1]
        metadata = item[2][0]
        center_freq = metadata["center_freq"]
        bandwidth = metadata["bandwidth"]

        # calculate the modulation order, ex: the "4" in "4-FSK"
        const = freq_map[const_name]
        mod_order = len(const)

        # samples per symbol presumably used as a bandwidth measure (ex: BW=1/SPS),
        # but does not work for FSK. samples per symbol is redefined into
        # the "oversampling rate", and samples per symbol is instead derived
        # from the modulation order
        oversampling_rate = np.copy(self.iq_samples_per_symbol)
        samples_per_symbol_recalculated = int(mod_order * oversampling_rate)

        # scale the frequency map by the oversampling rate such that the tones
        # are packed tighter around f=0 the larger the oversampling rate
        const_oversampled = const / oversampling_rate

        orig_state = np.random.get_state()
        if not self.random_data:
            np.random.seed(index)

        # get the modulation index
        mod_idx = self._mod_index(const_name)

        # calculate the resampling rate to convert from the oversampling rate specified by
        # self.iq_samples_per_symbol into the proper bandwidth
        resampleRate = bandwidth*mod_idx/(1/oversampling_rate)

        # calculate the indexes into symbol table
        symbol_nums = np.random.randint(0, len(const_oversampled), int(np.ceil((self.num_iq_samples / samples_per_symbol_recalculated) * (1/resampleRate)) ))
        # produce data symbols
        symbols = const_oversampled[symbol_nums]
    
        # if "g" not in const_name and self.pulse_shaping_filter is not None:
        #     pulse_shape = self.pulse_shaping_filter
        # else:
        # # rectangular pulse shape
        #     pulse_shape = np.ones(samples_per_symbol_recalculated)

        if "g" in const_name:
            # GMSK, GFSK
            pulse_shape = np.ones(samples_per_symbol_recalculated)
            taps = gaussian_taps(samples_per_symbol_recalculated, bandwidth)
            pulse_shape = np.convolve(taps,pulse_shape)
            filtered = sp.upfirdn(pulse_shape,symbols,up=samples_per_symbol_recalculated,down=1)
        # upsample symbols and apply pulse shaping
        # filtered = sp.upfirdn(pulse_shape,symbols,up=samples_per_symbol_recalculated,down=1)
        # we pass in bandwidth generically
    
        if 'g' not in const_name:
            if self.pulse_shaping_filter is not None:
                # if self.pulse_shaping_filter.name == 'rrc':
                #     metadata['excess_bandwidth'] = self.pulse_shaping_filter.bandwidth
                filtered = self.pulse_shaping_filter(symbols, samples_per_symbol_recalculated, bandwidth)

            else:
                pulse_shape = np.ones(samples_per_symbol_recalculated)
                filtered = sp.upfirdn(pulse_shape,symbols,up=samples_per_symbol_recalculated,down=1)


        # insert a zero at first sample to start at zero phase
        filtered = np.insert(filtered, 0, 0)

        phase = np.cumsum(np.array(filtered) * 1j * mod_idx * np.pi)
        modulated = np.exp(phase)

        if self.random_pulse_shaping:
            taps = low_pass(cutoff=bandwidth / 2, transition_bandwidth=(0.5 - bandwidth / 2) / 4)
            # apply the filter
            modulated = convolve(modulated, taps)

        # apply resampling
        modulated = irrational_rate_resampler ( modulated, resampleRate )

        # apply center frequency shifting
        modulated *= np.exp(2j*np.pi*center_freq*np.arange(0,len(modulated)))

        # determine the boundaries for where the signal currently resides.
        # these values are used to determine if aliasing has occured
        upperSignalEdge = center_freq + (bandwidth/2)
        lowerSignalEdge = center_freq - (bandwidth/2)

        # check to see if aliasing has occured due to upconversion. if so, then apply
        # a filter to minimize it
        if ( upperSignalEdge > 0.5 or lowerSignalEdge < -0.5):

            # the signal has overlaped either the -fs/2 or +fs/2 boundary and therefore
            # a BPF filter will be applied to attenuate the portion of the signal that
            # is overlapping the -fs/2 or +fs/2 boundary to minimize aliasing
            modulated = upconversionAntiAliasingFilter ( modulated, center_freq, bandwidth )

        if not self.random_data:
            np.random.set_state(orig_state)  # return numpy back to its previous state
   
        return modulated[:self.num_iq_samples]

    def _mod_index(self, const_name):
        # returns the modulation index based on the modulation
        if "gfsk" in const_name:
            # bluetooth
            mod_idx = 0.32
        elif "msk" in const_name:
            # MSK, GMSK
            mod_idx = 0.5
        else:
            # FSK
            mod_idx = 1.0
        return mod_idx






# apply an anti-aliasing filter to a signal which has aliased and wrapped around the
# -fs/2 or +fs/2 boundary due to upconversion
def upconversionAntiAliasingFilter ( input_signal, center_freq, bandwidth ):

    # determine the boundaries for where the signal currently resides.
    # these values are used to determine if aliasing has occured
    upperSignalEdge = center_freq + (bandwidth/2)
    lowerSignalEdge = center_freq - (bandwidth/2)

    # define the boundary for the upper and lower frequencies
    # upon which a BPF will be designed to limit aliasing
    upperBoundary = 0.48
    lowerBoundary = -upperBoundary

    # determine if aliasing has occured, and if so, which direction,
    # either +fs/2 or -fs/2
    if ( upperSignalEdge > 0.5): # aliasing occurs across +fs/2
        slicedUpperSignalEdge = upperBoundary
        slicedLowerSignalEdge = -0.5+center_freq
    elif ( lowerSignalEdge < -0.5): # aliasing occurs across -fs/2
        slicedLowerSignalEdge = lowerBoundary
        slicedUpperSignalEdge = 0.5+center_freq
    else: # no aliasing occurs
        slicedUpperSignalEdge = upperSignalEdge
        slicedLowerSignalEdge = lowerSignalEdge

    # compute the bandwidth and center frequency after the BPF is applied
    slicedBandwidth = slicedUpperSignalEdge - slicedLowerSignalEdge
    slicedCenterFreq = slicedLowerSignalEdge + (slicedBandwidth/2)

    # design a LPF then upconvert it to a BPF
    # 
    # calculate the transition bandwidth for the LPF with proportion to the
    # signal's bandwidth. a fixed ratio is used here in order to keep the 
    # transition bandwidth small
    transitionBandwidth = slicedBandwidth/16
    # the passband edge of the LPF is 1/2 of the post-filtered bandwidth. this
    # pushes the cutoff frequency past the 3 dB point in the signal bandwidth 
    # as to minimize the distortion of the underlying signal
    fPass = slicedBandwidth/2
    # calculate the filter cutoff location
    cutoff = fPass + (transitionBandwidth/2)
    # design the LPF
    LPFWeights = low_pass(cutoff=cutoff,transition_bandwidth=transitionBandwidth)
    # modulate the LPF to BPF
    numLPFWeights = len(LPFWeights)
    n = np.arange(-int(numLPFWeights-1)/2,((numLPFWeights-1)/2)+1)
    BPFWeights = LPFWeights * np.exp(2j*np.pi*slicedCenterFreq*n)
    # apply BPF
    output = np.convolve(BPFWeights,input_signal)
    return output


if __name__ == '__main__':
    # rrc_filter = RRCPulseShapeFilter(bandwidth_mode='fixed', bandwidth=0.1)
    # data = FSKDataset(
    #     modulations=['2fsk'],
    #     num_iq_samples=4096,
    #     num_samples_per_class=100,
    #     iq_samples_per_symbol=8,
    #     pulse_shaping_filter=rrc_filter,
    # )
    # print(data[0])

    data = FSKDigitalModulationDataset(modulations=['2fsk'],
                        num_iq_samples=4096,
                        pulse_shaping_filter_dict={'rrc':[0.1,0.2]})
    breakpoint()
    
    print(data[0])