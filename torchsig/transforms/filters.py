from typing import Any, Callable, List, Literal, Optional, Tuple, Union
import warnings
import numpy as np
from scipy import signal as sp
from torchsig.utils.dsp import convolve, gaussian_taps, low_pass, rrc_taps, irrational_rate_resampler


class AbstractPulseShapeFilter:
    """An abstract class representing a Pulse Shape Filter that can either work on
    targets or data

    """

    def __init__(self, name:str) -> None:
        warnings.warn(
            "Pulse Shape Filters only currently implemented for FSK constellations.", DeprecationWarning
        )
        self.string = self.__class__.__name__ + "()"
        self.random_generator = np.random.default_rng()
        self.name = None

    def __call__(self, symbols: Any, samples_per_symbol: int, bandwidth: float) -> Any:
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.string
    
    def name(self):
        return self.name
    
class RectangularPulseShapeFilter(AbstractPulseShapeFilter):

    def __init__(self, name='Rectangular'):
        super().__init__()
        self.name = name

    def __call__(self, symbols: Any, samples_per_symbol: int, bandwidth: float) -> Any:
        # upfirdn upsamples, applies the FIR filter, and then downsamples. The upsampling rate depends on the samples per symbol in the IQ (which is recalculated for FSK in the data generation phase, and the downsampling rate is fixed at one.)

        # Upsampling corresponds to expanding the symbol-sequence (IQ data) where the original samples are separated by sampler_per_symbol zeros. Then a low-pass filter is applied, which interpolates the zeros. The low-pass filter is equivalent to multiplying the signal by a rectangular function in the frequency domain.
        
        # Bandwidth is in the function input to match the abstract syntax but is not 
        # required for the rectangular pulse shape.

        filter = np.ones(samples_per_symbol)
        return sp.upfirdn(filter, symbols, up=samples_per_symbol, down=1)
    
class GaussianPulseShapeFilter(AbstractPulseShapeFilter):

    def __init__(self, name='Gaussian'):
        super().__init__()
        self.name = name
    

    def __call__(self, symbols: Any, samples_per_symbol: int, bandwidth: float) -> Any:
            pulse_shape = np.ones(samples_per_symbol)
            taps = gaussian_taps(samples_per_symbol, bandwidth)
            pulse_shape = np.convolve(taps,pulse_shape)
            return sp.upfirdn(pulse_shape,symbols,up=samples_per_symbol,down=1)

class RRCPulseShapeFilter(AbstractPulseShapeFilter):

    def __init__(self, name="rrc", size_in_symbols=4):
        super().__init__(name)
        self.name = name
        self.size_in_symbols = size_in_symbols

    def __call__(self, symbols: Any, samples_per_symbol: int, bandwidth: float) -> Any:
        pulse_shape = np.ones(samples_per_symbol)
        taps = rrc_taps(samples_per_symbol, self.size_in_symbols, bandwidth)
        pulse_shape = np.convolve(taps,pulse_shape)
        return sp.upfirdn(pulse_shape,symbols,up=samples_per_symbol,down=1)

    