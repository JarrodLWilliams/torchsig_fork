from typing import Any, Callable, List, Literal, Optional, Tuple, Union, Dict
import warnings
import numpy as np
from scipy import signal as sp
from torchsig.utils.dsp import convolve, gaussian_taps, low_pass, rrc_taps, irrational_rate_resampler

__all__  = ["RRCPulseShapeFilter", "GaussianPulseShapeFilter", "RectangularPulseShapeFilter"]

class AbstractPulseShapeFilter:
    """An abstract class representing a Pulse Shape Filter that can either work on
    targets or data

    """

    def __init__(self, name:str, bandwidth_mode=['fixed', 'variable'], bandwidth=None) -> None:
        warnings.warn(
            "Pulse Shape Filters only currently implemented for FSK constellations.", DeprecationWarning
        )
        self.string = self.__class__.__name__ + "()"
        self.random_generator = np.random.default_rng()
        self.name = None
        self.bandwidth_mode = bandwidth_mode
        self._bandwidth = bandwidth

    def __call__(self, symbols: Any, samples_per_symbol: int, bandwidth: Optional[float]=None) -> Any:
        match self.bandwidth_mode:
            case 'fixed':
                if self.bandwidth is None:
                    raise ValueError('Bandwidth mode is fixed and no bandwidth value was given.')
                return self._filter(symbols=symbols, samples_per_symbol=samples_per_symbol, bandwidth=self.bandwidth)
            case 'variable':
                return self._filter(symbols=symbols, samples_per_symbol=samples_per_symbol, bandwidth=bandwidth)
    
    def _filter(self, symbols: Any, samples_per_symbol: int, bandwidth: float):
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.string
    
    def name(self):
        return self.name
    
    @property
    def bandwidth(self):
        return self._bandwidth
    
    @bandwidth.setter
    def bandwidth(self, bandwidth):
        self._bandwidth = bandwidth
    
class AbstractCompositePulseShapeFilter:
    """An abstract class representing a Pulse Shape Filter that can either work on
    targets or data

    """

    def __init__(self, filters: Dict[str,AbstractPulseShapeFilter]) -> None:
        warnings.warn(
            "Pulse Shape Filters only currently implemented for FSK constellations.", DeprecationWarning
        )
        self.filters = filters
        self.name = None

    def __call__(self, cls:str|int, symbols: Any, samples_per_symbol: int, bandwidth: float) -> Any:
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.string
    
    def name(self):
        return self.name
    

    
class RectangularPulseShapeFilter(AbstractPulseShapeFilter):

    def __init__(self, name='Rectangular', bandwidth_mode='variable', bandwidth=None):
        super().__init__(name, bandwidth_mode, bandwidth)
        self.name = name
        

    def _filter(self, symbols: Any, samples_per_symbol: int, bandwidth: float) -> Any:
        # upfirdn upsamples, applies the FIR filter, and then downsamples. The upsampling rate depends on the samples per symbol in the IQ (which is recalculated for FSK in the data generation phase, and the downsampling rate is fixed at one.)

        # Upsampling corresponds to expanding the symbol-sequence (IQ data) where the original samples are separated by sampler_per_symbol zeros. Then a low-pass filter is applied, which interpolates the zeros. The low-pass filter is equivalent to multiplying the signal by a rectangular function in the frequency domain.
        
        # Bandwidth is in the function input to match the abstract syntax but is not 
        # required for the rectangular pulse shape.

        filter = np.ones(samples_per_symbol)
        return sp.upfirdn(filter, symbols, up=samples_per_symbol, down=1)
    

    
class GaussianPulseShapeFilter(AbstractPulseShapeFilter):

    def __init__(self, name='Gaussian', bandwidth_mode='variable', bandwidth=None):
        super().__init__(name=name, bandwidth_mode=bandwidth_mode, bandwidth=bandwidth)
        self.name = name
        

    def __call__(self, symbols: Any, samples_per_symbol: int, bandwidth: float) -> Any:
            pulse_shape = np.ones(samples_per_symbol)
            taps = gaussian_taps(samples_per_symbol, bandwidth)
            pulse_shape = np.convolve(taps,pulse_shape)
            return sp.upfirdn(pulse_shape,symbols,up=samples_per_symbol,down=1)

class RRCPulseShapeFilter(AbstractPulseShapeFilter):

    def __init__(self, name="rrc", size_in_symbols=4, bandwidth_mode='variable', bandwidth=None):
        super().__init__(name, bandwidth_mode, bandwidth)
        self.name = name
        self.size_in_symbols = size_in_symbols

    def _filter(self, symbols: Any, samples_per_symbol: int, bandwidth: float) -> Any:
        pulse_shape = np.ones(samples_per_symbol)
        taps = rrc_taps(samples_per_symbol, self.size_in_symbols, bandwidth)
        pulse_shape = np.convolve(taps,pulse_shape)
        return sp.upfirdn(pulse_shape,symbols,up=samples_per_symbol,down=1)
    
if __name__ == '__main__':
    filter = GaussianPulseShapeFilter(bandwidth_mode='fixed', bandwidth=5)
    print(filter)