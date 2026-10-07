import numpy as np
from scipy import signal

class StreamingCausalEEGFilter:
    """
    Streaming causal EEG bandpass filter using Second-Order Sections (SOS).
    
    Replaces non-causal offline `filtfilt` with a causal IIR filter that maintains
    persistent filter states (zi/zf) across streaming chunks.
    
    Default parameters match the verified AAD pipeline:
    - 2nd-order (4-pole) Butterworth bandpass from 1.0 to 6.0 Hz at fs=64 Hz.
    - Group delay in 1-6 Hz passband is ~6.47 samples (~101 ms), which aligns
      directly within the CA-TCN cross-correlation window (±8 samples / ±125 ms).
    """
    def __init__(self, lowcut: float = 1.0, highcut: float = 6.0, fs: float = 64.0, 
                 order: int = 2, n_channels: int = 8):
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order
        self.n_channels = n_channels
        
        # Design SOS filter
        self.sos = signal.butter(order, [lowcut, highcut], btype='bandpass', fs=fs, output='sos')
        self.n_sections = self.sos.shape[0]
        
        # Initialize filter state for each section and channel: [n_sections, 2, n_channels]
        self.zi = np.zeros((self.n_sections, 2, n_channels), dtype=np.float64)
        self.is_initialized = False

    def reset(self):
        """Resets the persistent filter state to zero."""
        self.zi = np.zeros((self.n_sections, 2, self.n_channels), dtype=np.float64)
        self.is_initialized = False

    def initialize_with_dc(self, initial_values: np.ndarray):
        """
        Initializes filter state to the steady-state response of the initial input DC offset.
        initial_values: [n_channels] or [1, n_channels]
        """
        initial_values = np.asarray(initial_values).reshape(-1)
        base_zi = signal.sosfilt_zi(self.sos) # shape: [n_sections, 2]
        # Multiply by initial per-channel value: [n_sections, 2, n_channels]
        self.zi = base_zi[:, :, np.newaxis] * initial_values[np.newaxis, np.newaxis, :]
        self.is_initialized = True

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """
        Processes a streaming chunk of multichannel EEG data.
        
        Parameters:
            chunk: np.ndarray of shape [n_samples, n_channels]
        Returns:
            filtered_chunk: np.ndarray of shape [n_samples, n_channels]
        """
        if chunk.ndim == 1:
            chunk = chunk[:, np.newaxis]
            
        n_samples, channels = chunk.shape
        if channels != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {channels}")
            
        if not self.is_initialized and n_samples > 0:
            # Auto-initialize with the first sample to mitigate startup transient
            self.initialize_with_dc(chunk[0])

        # sosfilt with axis=0 operates down the time dimension
        filtered_chunk, self.zi = signal.sosfilt(self.sos, chunk.astype(np.float64), axis=0, zi=self.zi)
        return filtered_chunk.astype(np.float32)

    def get_group_delay_samples(self) -> float:
        """Computes the mean group delay within the passband in samples."""
        w, gd = signal.group_delay(signal.sos2tf(self.sos), fs=self.fs)
        mask = (w >= self.lowcut) & (w <= self.highcut)
        if np.any(mask):
            return float(np.mean(gd[mask]))
        return float(np.mean(gd))

    def get_group_delay_ms(self) -> float:
        """Computes the mean group delay in milliseconds."""
        return self.get_group_delay_samples() * 1000.0 / self.fs


class DualBandCausalEEGFilter:
    """
    Simultaneous Dual-Band Streaming Causal EEG Filter.
    Extracts:
      1. ERP Phase Tracking Band (default: 1.0 - 6.5 Hz): captures slow cortical auditory potentials.
      2. Alpha Lateralization Band (default: 8.0 - 13.0 Hz): captures attentional parietal-temporal alpha suppression.
    
    Both sub-filters maintain independent causal SOS filter states (zi/zf) across chunks.
    """
    def __init__(
        self,
        erp_lowcut: float = 1.0,
        erp_highcut: float = 6.5,
        alpha_lowcut: float = 8.0,
        alpha_highcut: float = 13.0,
        fs: float = 64.0,
        order: int = 2,
        n_channels: int = 8
    ):
        self.erp_filter = StreamingCausalEEGFilter(
            lowcut=erp_lowcut, highcut=erp_highcut, fs=fs, order=order, n_channels=n_channels
        )
        self.alpha_filter = StreamingCausalEEGFilter(
            lowcut=alpha_lowcut, highcut=alpha_highcut, fs=fs, order=order, n_channels=n_channels
        )
        self.n_channels = n_channels
        self.fs = fs

    def reset(self):
        """Resets persistent filter states for both bands."""
        self.erp_filter.reset()
        self.alpha_filter.reset()

    def process_chunk(self, chunk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Processes streaming multichannel EEG chunk.
        Returns:
            erp_chunk: [n_samples, n_channels] (1.0 - 6.5 Hz)
            alpha_chunk: [n_samples, n_channels] (8.0 - 13.0 Hz)
        """
        erp_c = self.erp_filter.process_chunk(chunk)
        alpha_c = self.alpha_filter.process_chunk(chunk)
        return erp_c, alpha_c

