import numpy as np
from scipy import signal
from dataclasses import dataclass
import h5py
import os
from typing import Tuple, List, Optional

@dataclass
class SignalParams:
    frequency: float
    amplitude: float
    phase_shift: float = 1.0
    symbol_rate: float = 10
    constellation_size: int = 4

class SignalGenerator:
    def __init__(self, sample_rate=1000, duration=1.0):
        self.sample_rate = sample_rate
        self.duration = duration
        self.t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        
    def generate_psk(self, params: SignalParams) -> np.ndarray:
        """Generate PSK signal with given parameters"""
        symbols = np.random.choice([0, 1], size=int(self.duration * params.symbol_rate))
        signal = np.zeros_like(self.t)
        samples_per_symbol = int(self.sample_rate / params.symbol_rate)
        
        for i, symbol in enumerate(symbols):
            start_idx = i * samples_per_symbol
            end_idx = (i + 1) * samples_per_symbol
            if end_idx > len(self.t):
                break
            phase = params.phase_shift * np.pi * symbol
            signal[start_idx:end_idx] = params.amplitude * np.cos(
                2 * np.pi * params.frequency * self.t[start_idx:end_idx] + phase
            )
        return signal
    
    def generate_qam(self, params: SignalParams) -> np.ndarray:
        """Generate QAM signal with given parameters"""
        symbols_i = np.random.randint(0, int(np.sqrt(params.constellation_size)), 
                                    size=int(self.duration * params.symbol_rate))
        symbols_q = np.random.randint(0, int(np.sqrt(params.constellation_size)), 
                                    size=int(self.duration * params.symbol_rate))
        
        signal = np.zeros_like(self.t, dtype=complex)
        samples_per_symbol = int(self.sample_rate / params.symbol_rate)
        
        for i, (sym_i, sym_q) in enumerate(zip(symbols_i, symbols_q)):
            start_idx = i * samples_per_symbol
            end_idx = (i + 1) * samples_per_symbol
            if end_idx > len(self.t):
                break
            
            amplitude_i = (2 * sym_i - np.sqrt(params.constellation_size) + 1) * params.amplitude
            amplitude_q = (2 * sym_q - np.sqrt(params.constellation_size) + 1) * params.amplitude
            
            signal[start_idx:end_idx] = (
                amplitude_i * np.exp(1j * 2 * np.pi * params.frequency * self.t[start_idx:end_idx]) +
                amplitude_q * np.exp(1j * (2 * np.pi * params.frequency * self.t[start_idx:end_idx] + np.pi/2))
            )
        
        return np.real(signal)

    def add_jamming(self, signal: np.ndarray, noise_type: str = 'gaussian', 
                   noise_level: float = 0.5) -> np.ndarray:
        """Add jamming noise to the signal"""
        if noise_type == 'gaussian':
            noise = noise_level * np.random.normal(size=signal.shape)
        elif noise_type == 'pulse':
            noise = np.zeros_like(signal)
            pulse_positions = np.random.choice(len(signal), size=int(len(signal)*0.1))
            noise[pulse_positions] = noise_level * np.random.normal(size=len(pulse_positions))
        elif noise_type == 'swept':
            t = np.linspace(0, self.duration, len(signal))
            sweep_freq = np.linspace(0, 100, len(signal))
            noise = noise_level * np.sin(2 * np.pi * sweep_freq * t)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        return signal + noise

def generate_dataset(num_samples: int, output_path: str) -> None:
    """Generate dataset and save to HDF5 file"""
    generator = SignalGenerator()
    
    with h5py.File(output_path, 'w') as f:
        # Create datasets
        f.create_dataset('clean_signals', shape=(num_samples, len(generator.t)))
        f.create_dataset('jammed_signals', shape=(num_samples, len(generator.t)))
        f.create_dataset('labels', shape=(num_samples,))
        
        for i in range(num_samples):
            # Randomly select signal type and parameters
            signal_type = np.random.choice(['psk', 'qam'])
            params = SignalParams(
                frequency=np.random.uniform(10, 200),
                amplitude=np.random.uniform(0.1, 2.0),
                phase_shift=np.random.uniform(0, 2*np.pi),
                symbol_rate=np.random.uniform(5, 20)
            )
            
            # Generate clean signal
            if signal_type == 'psk':
                clean_signal = generator.generate_psk(params)
            else:
                clean_signal = generator.generate_qam(params)
                
            # Add jamming with random parameters
            noise_type = np.random.choice(['gaussian', 'pulse', 'swept'])
            noise_level = np.random.uniform(0.1, 3.0)
            jammed_signal = generator.add_jamming(clean_signal, noise_type, noise_level)
            
            # Store in dataset
            f['clean_signals'][i] = clean_signal
            f['jammed_signals'][i] = jammed_signal
            f['labels'][i] = 1 if noise_level > 1.0 else 0

if __name__ == "__main__":
    # Generate training and validation datasets
    os.makedirs("data", exist_ok=True)
    generate_dataset(1000, "data/train_data.h5")
    generate_dataset(200, "data/val_data.h5")