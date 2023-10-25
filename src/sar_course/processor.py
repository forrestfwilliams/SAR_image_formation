from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt

ERS_HEADER_LENGTH = 412


class Chirp:
    def __init__(self, freq_slope, pulse_length, sample_rate, freq_center=0, starting_phase=0, min_samples=0):
        self.freq_center = freq_center
        self.freq_slope = freq_slope
        self.pulse_length = pulse_length
        self.sample_rate = sample_rate
        self.min_samples = min_samples
        self.sample_interval = 1 / sample_rate
        self.n_points = int(self.pulse_length * self.sample_rate)

        time = self.sample_interval * np.arange(-self.n_points / 2, self.n_points / 2)
        phase = np.pi * freq_slope * (time**2) + 2 * np.pi * freq_center * time + starting_phase
        chirp = np.exp(1j * phase)
        n_missing = min_samples - chirp.shape[0]
        if n_missing > 0:
            chirp = np.pad(chirp, (0, n_missing))
        self.chirp = chirp
        self.time = self.sample_interval * np.arange(0, self.chirp.shape[0])

    def plot(self, show=False):
        chirp_power = np.abs(np.fft.fftshift(np.fft.fft(self.chirp)))
        chirp_power_db = 20 * np.log10(chirp_power)
        f, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.plot(self.time, chirp_power_db)
        ax.set(xlabel='Pulse Time (s)', ylabel='Power (dB)')
        if show:
            plt.show()
        else:
            plt.savefig('../../assets/chirp_spectrum.png')
        plt.close('all')


def matched_filter(signal, reference):
    signal_fft = np.fft.fft(signal)
    reference_fft = np.fft.fft(reference)
    spectrum = signal_fft * np.conjugate(reference_fft)
    compressed = np.fft.ifft(spectrum)
    return spectrum, compressed
