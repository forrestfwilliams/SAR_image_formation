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


def parse_ers_line(line, pad_to=None):
    digital_number = np.frombuffer(line[412:], dtype=np.int8).astype(float)
    digital_number -= 15.5
    data = digital_number[::2] + 1j * digital_number[1::2]
    if pad_to is not None:
        padded = np.zeros((pad_to), dtype=np.complex64)
        padded[: data.shape[0]] = data.copy()
        data = padded.copy()
    return data


def parse_ers(file_path, n_lines, n_samples=4903, pad_to=0):
    ers_data = np.zeros((n_lines, max(n_samples, pad_to)), dtype=np.csingle)
    with open(file_path, 'rb') as f:
        for i in range(n_lines):
            # each complex number takes 2 bytes to represent
            line = f.read(ERS_HEADER_LENGTH + n_samples * 2)
            row = parse_ers_line(line, pad_to=ers_data.shape[1])
            ers_data[i, :] = row
    return ers_data
