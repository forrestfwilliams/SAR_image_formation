from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt

ERS_HEADER_LENGTH = 412
DATA_PATH = Path('~/Documents/SAR_theory_course/data').expanduser()


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

        # start_time = -self.pulse_length / 2
        # end_time = self.pulse_length / 2
        # self.time = np.linspace(start_time, end_time, self.n_valid, endpoint=False)
        # chirp = np.exp(0 - 1j * (np.pi * self.freq_slope * self.time**2 + 2 * np.pi * self.freq_center * self.time))
        # n_missing = min_samples - chirp.shape[0]
        # if n_missing > 0:
        #     zero_fill = np.zeros((n_missing), dtype=np.complex64)
        #     chirp = np.append(chirp, zero_fill)
        #     added_time = np.linspace(end_time, end_time + (n_missing * self.sample_interval), n_missing, endpoint=False)
        #     self.time = np.append(self.time, added_time)
        #
        # self.chirp = chirp
        # self.n_samples = self.chirp.shape[0]
        # self.total_time = (self.n_samples / self.n_valid) * self.pulse_length

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


def plot_auto_convolve(chirp, show=False):
    _, compressed = matched_filter(chirp.chirp, chirp.chirp)
    compressed_power_db = 20 * np.log10(np.fft.ifftshift(np.abs(compressed)))

    f, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(chirp.time, compressed_power_db)
    ax.set(xlabel='', ylabel='Power (dB)')
    if show:
        plt.show()
    else:
        ax.set_xlim(965, 1031)
        ax.set_ylim(0, 65)
        plt.savefig('../../assets/convolved_spectrum.png')
    plt.close('all')


def make_multi_chirp(show=False):
    base_chirp = Chirp(1e12, 1e-5, 1e8)
    signal = np.zeros(2048, dtype=np.complex64)
    signal[100 : 100 + base_chirp.chirp.shape[0]] += 1 * base_chirp.chirp
    signal[400 : 400 + base_chirp.chirp.shape[0]] += 5 * base_chirp.chirp
    signal[500 : 500 + base_chirp.chirp.shape[0]] += 2 * base_chirp.chirp
    
    padded_chirp = Chirp(1e12, 1e-5, 1e8, min_samples=2048)
    _, compressed = matched_filter(signal, padded_chirp.chirp)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    ax1.plot(padded_chirp.time, signal)
    ax2.plot(padded_chirp.time, compressed)
    if show:
        plt.show()
    else:
        plt.savefig('../../assets/range_compression.png')
    plt.close('all')
    return signal


def parse_ers_line(line, pad_to=None):
    digital_number = np.frombuffer(line[412:], dtype=np.int8).astype(float)
    digital_number -= 15.5
    data = digital_number[::2] + 1j * digital_number[1::2]
    if pad_to is not None:
        padded = np.zeros((pad_to), dtype=np.complex64)
        padded[: data.shape[0]] = data.copy()
        data = padded.copy()
    return data


def parse_ers(file_path, n_lines=1024, n_samples=4903, pad_to=8192):
    ers_data = np.zeros((n_lines, pad_to), dtype=np.csingle)
    with open(file_path, 'rb') as f:
        for i in range(n_lines):
            # each complex number takes 2 bytes to represent
            line = f.read(ERS_HEADER_LENGTH + n_samples * 2)
            row = parse_ers_line(line, pad_to=pad_to)
            ers_data[i, :] = row
    return ers_data


def plot_spectral_average(data, show=False):
    average = np.abs(np.fft.fft(data)).mean(axis=0)
    f, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(list(range(average.shape[0])), 20 * np.log10(average))
    ax.set(xlabel='Range', ylabel='Power (dB)')
    if show:
        plt.show()
    plt.close('all')


def plot_image(image, show=False):
    f, ax = plt.subplots(1, 1, figsize=(20, 6))
    ax.imshow(np.abs(image), cmap='magma', vmin=0, vmax=600)
    plt.tight_layout()
    if show:
        plt.show()
    plt.close('all')


if __name__ == '__main__':
    show = True

    chirp = Chirp(1e12, 1e-5, 1e8, min_samples=2048)
    chirp.plot(show=show)
    plot_auto_convolve(chirp, show=show)
    make_multi_chirp(show=show)

    data = parse_ers(DATA_PATH / 'ersdata.dat')
    chirp = Chirp(4.189166e11, 37.12e-6, 18.96e6, min_samples=data.shape[1])
    chirp.plot(show=show)
    plot_spectral_average(data, show=show)
    _, compressed = matched_filter(data, chirp.chirp)
    plot_image(compressed, show=show)
