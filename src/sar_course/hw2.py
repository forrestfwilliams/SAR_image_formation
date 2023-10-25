from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt

ERS_HEADER_LENGTH = 412
DATA_PATH = Path('~/Documents/SAR_theory_course/data').expanduser()


def range_compress(signal, chirp, output_domain='time'):
    convolved = np.fft.fft(signal) * np.conj(np.fft.fft(chirp))
    if output_domain == 'time':
        convolved = np.fft.ifft(convolved)
    return convolved


class Chirp:
    def __init__(self, freq_center, freq_slope, pulse_length, sample_rate, min_samples=0):
        self.freq_center = freq_center
        self.freq_slope = freq_slope
        self.pulse_length = pulse_length
        self.sample_rate = sample_rate
        self.sample_spacing = 1 / sample_rate
        self.min_samples = min_samples
        self.n_valid = int(self.pulse_length * self.sample_rate)

        start_time = -self.pulse_length / 2
        end_time = self.pulse_length / 2
        self.time = np.linspace(start_time, end_time, self.n_valid, endpoint=False)
        chirp = np.exp(0 - 1j * (np.pi * self.freq_slope * self.time**2 + 2 * np.pi * self.freq_center * self.time))
        n_missing = min_samples - chirp.shape[0]
        if n_missing > 0:
            zero_fill = np.zeros((n_missing), dtype=np.complex64)
            chirp = np.append(chirp, zero_fill)
            added_time = np.linspace(end_time, end_time + (n_missing * self.sample_spacing), n_missing, endpoint=False)
            self.time = np.append(self.time, added_time)

        self.chirp = chirp
        self.n_samples = self.chirp.shape[0]
        self.total_time = (self.n_samples / self.n_valid) * self.pulse_length

    def plot(self, show=False):
        chirp_power = np.fft.fftshift(np.fft.fft(self.chirp))
        chirp_power_db = 20 * np.log10(chirp_power)
        f, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.plot(self.time, chirp_power_db)
        ax.set(xlabel='Pulse Time (s)', ylabel='Power (dB)')
        if show:
            plt.show()
        else:
            plt.savefig('../../assets/chirp_spectrum.png')
        plt.close('all')

    def auto_convolve(self):
        # convolved = np.convolve(self.chirp, np.conj(self.chirp[::-1]), mode='full')
        # convolved = np.fft.ifft(np.fft.fft(self.chirp) * np.fft.fft(np.conj(self.chirp[::-1])))
        return range_compress(self.chirp, self.chirp)


def plot_auto_convolve(chirp, show=False):
    con_power = np.fft.fftshift(chirp.auto_convolve())
    con_power_db = 20 * np.log10(con_power)

    f, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(list(range(len(con_power_db))), con_power_db)
    ax.set(xlabel='', ylabel='Power (dB)')
    if show:
        plt.show()
    else:
        ax.set_xlim(965, 1031)
        ax.set_ylim(0, 65)
        plt.savefig('../../assets/convolved_spectrum.png')
    plt.close('all')


def make_multi_chirp(show=False):
    base_chirp = Chirp(1.5e9, 1e12, 1e-5, 1e8, min_samples=0)
    signal = np.zeros(2048, dtype=np.complex64)
    signal[100 : 100 + base_chirp.chirp.shape[0]] += 1 * base_chirp.chirp
    signal[400 : 400 + base_chirp.chirp.shape[0]] += 5 * base_chirp.chirp
    signal[500 : 500 + base_chirp.chirp.shape[0]] += 2 * base_chirp.chirp

    convolved = np.convolve(signal, np.conj(base_chirp.chirp[::-1]), mode='full')

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    ax1.plot(list(range(len(signal))), signal)
    ax2.plot(list(range(len(convolved))), convolved)
    if show:
        plt.show()
    else:
        plt.savefig('../../assets/range_compression.png')
    plt.close('all')
    return signal


def parse_ers_line(line):
    digital_number = np.frombuffer(line[412:], dtype=np.int8).astype(float)
    digital_number -= 15.5
    data = digital_number[::2] + 1j * digital_number[1::2]
    return data


def parse_ers(file_path, n_lines=1024, n_samples=4903):
    ers_data = np.zeros((n_lines, n_samples), dtype=np.csingle)
    with open(file_path, 'rb') as f:
        for i in range(n_lines):
            # each complex number takes 2 bytes to represent
            line = f.read(ERS_HEADER_LENGTH + n_samples * 2)
            row = parse_ers_line(line)
            ers_data[i, :] = row
    return ers_data


def plot_spectral_average(data):
    average = np.real(data).mean(axis=0)
    f, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(list(range(average.shape[0])), average)
    ax.set(xlabel='Range', ylabel='Power (dB)')
    plt.show()
    plt.close('all')


def range_compress_image(image, chirp):
    range_compressed = np.zeros_like(image)
    for i in range(image.shape[0]):
        range_compressed[i, :] = np.fft.fftshift(range_compress(image[i, :], chirp))
    return range_compressed


def plot_image(image):
    f, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(20 * np.log10(np.abs(image)), cmap='gray')
    plt.show()
    plt.close('all')


if __name__ == '__main__':
    show = True

    chirp = Chirp(0, 1e12, 1e-5, 1e8, 2048)
    chirp.plot(show=show)
    plot_auto_convolve(chirp, show=show)
    make_multi_chirp(show=show)

    data = parse_ers(DATA_PATH / 'ersdata.dat')
    chirp = Chirp(0, 4.189166e11, 37.12e-6, 18.96e6, min_samples=data.shape[1])
    chirp.plot(show=show)
    plot_spectral_average(data)
    range_compressed = range_compress_image(data, chirp.chirp)
    plot_image(range_compressed)
