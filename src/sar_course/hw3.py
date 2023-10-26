from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from sar_course.processor import Chirp, matched_filter, parse_ers

DATA_PATH = Path('~/Documents/SAR_theory_course/data').expanduser()


def get_spectral_average(data, min, max):
    average = np.abs(np.fft.fftshift(data[:, min:max])).mean(axis=1)
    return average


def find_highest_peak(range_average):
    peaks, _ = find_peaks(range_average)
    peak_heights = [range_average[loc] for loc in peaks]
    max_index = np.array(peak_heights).argmax()
    peak_point = (peaks[max_index], peak_heights[max_index])
    return peak_point


def plot_spectral_average(data, min, max, show=False):
    average = get_spectral_average(data, min, max)
    peak_point = find_highest_peak(average)
    f, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(list(range(average.shape[0])), average, zorder=1)
    ax.scatter(peak_point[0], peak_point[1], color='red', zorder=2)
    if show:
        plt.show()
    plt.close('all')


def plot_image(image, show=False):
    f, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(np.abs(image), cmap='magma', vmin=0, vmax=600)
    plt.tight_layout()
    if show:
        plt.show()
    plt.close('all')


def calculate_doppler_centroid(data, min, max, prf):
    average = get_spectral_average(data, min, max)
    peak_point = find_highest_peak(average)
    delta_freq = prf / average.shape[0]
    doppler_centroid = peak_point[0] * delta_freq
    return doppler_centroid


class UnfocusedSAR:
    def __init__(self, prf, center_range, velocity, wavelength, antenna_length, min_pulses=64):
        self.prf = prf
        self.center_range = center_range
        self.velocity = velocity
        self.wavelength = wavelength
        self.antenna_length = antenna_length

        self.azimuth_width = (self.center_range * self.wavelength) / self.antenna_length
        self.cycle_time = self.azimuth_width / self.velocity
        self.azimuth_resolution = np.sqrt(self.wavelength * self.center_range)

        self.max_sight_distance = self.azimuth_width / 2
        self.max_doppler = (2 * self.velocity * self.max_sight_distance) / (self.wavelength * self.center_range)
        self.doppler_prf = self.max_doppler * 2

        # n_pulses should be a power of two
        self.n_pulses = max(min_pulses, int(2 ** np.ceil(np.log2(self.azimuth_width / self.azimuth_resolution))))
        self.pulse_time = self.n_pulses / self.prf
        self.pulse_distance = self.pulse_time * self.velocity
        self.pulse_spacing = self.cycle_time - self.pulse_time

        self.pulse_delta_frequency = self.prf / self.n_pulses
        self.pulse_shift_distance = (self.pulse_delta_frequency * self.wavelength * self.center_range) / (
            2 * self.velocity
        )
        self.pulse_shift_pixels = int(np.ceil(self.pulse_distance / self.pulse_shift_distance))

    def __repr__(self):
        return (
            f'UnfocusedSAR(prf={self.prf}, center_range={self.center_range}, velocity={self.velocity}, '
            f'wavelength={self.wavelength}, antenna_length={self.antenna_length} '
            f'azimuth_width={self.azimuth_width}, cycle_time={self.cycle_time}, '
            f'azimuth_resolution={self.azimuth_resolution}, n_pulses={self.n_pulses}, '
            f'pulse_time={self.pulse_time}, pulse_distance={self.pulse_distance}, '
            f'pulse_spacing={self.pulse_spacing})'
        )


def azimuth_defocus(patch, prf, doppler_centroid):
    defocused_patch = np.zeros_like(patch)
    for i in range(patch.shape[1]):
        shifted_line = patch[:, i] * np.exp(-1j * 2 * np.pi * doppler_centroid * i / prf)
        frequency_line = np.fft.fft(shifted_line)
        shifted_frequency_line = np.fft.fftshift(frequency_line)
        defocused_patch[:, i] = shifted_frequency_line
    return defocused_patch


def focus(sar, chirp, data):
    spectrum, compressed = matched_filter(data, chirp.chirp)
    # plot_spectral_average(spectrum, 2000, 2512, show=True)
    doppler_centroid = calculate_doppler_centroid(spectrum, 2000, 2512, sar.prf)
    n_patches = int(np.floor(compressed.shape[0] / sar.n_pulses))
    unfocused = np.zeros_like(compressed)
    for i in range(n_patches):
        start = i * sar.n_pulses
        stop = start + sar.n_pulses
        patch = compressed[start:stop, :].copy()
        unfocused_patch = azimuth_defocus(patch, sar.prf, doppler_centroid)
        unfocused[start:stop, :] = unfocused_patch


# data = parse_ers(DATA_PATH / 'ersdata.hw3', n_lines=10_100)
# chirp = Chirp(4.189166e11, 37.12e-6, 18.96e6, min_samples=data.shape[1])
# spectrum, compressed = matched_filter(data, chirp.chirp)
# plot_image(compressed, show=True)
# plot_spectral_average(spectrum, 2000, 2512, show=True)
# doppler_centroid = calculate_doppler_centroid(compressed, 2000, 2512, 1679.9)
# print(doppler_centroid)
# sar = UnfocusedSAR(1679.9, 830000, 7550, 0.0566, 10)


data = parse_ers(DATA_PATH / 'ersdata.hw3', n_lines=10_100)
chirp = Chirp(4.189166e11, 37.12e-6, 18.96e6, min_samples=data.shape[1])
sar = UnfocusedSAR(1679.9, 830000, 7550, 0.0566, 10)
out = focus(sar, chirp, data)
