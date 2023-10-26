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
    def __init__(self, prf, center_range, velocity, wavelength, antenna_length):
        self.prf = prf
        self.center_range = center_range
        self.velocity = velocity
        self.wavelength = wavelength
        self.antenna_length = antenna_length

        # self.max_offset = 450
        # self.doppler_max = (2 * self.velocity * self.max_offset) / (self.wavelength * self.center_range)
        # self.doppler_bandwidth = 2*self.doppler_max

        self.azimuth_width = (self.center_range * self.wavelength) / self.antenna_length
        self.cycle_time = self.azimuth_width / self.velocity
        self.azimuth_resolution = np.sqrt(self.wavelength * self.center_range)
        # n_pulses should be a power of two
        self.n_pulses = 2 ** np.ceil(np.log2(self.azimuth_width / self.azimuth_resolution))
        self.pulse_time = self.n_pulses / self.prf
        self.pulse_distance = self.pulse_time * self.velocity
        self.pulse_spacing = self.cycle_time - self.pulse_time


data = parse_ers(DATA_PATH / 'ersdata.hw3', n_lines=10_100)
chirp = Chirp(4.189166e11, 37.12e-6, 18.96e6, min_samples=data.shape[1])
spectrum, compressed = matched_filter(data, chirp.chirp)
plot_image(compressed, show=True)
plot_spectral_average(spectrum, 2000, 2512, show=True)
doppler_centroid = calculate_doppler_centroid(compressed, 2000, 2512, 1679.9)
print(doppler_centroid)
sar = UnfocusedSAR(1679.9, 830000, 7550, 0.0566, 10)
