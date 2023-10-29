from pathlib import Path
import numpy as np

from scipy.signal import find_peaks

from sar_course.processor import Chirp, matched_filter, parse_ers
from sar_course.utils import plot_img, plot_freq, to_mag_db

DATA_PATH = Path('~/Documents/SAR_theory_course/data').expanduser()


def get_spectral_average(data, min, max):
    average = np.abs(np.fft.fft(data[:, min:max], axis=0)).mean(axis=1)
    return average


def find_highest_peak(range_average):
    peaks, _ = find_peaks(range_average)
    peak_heights = [range_average[loc] for loc in peaks]
    max_index = np.array(peak_heights).argmax()
    peak_point = (peaks[max_index], peak_heights[max_index])
    return peak_point


def doppler_phase_shift(range_focused, prf):
    n_azimuth, n_range = range_focused.shape
    psRg = np.zeros(n_range, dtype=np.complex128)
    for i in np.arange(2, n_azimuth):
        psRg += range_focused[i, :] * np.conjugate(range_focused[i - 1, :])
    # compute phase shift at each range bin
    phi = np.arctan(np.imag(psRg) / np.real(psRg))
    phi_avg = np.mean(phi)
    fd = prf * phi_avg / (2 * np.pi)
    return fd, phi_avg


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

        self.azimuth_width = (self.center_range * self.wavelength) / self.antenna_length
        self.repeat_cycle_time = self.azimuth_width / self.velocity
        self.azimuth_resolution = np.sqrt(self.wavelength * self.center_range)

        self.pulse_spacing = self.velocity / self.prf
        self.min_n_pulses = self.azimuth_resolution / self.pulse_spacing

        # n_pulses should be a power of two
        self.n_pulses = int(2 ** np.ceil(np.log2(self.min_n_pulses)))
        self.burst_length_time = self.n_pulses / self.prf
        self.frequency_resolution = self.prf / self.n_pulses
        self.spatial_resolution = self.frequency_resolution * self.center_range * self.wavelength / (2 * self.velocity)

    def get_patch_spacings(self, n_lines):
        patch_pixel_shift = self.pulse_spacing * self.n_pulses / self.spatial_resolution
        n_patches = int(n_lines / self.n_pulses)
        out_n_azimuth_pixels = int(self.n_pulses + (n_patches - 1) * patch_pixel_shift)
        return patch_pixel_shift, n_patches, out_n_azimuth_pixels

    def __repr__(self):
        return (
            f'UnfocusedSAR(prf={self.prf}, center_range={self.center_range}, velocity={self.velocity}, '
            f'wavelength={self.wavelength}, antenna_length={self.antenna_length} '
            f'azimuth_width={self.azimuth_width}, repeat_cycle_time={self.repeat_cycle_time}, '
            f'azimuth_resolution={self.azimuth_resolution}, n_pulses={self.n_pulses})'
        )


def apply_phase_shift(data, phase_shift, prf):
    rcompfd = np.zeros(data.shape, dtype=np.complex128)

    for i in range(data.shape[0]):
        rcompfd[i, :] = data[i, :] * np.exp(-1j * 2 * np.pi * fd * i / prf)
    return rcompfd


def multi_range(data, nlk):
    bin_azimuth = int(data.shape[0])
    bin_range = int(data.shape[1] / nlk)
    data_out = np.zeros([bin_azimuth, bin_range])
    for i in range(bin_range):
        data_out[:, i] = np.mean(data[:, i * nlk : (i + 1) * nlk], axis=1)
    return data_out


def azimuth_defocus(data, n_patches, n_az_pixels, n_pulses, patch_pixel_shift):
    bins_range = data.shape[1]
    unfocused = np.zeros([n_az_pixels, bins_range], dtype=np.complex128)
    count_array = np.zeros([n_az_pixels, bins_range], dtype=np.complex128)
    for i in np.arange(n_patches):
        shift = int(np.round(i * patch_pixel_shift))
        patch = data[i * n_pulses : (i + 1) * n_pulses, :]
        unfocused_patch = np.abs(np.fft.fftshift(np.fft.fft(patch, axis=0), axes=0)) ** 2
        unfocused[shift : shift + n_pulses, :] += unfocused_patch
        count_array[shift : shift + n_pulses, :] += 1
    unfocused /= count_array
    return unfocused


sar = UnfocusedSAR(1679.9, 830000, 7550, 0.0566, 10)
data = parse_ers(DATA_PATH / 'ersdata.hw3', n_lines=10_100, pad_to=0)
chirp = Chirp(4.189166e11, 37.12e-6, 18.96e6, min_samples=data.shape[1])

spectrum, compressed = matched_filter(data, chirp.chirp)
n_valid = 4903 - int(chirp.sample_rate * chirp.pulse_length)
compressed = compressed[:, :n_valid].copy()
plot_img(np.abs(compressed), vlim=[0, 2000], title='Magnitude of range compressed image')

fd, _ = doppler_phase_shift(compressed, sar.prf)
corrected = apply_phase_shift(compressed, fd, sar.prf)

az_db = to_mag_db(np.mean(np.abs(np.fft.fftshift(np.fft.fft(corrected, axis=0))), axis=1))
plot_freq(list(range(az_db.shape[0])), az_db, 'Average Aximuth Spectrum after Doppler Centroid Correction', unit='Hz')
patch = corrected[: sar.n_pulses, :]
unfocused_patch = np.abs(np.fft.fftshift(np.fft.fft(patch, axis=0), axes=0)) ** 2
multilooked_patch = multi_range(unfocused_patch, 4)
plot_img(multilooked_patch, title='First Patch, range compressed & unfocused in azimuth', vlim=[0, 2e7])

patch_pixel_shift, n_patches, n_azimuth_pixels = sar.get_patch_spacings(10100)
unfocused = azimuth_defocus(corrected, n_patches, n_azimuth_pixels, sar.n_pulses, patch_pixel_shift)
unfocused_multilook = multi_range(unfocused, 4)
plot_img(unfocused_multilook, title='Multi-looked uncofused image', vlim=[0, 2e7])
