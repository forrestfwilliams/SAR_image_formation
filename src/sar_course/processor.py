import numpy as np

import matplotlib.pyplot as plt

EARTH_RADIUS = 6378 * 1e3
SPEED_OF_LIGHT = 2.99792458e8


class Chirp:
    def __init__(self, freq_slope, pulse_length, sample_rate, freq_center=0, starting_phase=0, min_samples=None):
        self.freq_center = freq_center
        self.freq_slope = freq_slope
        self.pulse_length = pulse_length
        self.sample_rate = sample_rate
        self.min_samples = min_samples

        self.bandwidth = self.freq_slope * self.pulse_length
        self.sample_interval = 1 / self.sample_rate
        self.n_points = self.pulse_length * self.sample_rate
        time = self.sample_interval * np.arange(-self.n_points / 2, self.n_points / 2)

        phase = np.pi * freq_slope * (time**2) + 2 * np.pi * self.freq_center * time + starting_phase
        chirp = np.exp(1j * phase)
        total_points = chirp.shape[0] if min_samples is None else min_samples
        chirp = np.pad(chirp, (0, total_points - chirp.shape[0]))

        self.chirp = chirp
        self.time = self.sample_interval * np.arange(0, total_points)

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


class SARPlatform:
    def __init__(self, prf, start_range, velocity, wavelength, antenna_length, grazing_angle, signal):
        self.prf = prf
        self.start_range = start_range
        self.velocity = velocity
        self.wavelength = wavelength
        self.antenna_length = antenna_length
        self.grazing_angle = np.deg2rad(grazing_angle)
        self.signal = signal

        self.range_n_valid = int(self.signal.chirp.shape[0] - self.signal.pulse_length * self.signal.sample_rate + 1)
        self.slant_range_spacing = SPEED_OF_LIGHT / (2 * self.signal.sample_rate)
        self.slant_range_resolution = SPEED_OF_LIGHT / (2 * self.signal.bandwidth)

        self.center_range = self.start_range + (SPEED_OF_LIGHT * self.range_n_valid / (4 * self.signal.sample_rate))
        altitude_part1 = self.center_range * np.cos(self.grazing_angle)
        altitude_part2 = np.sqrt(EARTH_RADIUS**2 - (self.center_range**2 * (np.sin(self.grazing_angle) ** 2)))
        self.altitude = altitude_part1 - EARTH_RADIUS + altitude_part2
        self.effective_velocity = self.velocity * np.sqrt(EARTH_RADIUS / (self.altitude + EARTH_RADIUS))

        self.earth_center_angle = np.arcsin(self.center_range * np.sin(self.grazing_angle) / EARTH_RADIUS)
        self.flat_earth_incidence = self.grazing_angle + self.earth_center_angle
        self.ground_range_spacing = self.slant_range_spacing / np.sin(self.flat_earth_incidence)
        self.ground_range_resolution = self.slant_range_resolution / np.sin(self.flat_earth_incidence)

    def get_range_centroid(self, doppler_centroid, starting_range):
        offset = doppler_centroid * self.wavelength * starting_range / (2 * self.effective_velocity)
        range_centroid = np.sqrt(starting_range**2 + offset**2)
        return range_centroid

    def get_azimuth_rate(self, range_centroid):
        return -2 * self.effective_velocity**2 / (self.wavelength * range_centroid)

    def get_azimuth_pulse_length(self, range_centroid):
        return range_centroid * self.wavelength / (self.effective_velocity * self.antenna_length)

    def get_azimuth_n_valid(self, patch_size, doppler_centroid, scale_factor):
        max_range = self.start_range + (self.range_n_valid - 1) * self.slant_range_spacing
        max_range_centroid = self.get_range_centroid(doppler_centroid, max_range)
        max_pulse_length = scale_factor * self.get_azimuth_pulse_length(max_range_centroid)
        max_ref_points = int(max_pulse_length * self.prf)
        azimuth_n_valid = int(patch_size - max_ref_points)
        return azimuth_n_valid

    def get_patch_properties(self, n_lines, patch_size, doppler_centroid, scale_factor):
        azimuth_n_valid = self.get_azimuth_n_valid(patch_size, doppler_centroid, scale_factor)
        n_patches = int(np.round(n_lines / azimuth_n_valid))
        out_n_lines = int(np.round(azimuth_n_valid * n_patches))
        return azimuth_n_valid, n_patches, out_n_lines

    def get_multilook_properties(self, patch_size, doppler_centroid, scale_factor):
        n_valid = self.get_azimuth_n_valid(patch_size, doppler_centroid, scale_factor)
        azimuth_spacing = (self.velocity / self.prf) * (EARTH_RADIUS / (EARTH_RADIUS + self.altitude))
        n_looks = int(np.round(self.ground_range_spacing / azimuth_spacing))
        patch_n_lines = int(np.floor(n_valid / n_looks))
        return patch_n_lines, n_valid, n_looks


def matched_filter(signal, reference):
    signal_fft = np.fft.fft(signal)
    reference_fft = np.fft.fft(reference)
    spectrum = signal_fft * np.conjugate(reference_fft)
    compressed = np.fft.ifft(spectrum)
    return spectrum, compressed


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


def create_azimuth_chirp(starting_range, sar, doppler_centroid, min_samples, scale_factor):
    starting_range = float(starting_range)
    range_centroid = sar.get_range_centroid(doppler_centroid, starting_range)
    azimuth_rate = sar.get_azimuth_rate(range_centroid)
    azimuth_pulse_length = 0.8 * sar.get_azimuth_pulse_length(range_centroid)
    chirp = Chirp(azimuth_rate, azimuth_pulse_length, sar.prf, doppler_centroid, min_samples=min_samples).chirp
    return chirp


def azimuth_focus(data, patch_size, sar, doppler_centroid, scale_factor=0.8):
    ranges = np.arange(0, data.shape[1]) * sar.slant_range_spacing + sar.start_range
    chirp_args = [sar, doppler_centroid, patch_size, scale_factor]
    chirps = np.apply_along_axis(create_azimuth_chirp, 0, np.expand_dims(ranges, axis=0), *chirp_args)
    chirp_ffts = np.fft.fft(chirps, axis=0)
    chirp_fft_conjugates = np.conjugate(chirp_ffts)

    azimuth_n_valid, n_patches, out_n_lines = sar.get_patch_properties(
        data.shape[0], patch_size, doppler_centroid, scale_factor
    )

    padded_n_lines = int((azimuth_n_valid * (n_patches - 1)) + patch_size) - data.shape[0]
    padded = np.pad(data, [(0, padded_n_lines), (0, 0)])
    focused = np.zeros((out_n_lines, sar.range_n_valid), dtype=np.complex128)
    for i in range(n_patches):
        start_line = i * azimuth_n_valid
        patch_stop_line = start_line + patch_size
        focused_stop_line = (i + 1) * azimuth_n_valid
        patch = padded[start_line:patch_stop_line, :].copy()
        patch_fft = np.fft.fft(patch, axis=0)
        patch_focused = np.fft.ifft(patch_fft * chirp_fft_conjugates, axis=0)
        focused[start_line:focused_stop_line, :] = patch_focused[:azimuth_n_valid, :]

    return focused, n_patches


def azimuth_multilook(data, n_patches, patch_n_lines, n_valid, n_looks):
    multilooked = np.zeros((n_patches * patch_n_lines, data.shape[1]), dtype=float)
    for k in range(n_patches):
        for j in np.arange(patch_n_lines):
            patch_start = k * n_valid
            patch = data[patch_start + j * n_looks : patch_start + (j + 1) * n_looks, :]
            multilooked[j + k * patch_n_lines, :] = np.mean(np.abs(patch), axis=0)
    return multilooked
