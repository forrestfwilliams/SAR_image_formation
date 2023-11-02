import numpy as np
from pathlib import Path

from sar_course import processor, utils

DATA_PATH = Path('~/Documents/SAR_theory_course/data').expanduser()


def get_time_difference(observed, test):
    half_length = test.pulse_length / 2
    lower_half = processor.Chirp(
        test.freq_slope, half_length, test.sample_rate, 0 - test.bandwidth / 4, min_samples=observed.total_points
    )
    start_position = int(np.round(half_length * test.sample_rate))
    upper_half = processor.Chirp(
        test.freq_slope,
        half_length,
        test.sample_rate,
        0 + test.bandwidth / 4,
        start_position,
        min_samples=observed.total_points,
    )
    _, lower_compressed = processor.matched_filter(observed.chirp, lower_half.chirp)
    _, upper_compressed = processor.matched_filter(observed.chirp, upper_half.chirp)
    lower_fft_mag = np.fft.fft(np.abs(lower_compressed))
    upper_fft_mag = np.fft.fft(np.abs(upper_compressed))

    cross_correlation = np.fft.fftshift(np.fft.ifft(lower_fft_mag * np.conjugate(upper_fft_mag)))
    pixel_difference = np.argmax(np.abs(cross_correlation)) - (observed.total_points / 2)
    time_difference = pixel_difference / observed.sample_rate
    return time_difference


def calculate_slope_error(time_difference, test_slope, bandwidth):
    return 2 * time_difference * test_slope**2 / bandwidth


def sub_aperture_autofocus(observed, test):
    time_diff = get_time_difference(observed, test)
    slope_error = calculate_slope_error(time_diff, test.freq_slope, test.bandwidth)
    corrected_slope = test.freq_slope + slope_error
    return slope_error, corrected_slope


pulse_length = 1e-5
sample_rate = 1e8
min_samples = 2048

observed = processor.Chirp(1e12, pulse_length, sample_rate, min_samples=min_samples)
test1 = processor.Chirp(1e12, pulse_length, sample_rate, min_samples=min_samples)
test2 = processor.Chirp(1.01e12, pulse_length, sample_rate, min_samples=min_samples)
test3 = processor.Chirp(1.03e12, pulse_length, sample_rate, min_samples=min_samples)
test4 = processor.Chirp(0.98e12, pulse_length, sample_rate, min_samples=min_samples)

for test in [test1, test2, test3, test4]:
    slope_error, slope = sub_aperture_autofocus(observed, test)
    print('Slope error = {:.3e} Hz/s'.format(slope_error))
    print('Corrected slope = {:.2e} Hz/s'.format(slope))

raw = utils.read_alos(DATA_PATH / 'simlband.dat', 2048, 2048, byteswap=True)
utils.plot_img(np.abs(raw), title='Magnitude of the raw image')

signal = processor.Chirp(1e12, 1e-5, 2.4e7, min_samples=2048)
sar = processor.SARPlatform(250, 4_653, 250, 0.25, 2, 23, signal)
_, range_focused = processor.matched_filter(raw, signal.chirp)
range_n_valid = int(signal.chirp.shape[0] - signal.pulse_length * signal.sample_rate + 1)
range_focused = range_focused[:, :range_n_valid].copy()
utils.plot_img(np.abs(range_focused), title='Range-focused Image')
utils.plot_img(np.abs(range_focused), title='Zoom-in', lim=[250, 350, None, None], aspect='auto')

azimuth_transformed = np.fft.fftshift(np.fft.fft(range_focused, axis=0), axes=0)
utils.plot_img(np.abs(azimuth_transformed), title='Transformed zoom-in', lim=[250, 350, None, None], aspect='auto')
