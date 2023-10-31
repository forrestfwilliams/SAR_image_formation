import numpy as np
from pathlib import Path

from sar_course import processor, utils

DATA_PATH = Path('~/Documents/SAR_theory_course/data').expanduser()

signal_chirp = processor.Chirp(4.189166e11, 37.12e-6, 18.96e6, min_samples=4903)
sar = processor.SARPlatform(1679.9, 830_000, 7550, 0.0566, 10, 23, signal_chirp)

print(f'Valid range length: {sar.range_n_valid} samples')

min_fft_size = int(2 ** np.ceil(np.log2(sar.range_n_valid)))
print(f'Minimum fft size: {min_fft_size} samples')

print(f'v_eff is: {sar.effective_velocity:.2f} m/s')
print(f'Slant range resolution: {sar.slant_range_resolution:.2f}')
print(f'Ground range resolution: {sar.ground_range_resolution:.2f}')

data = utils.parse_ers(DATA_PATH / 'ersdata.hw3', n_lines=10_100, pad_to=0)
spectrum, compressed = processor.matched_filter(data, sar.signal.chirp)
compressed = compressed[:, : sar.range_n_valid].copy()

doppler_center_phase_shift, _ = processor.doppler_phase_shift(compressed, sar.prf)
print(f'Dopper Centroid (phase shift method): {doppler_center_phase_shift}')

focused, n_patches = processor.azimuth_focus(compressed, 2048, sar, doppler_center_phase_shift)
utils.plot_img(
    np.abs(focused), title='Focussed image', vlim=[0, 3e4], origin='lower', aspect=0.26, interpolation='antialiased'
)

patch_n_lines, n_valid, n_looks = sar.get_multilook_properties(2048, doppler_center_phase_shift, 0.8)
multilook = processor.azimuth_multilook(focused, n_patches, patch_n_lines, n_valid, n_looks)
utils.plot_img(
    np.abs(multilook),
    title='Focussed image (azimuth looks={})'.format(n_looks),
    vlim=[0, 3e4],
    origin='lower',
    interpolation='antialiased',
)
