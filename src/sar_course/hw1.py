import numpy as np

FOURTH_PI = -11
BOLTZMAN = 1.38e-23
SPEED_OF_LIGHT = 2.998e8


def to_db(value):
    return 10 * np.log10(value)


def get_gain(antenna_area, wavelength):
    return (4 * np.pi * antenna_area) / (wavelength**2)


def altitude_to_distance(altitude, grazing_angle_degrees):
    return altitude / np.cos(np.deg2rad(grazing_angle_degrees))


def get_impulse_scattering_area(wavelength, pulse_length, grazing_angle_degrees, altitude, width):
    distance = altitude_to_distance(altitude, grazing_angle_degrees)
    return (SPEED_OF_LIGHT * pulse_length * distance * wavelength) / (
        2 * width * np.sin(np.deg2rad(grazing_angle_degrees))
    )


def get_snr_continuous(
    transmit_power,
    cable_loss_db,
    antenna_efficiency,
    antenna_area,
    wavelength,
    distance,
    object_size,
    object_reflectivity_db,
    noise_temp,
    noise_bandwidth,
):
    """Calculate SNR, use units of Hz, meters, K only"""
    signal_components = np.array(
        [
            to_db(transmit_power),
            cable_loss_db,
            to_db(get_gain(antenna_area, wavelength)),
            FOURTH_PI,
            to_db(1 / distance**2),
            to_db(object_size),
            object_reflectivity_db,
            FOURTH_PI,
            to_db(1 / distance**2),
            to_db(antenna_efficiency),
        ]
    )
    noise_components = [to_db(x) for x in [noise_temp, noise_bandwidth, BOLTZMAN]]

    return np.array(signal_components).sum() - np.array(noise_components).sum()


if __name__ == '__main__':
    base_case = get_snr_continuous(2500, 1, 0.5, 0.5, 0.24, 15_000, 50, -15, 900, 1_000_000)
    print(f'Base Case: {base_case:.2f} dB')

    new_wavelength_case = get_snr_continuous(2500, 1, 0.5, 0.5, 0.03, 15_000, 50, -15, 900, 1_000_000)
    print(f'3 cm Case: {new_wavelength_case:.2f} dB')

    # Fixed
    altitude = 8000
    grazing_angle = 45

    # L Band
    wavelength = 0.24
    power = 350
    area = get_impulse_scattering_area(wavelength, 1e-6, grazing_angle, altitude, 5000)
    distance = altitude_to_distance(altitude, grazing_angle)
    design1 = get_snr_continuous(power, 1, 0.5, 2, wavelength, distance, area, -15, 900, 1_000_000)
    print(f'L-band Design: {design1:.2f} dB')

    # C-band
    wavelength = 0.06
    power = 90
    area = get_impulse_scattering_area(wavelength, 1e-6, grazing_angle, altitude, 5000)
    distance = altitude_to_distance(altitude, grazing_angle)
    design1 = get_snr_continuous(power, 1, 0.5, 2, wavelength, distance, area, -15, 900, 1_000_000)
    print(f'C-band Design: {design1:.2f} dB')

    # K-band
    wavelength = 0.02
    power = 30
    area = get_impulse_scattering_area(wavelength, 1e-6, grazing_angle, altitude, 5000)
    distance = altitude_to_distance(altitude, grazing_angle)
    design1 = get_snr_continuous(power, 1, 0.5, 2, wavelength, distance, area, -15, 900, 1_000_000)
    print(f'K-band Design: {design1:.2f} dB')
