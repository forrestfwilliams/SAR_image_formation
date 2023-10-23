import numpy as np
import matplotlib.pyplot as plt

FOURTH_PI = -11
BOLTZMAN = 1.38e-23
SPEED_OF_LIGHT = 2.998e8


def taylor_young_step(t, frequency, distance, boat_start=1000, speed=5):
    reciever_travel_time = distance / SPEED_OF_LIGHT
    reciever_value = np.cos(2 * np.pi * frequency * (t + reciever_travel_time))
    boat_distance = 2 * np.sqrt(((distance / 2) ** 2) + ((boat_start - (speed * t)) ** 2))
    boat_time = boat_distance / SPEED_OF_LIGHT
    boat_value = -0.5 * np.cos(2 * np.pi * frequency * (t + boat_time))
    return reciever_value + boat_value


def taylor_young(wavelength=5, distance=5000, show=False):
    frequency = SPEED_OF_LIGHT / wavelength
    steps = np.linspace(0, 1e-7, 1000)
    interference_values = [taylor_young_step(t, frequency, distance) for t in steps]
    f, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(np.array(steps), interference_values)
    ax.set(xlabel='Time (s)', ylabel='Amplitude')
    if show:
        plt.show()
    else:
        plt.savefig('../../assets/taylor_young.png')
    return interference_values


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
    print(np.array(signal_components).sum(), np.array(noise_components).sum())
    return np.array(signal_components).sum() - np.array(noise_components).sum()


def question2():
    base_case = get_snr_continuous(2500, 1, 0.5, 0.5, 0.24, 15_000, 50, -15, 900, 1_000_000)
    print(f'Base Case: {base_case:.2f} dB')

    new_wavelength_case = get_snr_continuous(2500, 1, 0.5, 0.5, 0.03, 15_000, 50, -15, 900, 1_000_000)
    print(f'3 cm Case: {new_wavelength_case:.2f} dB')


def question3():
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


if __name__ == '__main__':
    taylor_young(show=True)
    question2()
    question3()
