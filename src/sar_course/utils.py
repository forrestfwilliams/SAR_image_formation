import matplotlib.pyplot as plt
import numpy as np


def plot_freq(
    freq,
    val,
    title,
    x='Frequency',
    y='20*log10(|spectrum|), [dB]',
    xlim=[None, None],
    ylim=[None, None],
    unit='MHz',
    shift=False,
):
    x += ' [{}]'.format(unit)
    if unit == 'MHz':
        u = 1e-6
    elif unit == 'Hz':
        u = 1
    if shift:
        val = np.fft.fftshift(val)
    plt.figure(figsize=[14, 4])
    plt.plot(freq * u, val)
    plt.title(title)
    plt.xlim(min(freq) * u, max(freq) * u)
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()


def plot_time(
    t, val, title, x=r'Time', y='amplitude [-]', xlim=[None, None], ylim=[None, None], unit='micros', shift=False
):
    if unit == 'micros':
        x += r' [$\mu$ s]'
        u = 1e6
    elif unit == 's':
        x += r' [s]'
        u = 1
    if shift:
        val = np.fft.fftshift(val)
    plt.figure(figsize=[14, 4])
    plt.plot(t * u, val)
    plt.title(title)
    plt.xlim(min(t) * u, max(t) * u)
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()


def plot_img(data, nhdr=0, title='Data', scale=1, vlim=[None, None], origin='upper', savetif=None):
    if scale > 1:
        clabel = 'Value * {} [-]'.format(scale)
    else:
        clabel = 'Value [-]'

    # Adjust the data part for better visualization
    val = np.array(data)
    val[:, nhdr:] = scale * val[:, nhdr:]

    # plot the 2D image
    plt.figure(figsize=[14, 14])
    im = plt.imshow(val, cmap='gray', interpolation='none', vmin=vlim[0], vmax=vlim[1], origin=origin)
    cbar = plt.colorbar(im, shrink=0.3, pad=0.02)
    cbar.set_label(clabel, rotation=270, labelpad=30)
    plt.title(title)
    plt.xlabel('Range [samples]')
    plt.ylabel('Azimuth [lines]')
    if savetif is not None:
        plt.savefig('{}'.format(savetif), format='tif')
    plt.show()


def to_mag_db(data):
    mag_db = 20 * np.log10(np.abs(data) + 1e-30)
    return mag_db
