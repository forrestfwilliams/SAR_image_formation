import matplotlib.pyplot as plt
import numpy as np

ERS_HEADER_LENGTH = 412


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
    plt.close('all')


def plot_time(
    t,
    val,
    title,
    x=r'Time',
    y='amplitude [-]',
    xlim=[None, None],
    ylim=[None, None],
    unit='micros',
    shift=False,
    plotcomplex=False,
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
    if plotcomplex:
        plt.plot(t * u, np.real(val), label='Real part')
        plt.plot(t * u, np.imag(val), label='Imaginary part')
        plt.legend(loc='upper right')
    else:
        plt.plot(t * u, val)
    plt.title(title)
    plt.xlim(min(t) * u, max(t) * u)
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()


def plot_img(
    data,
    nhdr=0,
    title='Data',
    scale=1,
    cmap='gray',
    vlim=[None, None],
    origin='upper',
    aspect="equal",
    xlabel='Range [bins]',
    ylabel='Azimuth [lines]',
    interpolation='none',
    savetif=None,
    figsize=[6, 6],
    lim=[None, None, None, None],
    ex=None,
    yticks=None,
):
    if scale > 1:
        clabel = 'Value * {} [-]'.format(scale)
    else:
        clabel = 'Value [-]'

    # Adjust the data part for better visualization
    val = np.array(data)
    val[:, nhdr:] = scale * val[:, nhdr:]

    # plot the 2D image
    plt.figure(figsize=figsize)
    im = plt.imshow(
        val, cmap=cmap, interpolation=interpolation, vmin=vlim[0], vmax=vlim[1], origin=origin, aspect=aspect, extent=ex
    )
    cbar = plt.colorbar(im, shrink=0.3, pad=0.02)
    cbar.set_label(clabel, rotation=270, labelpad=30)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(lim[0], lim[1])
    plt.ylim(lim[2], lim[3])
    if (yticks is not None) and (len(yticks) == 3):
        plt.yticks(np.linspace(lim[2], lim[3], yticks[2]), np.linspace(yticks[0], yticks[1], yticks[2]))
    if savetif is not None:
        plt.savefig('{}'.format(savetif), format='tif')
    plt.show()


def to_mag_db(data):
    mag_db = 20 * np.log10(np.abs(data) + 1e-30)
    return mag_db


def parse_ers_line(line, pad_to=None):
    digital_number = np.frombuffer(line[412:], dtype=np.int8).astype(float)
    digital_number -= 15.5
    data = digital_number[::2] + 1j * digital_number[1::2]
    if pad_to is not None:
        padded = np.zeros((pad_to), dtype=np.complex64)
        padded[: data.shape[0]] = data.copy()
        data = padded.copy()
    return data


def parse_ers(file_path, n_lines, n_samples=4903, pad_to=0):
    ers_data = np.zeros((n_lines, max(n_samples, pad_to)), dtype=np.csingle)
    with open(file_path, 'rb') as f:
        for i in range(n_lines):
            # each complex number takes 2 bytes to represent
            line = f.read(ERS_HEADER_LENGTH + n_samples * 2)
            row = parse_ers_line(line, pad_to=ers_data.shape[1])
            ers_data[i, :] = row
    return ers_data


def read_alos(filepath, n_samp, nlines, dtype=np.csingle, byteswap=False):
    with open(filepath, 'rb') as fn:
        load_arr = np.frombuffer(fn.read(), dtype=dtype)
        load_arr = load_arr.reshape((nlines, n_samp))
    if byteswap:
        load_arr = np.array(load_arr.byteswap())
    return np.array(load_arr)
