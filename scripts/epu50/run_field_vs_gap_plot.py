#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as optimize

from imaids import utils as ima_utils

from utils import ID_PERIOD
from utils import GAPS, PHASES
from run_rk_traj import load_rk_traj


def calc_eff_field(rz, field):
    """."""
    freqs = (2*np.pi/ID_PERIOD) * np.array([1, 3, 5])
    amps, *_ = ima_utils.fit_fourier_components(field, freqs, rz)
    return amps


def function_field_vs_gap(gap_over_lambda, a, b):
    """."""
    amp = a*np.exp(-b * gap_over_lambda)
    return amp


def calc_field_vs_gap_coeffs(traj_data, phase, datatype='fmap'):
    """."""
    data = traj_data[phase]
    rz = data['fmaprz']
    if datatype == 'fmap':
        bx, by = data['fmapbx'], data['fmapby']
        rz = data['fmaprz']
    else:
        bx, by = data['bx'], data['by']
        rz = data['rz']

    beffx, beffy = [], []
    keffx, keffy = [], []
    for gap in GAPS:
        rz_ = rz[gap]
        bx_, by_, = bx[gap], by[gap]
        fraction = int(len(rz_)/4)

        amps = calc_eff_field(
            rz_[fraction:3*fraction], bx_[fraction:3*fraction])
        beff = np.sqrt(amps[0]**2+amps[1]**2+amps[2]**2)
        keff = ima_utils.calc_deflection_parameter(beff, ID_PERIOD/1000)
        beffx.append(beff)
        keffx.append(keff)

        amps = calc_eff_field(
            rz_[fraction:3*fraction], by_[fraction:3*fraction])
        beff = np.sqrt(amps[0]**2+amps[1]**2+amps[2]**2)
        keff = ima_utils.calc_deflection_parameter(beff, ID_PERIOD/1000)
        beffy.append(beff)
        keffy.append(keff)

    g0 = min(float(gap) for gap in GAPS)
    gap_array = np.array([float(gap) for gap in GAPS])

    # plot amplitude bx
    curve_fit = optimize.curve_fit(
        function_field_vs_gap, gap_array/ID_PERIOD, beffx)[0]
    a, b = curve_fit[:2]
    ax, bx = a*np.exp(-b*g0/ID_PERIOD), b/np.pi

    # plot amplitude by
    curve_fit = optimize.curve_fit(
        function_field_vs_gap, gap_array/ID_PERIOD, beffy)[0]
    a, b = curve_fit[:2]
    ay, by = a*np.exp(-b*g0/ID_PERIOD), b/np.pi

    return ax, bx, ay, by, beffx, keffx, beffy, keffy


def plot_field_vs_gap(traj_data, phase, show_flag=False, datatype='fmap'):
    """."""
    ax, bx, ay, by, beffx, keffx, beffy, keffy = calc_field_vs_gap_coeffs(
        traj_data, phase, datatype=datatype)

    g0 = min(float(gap) for gap in GAPS)
    gap_array = np.array([float(gap) for gap in GAPS])
    gaps_fit = np.arange(22, 42, 1)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # plot amplitude bx
    curve_fit = optimize.curve_fit(
        function_field_vs_gap, gap_array/ID_PERIOD, beffx)[0]
    a, b = curve_fit[:2]
    ax, bx = a*np.exp(-b*g0/ID_PERIOD), b/np.pi
    label = 'Bx = {:.2f}*exp(- {:.2f} π (g - g0) / λ)'.format(ax, bx)
    fitted_curve = function_field_vs_gap(gaps_fit/ID_PERIOD, a, b)
    ax1.plot(gaps_fit, fitted_curve, '--', color='C0', label=label)
    ax1.plot(gap_array, beffx, 'o', color='C0')
    ax2.plot(gap_array, keffx, 'o', color='C0')

    # plot amplitude by
    curve_fit = optimize.curve_fit(
        function_field_vs_gap, gap_array/ID_PERIOD, beffy)[0]
    a, b = curve_fit[:2]
    ay, by = a*np.exp(-b*g0/ID_PERIOD), b/np.pi
    label = 'By = {:.2f}*exp(- {:.2f} π (g - g0) / λ)'.format(ay, by)
    fitted_curve = function_field_vs_gap(gaps_fit/ID_PERIOD, a, b)
    ax1.plot(gaps_fit, fitted_curve, '--', color='C1', label=label)
    ax1.plot(gap_array, beffy, 'o', color='C1')
    ax2.plot(gap_array, keffy, 'o', color='C1')

    # configure plot
    fig_path = 'results/phase-organized/' + phase + '/'
    ax1.set_xlabel('Gap [mm]')
    ax1.set_ylabel('B [T]')
    ax2.set_ylabel('K')
    ax1.legend()
    ax1.grid()
    title = 'Field amplitudes for phase ' + phase + ' mm'
    if phase == '-25.00':
        title = 'Campo magnético do EPU 50 (Polarização Horizontal)'
    elif phase == '+00.00':
        title = 'Campo magnético do EPU 50 (Polarização Vertical)'
    elif phase == '-16.39':
        title = 'Campo magnético do EPU 50 (Polarização Circular)'

    plt.title(title)
    plt.savefig(fig_path + 'field-amplitude-vs-gap.png', dpi=300)
    if show_flag:
        plt.show()
    plt.close()


if __name__ == "__main__":
    """."""
    traj_data, *_ = load_rk_traj()

    for phase in PHASES:
        plot_field_vs_gap(traj_data, phase, show_flag=False)
