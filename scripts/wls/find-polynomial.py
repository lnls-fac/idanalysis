#!/usr/bin/env python-sirius

"""Script to check kickmap"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from idanalysis.analysis import Tools
from idanalysis import IDKickMap

import utils



KICKX_POLYEXPS = [
    (2, 0), (0, 4), (3, 0), (2, 2), (4, 2), (1, 0), (0, 0), (6, 9),
    (1, 5), (0, 9), (8, 9), (2, 9), (6, 0), (8, 2), (4, 0), (8, 0), (0, 8)]


KICKX_POLYEXPS = [
# 3.256972800016463
    (2, 0), (0, 4), (3, 0), (2, 2), (4, 2), (1, 0), (0, 0), (6, 9), (1, 5),
    (0, 9), (8, 9), (2, 9), (6, 0), (8, 2), (4, 0), (8, 0), (4, 9), (1, 2),
    (5, 0), (7, 0), (9, 8), (8, 8), (2, 4), (4, 6), (0, 6), (2, 6), (3, 4),
    (9, 9), (7, 8), (0, 2), (3, 7), (6, 6), (6, 2), (8, 4), (5, 8), (3, 2)]


def kickx_fit(point, *coeffs):
    
    y, x = point.T
    fit = 0 * x
    for coeff, exp in zip(coeffs, KICKX_POLYEXPS):
        fit += coeff * x**exp[0] * y**exp[1]
    return fit


def run_fit_kickx(kmap, print_flag=False, nrpts=None):

    if print_flag:
        print(KICKX_POLYEXPS)

    brho = 10  # [T.m]

    posx_orig = kmap.posx.copy()
    posy_orig = kmap.posy.copy()
    kick_orig = (1e6/brho**2) * kmap.kickx.copy()

    if max(abs(posx_orig)) > 0:
        posx = posx_orig / max(abs(posx_orig))
    if max(abs(posy_orig)) > 0:
        posy = posy_orig / max(abs(posy_orig))

    xdata = np.array([(posy[i], posx[j]) for i in range(len(posy)) for j in range(len(posx))])
    ydata = np.array([kick_orig[i, j] for i in range(len(posy)) for j in range(len(posx))])

    p0 = np.zeros(len(KICKX_POLYEXPS))
    popt, pcov, infodict, *_ = curve_fit(
        kickx_fit, xdata, ydata, p0=p0, full_output=True)
    if print_flag:
        print(popt)
    kick_fit = kickx_fit(xdata, *popt)
    kick_fit.shape = kick_orig.shape

    residue0 = np.sqrt(np.mean((kick_orig)**2))
    residue = np.sqrt(np.mean((kick_orig - kick_fit)**2))
    if print_flag:
        print('residue0: ', residue0)
        print('residue : ', residue)

    if print_flag:
        if nrpts is None:
            posx_plot = posx_orig
            posy_plot = posy_orig
        else:
            posx_plot = np.linspace(min(posx_orig), max(posx_orig), nrpts)
            posx_fit = np.linspace(min(posx), max(posx), nrpts)
            posy_plot = np.linspace(min(posy_orig), max(posy_orig), nrpts)
            posy_fit = np.linspace(min(posy), max(posy), nrpts)

        for idy, _ in enumerate(posy):
            xdata = np.array([(posy[idy], posx_) for posx_ in posx_fit])
            kick_plot = kickx_fit(xdata, *popt)
            plt.plot(1e3*posx_orig, kick_orig[idy, :], label='data')
            plt.plot(1e3*posx_plot, kick_plot, label='fit')
            plt.title(f'idy:{idy} posy:{1e3*posy_orig[idy]}')
            plt.xlabel('posx [mm]')
            plt.show()

        for idx, _ in enumerate(posx):
            xdata = np.array([(posy_, posx[idx]) for posy_ in posy_fit])
            kick_plot = kickx_fit(xdata, *popt)
            plt.plot(1e3*posy_orig, kick_orig[:, idx], label='data')
            plt.plot(1e3*posy_plot, kick_plot, label='fit')
            plt.title(f'idx:{idx} posx:{1e3*posx_orig[idx]}')
            plt.xlabel('posy [mm]')
            plt.show()

    return residue


if __name__ == '__main__':

    gaps = utils.gaps
    phases = utils.phases
    widths = utils.widths

    fname = Tools._get_kmap_filename(
        width=widths[0], gap=gaps[-1], phase=phases[0], shift_flag=True)
    kmap = IDKickMap(kmap_fname=fname)

    run_fit_kickx(kmap, True, 101)
    raise ValueError

    all_exps = list()
    while True:
        residue_min = 1e30
        for expx in np.arange(10):
            for expy in np.arange(10):
                if (expx, expy) not in all_exps:
                    KICKX_POLYEXPS = all_exps + [(expx, expy), ]
                # print()
                residue = run_fit_kickx(kmap)
                if residue < residue_min:
                    residue_min = residue
                    this_exp = list(KICKX_POLYEXPS)
                    print(residue, this_exp)
        all_exps = this_exp
