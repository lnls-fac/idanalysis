#!/usr/bin/env python-sirius

"""Script to check kickmap"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from idanalysis.analysis import Tools
from idanalysis import IDKickMap
from fieldmaptrack import Beam

import utils

BRHO, *_ = Beam.calc_brho(3)  # [T.m]


class EXPS:
    # 3.5085511562608147
    kickx_polyexps = [
        (2, 0), (0, 4), (3, 0)]#, (2, 2), (4, 2), (1, 0),
        # (0, 0), (6, 9), (1, 5), (0, 9), (8, 9), (2, 9),
        # (6, 0), (8, 2), (4, 0), (8, 0), (0, 8)]

    # 4.447143472886692
    kicky_polyexps = [
        (0, 1), (1, 1), (5, 1)]#, (0, 0), (4, 1),
        # (9, 3), (8, 2), (3, 1), (7, 1), (9, 1),
        # (1, 0), (9, 2), (8, 1), (2, 1), (6, 1),
        # (2, 0), (6, 2)]

    poly_exps = {'x': kickx_polyexps,
                 'y': kicky_polyexps}


class FUNCTIONS:

    def kickx_fit(point, *coeffs):
        y, x = point.T
        fit = 0 * x
        for coeff, exp in zip(coeffs, EXPS.poly_exps['x']):
            fit += coeff * x**exp[0] * y**exp[1]
        return fit

    def kicky_fit(point, *coeffs):
        y, x = point.T
        fit = 0 * x
        for coeff, exp in zip(coeffs, EXPS.poly_exps['y']):
            fit += coeff * x**exp[0] * y**exp[1]
        return fit

    def kick_fit_selected(kick_component, exponents, point, *coeffs):

        y, x = point.T
        fit = 0 * x
        for coeff, exp in zip(coeffs, EXPS.poly_exps[kick_component]):
            if exp in exponents:
                fit += coeff * x**exp[0] * y**exp[1]
        return fit

    kick_fit = {'x': kickx_fit,
                'y': kicky_fit}


def get_xdata(kmap, kick_component):

    posx_orig = kmap.posx.copy()
    posy_orig = kmap.posy.copy()

    if kick_component == 'x':
        kick_orig = (1e6/BRHO**2) * kmap.kickx.copy()
    elif kick_component == 'y':
        kick_orig = (1e6/BRHO**2) * kmap.kicky.copy()
    else:
        raise ValueError

    if max(abs(posx_orig)) > 0:
        posx = posx_orig / max(abs(posx_orig))
    if max(abs(posy_orig)) > 0:
        posy = posy_orig / max(abs(posy_orig))

    xdata = np.array(
        [(posy[i], posx[j])
            for i in range(len(posy))
            for j in range(len(posx))])
    ret = (posx_orig, posy_orig, posx, posy, kick_orig, xdata)
    return ret


def run_fit_kick(kmap, kick_component='x',
                 print_flag=False, plot_flag=False, nrpts=None):

    ret = get_xdata(kmap, kick_component)
    posx_orig, posy_orig, posx, posy, kick_orig, xdata = ret
    exps = EXPS.poly_exps[kick_component]

    ydata = np.array(
        [kick_orig[i, j]
            for i in range(len(posy))
            for j in range(len(posx))])

    p0 = np.zeros(len(exps))
    popt, pcov, infodict, *_ = curve_fit(
        FUNCTIONS.kick_fit[kick_component], xdata, ydata,
        p0=p0, full_output=True)

    kick_fit = FUNCTIONS.kick_fit[kick_component](xdata, *popt)
    kick_fit.shape = kick_orig.shape

    residue0 = np.sqrt(np.mean((kick_orig)**2))
    residue = np.sqrt(np.mean((kick_orig - kick_fit)**2))
    percentual_error = 100*np.sqrt(
                       np.mean(((kick_orig - kick_fit)**2)/kick_orig**2))

    if print_flag:
        print('Exponents:')
        print(exps)
        print()

        print('Coefficients:')
        print(popt)
        print()

        print('residue0: ', residue0)
        print('residue : ', residue)
        print('percentual_error : ', percentual_error)

    if plot_flag:
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
            kick_plot = FUNCTIONS.kick_fit[kick_component](xdata, *popt)
            plt.plot(1e3*posx_orig, kick_orig[idy, :], label='data')
            plt.plot(1e3*posx_plot, kick_plot, label='fit')
            plt.title(f'idy:{idy} posy:{1e3*posy_orig[idy]}')
            plt.xlabel('posx [mm]')
            plt.ylabel(f'Kick {kick_component} [urad]')
            plt.show()

        for idx, _ in enumerate(posx):
            xdata = np.array([(posy_, posx[idx]) for posy_ in posy_fit])
            kick_plot = FUNCTIONS.kick_fit[kick_component](xdata, *popt)
            plt.plot(1e3*posy_orig, kick_orig[:, idx], label='data')
            plt.plot(1e3*posy_plot, kick_plot, label='fit')
            plt.title(f'idx:{idx} posx:{1e3*posx_orig[idx]}')
            plt.xlabel('posy [mm]')
            plt.ylabel(f'Kick {kick_component} [urad]')
            plt.show()

    return residue, kick_fit, popt


def get_kick_selected(kmap, kick_component, exponents,
                      plot_flag, popt, nrpts):

    ret = get_xdata(kmap, kick_component)
    posx_orig, posy_orig, posx, posy, kick_orig, xdata = ret

    kick_sel = FUNCTIONS.kick_fit_selected(kick_component, exponents,
                                           xdata, *popt)
    kick_sel.shape = kick_orig.shape

    residue0 = np.sqrt(np.mean((kick_orig)**2))
    residue = np.sqrt(np.mean((kick_orig - kick_sel)**2))

    print('Exponents:')
    print(exponents)
    print()

    print('Coefficients:')
    print(popt)
    print()

    print('residue0: ', residue0)
    print('residue : ', residue)
    print()

    if plot_flag:
        _, kick_fit, _ = run_fit_kick(kmap, kick_component,
                                      print_flag=False, plot_flag=False,
                                      nrpts=nrpts)
        for idy in range(len(kmap.posy)):
            plt.plot(1e3*kmap.posx, kick_fit[idy, :], label='All terms')
            plt.plot(1e3*kmap.posx, kick_sel[idy, :], label='Selected terms')
            plt.suptitle('Kickmap fitting')
            plt.title(f'idy : {idy}')
            plt.xlabel('posx [mm]')
            plt.ylabel(f'Kick {kick_component} [urad]')
            plt.legend()
            plt.grid()
            plt.show()
        for idx in range(len(kmap.posx)):
            plt.plot(1e3*kmap.posy, kick_fit[:, idx], label='All terms')
            plt.plot(1e3*kmap.posy, kick_sel[:, idx], label='Selected terms')
            plt.suptitle('Kickmap fitting')
            plt.title(f'idx : {idx}')
            plt.xlabel('posy [mm]')
            plt.ylabel(f'Kick {kick_component}[urad]')
            plt.legend()
            plt.grid()
            plt.show()

    return kick_sel


def get_kickmap_fit(kmap, exps_to_remove, nrpts,
                    plot_flag=False, save_map=False, sulfix='-filtered'):

    _, _, poptx = run_fit_kick(kmap, kick_component='x',
                               print_flag=False, plot_flag=False, nrpts=nrpts)
    _, _, popty = run_fit_kick(kmap, kick_component='y',
                               print_flag=False, plot_flag=False, nrpts=nrpts)

    poly_exps = EXPS.poly_exps['x']
    exponents = [elem for elem in poly_exps if elem not in exps_to_remove['x']]
    kickx_sel = get_kick_selected(kmap, 'x', exponents,
                                  plot_flag, poptx, nrpts)

    poly_exps = EXPS.poly_exps['y']
    exponents = [elem for elem in poly_exps if elem not in exps_to_remove['y']]
    kicky_sel = get_kick_selected(kmap, 'y', exponents,
                                  plot_flag, popty, nrpts)

    kmap.kickx = (1e-6*BRHO**2)*kickx_sel
    kmap.kicky = (1e-6*BRHO**2)*kicky_sel

    if save_map:
        sulfix_ = sulfix + '.txt'
        fname = fname.replace('.txt', sulfix_)
        kmap.save_kickmap_file(fname)


def do_fitting(iter, kick_component):
    all_exps = list()
    stg = ''
    for i in range(iter):
        residue_min = 1e30
        for expx in np.arange(10):
            for expy in np.arange(10):
                if (expx, expy) not in all_exps:
                    EXPS.poly_exps[kick_component] =\
                        all_exps + [(expx, expy), ]
                residue, *_ = run_fit_kick(kmap, kick_component=kick_component)
                if residue < residue_min:
                    residue_min = residue
                    this_exp = list(EXPS.poly_exps[kick_component])
                    stg += str(residue)
                    stg += ' '
                    stg += str(this_exp)
                    stg += '\n'
                    print('New residue found!')
        print('iteraction: ', i)
        all_exps = this_exp
    print('\n\n')
    print(stg)


if __name__ == '__main__':

    gaps = utils.gaps
    phases = utils.phases
    widths = utils.widths

    fname = Tools._get_kmap_filename(
        width=widths[0], gap=gaps[-1], phase=phases[0], shift_flag=True)
    kmap = IDKickMap(kmap_fname=fname)

    do_fitting(iter=15, kick_component='y')

    nrpts = 101
    exps_to_remove = {'x': [(0, 4)],
                      'y': [(1, 1), (5, 1)]}
    # get_kickmap_fit(kmap, exps_to_remove, nrpts, False)
