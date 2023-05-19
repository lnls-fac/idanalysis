#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from idanalysis import orbcorr as orbcorr

from utils import EXCDATA


def bilinear_fit(x1, x2, y):
    """."""
    one = np.ones(x1.shape)
    matrix = np.array([
        [np.dot(one, one), np.dot(one, x1), np.dot(one, x2)],
        [np.dot(x1, one), np.dot(x1, x1), np.dot(x1, x2)],
        [np.dot(x2, one), np.dot(x2, x1), np.dot(x2, x2)],
        ])
    vector = np.array(
        [np.dot(y, one), np.dot(y, x1), np.dot(y, x2)])
    [a, b, c] = np.dot(np.linalg.inv(matrix), vector)
    return [a, b, c]


def fit_function(x, a, b):
    """."""
    f = a + b*x
    return f


def optimize(itotal,iby):
    opt = curve_fit(fit_function, itotal, iby)[0]
    return opt


def run_separate_correctors(gap):
    """."""
    data = EXCDATA[gap]
    file, iup, idown, iby = \
        data['FILE'], data['I_UP'], data['I_DOWN'], data['IBY']
    iby *= 1e-6  # [g.cm] -> [T.m]

    # separate correctors
    a, b, c = bilinear_fit(iup, idown, iby)
    iby1_down_off = iby - c * idown
    iby2_up_off = iby - b * iup

    # correctors together
    itot = iup + idown
    al, dl, *_ = optimize(itot, iby)

    curr_fit = np.array([-5, +5])
    iby_fit = 0 * al + dl * curr_fit
    zero = format(0, '+1.4e')
    ip1 = format(curr_fit[0], '+08.3f')
    iby_p1 = format(iby_fit[0], '+1.4e')
    ip2 = format(curr_fit[1], '+08.3f')
    iby_p2 = format(iby_fit[1], '+1.4e')
    print('# EXCITATION DATA')
    print('# ===============')
    print('# for gap {} mm'.format(gap))
    print(ip1, ' ', iby_p1, zero)
    print(ip2, ' ', iby_p2, zero)
    print('')
    print('# COMMENTS')
    print('# ========')
    print('# 1. data from ')
    print('#    {}'.format(file))
    print('# 2. linear fitting done with "idanalysis/scripts/wiggler/correctors_excdata.py"')

    curr = np.array([-2.0, 2.5])
    iby1_fit = a + b * curr
    iby2_fit = a + c * curr
    iby3_fit = al + dl * curr
    label1 = 'Upstream Corrector'
    label1_fit = 'fitting: {:+.2f} + {:.2f} I'.format(1e6*a, 1e6*b)
    label2 = 'Downstream Corrector'
    label2_fit = 'fitting: {:+.2f} + {:.2f} I'.format(1e6*a, 1e6*c)
    label3 = 'Upstream + Downstream Correctors'
    label3_fit = 'fitting: {:+.2f} + {:.2f} I'.format(1e6*al, 1e6*dl)
    plt.plot(iup, 1e6*iby1_down_off, 'o', color='C0', label=label1)
    plt.plot(curr, 1e6*iby1_fit, '--', color='C0', label=label1_fit)
    plt.plot(idown, 1e6*iby2_up_off, 'o', color='C1', label=label2)
    plt.plot(curr, 1e6*iby2_fit, '--', color='C1', label=label2_fit)
    plt.plot(itot, 1e6*iby, 'o', color='C2', label=label3)
    plt.plot(curr, 1e6*iby3_fit, '--', color='C2', label=label3_fit)
    plt.legend()
    plt.xlabel('Current [A]')
    plt.ylabel('By integral [G.cm]')
    plt.title('Wiggler 180mm Correctors Excitation Curve (gap {} mm)'.format(gap))
    plt.savefig(
        'results/si-corrector-wig180-gap{}.png'.format(gap.replace('.','p')))
    plt.show()


def run(gap):
    """."""
    data = EXCDATA[gap]
    iup, idown, iby = data['I_UP'], data['I_DOWN'], data['IBY']
    iby *= 1e-6  # [g.cm] -> [T.m]
    itotal = iup + idown
    opt = optimize(itotal, iby)
    print('a = ',opt[0])
    print('b = ',opt[1])
    iby_p1 = fit_function(-5, 1 * opt[0], opt[1])
    iby_p2 = fit_function(+5, 1 * opt[0], opt[1])
    ip1 = format(-5,'+08.3f')
    iby_p1 = format(iby_p1,'+1.4e')
    zero = format(0,'+1.4e')
    ip2 = format(+5,'+08.3f')
    iby_p2 = format(iby_p2,'+1.4e')
    print('# EXCITATION DATA')
    print('# ===============')
    print(ip1, iby_p1, zero)
    print(ip2, iby_p2, zero)
    by_fit = fit_function(itotal,opt[0],opt[1])
    plt.plot(itotal,iby,'o',color = 'C0', label='Measurement')
    plt.plot(itotal,by_fit,'-',color = 'C1', label='Fitting a+bx: a={:.2e} b={:.6f}'.format(opt[0],opt[1]))
    plt.xlabel('Current [A]')
    plt.ylabel('Integrated field [Tm]')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    """."""

    # run('59.60')
    run_separate_correctors('45.00')
    run_separate_correctors('59.60')
