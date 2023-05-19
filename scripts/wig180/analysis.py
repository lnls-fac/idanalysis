#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

import pyaccel
from pyaccel import optics as pyacc_opt

import pymodels

from idanalysis import orbcorr as orbcorr
from idanalysis import model as model
from idanalysis import optics as optics

import utils


def plot_beta_beating(twiss0, twiss1, twiss2, twiss3, idconfig, plot_flag=True):
  if plot_flag:
    #Compare optics between nominal value and uncorrect optics due ID insertion
    dtunex, dtuney, bbeatx, bbeaty, bbeatx_rms, bbeaty_rms, bbeatx_absmax, bbeaty_absmax = calc_dtune_betabeat(twiss0,twiss1)
    print('\n')
    print('Not symmetrized optics :')
    print(f'dtunex: {dtunex:+.6f}')
    print(f'dtunex: {dtuney:+.6f}')
    print(f'bbetax: {bbeatx_rms:04.1f} % rms, {bbeatx_absmax:04.1f} % absmax')
    print(f'bbetay: {bbeaty_rms:04.1f} % rms, {bbeaty_absmax:04.1f} % absmax')
    print()
    title = 'Beta Beating from ' + idconfig

    plt.figure(1)
    blue, red = (0.4,0.4,1), (1,0.4,0.4)
    labelx = f'X ({bbeatx_rms:.1f} % rms)'
    labely = f'Y ({bbeaty_rms:.1f} % rms)'
    plt.plot(twiss0.spos, bbeatx, color=blue, alpha=0.8, label=labelx)
    plt.plot(twiss0.spos, bbeaty, color=red, alpha=0.8, label=labely)
    plt.xlabel('spos [m]')
    plt.ylabel('Beta Beat [%]')
    plt.title(title)
    suptitle = 'Orbit with local correction + SOFB - not symmetrized optics'
    # suptitle = 'Orbit with local correction - not symmetrized optics'
    # suptitle = 'Orbit without correction - not symmetrized optics'
    plt.suptitle(suptitle)
    plt.ylim(-15,15)
    plt.legend()
    plt.grid()
    plt.savefig('results/' + idconfig + '/' + suptitle + '_' + idconfig + '.png',dpi=300)

    #Compare optics between nominal value and symmetrized optics
    dtunex, dtuney, bbeatx, bbeaty, bbeatx_rms, bbeaty_rms, bbeatx_absmax, bbeaty_absmax = calc_dtune_betabeat(twiss0,twiss2)
    print('symmetrized optics but uncorrect tunes:')
    print(f'dtunex: {dtunex:+.6f}')
    print(f'dtunex: {dtuney:+.6f}')
    print(f'bbetax: {bbeatx_rms:04.1f} % rms, {bbeatx_absmax:04.1f} % absmax')
    print(f'bbetay: {bbeaty_rms:04.1f} % rms, {bbeaty_absmax:04.1f} % absmax')
    print()

    plt.figure(2)
    blue, red = (0.4,0.4,1), (1,0.4,0.4)
    labelx = f'X ({bbeatx_rms:.1f} % rms)'
    labely = f'Y ({bbeaty_rms:.1f} % rms)'
    plt.plot(twiss0.spos, bbeatx, color=blue, alpha=0.8, label=labelx)
    plt.plot(twiss0.spos, bbeaty, color=red, alpha=0.8, label=labely)
    plt.xlabel('spos [m]')
    plt.ylabel('Beta Beat [%]')
    plt.title(title)
    plt.suptitle('Symmetrized optics and uncorrect tunes')
    plt.legend()
    plt.grid()


    #Compare optics between nominal value and symmetrized optics with tune correction
    dtunex, dtuney, bbeatx, bbeaty, bbeatx_rms, bbeaty_rms, bbeatx_absmax, bbeaty_absmax = calc_dtune_betabeat(twiss0,twiss3)
    print('symmetrized optics and correct tunes:')
    print(f'dtunex: {dtunex:+.6f}')
    print(f'dtunex: {dtuney:+.6f}')
    print(f'bbetax: {bbeatx_rms:04.1f} % rms, {bbeatx_absmax:04.1f} % absmax')
    print(f'bbetay: {bbeaty_rms:04.1f} % rms, {bbeaty_absmax:04.1f} % absmax')

    plt.figure(3)
    blue, red = (0.4,0.4,1), (1,0.4,0.4)
    labelx = f'X ({bbeatx_rms:.1f} % rms)'
    labely = f'Y ({bbeaty_rms:.1f} % rms)'
    plt.plot(twiss0.spos, bbeatx, color=blue, alpha=0.8, label=labelx)
    plt.plot(twiss0.spos, bbeaty, color=red, alpha=0.8, label=labely)
    plt.xlabel('spos [m]')
    plt.ylabel('Beta Beat [%]')
    plt.title(title)
    plt.suptitle('Symmetrized optics and correct tunes')
    plt.ylim(-15,15)
    plt.legend()
    plt.grid()
    plt.savefig('results/' + idconfig + '/full_correction.png', dpi=300)
    plt.show()

def calc_dtune_betabeat(twiss0, twiss1):
    dtunex = (twiss1.mux[-1] - twiss0.mux[-1]) / 2 / np.pi
    dtuney = (twiss1.muy[-1] - twiss0.muy[-1]) / 2 / np.pi
    bbeatx = 100 * (twiss1.betax - twiss0.betax) / twiss0.betax
    bbeaty = 100 * (twiss1.betay - twiss0.betay) / twiss0.betay
    bbeatx_rms = np.std(bbeatx)
    bbeaty_rms = np.std(bbeaty)
    bbeatx_absmax = np.max(np.abs(bbeatx))
    bbeaty_absmax = np.max(np.abs(bbeaty))
    return (
      dtunex, dtuney, bbeatx, bbeaty,
      bbeatx_rms, bbeaty_rms, bbeatx_absmax, bbeaty_absmax)


def create_model_bare():
    """."""
    print('--- model bare ---')
    model = pymodels.si.create_accelerator()
    twiss, *_ = pyaccel.optics.calc_twiss(model, indices='closed')
    print('length : {:.4f} m'.format(model.length))
    print('tunex  : {:.6f}'.format(twiss.mux[-1]/2/np.pi))
    print('tuney  : {:.6f}'.format(twiss.muy[-1]/2/np.pi))
    return model, twiss


def create_model_ids(idconfig):
    """."""
    print('--- model with kick-model wiggler ---')
    ids = utils.create_ids(idconfig=idconfig, rescale_kicks=1)
    model = pymodels.si.create_accelerator(ids=ids)
    model.cavity_on = False
    model.radiation_on = 0
    twiss, *_ = pyaccel.optics.calc_twiss(model, indices='closed')
    print('length : {:.4f} m'.format(model.length))
    print('tunex  : {:.6f}'.format(twiss.mux[-1]/2/np.pi))
    print('tuney  : {:.6f}'.format(twiss.muy[-1]/2/np.pi))
    return model, twiss, ids


def run(idconfig):
    """."""
    # bare lattice
    ring0, twiss0 = create_model_bare()
    print()
    ring0.cavity_on = False
    ring0.radiation_on = 0
    # lattice with IDs
    ring1, twiss1, ids = create_model_ids(idconfig=idconfig)
    subsec = str(ids[0].subsec)[2:4]
    straight_nr = int(subsec)
    _, knobs, _ = optics.symm_get_knobs(ring0, straight_nr)  # get knobs and beta locations
    locs_beta = optics.symm_get_locs_beta(knobs)
    print()

    # goal parameters (bare lattice)
    goal_tunes = np.array([twiss0.mux[-1] / 2 / np.pi, twiss0.muy[-1] / 2 / np.pi])
    goal_beta = np.array([twiss0.betax[locs_beta], twiss0.betay[locs_beta]])
    goal_alpha = np.array([twiss0.alphax[locs_beta], twiss0.alphay[locs_beta]])

    # correct orbit locally with ID correctors
    print('--- local orbit correction ---')
    ret = orbcorr.correct_orbit_local(
        ring0, ring1, 'WIG180',
        correction_plane='x', plot=False)
    dkickx1, dkickx2, dkicky1, dkicky2 = ret[0]
    fmts = 'correctors dk {:<10s} : {:+06.1f} {:+06.1f} urad'
    print(fmts.format('horizontal', dkickx1*1e6, dkickx2*1e6))
    print(fmts.format('vertical', dkicky1*1e6, dkicky2*1e6))
    # correct orbit residual globally with SOFB
    # orbcorr.correct_orbit_sofb(
        # ring0, ring1, 'WIG180')
    twiss1, *_ = pyacc_opt.calc_twiss(ring1, indices='closed')

    # symmetrize optics (local quad fam knobs)
    dk_tot = np.zeros(len(knobs))
    for i in range(7):
        dk = optics.correct_symmetry_withbeta(
            ring1, straight_nr, goal_beta, goal_alpha)
        print('iteration #{}, dK: {}'.format(i+1, dk))
        dk_tot += dk
    for i, fam in enumerate(knobs):
        print('{:<9s} dK: {:+9.6f} 1/m²'.format(fam, dk_tot[i]))
    twiss2, *_ = pyacc_opt.calc_twiss(ring1, indices='closed')
    print()

    # correct tunes
    tunes = twiss1.mux[-1]/np.pi/2, twiss1.muy[-1]/np.pi/2
    print('init    tunes: {:.9f} {:.9f}'.format(tunes[0], tunes[1]))
    for i in range(2):
        optics.correct_tunes_twoknobs(ring1, goal_tunes)
        twiss, *_ = pyacc_opt.calc_twiss(ring1)
        tunes = twiss.mux[-1]/np.pi/2, twiss.muy[-1]/np.pi/2
        print('iter #{} tunes: {:.9f} {:.9f}'.format(i+1, tunes[0], tunes[1]))
    print('goal    tunes: {:.9f} {:.9f}'.format(goal_tunes[0], goal_tunes[1]))
    twiss3, *_ = pyacc_opt.calc_twiss(ring1, indices='closed')
    print()

    plot_beta_beating(twiss0, twiss1, twiss2, twiss3, idconfig, plot_flag=True)


if __name__ == '__main__':
    """."""
    # run(idconfig='ID3979')  # gap 59.6 mm, correctors with zero current
    # run(idconfig = 'ID4017')  # gap 59.6 mm, correctors with best current
    run(idconfig = 'ID4020')  # gap 45.0 mm, correctors with zero current
