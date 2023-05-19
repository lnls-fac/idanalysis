#!/usr/bin/env python-sirius

import pickle
import numpy as np
import matplotlib.pyplot as plt

import pyaccel
from pymodels import si

from idanalysis import optics
from idanalysis.model import get_id_sabia_list

import idanalysis.utils as utils
#utils.FOLDER_BASE = '/home/ximenes/repos-dev/'
utils.FOLDER_BASE = '/home/gabriel/repos-sirius/'


def create_model(ids, goal_tunes=None, straight_nr=None):

    print('--- create model')
    ring = si.create_accelerator(ids=ids)
    ring.vchamber_on = optics.CHAMBER_ON

    if goal_tunes is None:
        tw0, *_ = pyaccel.optics.calc_twiss(ring)
        goal_tunes = np.array([tw0.mux[-1] / 2 / np.pi, tw0.muy[-1] / 2 / np.pi])
    print()

    if straight_nr is not None:
        _, knobs, _ = optics.symm_get_knobs(ring, straight_nr)
        locs_beta = optics.symm_get_locs_beta(knobs)
        goal_beta = np.array([tw0.betax[locs_beta], tw0.betay[locs_beta]])
        goal_alpha = np.array([tw0.alphax[locs_beta], tw0.alphay[locs_beta]])
    else:
        goal_beta = None
        goal_alpha = None

    return ring, goal_tunes, goal_beta, goal_alpha


def correct_optics(ring, straight_nr, goal_tunes, goal_beta, goal_alpha, weight=False):

    ring0 = ring[:]

    tw, *_ = pyaccel.optics.calc_twiss(ring)

    print('--- correct cod')
    id_famname = None  # NOTE: set!
    cod = optics.correct_orbit(ring, id_famname, False)
    print('kicks [urad]: {}'.format(1e6*cod[0]))
    ring1 = ring[:]
    print()

    print('--- symmetrize optics')
    _, knobs, _ = optics.symm_get_knobs(ring, straight_nr)

    dk_tot = np.zeros(len(knobs))
    for i in range(5):
        dk = optics.correct_symmetry_withbeta(ring, straight_nr, goal_beta, goal_alpha, weight=weight)
        print('iteration #{}, dK: {}'.format(i+1, dk))
        dk_tot += dk
    for i, fam in enumerate(knobs):
        print('{:<9s} dK: {:+9.6f} 1/m²'.format(fam, dk_tot[i]))
    ring2 = ring[:]
    print()

    print('--- correct tunes')
    tw, *_ = pyaccel.optics.calc_twiss(ring)
    tunes = tw.mux[-1]/np.pi/2, tw.muy[-1]/np.pi/2
    print('init    tunes: {:.9f} {:.9f}'.format(tunes[0], tunes[1]))
    for i in range(2):
        optics.correct_tunes_twoknobs(ring, goal_tunes)
        tw, *_ = pyaccel.optics.calc_twiss(ring)
        tunes = tw.mux[-1]/np.pi/2, tw.muy[-1]/np.pi/2
        print('iter #{} tunes: {:.9f} {:.9f}'.format(i+1, tunes[0], tunes[1]))
    print('goal    tunes: {:.9f} {:.9f}'.format(goal_tunes[0], goal_tunes[1]))
    ring3 = ring[:]
    print()

    return ring0, ring1, ring2, ring3


def save_models(configs):

    ring_nom, goal_tunes, goal_beta, goal_alpha = create_model(ids=None, goal_tunes=None, straight_nr=10)
    tw_nom, *_ = pyaccel.optics.calc_twiss(ring_nom)

    models = dict()
    for config in configs:
        
        config_label = configs.get_config_label(config)
        print('=== {}\n'.format(config_label))

        fname = configs.get_kickmap_filename(config)
        ids = get_id_sabia_list(fname)
        
        ring, *_ = create_model(ids=ids, goal_tunes=goal_tunes)
        ring0, ring1, ring2, ring3 = correct_optics(ring, 10, goal_tunes, goal_beta, goal_alpha)

        models[config_label] = (ring0, ring1, ring2, ring3)

        tw2, *_ = pyaccel.optics.calc_twiss(ring2)
        tw3, *_ = pyaccel.optics.calc_twiss(ring3)

        # betax
        plt.plot(tw_nom.spos, tw_nom.betax, label='nom')
        plt.plot(tw2.spos, 1e3*(tw2.betax - tw_nom.betax), label='1000 x (symm - nom)')
        plt.plot(tw3.spos, 1e3*(tw3.betax - tw_nom.betax), label='1000 x (symm+tune - nom)')
        plt.legend()
        plt.xlabel('posz [m]')
        plt.xlabel('beta [m]')
        plt.title('BetaX')
        plt.show()

        # betay
        plt.plot(tw_nom.spos, tw_nom.betay, label='nom')
        plt.plot(tw2.spos, 1e3*(tw2.betay - tw_nom.betay), label='1000 x (symm - nom)')
        plt.plot(tw3.spos, 1e3*(tw3.betay - tw_nom.betay), label='1000 x (symm+tune - nom)')
        plt.legend()
        plt.xlabel('posz [m]')
        plt.xlabel('beta [m]')
        plt.title('BetaY')
        plt.show()

    pickle.dump(models, open('models.pickle', 'wb'))


def test_symm():

    ring_nom, goal_tunes, goal_beta, goal_alpha = create_model(ids=None, goal_tunes=None, straight_nr=10)
    
    # locs = symm_get_locs(ring_nom)
    # _, knobs, _ = symm_get_knobs(ring_nom, 10)
    # locs_beta = symm_get_locs_beta(knobs)
    # alpha1 = symm_calc_residue(ring_nom, locs, locs_beta, goal_beta, goal_alpha, weight=False)
    # ring, goal_tunes, *_ = create_model(ids=ids, goal_tunes=goal_tunes)
    # alpha2 = symm_calc_residue(ring, locs, locs_beta, goal_beta, goal_alpha, weight=False)

    # plt.plot(1000*alpha1, label='alpha nominal (x1000)')
    # plt.plot(alpha2, label='alpha with uncorrected ID')
    # plt.legend()
    # plt.show()
    # return

    tw_nom, *_ = pyaccel.optics.calc_twiss(ring_nom)
    ring, goal_tunes, *_ = create_model(ids=ids, goal_tunes=goal_tunes)
    ring_ = ring[:]
    ring0, ring1, ring2, ring3 = correct_optics(ring, 10, goal_tunes, goal_beta, goal_alpha, False)
    tw0, *_ = pyaccel.optics.calc_twiss(ring2)
    ring0, ring1, ring2, ring3 = correct_optics(ring_, 10, goal_tunes, goal_beta, goal_alpha, True)
    tw1, *_ = pyaccel.optics.calc_twiss(ring2)

    plt.plot(tw_nom.spos, 1e3*(tw0.betax - tw_nom.betax), label='without beta weigths')
    plt.plot(tw_nom.spos, 1e3*(tw1.betax - tw_nom.betax), label='with large beta weights')
    plt.xlabel('spos [m]')
    plt.ylabel('betax - betax_nom [mm]')
    plt.legend()
    plt.title('With BC symmetry points')
    plt.show()

    plt.plot(tw_nom.spos, 1e3*(tw0.betay - tw_nom.betay), label='without beta weigths')
    plt.plot(tw_nom.spos, 1e3*(tw1.betay - tw_nom.betay), label='with large beta weights')
    plt.xlabel('spos [m]')
    plt.ylabel('betay - betay_nom [mm]')
    plt.legend()
    plt.title('With BC symmetry points')
    plt.show()


# --- legacy (to be adapted and tested in new version lib version)

def tune_shift(models):

    # models = load_models(folder='./results/vchamber-off/')
    nominal_ring = si.create_accelerator(ids=None)
    tw, *_ = pyaccel.optics.calc_twiss(nominal_ring)
    goal_tunes = np.array([tw.mux[-1] / 2 / np.pi, tw.muy[-1] / 2 / np.pi])

    for config, rings in models.items():
        ring0, *_ = rings
        tw0, *_ = pyaccel.optics.calc_twiss(ring0)
        tunes0 = np.array([tw0.mux[-1] / 2 / np.pi, tw0.muy[-1] / 2 / np.pi])
        dtunes = tunes0 - goal_tunes
        print('{:<40s}: {:+.8f} {:+.8f}'.format(config, dtunes[0], dtunes[1]))


def calc_dynapt_area(dynapt):
    vx, vy = dynapt
    area = 0
    for i in range(len(vx)-1):
        v1 = np.array([vx[i], vy[i]])
        v2 = np.array([vx[i+1], vy[i+1]])
        area += np.linalg.norm(np.cross(v1, v2))
    return area


def closed_orbit(models, modelname=None):

    # models = load_models(folder='./results/vchamber-off/')
    for config, rings in models.items():
        config = config.replace('Linear', 'L')
        config = config.replace('Elliptical', 'E')
        ring0, ring1, ring2, ring3 = rings
        if modelname == 'ring0' or modelname is None:
            ring = ring0
        elif modelname == 'ring1':
            ring = ring1
        elif modelname == 'ring2':
            ring = ring2
        elif modelname == 'ring3':
            ring = ring3
        tw0, *_ = pyaccel.optics.calc_twiss(ring)
        cod = 1e6*tw0.co[2]
        print('{:<40s}: std: {:+06.1f} mm,  maxabs: {:+06.1f} um'.format(config, np.std(cod), np.max(abs(cod))))
        plt.plot(tw0.spos, cod, '-', label=config + ' ({:.1f} um)'.format(np.max(abs(cod))))

    plt.title('Vertical COD - {}'.format(modelname))
    plt.xlabel('spos [m]')
    plt.ylabel('COD [um]')
    plt.legend()
    plt.grid()
    plt.show()

    for config, rings in models.items():
        config = config.replace('Linear', 'L')
        config = config.replace('Elliptical', 'E')
        ring0, ring1, ring2, ring3 = rings
        if modelname == 'ring0' or modelname is None:
            ring = ring0
        elif modelname == 'ring1':
            ring = ring1
        elif modelname == 'ring2':
            ring = ring2
        elif modelname == 'ring3':
            ring = ring3
        tw0, *_ = pyaccel.optics.calc_twiss(ring)
        cod = 1e6*tw0.co[0]
        print('{:<40s}: std: {:+06.2f} mm,  maxabs: {:+06.1f} mm'.format(config, np.std(cod), np.max(abs(cod))))
        plt.plot(tw0.spos, cod, '-', label=config + ' ({:.1f} mm)'.format(np.max(abs(cod))))

    plt.title('Horizontal COD - {}'.format(modelname))
    plt.xlabel('spos [m]')
    plt.ylabel('COD [um]')
    plt.legend()
    plt.grid()
    plt.show()

    
def kick_strengths(models):
    # models = load_models(folder='./results/vchamber-off/')
    for config, rings in models.items():
        _, ring1, *_ = rings
        idx = pyaccel.lattice.find_indices(ring1, 'fam_name', 'IDC')
        hkick = pyaccel.lattice.get_attribute(ring1, 'hkick', idx)
        vkick = pyaccel.lattice.get_attribute(ring1, 'vkick', idx)
        print('{:<40s}: {:+.2f} urad {:+.2f} urad'.format(config, 1e6*max(abs(hkick)), 1e6*max(abs(vkick))))


def quadrupole_strengths(models):

    # models = load_models(folder='./results/vchamber-off/')
    ring_nom = si.create_accelerator(ids=None)
    k0 = list()
    _, knobs, _ = symm_get_knobs(ring_nom, 10)
    for inds in knobs.values():
        k0 += list(pyaccel.lattice.get_attribute(ring_nom, 'polynom_b', inds[0], 1))
    k0 = np.array(k0)
    
    for config, rings in models.items():
        _, _, ring2, *_ = rings
        dk1 = list()
        for inds in knobs.values():
            dk1 += list(pyaccel.lattice.get_attribute(ring2, 'polynom_b', inds[0], 1))
        dk1 = np.array(dk1) - k0
        print('{:<40s}: '.format(config), end='')
        for dk in dk1:
            print('{:+.5f} '.format(dk), end='')
        print()


def beta_difference_model(config, rings):
    
    ring_nom = si.create_accelerator(ids=None)
    tw_nom, *_ = pyaccel.optics.calc_twiss(ring_nom)
    ring0, ring1, ring2, ring3 = rings

    tw, *_ = pyaccel.optics.calc_twiss(ring0)
    dbetax0 = 1e3*(tw.betax - tw_nom.betax)    
    dbetay0 = 1e3*(tw.betay - tw_nom.betay)    
    tw, *_ = pyaccel.optics.calc_twiss(ring1)
    dbetax1 = 1e3*(tw.betax - tw_nom.betax)
    dbetay1 = 1e3*(tw.betay - tw_nom.betay)
    tw, *_ = pyaccel.optics.calc_twiss(ring2)
    dbetax2 = 1e3*(tw.betax - tw_nom.betax)
    dbetay2 = 1e3*(tw.betay - tw_nom.betay)
    tw, *_ = pyaccel.optics.calc_twiss(ring3)
    dbetax3 = 1e3*(tw.betax - tw_nom.betax)
    dbetay3 = 1e3*(tw.betay - tw_nom.betay)

    plt.plot(tw_nom.spos, dbetax0, label='uncorr')
    plt.plot(tw_nom.spos, dbetax1, label='cod corr')
    plt.plot(tw_nom.spos, dbetax3, label='tune corr')
    plt.plot(tw_nom.spos, dbetax2, label='symm corr')
    plt.xlabel('spos [m]')
    plt.ylabel('dbeta [mm]')
    plt.grid()
    plt.legend()
    plt.title('BetaX Variation from nominal ({})'.format(config))
    plt.savefig('dbetax-{}.svg'.format(config))
    plt.show()

    plt.plot(tw_nom.spos, dbetay0, label='uncorr')
    plt.plot(tw_nom.spos, dbetay1, label='cod corr')
    plt.plot(tw_nom.spos, dbetay3, label='tune corr')
    plt.plot(tw_nom.spos, dbetay2, label='symm corr')
    plt.xlabel('spos [m]')
    plt.ylabel('dbeta [mm]')
    plt.grid()
    plt.legend()
    plt.title('BetaY Variation from nominal ({})'.format(config))
    plt.savefig('dbetay-{}.svg'.format(config))
    plt.show()


def beta_difference(models, modelname):

    # models = load_models(folder='./results/vchamber-off/')
    ring_nom = si.create_accelerator(ids=None)
    tw_nom, *_ = pyaccel.optics.calc_twiss(ring_nom)

    for config, rings in models.items():
        ring0, ring1, ring2, ring3 = rings
        if modelname == 'ring0':
            ring = ring0
        elif modelname == 'ring1':
            ring = ring1
        elif modelname == 'ring2':
            ring = ring2
        elif modelname == 'ring3':
            ring = ring3
        tw, *_ = pyaccel.optics.calc_twiss(ring)
        dbeta = 1e3*(tw.betax - tw_nom.betax)
        plt.plot(tw_nom.spos, dbeta, label=config)
    plt.xlabel('spos [m]')
    plt.ylabel('beta [mm]')
    plt.grid()
    plt.legend()
    plt.title('BetaX Variation ({})'.format(modelname))
    plt.show()

    for config, rings in models.items():
        ring0, ring1, ring2, ring3 = rings
        if modelname == 'ring0':
            ring = ring0
        elif modelname == 'ring1':
            ring = ring1
        elif modelname == 'ring2':
            ring = ring2
        elif modelname == 'ring3':
            ring = ring3
        tw, *_ = pyaccel.optics.calc_twiss(ring)
        dbeta = 1e3*(tw.betay - tw_nom.betay)
        plt.plot(tw_nom.spos, dbeta, label=config)
    plt.xlabel('spos [m]')
    plt.ylabel('beta [mm]')
    plt.legend()
    plt.title('BetaY Variation ({})'.format(modelname))
    plt.show()


def plot_dynapt_xy(fname, folder='./results/'):
    dynapt = pickle.load(open(folder + 'dynapt_xy_nominal.pickle', 'rb'))['ring0']
    area = 1e6*calc_dynapt_area(dynapt)
    plt.plot(1e3*dynapt[0], 1e3*dynapt[1], label='nominal ({:.1f} mm²)'.format(area))
    config = get_label_delta(fname)
    dynapt = pickle.load(open(folder + 'dynapt_xy_' + config + '.pickle', 'rb'))
    area = 1e6*calc_dynapt_area(dynapt['ring0'])
    plt.plot(1e3*dynapt['ring0'][0], 1e3*dynapt['ring0'][1], label='uncorr ({:.1f} mm²)'.format(area))
    area = 1e6*calc_dynapt_area(dynapt['ring1'])
    plt.plot(1e3*dynapt['ring1'][0], 1e3*dynapt['ring1'][1], label='cod corr ({:.1f} mm²)'.format(area))
    area = 1e6*calc_dynapt_area(dynapt['ring2'])
    plt.plot(1e3*dynapt['ring2'][0], 1e3*dynapt['ring2'][1], label='symm corr ({:.1f} mm²)'.format(area))
    area = 1e6*calc_dynapt_area(dynapt['ring3'])
    plt.plot(1e3*dynapt['ring3'][0], 1e3*dynapt['ring3'][1], label='tune corr ({:.1f} mm²)'.format(area))
    plt.title('Dynamical Aperture XY for {}'.format(config))
    plt.legend()
    plt.xlabel('posx [mm]')
    plt.ylabel('posy [mm]')
    plt.grid()
    plt.xlim([-12,12])
    plt.ylim([0, 6])
    plt.savefig('dynapt-xy-' + config + '.svg')
    plt.show()


def plot_dynapt_ex(fname, folder='./results/'):
    dynapt = pickle.load(open(folder + 'dynapt_ex_nominal.pickle', 'rb'))['ring0']
    plt.plot(1e2*dynapt[0], 1e3*dynapt[1], label='nominal')
    config = get_label_delta(fname)
    dynapt = pickle.load(open(folder + 'dynapt_ex_' + config + '.pickle', 'rb'))
    plt.plot(1e2*dynapt['ring0'][0], 1e3*dynapt['ring0'][1], label='uncorr')
    plt.plot(1e2*dynapt['ring1'][0], 1e3*dynapt['ring1'][1], label='cod corr')
    plt.plot(1e2*dynapt['ring2'][0], 1e3*dynapt['ring2'][1], label='symm corr')
    plt.plot(1e2*dynapt['ring3'][0], 1e3*dynapt['ring3'][1], label='tune corr')
    plt.title('Dynamical Aperture dE-X for {}'.format(config))
    plt.legend()
    plt.xlabel('de [%]')
    plt.ylabel('posx [mm] @ y = 1 mm')
    plt.grid()
    plt.xlim([-5, 5])
    plt.ylim([-16, 0])
    plt.savefig('dynapt-ex-' + config + '.svg')
    plt.show()


def plot_dynapt_xy_all_models(folder='./results/'):
    data = pickle.load(open(folder + 'dynapt_xy_' + 'nominal' + '.pickle', 'rb'))
    dynapt = data['ring0']
    area_nom = 1e6*calc_dynapt_area(dynapt)
    for config in delta_configs:
        label = get_label_delta(id_sabia + config)
        data = pickle.load(open(folder + 'dynapt_xy_' + label + '.pickle', 'rb'))
        dynapt = data['ring0']
        area0 = 1e6*calc_dynapt_area(dynapt) - area_nom
        dynapt = data['ring1']
        area1 = 1e6*calc_dynapt_area(dynapt) - area_nom
        dynapt = data['ring2']
        area2 = 1e6*calc_dynapt_area(dynapt) - area_nom
        dynapt = data['ring3']
        area3 = 1e6*calc_dynapt_area(dynapt) - area_nom
        print('{:<40s}: {:+05.1f} mm²  {:+06.2f} %'.format(label, area3, 100*area3/area_nom))
        plt.plot([area0, area1, area2, area3], '-o', label=label)
    
    plt.title('DynApt XY Area Reduction w.r.t. Nominal ({:.1f} mm²)'.format(area_nom))
    # plt.legend()
    plt.xlabel('0: uncorr, 1:cod, 2:symm, 3:tune')
    plt.ylabel('dynapt delta area [mm²]')
    plt.grid()
    plt.savefig('dynapt-xy-all-models.svg')
    plt.show()


def plot_dynapt_xy_all(modelname, folder='./results/'):
    print('model name: {}'.format(modelname))
    label = 'nominal'
    dynapt = pickle.load(open(folder + 'dynapt_xy_nominal.pickle', 'rb'))['ring0']
    area = 1e6*calc_dynapt_area(dynapt)
    print('{:<40s}: {:5.1f} mm², {:+06.2f} mm, {:+06.2f} mm'.format(label, area, 1e3*dynapt[0][-1], 1e3*dynapt[0][-2]))
    plt.plot(1e3*dynapt[0], 1e3*dynapt[1], label='{} ({:.1f} mm²)'.format(label, area))
    for config in delta_configs:
        label = get_label_delta(id_sabia + config)
        dynapt = pickle.load(open(folder + 'dynapt_xy_' + label + '.pickle', 'rb'))[modelname]
        area = 1e6*calc_dynapt_area(dynapt)
        print('{:<40s}: {:5.1f} mm², {:+06.2f} mm, {:+06.2f} mm'.format(label, area, 1e3*dynapt[0][-1], 1e3*dynapt[0][-2]))
        plt.plot(1e3*dynapt[0], 1e3*dynapt[1], label='{} ({:.1f} mm²)'.format(label, area))
    plt.title('Dynamical Aperture XY for {}'.format(modelname))
    plt.legend()
    plt.xlabel('posx [mm]')
    plt.ylabel('posy [mm]')
    plt.grid()
    plt.xlim([-12,12])
    plt.ylim([0, 6])
    plt.show()


def plot_dynapt_ex_all(modelname, folder='./results/'):
    print('model name: {}'.format(modelname))
    label = 'nominal'
    dynapt = pickle.load(open(folder + 'dynapt_ex_nominal.pickle', 'rb'))['ring0']
    plt.plot(1e2*dynapt[0], 1e3*dynapt[1], label='{}'.format(label))
    for config in delta_configs:
        label = get_label_delta(id_sabia + config)
        dynapt = pickle.load(open(folder + 'dynapt_ex_' + label + '.pickle', 'rb'))[modelname]
        plt.plot(1e2*dynapt[0], 1e3*dynapt[1], label='{}'.format(label))
    plt.title('Dynamical Aperture dE-X for {}'.format(modelname))
    plt.legend()
    plt.xlabel('de [%]')
    plt.ylabel('posx [mm] @ y = 1 mm')
    plt.grid()
    plt.xlim([-5, 5])
    plt.ylim([-16, 0])
    plt.show()


def coupling(models):

    # models = load_models(folder='./results/vchamber-off/')

    # ring = si.create_accelerator(ids=None)
    # traj, *_ = pyaccel.tracking.ring_pass(ring, [0e-7, 0, 1e-5, 0, 0, 0], 1000, turn_by_turn=True)
    # plt.plot(1e3*traj[0, :], 1e3*traj[1, :], '.', color=[0,0,1])
    # plt.plot(1e3*traj[2, :], 1e3*traj[3, :], '.', color=[1,0,0])
    # plt.show()

    for config, rings in models.items():

        *_, ring3 = rings

        tw, *_ = pyaccel.optics.calc_twiss(ring3)
        ep = pyaccel.optics.EquilibriumParametersOhmiFormalism(ring3)
        coup1 = 100 * ep.emity / ep.emitx
        traj, *_ = pyaccel.tracking.ring_pass(ring3, [1e-3, 0, 0, 0, 0, 0], 1000, turn_by_turn=True)
        Jx = max(traj[0,:])**2/tw.betax[0]
        Jy = max(traj[2,:])**2/tw.betay[0]
        coup2 = 100 * Jy/Jx
        print('{:<40s}: {:.4f} %  {:.4f} %'.format(config, coup2, coup1))
        plt.plot(1e3*traj[0, :], 1e3*traj[1, :], '.', color=[0,0,1], label='trajx')
        plt.plot(50*1e3*traj[2, :], 50*1e3*traj[3, :], '.', color=[1,0,0], label='trajy x 50')
        plt.xlabel('pos [mm]')
        plt.ylabel('ang [mrad]')
        plt.legend()
        plt.title(config + '\nJy/Jx = {:.4f} % '.format(coup2))
        plt.savefig('coupling-' + config + '.svg')
        plt.show()


if __name__ == '__main__':

    deltadata = utils.create_deltadata()
    save_models(deltadata)

    # nominal
    # save_dynapt_xy({})
    # save_dynapt_ex({})
    
    # models = load_models(folder='./results/vchamber-on/')
    # configs = list(models.keys())

    # for config in configs[0:4]:
    #     save_dynapt_xy({config:models[config]})
    # for config in configs[4:8]:
    #     save_dynapt_xy({config:models[config]})
    # for config in configs[8:12]:
    #     save_dynapt_xy({config:models[config]})
    # for config in configs[12:16]:
    #     save_dynapt_xy({config:models[config]})
    # for config in configs[16:20]:
    #     save_dynapt_xy({config:models[config]})
    # for config in configs[20:22]:
    #     save_dynapt_xy({config:models[config]})

    # for config in configs[0:4]:
    #     save_dynapt_ex({config:models[config]})
    # for config in configs[4:8]:
    #     save_dynapt_ex({config:models[config]})
    # for config in configs[8:12]:
    #     save_dynapt_ex({config:models[config]})
    # for config in configs[12:16]:
    #     save_dynapt_ex({config:models[config]})
    # for config in configs[16:20]:
    #     save_dynapt_ex({config:models[config]})
    # for config in configs[20:22]:
    #     save_dynapt_ex({config:models[config]})
    
    # ring = si.create_accelerator(ids=None)
    # ring.vchamber_on = optics.CHAMBER_ON
    # nrturns = 4000
    # calc_dynapt_ex(ring, nrturns, demax=0.05, nrpts=9, mindeltax=0.1e-3, xmin=-30e-3, y=1e-3)

    # test_symm()

    # ring = si.create_accelerator(ids=ids)
    # idx= pyaccel.lattice.find_indices(ring, 'fam_name', 'DELTA52')
    # print(idx)
    # pickle.dump(ring, open('test.pickle', 'wb'))
    # ring = pickle.load(open('test.pickle', 'rb'))
    # print(ring[2909].trackcpp_e.length)
    # print(ring[2909].trackcpp_e.kicktable_idx)
    # print(ring[2909].trackcpp_e.rescale_kicks)


    # models = load_models()
    # configs = list(models.keys())

    # ring0, ring1, ring2, ring3 = models[configs[0]]
    # idx= pyaccel.lattice.find_indices(ring0, 'fam_name', 'DELTA52')
    # print(idx)
    # el = ring0[2909]
    # print(el.)
    # print()
    # rx = 0
    # ry = 0
    # print(models[configs[0]])
    # _, lost, _, _, _ = pyaccel.tracking.ring_pass(ring0, [rx, 0, ry, 0, 0, 0], 1000)
    # print(lost)
    pass
