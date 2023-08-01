#!/usr/bin/env python-sirius

from fieldmaptrack.common_analysis import multipoles_analysis
import numpy as np
import matplotlib.pyplot as plt

from idanalysis import IDKickMap
from mathphys.functions import save_pickle, load_pickle


def create_fmap_fname(current, i):
    fname = './meas-data/2023-06-12_SWLS_Sirius_Model9.0_standard_I={}A_Aibox2m_X=-10_10mm_Y=-3_3mm_Z=-1000_1000mm_ID=000{}.txt'.format(current, i+1)
    return fname


def create_idkickmap(current, i):
    """."""
    # create  IDKickMap and set beam energy, fieldmap filename and RK step
    idkickmap = IDKickMap()
    fmap_fname = create_fmap_fname(current, i)
    idkickmap.fmap_fname = fmap_fname
    idkickmap.beam_energy = 3.0  # [GeV]
    # # print(idkickmap.brho)

    # set various fmap_configurations
    idkickmap.fmap_config.traj_init_rz = 1 * min(idkickmap.fmap.rz)
    idkickmap.fmap_config.traj_rk_min_rz = 1 * max(idkickmap.fmap.rz)

    return idkickmap


def calc_rk_traj(current, i, traj_init_rx, traj_init_ry, rk_s_step=0.2):
    """."""
    # create IDKickMap
    idkickmap = create_idkickmap(current, i)

    idkickmap.rk_s_step = rk_s_step
    print('Calculating trajectory...')
    idkickmap.fmap_calc_trajectory(
        traj_init_rx=traj_init_rx, traj_init_ry=traj_init_ry)

    # multipolar analysis
    idkickmap.fmap_config.multipoles_perpendicular_grid = np.linspace(-3, 3, 7)
    # idkickmap.fmap_config.multipoles_perpendicular_grid = \
        # [-3, -2, -1, 0, 1, 2, -3]
    idkickmap.fmap_config.multipoles_normal_field_fitting_monomials =\
        np.arange(0, 3, 1).tolist()
    idkickmap.fmap_config.multipoles_skew_field_fitting_monomials =\
        np.arange(0, 3, 1).tolist()
    idkickmap.fmap_config.multipoles_r0 = 12  # [mm]
    idkickmap.fmap_config.normalization_monomial = 0
    print('Calculating multipoles...')
    IDKickMap.multipoles_analysis(idkickmap.fmap_config)
    save_pickle(idkickmap, './results/model/data/general/data_WLS_I=' + str(current), overwrite=True)
    return idkickmap


def plot_results(idkickmap, current):
    multipoles = idkickmap.fmap_config.multipoles
    traj = idkickmap.fmap_config.traj
    plt.figure(1)
    i_nquad = multipoles.normal_multipoles_integral[1]
    i_squad = multipoles.skew_multipoles_integral[1]
    labeln = 'Normal - Integrated : {:.3f} T'.format(i_nquad)
    labels = 'Skew - Integrated : {:.3f} T'.format(i_squad)
    plt.plot(idkickmap.fmap_config.traj.rz,
             multipoles.normal_multipoles[1, :], '-', label=labeln)
    plt.plot(idkickmap.fmap_config.traj.rz,
             multipoles.skew_multipoles[1, :], '-', label=labels)
    plt.title('Quadrupolar component (I = {} A)'.format(current))
    plt.xlabel('z [mm]')
    plt.ylabel('Quadrupolar term [T/m]')
    plt.legend()
    plt.grid()
    if current == 236.2:
        current = 236
    filename = 'I_' + str(current) + 'quadrupoles'
    plt.savefig(filename, dpi=300)
    plt.clf()

    plt.figure(2)
    i_nsext = multipoles.normal_multipoles_integral[2]
    i_ssext = multipoles.skew_multipoles_integral[2]
    labeln = 'Normal - Integrated : {:.3f} T/m'.format(i_nsext)
    labels = 'Skew - Integrated : {:.3f} T/m'.format(i_ssext)
    plt.plot(idkickmap.fmap_config.traj.rz,
             multipoles.normal_multipoles[2, :], '-', label=labeln)
    plt.plot(idkickmap.fmap_config.traj.rz,
             multipoles.skew_multipoles[2, :], '-', label=labels)
    plt.title('Sextupolar component (I = {} A)'.format(current))
    plt.xlabel('z [mm]')
    plt.ylabel('Sextupolar term [T/m²]')
    plt.legend()
    plt.grid()
    if current == 236.2:
        current = 236
    filename = 'I_' + str(current) + 'sextupoles'
    plt.savefig(filename, dpi=300)
    plt.clf()

    # rz = traj.rz
    # bx = traj.bx
    # by = traj.by
    # bz = traj.bz
    # plt.figure(3)
    # plt.plot(rz, bx, color='b', label="Bx")
    # plt.plot(rz, by, color='C1', label="By")
    # plt.plot(rz, bz, color='g', label="Bz")
    # plt.xlabel('rz [mm]')
    # plt.ylabel('Field [T]')
    # plt.title('Field on traj (I = {} A)'.format(current))
    # plt.legend()

    # plt.figure(4)
    # labelx = 'rx @ end: {:+.1f} um'.format(1e3*traj.rx[-1])
    # labely = 'ry @ end: {:+.1f} um'.format(1e3*traj.ry[-1])
    # plt.plot(traj.rz, 1e3*traj.rx, '.-', label=labelx)
    # plt.plot(traj.rz, 1e3*traj.ry, '.-', label=labely)
    # plt.xlabel('rz [mm]')
    # plt.ylabel('pos [um]')
    # plt.legend()
    # plt.title('Runge-Kutta Trajectory Pos (I = {} A)'.format(current))

    # plt.figure(5)
    # labelx = 'px @ end: {:+.1f} urad'.format(1e6*traj.px[-1])
    # labely = 'py @ end: {:+.1f} urad'.format(1e6*traj.py[-1])
    # plt.plot(traj.rz, 1e6*traj.px, '.-', label=labelx)
    # plt.plot(traj.rz, 1e6*traj.py, '.-', label=labely)
    # plt.xlabel('rz [mm]')
    # plt.ylabel('ang [urad]')
    # plt.legend()
    # plt.title('Runge-Kutta Trajectory Ang (I = {} A)'.format(current))
    # plt.show()


if __name__ == "__main__":
    """."""
    currents = [20, 60, 100, 150, 200, 236.2, 240]
    traj_init_rx = 0.0  # [mm]
    traj_init_ry = 0.0  # [mm]
    rk_s_step = 0.2  # [mm]

    for i, current in enumerate(currents):
        # idkickmap = calc_rk_traj(current, i, traj_init_rx, traj_init_ry, rk_s_step)
        idkickmap = load_pickle('./results/model/data/general/data_WLS_I=' + str(current))
        plot_results(idkickmap, current)
