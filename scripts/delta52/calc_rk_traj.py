#!/usr/bin/env python-sirius

from fieldmaptrack.common_analysis import multipoles_analysis
from fieldmaptrack import Multipoles
import numpy as np
import matplotlib.pyplot as plt
import utils

from idanalysis import IDKickMap
from mathphys.functions import save_pickle, load_pickle


def create_model(phase, gap):
    model = utils.generate_radia_model(
                        phase=phase,
                        gap=gap,
                        solve=utils.SOLVE_FLAG)
    return model


def create_idkickmap(phase, gap):
    """."""
    # create  IDKickMap and set beam energy, fieldmap filename and RK step
    idkickmap = IDKickMap()
    print('creating model...')
    idkickmap.radia_model = create_model(phase, gap)
    idkickmap.beam_energy = 3.0
    idkickmap.rk_s_step = rk_s_step
    idkickmap.traj_init_rz = -1*(utils.ID_PERIOD*utils.NR_PERIODS + 40)
    idkickmap.traj_rk_min_rz = utils.ID_PERIOD*utils.NR_PERIODS + 40
    idkickmap.kmap_idlen = 1.2
    # # print(idkickmap.brho)

    # set various fmap_configurations

    return idkickmap


def calc_rk_traj(phase, gap, traj_init_rx, traj_init_ry, rk_s_step=0.2):
    """."""
    # create IDKickMap
    idkickmap = create_idkickmap(phase, gap)
    traj_init_rz = -1*(utils.ID_PERIOD*utils.NR_PERIODS + 40)
    traj_rk_min_rz = utils.ID_PERIOD*utils.NR_PERIODS + 40
    idkickmap.rk_s_step = rk_s_step
    print('Calculating trajectory...')
    idkickmap.fmap_calc_trajectory(
            traj_init_rx=traj_init_rx, traj_init_ry=traj_init_ry,
            traj_init_rz=traj_init_rz,
            traj_rk_min_rz=traj_rk_min_rz)
    traj = idkickmap.traj

    # multipolar analysis
    harmonics = [0, 1, 2, 3]
    n_list = harmonics
    s_list = harmonics
    mp = Multipoles()
    mp.trajectory = traj
    mp.perpendicular_grid = np.linspace(-3, 3, 21)
    mp.normal_field_fitting_monomials = n_list
    mp.skew_field_fitting_monomials = s_list

    print('Calculating multipoles...')
    mp.calc_multipoles(is_ref_trajectory_flag=False)
    mp.calc_multipoles_integrals()
    fname = './results/model/data/general/traj_delta_p{}_g{}'.format(
        phase, gap)
    data = dict()
    data['traj'] = traj
    data['multipoles'] = mp
    save_pickle(data, fname)
    return data


def plot_results(traj, multipoles, phase, gap):
    plt.figure(1)
    i_nquad = multipoles.normal_multipoles_integral[1]
    i_squad = multipoles.skew_multipoles_integral[1]
    labeln = 'Integrated : {:.3f} T'.format(i_nquad)
    labels = 'Integrated : {:.3f} T'.format(i_squad)
    plt.plot(traj.rz,
             multipoles.normal_multipoles[1, :], '.-', label=labeln)
    plt.plot(traj.rz,
             multipoles.skew_multipoles[1, :], '.-', label=labels)
    plt.title('Quadrupolar component (dgv = {} mm)'.format(gap))
    plt.xlabel('rz [mm]')
    plt.ylabel('Quadrupolar term [T/m]')
    plt.legend()
    plt.grid()

    plt.figure(2)
    i_nsext = multipoles.normal_multipoles_integral[2]
    i_ssext = multipoles.skew_multipoles_integral[2]
    labeln = 'Integrated : {:.3f} T/m'.format(i_nsext)
    labels = 'Integrated : {:.3f} T/m'.format(i_ssext)
    plt.plot(traj.rz,
             multipoles.normal_multipoles[2, :], '.-', label=labeln)
    plt.plot(traj.rz,
             multipoles.skew_multipoles[2, :], '.-', label=labels)
    plt.title('Sextupolar component (dgv = {} mm)'.format(gap))
    plt.xlabel('rz [mm]')
    plt.ylabel('Sextupolar term [T/m²]')
    plt.legend()
    plt.grid()

    rz = traj.rz
    bx = traj.bx
    by = traj.by
    bz = traj.bz
    plt.figure(3)
    plt.plot(rz, bx, color='b', label="Bx")
    plt.plot(rz, by, color='C1', label="By")
    plt.plot(rz, bz, color='g', label="Bz")
    plt.xlabel('rz [mm]')
    plt.ylabel('Field [T]')
    plt.title('Field on traj (dgv = {} mm)'.format(gap))
    plt.legend()

    plt.figure(4)
    labelx = 'rx @ end: {:+.1f} um'.format(1e3*traj.rx[-1])
    labely = 'ry @ end: {:+.1f} um'.format(1e3*traj.ry[-1])
    plt.plot(traj.rz, 1e3*traj.rx, '.-', label=labelx)
    plt.plot(traj.rz, 1e3*traj.ry, '.-', label=labely)
    plt.xlabel('rz [mm]')
    plt.ylabel('pos [um]')
    plt.legend()
    plt.title('Runge-Kutta Trajectory Pos (dgv = {} mm)'.format(gap))

    plt.figure(5)
    labelx = 'px @ end: {:+.1f} urad'.format(1e6*traj.px[-1])
    labely = 'py @ end: {:+.1f} urad'.format(1e6*traj.py[-1])
    plt.plot(traj.rz, 1e6*traj.px, '.-', label=labelx)
    plt.plot(traj.rz, 1e6*traj.py, '.-', label=labely)
    plt.xlabel('rz [mm]')
    plt.ylabel('ang [urad]')
    plt.legend()
    plt.title('Runge-Kutta Trajectory Ang (dgv = {} mm)'.format(gap))
    plt.show()


if __name__ == "__main__":
    """."""
    gap = 26.250
    phase = -13.125
    traj_init_rx = 0.0  # [mm]
    traj_init_ry = 0.0  # [mm]
    rk_s_step = 0.2  # [mm]
    fname = './results/model/data/general/traj_delta_p{}_g{}'.format(
        phase, gap)

    data = calc_rk_traj(phase, gap, traj_init_rx, traj_init_ry, rk_s_step)
    data = load_pickle(fname)
    traj = data['traj']
    multipoles = data['multipoles']
    plot_results(traj, multipoles, phase, gap)
