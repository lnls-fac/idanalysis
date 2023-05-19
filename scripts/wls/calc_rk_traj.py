#!/usr/bin/env python-sirius

from fieldmaptrack.common_analysis import multipoles_analysis
import numpy as np
import matplotlib.pyplot as plt

from idanalysis import IDKickMap

from utils import FOLDER_BASE
from utils import DATA_PATH
from utils import ID_CONFIGS


def create_idkickmap(idconfig):
    """."""
    # get fieldmap file name
    MEAS_FILE = ID_CONFIGS[idconfig]
    _, meas_id = MEAS_FILE.split('I')
    meas_id = meas_id.replace('.dat', '')
    fmap_fname = FOLDER_BASE + DATA_PATH + MEAS_FILE

    # create  IDKickMap and set beam energy, fieldmap filename and RK step
    idkickmap = IDKickMap()
    idkickmap.fmap_fname = fmap_fname
    idkickmap.beam_energy = 3.0  # [GeV]
    # # print(idkickmap.brho)

    # set various fmap_configurations
    idkickmap.fmap_config.traj_init_rz = 1 * min(idkickmap.fmap.rz)
    idkickmap.fmap_config.traj_rk_min_rz = 1 * max(idkickmap.fmap.rz)

    return idkickmap


def plot_rk_traj(idconfig, traj_init_rx, traj_init_ry, rk_s_step=0.2):
    """."""
    # create IDKickMap
    idkickmap = create_idkickmap(idconfig)

    idkickmap.rk_s_step = rk_s_step
    idkickmap.fmap_calc_trajectory(
        traj_init_rx=traj_init_rx, traj_init_ry=traj_init_ry)

    # multipolar analysis
    idkickmap.fmap_config.multipoles_perpendicular_grid = np.linspace(-3,3,7)
    # idkickmap.fmap_config.multipoles_perpendicular_grid = \
        # [-3, -2, -1, 0, 1, 2, -3]
    idkickmap.fmap_config.multipoles_normal_field_fitting_monomials = np.arange(0, 3, 1).tolist()
    idkickmap.fmap_config.multipoles_skew_field_fitting_monomials = np.arange(0, 3, 1).tolist()
    idkickmap.fmap_config.multipoles_r0 = 12  # [mm]
    idkickmap.fmap_config.normalization_monomial = 0
    IDKickMap.multipoles_analysis(idkickmap.fmap_config)
    multipoles = idkickmap.fmap_config.multipoles
    # print(idkickmap.fmap_config.multipoles)
    plt.figure(1)
    plt.plot(idkickmap.fmap_config.traj.rz, multipoles.normal_multipoles[2, :],'.-')


    fmap = idkickmap.fmap_config.fmap
    rz = fmap.rz
    bx = fmap.bx[fmap.ry_zero][fmap.rx_zero][:]
    by = fmap.by[fmap.ry_zero][fmap.rx_zero][:]
    bz = fmap.bz[fmap.ry_zero][fmap.rx_zero][:]
    plt.figure(2)
    plt.plot(rz, bx, color='b', label="Bx")
    plt.plot(rz, by, color='C1', label="By")
    plt.plot(rz, bz, color='g', label="Bz")
    plt.xlabel('rz [mm]')
    plt.ylabel('Field [T]')
    plt.legend()
    plt.show()

    # labelx = 'rx @ end: {:+.1f} um'.format(1e3*traj.rx[-1])
    # labely = 'ry @ end: {:+.1f} um'.format(1e3*traj.ry[-1])
    # plt.figure(2)
    # plt.plot(traj.rz, 1e3*traj.rx, '.-', label=labelx)
    # plt.plot(traj.rz, 1e3*traj.ry, '.-', label=labely)
    # plt.xlabel('rz [mm]')
    # plt.ylabel('pos [um]')
    # plt.legend()
    # plt.title('Runge-Kutta Trajectory Pos ({})'.format(idconfig))
    # plt.show()

    # labelx = 'px @ end: {:+.1f} urad'.format(1e6*traj.px[-1])
    # labely = 'py @ end: {:+.1f} urad'.format(1e6*traj.py[-1])
    # plt.plot(traj.rz, 1e6*traj.px, '.-', label=labelx)
    # plt.plot(traj.rz, 1e6*traj.py, '.-', label=labely)
    # plt.xlabel('rz [mm]')
    # plt.ylabel('ang [urad]')
    # plt.legend()
    # plt.title('Runge-Kutta Trajectory Ang ({})'.format(idconfig))
    # plt.show()


if __name__ == "__main__":
    """."""
    idconfig = 'I228A'

    traj_init_rx = 0.0  # [mm]
    traj_init_ry = 0.0  # [mm]
    rk_s_step = 0.2  # [mm]

    plot_rk_traj(idconfig, traj_init_rx, traj_init_ry, rk_s_step)
