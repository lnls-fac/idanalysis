#!/usr/bin/env python-sirius

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
    _, meas_id =  MEAS_FILE.split('ID=')
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

    traj = idkickmap.traj

    labelx = 'rx @ end: {:+.1f} um'.format(1e3*traj.rx[-1])
    labely = 'ry @ end: {:+.1f} um'.format(1e3*traj.ry[-1])
    plt.plot(traj.rz, 1e3*traj.rx, '.-', label=labelx)
    plt.plot(traj.rz, 1e3*traj.ry, '.-', label=labely)
    plt.xlabel('rz [mm]')
    plt.ylabel('pos [um]')
    plt.legend()
    plt.title('Runge-Kutta Trajectory Pos ({})'.format(idconfig))
    plt.show()

    labelx = 'px @ end: {:+.1f} urad'.format(1e6*traj.px[-1])
    labely = 'py @ end: {:+.1f} urad'.format(1e6*traj.py[-1])
    plt.plot(traj.rz, 1e6*traj.px, '.-', label=labelx)
    plt.plot(traj.rz, 1e6*traj.py, '.-', label=labely)
    plt.xlabel('rz [mm]')
    plt.ylabel('ang [urad]')
    plt.legend()
    plt.title('Runge-Kutta Trajectory Ang ({})'.format(idconfig))
    plt.show()


if __name__ == "__main__":
    """."""
    idconfig = 'ID4019'  # gap 49.73 mm, correctors with zero current
    # idconfig = 'ID3979'  # gap 59.6 mm, correctors with zero current
    # idconfig = 'ID4017'  # gap 59.6 mm, correctors with best current
    # idconfig = 'ID4020'  # gap 45.0 mm, correctors with zero current
    # idconfig = 'ID3969'  #gap 59.6 mm, without correctors

    traj_init_rx = 0.0  # [mm]
    traj_init_ry = 0.0  # [mm]
    rk_s_step = 0.2  # [mm]
    plot_rk_traj(idconfig, traj_init_rx, traj_init_ry, rk_s_step)
