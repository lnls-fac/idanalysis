#!/usr/bin/env python-sirius

import numpy as np
import utils

from idanalysis import IDKickMap

from utils import FOLDER_BASE
from utils import MEAS_DATA_PATH
from utils import ID_CONFIGS

from run_rk_traj import PHASES, GAPS


def run(idconfig, posx, posy):

    idconfig = utils.get_meas_idconfig(phase, gap)
    MEAS_FILE = ID_CONFIGS[idconfig]

    _, meas_id = MEAS_FILE.split('ID=')
    meas_id = meas_id.replace('.dat', '')
    idkickmap = IDKickMap()
    fmap_fname = FOLDER_BASE + MEAS_DATA_PATH + MEAS_FILE
    idkickmap.fmap_fname = fmap_fname
    idkickmap.beam_energy = 3.0  # [GeV]
    idkickmap.rk_s_step = 2  # [mm]
    idkickmap.fmap_config.traj_init_px = 0
    idkickmap.fmap_config.traj_init_py = 0
    # idkickmap.calc_id_termination_kicks(period_len=50, kmap_idlen=2.773)
    # print(idkickmap.fmap_config)
    idkickmap.fmap_calc_kickmap(posx=posx, posy=posy)
    fname = './results/kickmap-ID{}.txt'.format(meas_id)
    idkickmap.save_kickmap_file(kickmap_filename=fname)


if __name__ == "__main__":
    """."""
    global phase, gap
    phase, gap = PHASES[0], GAPS[0]
    posx = np.arange(-18, +19, 1) / 1000  # [m]
    posy = np.linspace(-12, +12, 3) / 1000  # [m]
