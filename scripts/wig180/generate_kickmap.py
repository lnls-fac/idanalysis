#!/usr/bin/env python-sirius

import numpy as np

from idanalysis import IDKickMap

from utils import FOLDER_BASE
from utils import DATA_PATH
from utils import ID_CONFIGS


def run(idconfig, posx, posy):

    MEAS_FILE = ID_CONFIGS[idconfig]

    _, meas_id =  MEAS_FILE.split('ID=')
    meas_id = meas_id.replace('.dat', '')
    idkickmap = IDKickMap()
    fmap_fname = FOLDER_BASE + DATA_PATH + MEAS_FILE
    idkickmap.fmap_fname = fmap_fname
    idkickmap.beam_energy = 3.0  # [GeV]
    idkickmap.rk_s_step = 0.2  # [mm]
    idkickmap.calc_id_termination_kicks(fmap_fname=fmap_fname, period_len=180, kmap_idlen=2.654)
    idkickmap.fmap_calc_kickmap(posx = posx, posy = posy)
    fname = './results/{}/kickmap-ID{}.txt'.format(idconfig, meas_id)
    idkickmap.save_kickmap_file(kickmap_filename=fname)


if __name__ == "__main__":
    """."""
    posx = np.linspace(-15, +15, 31) / 1000  # [m]
    posy = np.linspace(-12, +12, 3) / 1000  # [m]
    idconfig = 'ID4019'  # gap 49.73 mm, correctors with zero current
    # idconfig = 'ID3979'  # gap 59.6 mm, correctors with zero current
    # idconfig = 'ID4017'  # gap 59.6 mm, correctors with best current
    # idconfig = 'ID4020'  # gap 45.0 mm, correctors with zero current
    # idconfig = 'ID3969'  #gap 59.6 mm, without correctors
    run(idconfig, posx, posy)
