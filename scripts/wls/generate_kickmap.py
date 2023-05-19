#!/usr/bin/env python-sirius

import numpy as np

from idanalysis import IDKickMap

from utils import FOLDER_BASE
from utils import DATA_PATH
from utils import WLS_CONFIGS


def run(idconfig, posx, posy):
    
    MEAS_FILE = WLS_CONFIGS[idconfig]
    fmap_fname = FOLDER_BASE + DATA_PATH + MEAS_FILE
    
    idkickmap = IDKickMap()
    # idkickmap.calc_id_termination_kicks(fmap_fname=fmap_fname, period_len=10, kmap_idlen=1.2)
    idkickmap.fmap_calc_kickmap(fmap_fname=fmap_fname, posx = posx, posy = posy)
    fname = './results/{}/kickmap-{}.txt'.format(idconfig, idconfig)
    idkickmap.save_kickmap_file(kickmap_filename=fname)


if __name__ == "__main__":
    """."""
    posx = np.linspace(-9, +9, 31) / 1000  # [m]
    posy = np.linspace(-12, +12, 3) / 1000  # [m]
    idconfig = 'I228A'
    run(idconfig, posx, posy)

