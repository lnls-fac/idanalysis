#!/usr/bin/env python-sirius

import numpy as np

from idanalysis import IDKickMap

import utils

MEAS_DATA_PATH = utils.MEAS_DATA_PATH


def run(idconfig, posx, posy):

    MEAS_FILE = utils.ID_CONFIGS[idconfig]
    idkickmap = IDKickMap()
    fmap_fname = MEAS_DATA_PATH + MEAS_FILE
    idkickmap.fmap_fname = fmap_fname
    idkickmap.beam_energy = 3.0  # [GeV]
    idkickmap.rk_s_step = 0.2  # [mm]
    idkickmap.fmap_calc_kickmap(posx=posx, posy=posy)
    fname = './results/measurements/kickmaps/kickmap-{}.txt'.format(idconfig)
    idkickmap.save_kickmap_file(kickmap_filename=fname)


if __name__ == "__main__":
    """."""
    posx = np.linspace(-4.9, +4.9, 11) / 1000  # [m]
    posy = np.linspace(-1, 1, 2)  # [m]
    configs = np.arange(58, 72, 1)
    for nrconfig in configs:
        idconfig = 'ID44' + str(nrconfig)
        run(idconfig, posx, posy)
