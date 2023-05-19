#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

from idanalysis import IDKickMap

from utils import FOLDER_BASE
from utils import MEAS_DATA_PATH
from utils import ID_CONFIGS
from utils import get_meas_idconfig

from imaids.models import AppleII as _AppleII
from imaids.models import AppleIISabia as _AppleIISabia
from imaids.blocks import Block

from calibrate_model import RadiaModelCalibration


def init_objects(phase, gap, mr=1.25, compare=None):
    """."""
    if compare == True:
        idconfig = get_meas_idconfig(phase, gap)
        MEAS_FILE = ID_CONFIGS[idconfig]
        _, meas_id = MEAS_FILE.split('ID=')
        meas_id = meas_id.replace('.dat', '')
        idkickmap = IDKickMap()
        fmap_fname = FOLDER_BASE + MEAS_DATA_PATH + MEAS_FILE
        idkickmap.fmap_fname = fmap_fname
        fmap = idkickmap.fmap_config.fmap
    else:
        fmap = 0
    nr_periods = 54
    period_length = 50
    block_shape = [[[0.1, 0], [40, 0], [40, -40], [0.1, -40]]]
    longitudinal_distance = 0.2
    block_len = period_length/4 - longitudinal_distance
    start_lengths = [block_len/4, block_len/2, 3*block_len/4, block_len]
    start_distances = [block_len/2, block_len/4, 0, longitudinal_distance]
    end_lenghts = start_lengths[-2::-1]  # Tirar último elemento e inverter
    end_distances = start_distances[-2::-1]  # Tirar último elemento e inverter
    gap = float(gap)
    epu = _AppleIISabia(
                gap=gap, nr_periods=nr_periods,
                period_length=period_length,
                mr=mr, block_shape=block_shape,
                block_subdivision=[[8, 4, 3]],
                start_blocks_length=start_lengths,
                start_blocks_distance=start_distances,
                end_blocks_length=end_lenghts,
                end_blocks_distance=end_distances)

    return epu, fmap


def run(posx, posy, gap, phase):

    epu, fmap = init_objects(phase='+00.00', gap='32.5', mr=1.25, compare=True)
    dp = float(phase)
    cm = RadiaModelCalibration(fmap, epu)
    cm.init_fields()

    # search for best shift and calc scale
    shifts = np.linspace(-0.25, 0.25, 31) * epu.period_length
    minshift, minscale, minresidue = shifts[0], 1.0, float('inf')
    for shift in shifts:
        residue, scale, _ = cm.shiftscale_calc_residue(shift=shift)
        print(
            'shift: {:+08.4f} mm -> residue: {:07.5f} T'.format(shift, residue))
        if residue < minresidue:
            minshift, minscale, minresidue = shift, scale, residue

    scale = minscale
    gap = str(gap)
    epu, _ = init_objects(phase=phase, gap=gap, mr=1.25*-1*scale)
    epu.dp = dp
    filename = './results/model/kickmap' + 'p' + phase + 'g' + gap + '.dat'
    epu.save_kickmap(
        filename, 3e6, posx, posy, zmin=-1800, zmax=1800, rkstep=3)


if __name__ == "__main__":
    """."""
    posx = np.arange(-12, +13, 1)  # [mm]
    posy = np.arange(-5, +6, 1)  # [mm]
    gap_list = [22.0, 23.3, 25.7, 29.3, 32.5, 40.9]
    # phase_list = ['-25.00', '-16.39', '+00.00','+16.39', '+25.00']
    phase_list = ['+00.00']
    for phase in phase_list:
        for gap in gap_list:
            run(posx, posy, gap, phase)
