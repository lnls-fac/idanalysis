#!/usr/bin/env python-sirius

"""Script to check kickmap"""
from idanalysis.analysis import AnalysisKickmap

import utils

if __name__ == '__main__':

    planes = ['x', 'y']
    kick_planes = ['x', 'y']
    dgvs = utils.dgv
    phase = utils.phases[0]

    kickanalysis = AnalysisKickmap()
    kickanalysis.save_flag = True
    kickanalysis.plot_flag = True
    kickanalysis.shift_flag = True
    kickanalysis.filter_flag = False
    kickanalysis.linear = False
    kickanalysis.meas_flag = True

    # kickanalysis.check_kick_at_plane(
        # phase=0, dgv=[0, 13.125, 26.25], mf='_True',
        # planes=planes, kick_planes=kick_planes)

    kickanalysis.check_kick_all_planes(
        phase=-13.125, dgv=26.25, mf='_True',
        planes=planes, kick_planes=kick_planes)

    # kickanalysis.check_kick_at_plane_trk(
    #    phase=-26.25, dgv=26.25, mf='_True')
