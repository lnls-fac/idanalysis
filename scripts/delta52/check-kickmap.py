#!/usr/bin/env python-sirius

"""Script to check kickmap"""
from idanalysis.analysis import AnalysisKickmap

import utils

if __name__ == '__main__':

    planes = ['x', 'y']
    kick_planes = ['x', 'y']
    gaps = utils.gaps
    phase = utils.phases[2]
    width = utils.widths[0]

    kickanalysis = AnalysisKickmap()
    kickanalysis.save_flag = True
    kickanalysis.plot_flag = True
    kickanalysis.shift_flag = utils.SHIFT_FLAG
    kickanalysis.filter_flag = utils.FILTER_FLAG
    kickanalysis.linear = False
    kickanalysis.meas_flag = utils.MEAS_FLAG

    kickanalysis.check_kick_at_plane(
        width=width, gap=gaps, phase=phase,
        planes=planes, kick_planes=kick_planes)

    # kickanalysis.check_kick_all_planes(
        # width=widths[0], phase=phase, gap=gap,
        # planes=planes, kick_planes=kick_planes)

    # kickanalysis.check_kick_at_plane_trk(
        # width=width, phase=phase, gap=gaps[0])
