#!/usr/bin/env python-sirius

"""Script to check kickmap"""
from idanalysis.analysis import AnalysisKickmap

import utils

if __name__ == '__main__':

    planes = ['x', 'y']
    kick_planes = ['x', 'y']
    gap = utils.gaps[0]
    phase = utils.phases[0]
    widths = utils.widths

    kickanalysis = AnalysisKickmap()
    kickanalysis.save_flag = True
    kickanalysis.plot_flag = True
    kickanalysis.shift_flag = False
    kickanalysis.filter_flag = False
    kickanalysis.linear = False

    kickanalysis.check_kick_at_plane(
        gap=gap, phase=phase,
        planes=planes, kick_planes=kick_planes)

    # kickanalysis.check_kick_all_planes(
        # width=widths[0], phase=phase, gap=gap,
        # planes=planes, kick_planes=kick_planes)

    # kickanalysis.check_kick_at_plane_trk(
        # width=widths[0], phase=phase, gap=gap)
