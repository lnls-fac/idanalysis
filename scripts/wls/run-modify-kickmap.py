#!/usr/bin/env python-sirius

from idanalysis.analysis import AnalysisKickmap

import utils
import numpy as np

if __name__ == "__main__":

    gaps = utils.gaps
    phases = utils.phases
    widths = utils.widths

    kickanalysis = AnalysisKickmap()
    kickanalysis.meas_flag = utils.MEAS_FLAG
    kickanalysis.run_shift_kickmap(gaps=gaps, phases=phases,
                                   widths=widths)
    rx = np.linspace(-4.0, +4.0, 21) / 1000  # [m]
    ry = np.linspace(-2.6, +2.6, 21) / 1000  # [m]
    kickanalysis.run_filter_kickmap(gaps=gaps, phases=phases,
                                    widths=widths, rx=rx, ry=ry)
