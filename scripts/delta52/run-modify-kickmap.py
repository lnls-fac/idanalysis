#!/usr/bin/env python-sirius

from idanalysis.analysis import AnalysisKickmap

import utils


if __name__ == "__main__":

    dgvs = utils.dgv
    phases = utils.phases


    kickanalysis = AnalysisKickmap()
    kickanalysis.meas_flag = False

    for phase in phases:
        for dgv in dgvs:
            kickanalysis.run_shift_kickmap(phase=phase, dgv=dgv)

    # kickanalysis.run_filter_kickmap(gaps=gaps, phases=phases,
                                    # widths=widths)
