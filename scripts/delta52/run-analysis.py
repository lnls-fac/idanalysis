#!/usr/bin/env python-sirius

import numpy as np
from idanalysis.analysis import AnalysisEffects
import utils


if __name__ == "__main__":

    analysis = AnalysisEffects()
    analysis.fitted_model = False
    analysis.shift_flag = True
    analysis.filter_flag = False
    analysis.calc_type = analysis.CALC_TYPES.symmetrized
    analysis.orbcorr_plot_flag = False
    analysis.bb_plot_flag = True
    analysis.linear = False
    analysis.meas_flag = False

    dgvs = utils.dgv
    for phase in utils.phases:
        for dgv in dgvs:
            analysis.run_analysis_dynapt(phase=phase, dgv=dgv)
