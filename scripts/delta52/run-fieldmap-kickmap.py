#!/usr/bin/env python-sirius

import numpy as np
from idanalysis.analysis import FieldAnalysisFromFieldmap
import utils

if __name__ == "__main__":

    fmap_fanalysis = FieldAnalysisFromFieldmap()
    fmap_fanalysis.kmap_idlen = utils.ID_KMAP_LEN
    fmap_fanalysis.config_idx = 2

    # Grid for low beta
    fmap_fanalysis.gridx = list(np.linspace(-4.0, +4.0, 21) / 1000)  # [m]
    fmap_fanalysis.gridy = list(np.linspace(-1.0, +1.0, 3) / 1000)  # [m]

    fmap_fanalysis.run_calc_fields()
    phase = utils.phases[2]
    sulfix = '-phase' + fmap_fanalysis._get_phase_str(phase)
    fmap_fanalysis.run_plot_data(phase=phase,
                                 gap=utils.gaps[0],
                                 sulfix=sulfix)
    fmap_fanalysis.run_generate_kickmap()
