#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt
from idanalysis import AnalysisFromRadia

obj = AnalysisFromRadia()

obj.rz_max = 100
obj.rz_nrpts = 501

obj.rt_max = 10
obj.rt_nrpts = 301

obj.traj_init_rz = -100
obj.traj_max_rz = 100
obj.kmap_idlen = 1.5
obj.gridx = list(np.linspace(-10, +11, 3) / 1000)  # [m]
obj.gridy = list(np.linspace(-2, +2, 2) / 1000)  # [m]

obj.run_calc_fields()
obj.run_plot_data(width=50, gap=40)
obj.run_generate_kickmap()
