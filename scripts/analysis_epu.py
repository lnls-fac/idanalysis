#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

from idanalysis.trajectory import IDTrajectory

from pyaccel import lattice as pyacc_lat
from pyaccel import optics as pyacc_opt
from pyaccel.optics.edwards_teng import estimate_coupling_parameters
from idanalysis.epudata import EPUData
from idanalysis.model import calc_optics, create_model, get_id_epu_list

from utils import create_epudata

# create object with list of all possible EPU50 configurations
configs = create_epudata()

# take first configuration
config = configs[2]
print(configs.get_config_label(config))

# create list of IDs to be inserted
kmap_fname = configs.get_kickmap_filename(config)
ids = get_id_epu_list(kmap_fname)
print(ids[0])

# create model without IDs
model1 = create_model(ids=None)
twiss1, *_ = pyacc_opt.calc_twiss(model1)
mux1, muy1 = twiss1.mux, twiss1.muy
betax1, betay1 = twiss1.betax, twiss1.betay

# create model with IDs
model2 = create_model(ids=ids)
twiss2, *_ = pyacc_opt.calc_twiss(model2)
mux2, muy2 = twiss2.mux, twiss2.muy
betax2, betay2 = twiss2.betax, twiss2.betay

# check optics
epu50 = pyacc_lat.find_indices(model2, 'fam_name', 'EPU50')
print('EPU50 indices: ', epu50)
print('model length : {}'.format(model2.length))
print('dtunex       : {}'.format((mux2[-1] - mux1[-1]) / 2 / np.pi))
print('dtuney       : {}'.format((muy2[-1] - muy1[-1]) / 2 / np.pi))

# plot beta
# plt.plot(twiss1.spos, betax1, color=(0,0,0.5), label='w/o IDs')
# plt.plot(twiss2.spos, betax2, color=(0,0,1.0), label='with IDs')
plt.plot(twiss2.spos, 100*(betax2-betax1)/betax1, color=(0,0,1.0), label='with IDs')
plt.show()
