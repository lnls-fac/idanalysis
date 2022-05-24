#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt
from apsuite.orbcorr import OrbitCorr, CorrParams


from pymodels import si

import pyaccel



# create model and structures
model = si.create_accelerator()
famdata = si.get_family_data(model)
chs = [val[0] for val in famdata['CH']['index']]
bpms = [val[0] for val in famdata['CH']['index']]
bpms = famdata['BPM']['index']
spos = pyaccel.lattice.find_spos(model, indices=bpms)

# create orbit correcor
cparams = CorrParams()
cparams.tolerance = 1e-8  # [m]
cparams.maxnriters = 20

ocorr = OrbitCorr(model, 'SI')

# get unperturbed orbit
orb0 = ocorr.get_orbit()

# perturb orbit
model[chs[0]].hkick_polynom = 10e-6  # [rad]

# get perturbed orbit
orb1 = ocorr.get_orbit()

# calc closed orbit distortions (cod) before correction
codu = orb1 - orb0
codux = codu[:len(bpms)]
coduy = codu[len(bpms):]

# calc response matrix and correct orbit
respm = ocorr.get_jacobian_matrix()
if not ocorr.correct_orbit(jacobian_matrix=respm, goal_orbit=orb0):
    print('Could not correct orbit!')

# get corrected orbit
orb2 = ocorr.get_orbit()

# calc closed orbit distortions (cod) after correction
codc = orb2 - orb0
codcx = codc[:len(bpms)]
codcy = codc[len(bpms):]

# plt.plot(spos, 1e6*codux, label='Uncorrected')
plt.plot(spos, 1e6*codcx, label='Corrected')
plt.xlabel('spos [m]')
plt.ylabel('codx [um]')
plt.legend()
plt.show()




# # orb0 = pyaccel.tracking.find_orbit4(model, indices='closed')
# model[chs[0][0]].hkick_polynom = 10e-6  # [rad]
# # orb1 = pyaccel.tracking.find_orbit4(model, indices='closed')

