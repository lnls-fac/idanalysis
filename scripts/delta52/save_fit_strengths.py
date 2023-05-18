#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt
from mathphys.functions import load_pickle, save_pickle
import pyaccel
import pymodels

FIT_PATH = '/home/gabriel/Desktop/my-data-by-day/2023-05-15-SI_low_coupling/fitting_ref_config_before_low_coupling.pickle'

if __name__ == "__main__":

    model = load_pickle(FIT_PATH)['fit_model']
    famdata = pymodels.si.families.get_family_data(model)
    idcs_qn = np.array(famdata['QN']['index']).ravel()
    idcs_qs = np.array(famdata['QS']['index']).ravel()
    kl = pyaccel.lattice.get_attribute(model, 'KL', idcs_qn).ravel()
    ksl = pyaccel.lattice.get_attribute(model, 'KsL', idcs_qs).ravel()
    data = dict()
    data['index_qn'] = idcs_qn
    data['index_qs'] = idcs_qs
    data['KL'] = kl
    data['KsL'] = ksl

    kfile = FIT_PATH.replace('coupling.pickle', 'coupling_strengths.pickle')
    save_pickle(data, kfile, overwrite=True)
