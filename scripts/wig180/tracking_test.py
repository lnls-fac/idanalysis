#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

import pyaccel
from pyaccel import optics as pyacc_opt

import pymodels

from idanalysis import orbcorr as orbcorr
from idanalysis import model as model
from idanalysis import optics as optics

import utils



def create_model_ids(idconfig):
    """."""
    print('--- model with kick-model wiggler ---')
    ids = utils.create_ids(idconfig=idconfig, rescale_kicks=1)
    model = pymodels.si.create_accelerator(ids=ids)
    twiss, *_ = pyaccel.optics.calc_twiss(model, indices='closed')
    print('length : {:.4f} m'.format(model.length))
    print('tunex  : {:.6f}'.format(twiss.mux[-1]/2/np.pi))
    print('tuney  : {:.6f}'.format(twiss.muy[-1]/2/np.pi))
    return model, twiss, ids


def run(idconfig):
    """."""
   
    # lattice with IDs
    model,twiss, ids = create_model_ids(idconfig=idconfig)
    x0 = 0
    y0 = 0

    famdata = pymodels.si.get_family_data(model)
    idx = famdata['WIG180']['index']
    idx_end = idx[0][-1]
    print(idx_end)


    coord_ini = np.array([x0, 0, y0, 0, 0, 0])
    coord_fin, *_ = pyaccel.tracking.line_pass(model, coord_ini, indices='open')
    x = coord_fin[0,:]
    px = coord_fin[1,:]
    y = coord_fin[2,:]
    py = coord_fin[3,:]
    xf = x[idx_end+1]
    pxf = px[idx_end+1]
    yf = y[idx_end+1]
    pyf = py[idx_end+1]

    print(xf*1e3)
    print(yf*1e3)
    print(pxf*1e6)
    print(pyf*1e6)
    
    spos = pyaccel.lattice.find_spos(model)
    print(spos[idx_end+1])

    # plt.figure(1)
    # plt.plot(spos, x*1e6, ls='-', label='xpos')
    # plt.plot(spos, y*1e6, ls='-', label='ypos')
    # plt.xlabel('s [m]')
    # plt.ylabel("tranverse pos [um]")
    # plt.grid()
    # plt.show()

    # plt.figure(2)
    # plt.plot(spos, px*1e6, ls='-', label='xpos')
    # plt.plot(spos, py*1e6, ls='-', label='ypos')
    # plt.xlabel('s [m]')
    # plt.ylabel("tranverse ang [urad]")
    # plt.grid()
    # plt.show()

   


if __name__ == '__main__':
    """."""
    run(idconfig='ID3979')  # correctors with zero current
