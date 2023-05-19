#!/usr/bin/env python-sirius

"""Script to check kickmap through model tracking."""

import matplotlib.pyplot as plt
import numpy as np

import pymodels
import pyaccel
from idanalysis import IDKickMap
import utils
from idanalysis import IDKickMap
from pyaccel import lattice as pyacc_lat

def calc_idkmap_kicks(plane_idx=0, plot_flag=False, idkmap=None):

  brho = 10.007
  kickx_end = idkmap.kickx_upstream + idkmap.kickx_downstream
  kicky_end = idkmap.kicky_upstream + idkmap.kicky_downstream
  rx0 = idkmap.posx
  ry0 = idkmap.posy
  rxf = idkmap.fposx[0, :]
  ryf = idkmap.fposy[0, :]
  pxf = (idkmap.kickx[0, :] + kickx_end)/ brho**2
  pyf = (idkmap.kicky[0, :] + kicky_end) / brho**2
  if plot_flag:
    plt.plot(1e3*rx0, 1e6*pxf, label = 'Kick X', color='C1')
    plt.plot(1e3*rx0, 1e6*pyf, label = 'Kick Y', color='b')
    plt.xlabel('init rx [mm]')
    plt.ylabel('final px [urad]')
    plt.title('Kicks')
    plt.legend()
    plt.grid()
    plt.show()




  return rx0,ry0, pxf, pyf, rxf, ryf

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



if __name__ == '__main__':
  fname = "results/ID4019/kickmap-ID4019.txt"
  idconfig = fname[8:14]
  wiggler_kickmap = IDKickMap(fname)
  rx0, ry0, pxf, pyf, rxf, ryf = calc_idkmap_kicks(
    idkmap=wiggler_kickmap, plane_idx=0, plot_flag=True)

  print(pxf[15]*1e6)
  # print(pyf[15]*1e6)

  # lattice with IDs
  model,twiss, ids = create_model_ids(idconfig=idconfig)

  famdata = pymodels.si.get_family_data(model)
  idx = famdata['WIG180']['index']

  idx_end = idx[0][-1]
  idx_begin = idx[0][0]
  idx_dif = idx_end - idx_begin
  print(idx_dif)


  y0 = 0

  pxf_list = []
  pyf_list = []

  inds = pyacc_lat.find_indices(model, 'fam_name', 'WIG180')
  model = pyacc_lat.shift(model, start=idx_begin)
  for x0 in rx0:
    coord_ini = np.array([x0, 0, y0, 0, 0, 0])

    coord_fin, *_ = pyaccel.tracking.line_pass(model, coord_ini, indices='open')
    x = coord_fin[0,:]
    px = coord_fin[1,:]
    y = coord_fin[2,:]
    py = coord_fin[3,:]

    xf_t = x[idx_dif+1]
    yf_t = y[idx_dif+1]
    pxf_t = px[idx_dif+1]
    pyf_t = py[idx_dif+1]

    if x0==0:
      print(pxf_t*1e6)


    # xf_list.append(xf_t)
    # yf_list.append(yf_t)

    pxf_list.append(pxf_t)
    pyf_list.append(pyf_t)

  # xf_array = np.array(xf_list)
  # yf_array = np.array(yf_list)
  pxf_array = np.array(pxf_list)
  pyf_array = np.array(pyf_list)

  plt.plot(1e3*rx0, 1e6*pxf,'-', color='C1', label = 'Kick X - kickmap')
  plt.plot(1e3*rx0, 1e6*pyf,'-', color='b', label = 'Kick Y - kickmap')
  plt.plot(1e3*rx0, 1e6*pxf_array,'o', color='C1', label = 'Kick X - tracking')
  plt.plot(1e3*rx0, 1e6*pyf_array,'o', color='b', label = 'Kick Y - tracking')
  plt.xlabel('x0 [mm]')
  plt.ylabel('final px [urad]')
  plt.title('Kicks')
  plt.legend()
  plt.grid()
  plt.show()
