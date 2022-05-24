#!/usr/bin/env python-sirius

"""Script to check kickmap through model tracking."""

import numpy as np
import matplotlib.pyplot as plt

from pyaccel import lattice as pyacc_lat
from pyaccel import tracking as pyacc_track
from idanalysis.model import create_model, get_id_epu_list
from kickmaps import IDKickMap

from utils import create_epudata


def calc_idkmap_kicks(fname, yidx=0, plot_flag=False):
  idkmap = IDKickMap(fname)
  rx0 = idkmap.posx
  pxf = idkmap.kickx[yidx, :] / idkmap.brho**2
  pyf = idkmap.kicky[yidx, :] / idkmap.brho**2
  if plot_flag:
    plt.plot(1e3*rx0, 1e6*pxf)
    plt.xlabel('init rx [mm]')
    plt.ylabel('final px [urad]')
    plt.title('KickX')
    plt.show()

  return idkmap, rx0, pxf, pyf


def calc_model_kicks(
  fname, idkmap, yidx=0, nr_steps=40, at_end_idx=3, plot_flag=False):

  ids = get_id_epu_list(fname, nr_steps=nr_steps)

  # check model circumference
  model = create_model(ids=ids)
  spos = pyacc_lat.find_spos(model, indices='closed')
  circ = spos[-1]
  print(f'circumference: {circ:.4f} m')

  # shift model so as to start at EPU50.
  inds = pyacc_lat.find_indices(model, 'fam_name', 'EPU50')
  model = pyacc_lat.shift(model, start=inds[0])

  # initial tracking conditions (particles)
  rx0 = np.asarray(idkmap.posx)  # [m]
  ry0 = idkmap.posy[yidx]  # [m]
  pos0 = np.zeros((6, len(rx0)))
  pos0[0, :] = rx0
  pos0[2, :] = ry0

  # tracking
  posf, *_ = pyacc_track.line_pass(model, particles=pos0, indices=[at_end_idx])
  rxf = posf[0, :]
  pxf = posf[1, :]
  ryf = posf[2, :]
  pyf = posf[3, :]

  # plots
  if plot_flag:
    plt.plot(1e3*rx0, 1e6*pxf)
    plt.xlabel('init rx [mm]')
    plt.ylabel('final px [urad]')
    plt.title('KickX vc X')
    plt.show()

  return model, rx0, pxf, pyf, rxf, ryf


def model_tracking_kick_error():
  """."""

  # create object with list of all possible EPU50 configurations
  configs = create_epudata()

  # select ID config
  configname = configs[0]
  fname = configs.get_kickmap_filename(configname)
  print('configuration: ', configs.get_config_label(configname))

  # compare tracking anglex with kickmap kickx
  yidx = 8  # [y = 0 mm]
  idkmap, rx0_1, pxf_1, *_ = calc_idkmap_kicks(
    fname, yidx=yidx, plot_flag=False)
  at_end_idx = 1  # half ID, half kick
  model, rx0_2, pxf_2, *_ = calc_model_kicks(
    fname, idkmap, yidx=yidx, nr_steps=40, at_end_idx=at_end_idx,
    plot_flag=False)
  pxf_err = pxf_2*2 - pxf_1  # kick error for whole kickmap

  # plot comparison
  plt.plot(1e3*rx0_1, 1e6*pxf_1, label='input kickmap')
  plt.plot(1e3*rx0_2, 1e6*pxf_2*2, label='tracking w/ kickmap')
  # plt.plot(1e3*rx0_2, 1e6*pxf_err*1e14, label=r'error x 10$^{14}$')
  plt.xlabel('rx [mm]')
  plt.ylabel('px [urad]')
  plt.title('Midplane horizontal kick from model tracking')
  plt.legend()
  plt.show()


if __name__ == '__main__':
    model_tracking_kick_error()

