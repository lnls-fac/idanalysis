"""."""

import numpy as np
import matplotlib.pyplot as plt

import pyaccel

from pyaccel.optics import calc_edwards_teng
from pyaccel.optics.edwards_teng import estimate_coupling_parameters
from pyaccel.optics import EqParamsFromBeamEnvelope
from pymodels import si


def get_id_sabia_list(configname, ids=None):
    """."""
    delta52 = si.IDModel(
        subsec=si.IDModel.SUBSECTIONS.ID10SB, file_name=configname,
        fam_name='DELTA52', 
        nr_steps=40, rescale_kicks=1.0, rescale_length=1.2/1.6)
    ids = ids or dict()
    if not isinstance(configname, str):
        raise TypeError
    ids = [delta52, ]
    return ids


def get_id_epu_list(configname, ids=None, nr_steps=40):
    epu50 = si.IDModel(
        subsec=si.IDModel.SUBSECTIONS.ID11SP, file_name=configname,
        fam_name='EPU50',
        nr_steps=nr_steps, rescale_kicks=1.0, rescale_length=2.8/3.6)
    ids = ids or dict()
    if not isinstance(configname, str):
        raise TypeError
    ids = [epu50, ]
    return ids


def create_model(ids, vchamber_on=False):
    """."""
    ring = si.create_accelerator(ids=ids)
    ring.vchamber_on = vchamber_on
    return ring


def calc_optics(ids, vchamber_on=False):
    """."""
    ring = create_model(ids, vchamber_on)
    eqp = EqParamsFromBeamEnvelope(accelerator=ring)
    orb4 = pyaccel.tracking.find_orbit4(ring, indices='closed')
    plt.plot(1e6*orb4[0, :])
    plt.xlabel('element index')
    plt.ylabel('posx [um]')
    plt.show()

    edt, m66 = calc_edwards_teng(ring)
    tune_sep, emit_ratio = estimate_coupling_parameters(edt)
    data = dict()
    data['ring'] = ring
    data['edt'] = edt
    data['tune1'] = eqp.tune1
    data['tune2'] = eqp.tune2
    data['emit1'] = eqp.emit1
    data['emit2'] = eqp.emit2
    data['edt_tune1'] = edt[-1].mu1 / 2 / np.pi
    data['edt_tune2'] = edt[-1].mu2 / 2 / np.pi
    data['edt_tune_sep'] = tune_sep
    data['edt_emit_ratio'] = emit_ratio

    return data


