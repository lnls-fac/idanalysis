#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

import pyaccel
from pymodels import si
from apsuite.orbcorr import OrbitCorr, CorrParams


IDS_FAMNAMES = [
    'PAPU50', 'APU22', 'IVU18', 'VPU29', 'WIG180', 'APU58', 'EPU50', 'DELTA52']


def get_ids_indices(model):
    inds_list = list()
    for famname in IDS_FAMNAMES:
        inds = pyaccel.lattice.find_indices(model, 'fam_name', famname)
        for idx in inds:
            if model[idx].pass_method == 'kicktable_pass':
                inds_list.append(idx)
    return sorted(inds_list)


def correct_orbit_local(
        model0, model1, id_famname, correction_plane='both', plot=True):
    """."""

    delta_kick = 1e-6  # [rad]
    tol_svals = 1e-5

    orb0 = pyaccel.tracking.find_orbit4(model0, indices='open')
    orb0_pos = orb0[[0, 2], :]
    orb0_ang = orb0[[1, 3], :]

    # find idc1 and idc2 indices for local correctors
    inds = pyaccel.lattice.find_indices(model1, 'fam_name', id_famname)
    idinds = list()
    for idx in inds:
        if model1[idx].pass_method == 'kicktable_pass':
            idinds.append(idx)
    idc1, idc2 = idinds[0], idinds[-1]
    while idc1 >= 0 and model1[idc1].fam_name != 'IDC':
        idc1 -= 1
    while idc2 < len(model1) and model1[idc2].fam_name != 'IDC':
        idc2 += 1
    if idc1 < 0 or idc2 >= len(model1):
        raise ValueError('Could not find ID correctors!')
    cors = [idc1, idc2]

    # get indices
    bpms = pyaccel.lattice.find_indices(model1, 'fam_name', 'BPM')
    nrcors = len(cors)
    nrbpms = len(bpms)

    # t_in_original = model1[idinds[0]].t_in.copy()
    # t_out_original = model1[idinds[-1]].t_out.copy()
    # model1[idinds[0]].rescale_kicks *= 0
    # model1[idinds[-1]].rescale_kicks *= 0
    # model1[idinds[0]].t_in *= 0
    # model1[idinds[-1]].t_out *= 0

    # calc respm
    respm = np.zeros((2*nrbpms, 2*len(cors)))
    for i in range(nrcors):
        kick0 = model1[cors[i]].hkick_polynom
        model1[cors[i]].hkick_polynom = kick0 + delta_kick/2
        orb1 = pyaccel.tracking.find_orbit4(model1, indices='open')
        model1[cors[i]].hkick_polynom = kick0 - delta_kick/2
        orb2 = pyaccel.tracking.find_orbit4(model1, indices='open')
        orb_delta = (orb1 - orb2) / delta_kick
        orb_delta = orb_delta[[0, 2], :]
        orb_delta = orb_delta[:, bpms]  # select cod in BPMs
        respm[:, i] = orb_delta.flatten()
        model1[cors[i]].hkick_polynom = kick0
    for i in range(nrcors):
        kick0 = model1[cors[i]].vkick_polynom
        model1[cors[i]].vkick_polynom = kick0 + delta_kick/2
        orb1 = pyaccel.tracking.find_orbit4(model1, indices='open')
        model1[cors[i]].vkick_polynom = kick0 - delta_kick/2
        orb2 = pyaccel.tracking.find_orbit4(model1, indices='open')
        orb_delta = (orb1 - orb2) / delta_kick
        orb_delta = orb_delta[[0, 2], :]
        orb_delta = orb_delta[:, bpms]  # select cod in BPMs
        respm[:, nrcors + i] = orb_delta.flatten()
        model1[cors[i]].vkick_polynom = kick0

    if correction_plane == 'x':
        for i in range(nrcors):
            respm[:, 1*nrcors+i] *= 0
    elif correction_plane == 'y':
        for i in range(nrcors):
            respm[:, 0*nrcors+i] *= 0

    # model1[idinds[0]].rescale_kicks = 0.5
    # model1[idinds[-1]].rescale_kicks = 0.5
    # model1[idinds[0]].t_in = t_in_original
    # model1[idinds[-1]].t_out = t_out_original

    # inverse matrix
    umat, smat, vmat = np.linalg.svd(respm, full_matrices=False)
    sel_svals = abs(smat) > tol_svals
    ismat = np.zeros(smat.shape)
    ismat[sel_svals] = 1/smat[sel_svals]
    ismat = np.diag(ismat)
    invmat = -1 * np.dot(np.dot(vmat.T, ismat), umat.T)

    orb1 = pyaccel.tracking.find_orbit4(model1, indices='open')
    cod_u = orb1 - orb0
    cod0_pos = cod_u[[0, 2], :]
    cod0_ang = cod_u[[1, 3], :]

    # correct orbit gradually
    dk_total = np.zeros(2*nrcors)
    for j in np.arange(10):
        # calc dk
        cod = orb1[[0, 2], :] - orb0[[0, 2], :]
        dk = np.dot(invmat, cod[:, bpms].flatten())
        dk_total += dk
        # apply correction
        for i in range(nrcors):
            model1[cors[i]].hkick_polynom += dk[i]
            model1[cors[i]].vkick_polynom += dk[nrcors + i]
        orb1 = pyaccel.tracking.find_orbit4(model1, indices='open')

    print("Correctors's kicks: ")
    txt = 'IDC1 x: {:.2f} urad   IDC2 x: {:.2f} urad'.format(
        1e6*dk_total[0], 1e6*dk_total[1])
    print(txt)
    txt = 'IDC1 y: {:.2f} urad   IDC2 y: {:.2f} urad'.format(
        1e6*dk_total[2], 1e6*dk_total[3])
    print(txt)

    cod_c = orb1 - orb0
    cod_c_pos = cod_c[[0, 2], :]
    cod_c_ang = cod_c[[1, 3], :]
    # calc maximum cod distortion in the ID sectionn
    idx_x = np.argmax(np.abs(cod_c_pos[0, (cors[0]+1):cors[-1]]))
    idx_y = np.argmax(np.abs(cod_c_pos[1, (cors[0]+1):cors[-1]]))
    max_rx1_corr, max_ry1_corr = cod_c_pos[0, idx_x], cod_c_pos[1, idx_y]

    # stats for cod in the whole ring
    rms_rx0_ring = np.std(cod0_pos[0, :])*1e6
    rms_ry0_ring = np.std(cod0_pos[1, :])*1e6
    max_rx0_ring = np.max(np.abs(cod0_pos[0, :]))*1e6
    max_ry0_ring = np.max(np.abs(cod0_pos[1, :]))*1e6
    rms_rx1_ring = np.std(cod_c_pos[0, :])*1e6
    rms_ry1_ring = np.std(cod_c_pos[1, :])*1e6
    max_rx1_ring = np.max(np.abs(cod_c_pos[0, :]))*1e6
    max_ry1_ring = np.max(np.abs(cod_c_pos[1, :]))*1e6

    rms_px0_ring = np.std(cod0_ang[0, :])*1e6
    rms_py0_ring = np.std(cod0_ang[1, :])*1e6
    max_px0_ring = np.max(np.abs(cod0_ang[0, :]))*1e6
    max_py0_ring = np.max(np.abs(cod0_ang[1, :]))*1e6
    rms_px1_ring = np.std(cod_c_ang[0, :])*1e6
    rms_py1_ring = np.std(cod_c_ang[1, :])*1e6
    max_px1_ring = np.max(np.abs(cod_c_ang[0, :]))*1e6
    max_py1_ring = np.max(np.abs(cod_c_ang[1, :]))*1e6

    # stats for cod in the the bpms
    rms_rx0_bpms = np.std(cod0_pos[0, bpms])*1e6
    rms_ry0_bpms = np.std(cod0_pos[1, bpms])*1e6
    rms_rx1_bpms = np.std(cod_c_pos[0, bpms])*1e6
    rms_ry1_bpms = np.std(cod_c_pos[1, bpms])*1e6

    rms_px0_bpms = np.std(cod0_ang[0, bpms])*1e6
    rms_py0_bpms = np.std(cod0_ang[1, bpms])*1e6
    rms_px1_bpms = np.std(cod_c_ang[0, bpms])*1e6
    rms_py1_bpms = np.std(cod_c_ang[1, bpms])*1e6

    if plot:
        spos = pyaccel.lattice.find_spos(model1)
        cod0_rx = 1e6*cod0_pos[0, :]
        cod_c_rx = 1e6*cod_c_pos[0, :]
        cod0_ry = 1e6*cod0_pos[1, :]
        cod_c_ry = 1e6*cod_c_pos[1, :]
        cod0_px = 1e6*cod0_ang[0, :]
        cod_c_px = 1e6*cod_c_ang[0, :]
        cod0_py = 1e6*cod0_ang[1, :]
        cod_c_py = 1e6*cod_c_ang[1, :]

        # --- cod_rx ---

        # uncorrected
        label = (
                'Uncorrected - rms@bpms, max@ring, rms@ring : {:.2f}, '
                '{:.2f}, {:.2f} um'.format(
                    rms_rx0_bpms, max_rx0_ring, rms_rx0_ring))

        plt.plot(spos, cod0_rx, '-', color='C0')
        plt.plot(spos[bpms], cod0_rx[bpms], '.', color='C0', label=label)

        # corrected
        label = (
                'Corrected - rms@bpms, max@ring, rms@ring : {:.2f}, '
                '{:.2f}, {:.2f} um').format(
                    rms_rx1_bpms, max_rx1_ring, rms_rx1_ring)
        plt.plot(spos, cod_c_rx, '-', color='C1')
        plt.plot(spos[bpms], cod_c_rx[bpms], '.', color='C1', label=label)

        # corrected @ ID straight
        label = 'Corrected - ID straight'
        spos_corr = spos[cors[0]:cors[-1]+1]
        cod_c_rx_corr = cod_c_rx[cors[0]:cors[-1]+1]
        plt.plot(spos_corr, cod_c_rx_corr, '.-', color='C2', label=label)
        plt.legend()
        plt.title('Horizontal COD Position')
        plt.xlabel('spos [m]')
        plt.ylabel('Pos [um]')
        plt.grid()
        plt.show()

        # --- cod_ry ---

        # uncorrected
        label = (
                'Uncorrected - rms@bpms, max@ring, rms@ring :  {:.2f}, '
                '{:.2f}, {:.2f} um').format(
                    rms_ry0_bpms, max_ry0_ring, rms_ry0_ring)

        plt.plot(spos, cod0_ry, '-', color='C0')
        plt.plot(spos[bpms], cod0_ry[bpms], '.', color='C0', label=label)

        # corrected
        label = (
                'Corrected - rms@bpms, max@ring, rms@ring : {:.2f}, '
                '{:.2f}, {:.2f} um').format(
                    rms_ry1_bpms, max_ry1_ring, rms_ry1_ring)
        plt.plot(spos, cod_c_ry, '-', color='C1')
        plt.plot(spos[bpms], cod_c_ry[bpms], '.', color='C1', label=label)

        # corrected @ ID straight
        label = 'Corrected - ID straight'
        spos_corr = spos[cors[0]:cors[-1]+1]
        cod_c_ry_corr = cod_c_ry[cors[0]:cors[-1]+1]
        plt.plot(spos_corr, cod_c_ry_corr, '.-', color='C2', label=label)
        plt.legend()
        plt.title('Vertical COD Position')
        plt.xlabel('spos [m]')
        plt.ylabel('Pos [um]')
        plt.grid()
        plt.show()

        # --- cod_px ---

        # uncorrected
        label = (
                'Uncorrected - rms@bpms, max@ring, rms@ring : {:.2f}, '
                '{:.2f}, {:.2f} urad').format(
                    rms_px0_bpms, max_px0_ring, rms_px0_ring)

        plt.plot(spos, cod0_px, '-', color='C0')
        plt.plot(spos[bpms], cod0_px[bpms], '.', color='C0', label=label)

        # corrected
        label = (
                'Corrected - rms@bpms, max@ring, rms@ring : {:.2f}, '
                '{:.2f}, {:.2f} urad').format(
                    rms_px1_bpms, max_px1_ring, rms_px1_ring)
        plt.plot(spos, cod_c_px, '-', color='C1')
        plt.plot(spos[bpms], cod_c_px[bpms], '.', color='C1', label=label)

        # corrected @ ID straight
        label = 'Corrected - ID straight'
        spos_corr = spos[cors[0]:cors[-1]+1]
        cod_c_px_corr = cod_c_px[cors[0]:cors[-1]+1]
        plt.plot(spos_corr, cod_c_px_corr, '.-', color='C2', label=label)
        plt.legend()
        plt.title('Horizontal COD Angle')
        plt.xlabel('spos [m]')
        plt.ylabel('Angle [urad]')
        plt.grid()
        plt.show()

        # --- cod_py ---

        # uncorrected
        label = (
                'Uncorrected - rms@bpms, max@ring, rms@ring : {:.2f}, '
                '{:.2f}, {:.2f} urad').format(
                    rms_py0_bpms, max_py0_ring, rms_py0_ring)

        plt.plot(spos, cod0_py, '-', color='C0')
        plt.plot(spos[bpms], cod0_py[bpms], '.', color='C0', label=label)

        # corrected
        label = (
                'Corrected - rms@bpms, max@ring, rms@ring : {:.2f}, '
                '{:.2f}, {:.2f} urad').format(
                    rms_py1_bpms, max_py1_ring, rms_py1_ring)
        plt.plot(spos, cod_c_py, '-', color='C1')
        plt.plot(spos[bpms], cod_c_py[bpms], '.', color='C1', label=label)

        # corrected @ ID straight
        label = 'Corrected - ID straight'
        spos_corr = spos[cors[0]:cors[-1]+1]
        cod_c_py_corr = cod_c_py[cors[0]:cors[-1]+1]
        plt.plot(spos_corr, cod_c_py_corr, '.-', color='C2', label=label)
        plt.legend()
        plt.title('Vertical COD Angle')
        plt.xlabel('spos [m]')
        plt.ylabel('Angle [urad]')
        plt.grid()
        plt.show()

    ret = (dk_total,
           max_rx1_corr, max_ry1_corr,
           rms_rx0_ring, rms_ry0_ring,
           rms_rx0_bpms, rms_ry0_bpms,
           rms_rx1_ring, rms_ry1_ring,
           rms_rx1_bpms, rms_ry1_bpms,
           max_rx0_ring, max_ry0_ring,
           max_rx1_ring, max_ry1_ring,
           )
    return ret


def get_fofb_bpms_idx(bpms):
    """."""
    return bpms.reshape(20, -1)[:, [0, 3, 4, 7]].ravel()


def correct_orbit_fb(
        model0, model1, minsingval=0.2,
        nr_steps=1, corr_system='SOFB', plot_flag=False):
    """."""
    # calculate structures
    famdata = si.get_family_data(model1)
    bpms = np.array([idx[0] for idx in famdata['BPM']['index']])
    spos_bpms = pyaccel.lattice.find_spos(model1, indices=bpms)
    inds_list = get_ids_indices(model1)

    # create orbit corrector
    cparams = CorrParams()
    cparams.minsingval = minsingval
    cparams.tolerance = 1e-8  # [m]
    cparams.maxnriters = 20

    # get unperturbed orbit
    ocorr = OrbitCorr(model0, 'SI', params=cparams, corr_system=corr_system)
    orb0 = ocorr.get_orbit()

    factors = np.linspace(0, 1, nr_steps+1)
    rescale_kicks_orig = [model1[idx].rescale_kicks for idx in inds_list]

    for factor in factors:

        # rescale kicks
        for idx, rescale_kicks in zip(inds_list, rescale_kicks_orig):
            model1[idx].rescale_kicks = factor * rescale_kicks

        # get perturbed orbit
        ocorr = OrbitCorr(
            model1, 'SI', params=cparams, corr_system=corr_system)
        orb1 = ocorr.get_orbit()

        # calc closed orbit distortions (cod) before correction
        cod_u = orb1 - orb0
        codx_u = cod_u[:len(bpms)]
        cody_u = cod_u[len(bpms):]

        # calc response matrix and correct orbit
        if not ocorr.correct_orbit(goal_orbit=orb0):
            print('Could not correct orbit!')

        # get corrected orbit
        orb2 = ocorr.get_orbit()
        kicks = ocorr.get_kicks()

        # calc closed orbit distortions (cod) after correction
        cod_c = orb2 - orb0
        codx_c = cod_c[:len(bpms)]
        cody_c = cod_c[len(bpms):]

    if corr_system == 'FOFB':
        codx_c = codx_c[ocorr.params.enbllistbpm[:len(bpms)]]
        cody_c = cody_c[ocorr.params.enbllistbpm[len(bpms):]]
        codx_u = codx_u[ocorr.params.enbllistbpm[:len(bpms)]]
        cody_u = cody_u[ocorr.params.enbllistbpm[len(bpms):]]
        spos_bpms = spos_bpms[ocorr.params.enbllistbpm[:len(bpms)]]
    elif corr_system == 'SOFB':
        pass
    else:
        raise ValueError('Corretion system must be "SOFB" or "FOFB"')

    if plot_flag:
        rmsx_u = np.std(1e6*codx_u)
        rmsx_c = np.std(1e6*codx_c)
        rmsy_u = np.std(1e6*cody_u)
        rmsy_c = np.std(1e6*cody_c)

        plt.figure(1)
        plt.plot(spos_bpms, 1e6*codx_u,
                 label='Uncorrected - RMS = {:.3f}'.format(rmsx_u))
        plt.plot(spos_bpms, 1e6*codx_c,
                 label='Corrected - RMS = {:.3f}'.format(rmsx_c))
        plt.legend()
        plt.xlabel('pos [m]')
        plt.ylabel('codx [um]')

        plt.figure(2)
        plt.plot(spos_bpms, 1e6*cody_u,
                 label='Uncorrected - RMS = {:.3f}'.format(rmsy_u))
        plt.plot(spos_bpms, 1e6*cody_c,
                 label='Corrected - RMS = {:.3f}'.format(rmsy_c))
        plt.legend()
        plt.xlabel('pos [m]')
        plt.ylabel('cody [um]')
        plt.show()

    return kicks, spos_bpms, codx_c, cody_c, codx_u, cody_u, bpms
