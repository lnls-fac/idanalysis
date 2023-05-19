import sys
import pickle
import numpy as np
import pyaccel
import matplotlib.pyplot as plt

from apsuite.optics_analysis.tune_correction import TuneCorr


CHAMBER_ON = False


def get_id_straigh_index_interval(tr, straight_nr):

    if straight_nr and 1 <= straight_nr <= 20:
        mia = pyaccel.lattice.find_indices(tr, 'fam_name', 'mia')
        mib = pyaccel.lattice.find_indices(tr, 'fam_name', 'mib')
        mip = pyaccel.lattice.find_indices(tr, 'fam_name', 'mip')
        locs = sorted(mia + mib + mip)
        center = locs[straight_nr-1]
        # find indices of first upstream and downstream bracketing dipole segments
        for idx1 in range(center, -1, -1):
            if tr[idx1].angle != 0:
                break
        for idx2 in range(center, len(tr), +1):
            if tr[idx2].angle != 0:
                break
    else:
        idx1, idx2 = None, None
    return idx1, idx2


def symm_get_locs(tr):
    mia = pyaccel.lattice.find_indices(tr, 'fam_name', 'mia')
    mib = pyaccel.lattice.find_indices(tr, 'fam_name', 'mib')
    mip = pyaccel.lattice.find_indices(tr, 'fam_name', 'mip')
    # mc = pyaccel.lattice.find_indices(tr, 'fam_name', 'mc')
    # locs = mia + mib + mip + mc
    locs = mia + mib + mip
    return locs


def symm_get_locs_beta(knobs):
    minv, maxv = sys.maxsize, -sys.maxsize
    for inds in knobs.values():
        minv = min(minv, min(inds))
        maxv = max(maxv, max(inds))
    # return [minv-1, maxv+1]  # start of drift and end of last local quad
    return [minv, maxv+1]  # start of first local quad and end of last local quad


def symm_get_knobs(tr, straight_nr, allquads=False):

    idx1, idx2 = get_id_straigh_index_interval(tr, straight_nr)

    knobs, knobs_in, knobs_out = dict(), dict(), dict()

    if allquads:
        quadfams = ['QFA', 'QFB', 'QFP', 'QDA', 'QDB1', 'QDB2', 'QDP1', 'QDP2', 'Q1', 'Q2', 'Q3', 'Q4']
    else:
        quadfams = ['QFA', 'QFB', 'QFP', 'QDA', 'QDB1', 'QDB2', 'QDP1', 'QDP2']

    for fam in quadfams:

        # indices of all fam quad elements
        idx = np.array(pyaccel.lattice.find_indices(tr, 'fam_name', fam))

        knobs[fam] = list(idx)

        # find indices of fam quad elements inside and outside the ID section
        if None in (idx1, idx2):
            sel = []
        else:
            sel = np.where(np.logical_and(idx > idx1, idx < idx2))

        idx_in = list(idx[sel])  # inside
        idx_out = list(set(idx) - set(idx_in))  # outside

        # add knobs of fam quads in ID section
        if np.any(idx_in):
            knobs_in[fam+'_ID'] = idx_in

        # add knobs of fam quads outside ID section
        if np.any(idx_out):
            knobs_out[fam] = idx_out

    knobs['QFB'] += knobs['QFP']; knobs.pop('QFP')
    knobs['QDB1'] += knobs['QDP1']; knobs.pop('QDP1')
    knobs['QDB2'] += knobs['QDP2']; knobs.pop('QDP2')


    return knobs, knobs_in, knobs_out


def correct_tunes_twoknobs(tr, goal_tunes, knobs_out):

    tunecorr = TuneCorr(tr, 'SI', method='Proportional', grouping='TwoKnobs')
    tunemat = tunecorr.calc_jacobian_matrix()
    tunecorr.correct_parameters(
        model=tr, goal_parameters=goal_tunes, jacobian_matrix=tunemat)


def symm_calc_residue_withbeta(tr, locs, locs_beta, goal_beta, goal_alpha):
    tw, _ = pyaccel.optics.calc_twiss(tr)
    nrlocs = len(locs)
    nrlocs_beta = len(locs_beta)
    residue = np.zeros(2*nrlocs+4*nrlocs_beta)

    # residue components: preserve symmetry points
    residue[:nrlocs] = tw.alphax[locs] - 0
    residue[nrlocs:2*nrlocs] = tw.alphay[locs] - 0

    # residue components: restore beta/alpha values
    residue[2*nrlocs+0*nrlocs_beta:2*nrlocs+1*nrlocs_beta] = 1*(tw.betax[locs_beta] - goal_beta[0])
    residue[2*nrlocs+1*nrlocs_beta:2*nrlocs+2*nrlocs_beta] = 1*(tw.betay[locs_beta] - goal_beta[1])
    residue[2*nrlocs+2*nrlocs_beta:2*nrlocs+3*nrlocs_beta] = 1*(tw.alphax[locs_beta] - goal_alpha[0])
    residue[2*nrlocs+3*nrlocs_beta:2*nrlocs+4*nrlocs_beta] = 1*(tw.alphay[locs_beta] - goal_alpha[1])
    return residue


def correct_symmetry_withbeta(
        tr, straight_nr, goal_beta, goal_alpha, delta_k=1e-5):
    """."""

    # get symmetry point indices
    locs = symm_get_locs(tr)
    nrlocs = len(locs)

    # get dict with local quad indices
    _, knobs, _ = symm_get_knobs(tr, straight_nr)
    locs_beta = symm_get_locs_beta(knobs)
    nrlocs_beta = len(locs_beta)

    # calc respm
    respm = np.zeros((2*nrlocs+4*nrlocs_beta, len(knobs)))
    for i, fam in enumerate(knobs):
        inds = knobs[fam]
        k0 = pyaccel.lattice.get_attribute(tr, 'polynom_b', inds, 1)
        pyaccel.lattice.set_attribute(tr, 'polynom_b', inds, k0 + delta_k/2, 1)
        res1 = symm_calc_residue_withbeta(
            tr, locs, locs_beta, goal_beta, goal_alpha)
        pyaccel.lattice.set_attribute(tr, 'polynom_b', inds, k0 - delta_k/2, 1)
        res2 = symm_calc_residue_withbeta(
            tr, locs, locs_beta, goal_beta, goal_alpha)
        pyaccel.lattice.set_attribute(tr, 'polynom_b', inds, k0, 1)
        respm[:, i] = (res1 - res2)/delta_k

    # inverse matrix
    umat, smat, vmat = np.linalg.svd(respm, full_matrices=False)
    # print('singular values: ', smat)
    ismat = 1/smat
    for i in range(len(smat)):
        if smat[i]/max(smat) < 1e-4:
            ismat[i] = 0
    ismat = np.diag(ismat)
    invmat = -1 * np.dot(np.dot(vmat.T, ismat), umat.T)

    # calc dk
    alpha = symm_calc_residue_withbeta(
        tr, locs, locs_beta, goal_beta, goal_alpha)
    dk = np.dot(invmat, alpha.flatten())

    # apply correction
    k0 = list()
    for i, fam in enumerate(knobs):
        inds = knobs[fam]
        k0_ = pyaccel.lattice.get_attribute(tr, 'polynom_b', inds, 1)
        pyaccel.lattice.set_attribute(tr, 'polynom_b', inds, k0_ + 1*dk[i], 1)
        k0.append(k0_[0])

    return dk, np.array(k0)


def calc_dynapt_xy(
                   ring, nrturns, nrtheta=9, mindeltar=0.1e-3,
                   r1=0, r2=30e-3, print_flag=False):

    ang = np.linspace(0, np.pi, nrtheta)
    ang[0] += 0.0001
    ang[-1] -= 0.0001

    vx, vy = list(), list()
    for a in ang:
        r1_, r2_ = r1, r2
        while r2_-r1_ > mindeltar:
            rm = (r1_+r2_) / 2
            rx = rm * np.cos(a)
            ry = rm * np.sin(a)
            _, lost, _, _, _ = pyaccel.tracking.ring_pass(ring, [rx, 0, ry, 0, 0, 0], nrturns)
            if lost:
                r2_ = rm
            else:
                r1_ = rm
            if print_flag:
                print('ang:{:5.1f} r1:{:5.2f} r2:{:5.2f}'.format(a*180/np.pi, r1_*1e3, r2_*1e3))
        rm = 0.5*(r1_+r2_)
        rx = rm * np.cos(a)
        ry = rm * np.sin(a)
        vx.append(rx)
        vy.append(ry)
    return np.array(vx), np.array(vy)


def calc_dynapt_ex(
                   ring, nrturns, demax=0.05, nrpts=33,
                   mindeltax=0.1e-3, xmin=-30e-3, y=1e-3, print_flag=False):

    denergies = np.linspace(-demax, demax, nrpts)
    x = []
    for denergy in denergies:
        xmin_, xmax_ = xmin, 0.0
        while xmax_ - xmin_ > mindeltax:
            xm = (xmin_ + xmax_) / 2
            _, lost, _, _, _ = pyaccel.tracking.ring_pass(ring, [xm, 0, y, 0, denergy, 0], nrturns)
            if lost:
                xmin_ = xm
            else:
                xmax_ = xm
            if print_flag:
                print('ene:{:+4.1f} % xmin:{:+6.2f} xmax:{:+6.2f}'.format(denergy*100, xmin_*1e3, xmax_*1e3))
        xm = (xmin_ + xmax_) / 2
        x.append(xm)

    return denergies, np.array(x)


def save_dynapt_xy(models,
    nrturns=4000,
    nrtheta=33,
    mindeltar=0.1e-3,
    r1=0,
    r2=30e-3):

    # nominal_ring = si.create_accelerator(ids=None)
    # nominal_ring.vchamber_on = CHAMBER_ON
    # data = dict()
    # data['ring0'] = calc_dynapt_xy(nominal_ring, nrturns=nrturns, nrtheta=nrtheta, mindeltar=mindeltar, r1=r1, r2=r2)
    # pickle.dump(data, open('dynapt_xy_' + 'nominal' + '.pickle', 'wb'))
    # return

    for config, rings in models.items():
        print('=== {} ==='.format(config))
        ring0, ring1, ring2, ring3 = rings
        data = dict()
        print('--- ring0 ---')
        data['ring0'] = calc_dynapt_xy(ring0, nrturns=nrturns, nrtheta=nrtheta, mindeltar=mindeltar, r1=r1, r2=r2)
        print('--- ring1 ---')
        data['ring1'] = calc_dynapt_xy(ring1, nrturns=nrturns, nrtheta=nrtheta, mindeltar=mindeltar, r1=r1, r2=r2)
        print('--- ring2 ---')
        data['ring2'] = calc_dynapt_xy(ring2, nrturns=nrturns, nrtheta=nrtheta, mindeltar=mindeltar, r1=r1, r2=r2)
        print('--- ring3 ---')
        data['ring3'] = calc_dynapt_xy(ring3, nrturns=nrturns, nrtheta=nrtheta, mindeltar=mindeltar, r1=r1, r2=r2)
        pickle.dump(data, open('dynapt_xy_' + config + '.pickle', 'wb'))
        print()


def save_dynapt_ex(models,
    nrturns=4000,
    demax=0.05,
    nrpts=33,
    mindeltax=0.1e-3,
    xmin=-30e-3,
    y=1e-3):

    # nominal_ring = si.create_accelerator(ids=None)
    # nominal_ring.vchamber_on = CHAMBER_ON
    # data = dict()
    # data['ring0'] = calc_dynapt_ex(nominal_ring, nrturns=nrturns, demax=demax, nrpts=nrpts, mindeltax=mindeltax, xmin=xmin, y=y)
    # pickle.dump(data, open('dynapt_ex_' + 'nominal' + '.pickle', 'wb'))
    # return

    for config, rings in models.items():
        print('=== {} ==='.format(config))
        ring0, ring1, ring2, ring3 = rings
        data = dict()
        print('--- ring0 ---')
        data['ring0'] = calc_dynapt_ex(ring0, nrturns=nrturns, demax=demax, nrpts=nrpts, mindeltax=mindeltax, xmin=xmin, y=y)
        print('--- ring1 ---')
        data['ring1'] = calc_dynapt_ex(ring1, nrturns=nrturns, demax=demax, nrpts=nrpts, mindeltax=mindeltax, xmin=xmin, y=y)
        print('--- ring2 ---')
        data['ring2'] = calc_dynapt_ex(ring2, nrturns=nrturns, demax=demax, nrpts=nrpts, mindeltax=mindeltax, xmin=xmin, y=y)
        print('--- ring3 ---')
        data['ring3'] = calc_dynapt_ex(ring3, nrturns=nrturns, demax=demax, nrpts=nrpts, mindeltax=mindeltax, xmin=xmin, y=y)
        pickle.dump(data, open('dynapt_ex_' + config + '.pickle', 'wb'))
        print()


def load_models(folder='./results/', modelsname='models'):
    models = pickle.load(open(folder + modelsname + '.pickle', 'rb'))
    return models
