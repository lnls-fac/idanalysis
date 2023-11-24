#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt
from imaids.models import DeltaSabia
from idanalysis.analysis import Tools
import utils
import imaids
from scipy.fftpack import fft, ifft, fftfreq
from mathphys.functions import save_pickle, load_pickle


def generate_radia_model(phase=0, dgv=0, trf_on_blocks=False):
    """."""

    delta = DeltaSabia(trf_on_blocks=trf_on_blocks)
    delta.set_cassete_positions(dp=phase, dgv=dgv)

    return delta


def get_cassettes_names(model):
    return list(model.cassettes_ref.keys())


def get_fieldmap(phase, dgv):
    fmap_fname = Tools.get_fmap_fname(utils.ID_CONFIGS, utils.MEAS_DATA_PATH,
                                      config_keys=(phase, dgv),
                                      config_dict=utils.CONFIG_DICT)
    return Tools.get_fmap(fmap_fname)


def get_full_field(fmap):
    full_field = np.zeros((len(fmap.rx), len(fmap.rz), 3))
    full_field[:, :, 0] = fmap.bx[0]
    full_field[:, :, 1] = fmap.by[0]
    full_field[:, :, 2] = fmap.bz[0]
    return full_field


def get_full_field_model(rx, rz, model):
    field_model = np.zeros((len(rx), len(rz), 3))
    for i, x in enumerate(rx):
        field_model[i, :, :] = model.get_field(x=x, y=0, z=rz, nproc=24)
    return field_model


def calc_field_difference(full_field_meas, delta, rx, rz):
    full_field_model = get_full_field_model(rx, rz, delta)
    full_field_model = np.swapaxes(full_field_model, 0, 1)
    field_diff = (full_field_meas - full_field_model).ravel(order='F')
    return field_diff


def cas_2_angle(cas):
    cas_2_angle = dict()
    cas_2_angle['cid'] = -np.pi/4
    cas_2_angle['csd'] = -3*np.pi/4
    cas_2_angle['cie'] = np.pi/4
    cas_2_angle['cse'] = 3*np.pi/4
    return cas_2_angle[cas]


def do_block_displacement(delta, block, cas):
    block.shift([0, -delta.gap/2, 0])
    center = np.array(block.center_point)
    center[0] = 0
    center[1] = 0
    block.shift(-center)
    block.rotate([0, 0, 0], [0, 0, 1], cas_2_angle(cas))


def do_block_shift(block, shift_list, cas):
    x = shift_list[0]
    y = shift_list[1]
    shiftb = np.array([x, y])

    # Apply local shift to block
    theta = cas_2_angle(cas)
    M = np.zeros((2, 2))
    M[0, 0] = np.cos(theta)
    M[0, 1] = -np.sin(theta)
    M[1, 0] = np.sin(theta)
    M[1, 1] = np.cos(theta)
    shift = np.dot(M, shiftb)
    block.shift([shift[0], shift[1], 0])


def calculate_jac_elements(delta, rx, cas='cid', z_range=600, nr_pts=2401):
    blocks = [
        delta.cassettes_ref[cas].blocks[block_nr] for block_nr in
        np.arange(0, 8, 1)]
    blocks += [
            delta.cassettes_ref[cas].blocks[-4],
            delta.cassettes_ref[cas].blocks[-2]]
    nr_blocks = len(blocks)

    x = rx
    z = np.linspace(-z_range/2, z_range/2, nr_pts)

    dmag = 0.04
    dshift = 0.04

    delta_shifts = [np.array([dshift, 0, 0]),
                    np.array([0, dshift, 0])]

    delta_mags = [np.array([dmag, 0, 0]),
                  np.array([0, dmag, 0]),
                  np.array([0, 0, dmag]),
                  ]

    jac_elem = np.zeros(
        (len(z), len(x), len(delta_mags)+len(delta_shifts), nr_blocks, 3))

    for k, block in enumerate(blocks):
        print('block: ', k)
        for i, x_ in enumerate(x):
            for j, d_mag in enumerate(delta_mags):

                block.magnetization = np.array(block.magnetization) + d_mag/2
                do_block_displacement(delta, block, cas)
                field_p = block.get_field(x=x_, y=0, z=z, nproc=24)

                block.magnetization = np.array(block.magnetization) - d_mag
                do_block_displacement(delta, block, cas)
                field_n = block.get_field(x=x_, y=0, z=z, nproc=24)

                block.magnetization = np.array(block.magnetization) + d_mag/2
                do_block_displacement(delta, block, cas)

                diff = (field_p - field_n)/dmag
                jac_elem[:, i, j, k, :] = diff

            for j, d_shift in enumerate(delta_shifts):
                do_block_shift(block, d_shift/2, cas)
                field_p = block.get_field(x=x_, y=0, z=z, nproc=24)

                do_block_shift(block, -d_shift, cas)
                field_n = block.get_field(x=x_, y=0, z=z, nproc=24)

                do_block_shift(block, d_shift/2, cas)

                diff = (field_p - field_n)/dshift
                jac_elem[:, i, len(delta_mags) + j, k, :] = diff

    return jac_elem


def transform_cd_2_ce(jac_elem):
    jac_elem[:, :, :, :, :] = jac_elem[:, ::-1, :, :, :]  # Invert x grid

    # delta mx
    jac_elem[:, :, 0, :, 1] = -1*jac_elem[:, :, 0, :, 1]  # Change By signal
    jac_elem[:, :, 0, :, 2] = -1*jac_elem[:, :, 0, :, 2]  # Change Bz signal

    # delta my
    jac_elem[:, :, 1, :, 0] = -1*jac_elem[:, :, 1, :, 0]  # Change Bx signal

    # delta mz
    jac_elem[:, :, 2, :, 0] = -1*jac_elem[:, :, 2, :, 0]  # Change Bx signal

    # delta rx
    jac_elem[:, :, 3, :, 1] = -1*jac_elem[:, :, 3, :, 1]  # Change By signal
    jac_elem[:, :, 3, :, 2] = -1*jac_elem[:, :, 3, :, 2]  # Change Bz signal

    # delta ry
    jac_elem[:, :, 4, :, 0] = -1*jac_elem[:, :, 4, :, 0]  # Change Bx signal

    return jac_elem


def transform_ci_2_cs(jac_elem):
    # delta mx
    jac_elem[:, :, 0, :, 0] = -1*jac_elem[:, :, 0, :, 0]  # Change Bx signal
    jac_elem[:, :, 0, :, 2] = -1*jac_elem[:, :, 0, :, 2]  # Change Bz signal

    # delta my
    jac_elem[:, :, 1, :, 1] = -1*jac_elem[:, :, 1, :, 1]  # Change By signal

    # delta mz
    jac_elem[:, :, 2, :, 1] = -1*jac_elem[:, :, 2, :, 1]  # Change By signal

    # delta rx
    jac_elem[:, :, 3, :, 0] = -1*jac_elem[:, :, 3, :, 0]  # Change Bx signal
    jac_elem[:, :, 3, :, 2] = -1*jac_elem[:, :, 3, :, 2]  # Change Bz signal

    # delta ry
    jac_elem[:, :, 4, :, 1] = -1*jac_elem[:, :, 4, :, 1]  # Change By signal

    return jac_elem


def transform_cid_2_cse(jac_elem):
    jac_cie = transform_cd_2_ce(jac_elem)  # Transform to cie
    jac_cse = transform_ci_2_cs(jac_cie)  # Transform to cse
    return jac_cse


def create_cas_jacobian(jac_elem):
    cid = jac_elem.copy()
    cie = jac_elem.copy()
    csd = jac_elem.copy()
    cse = jac_elem.copy()

    csd = transform_ci_2_cs(csd)
    cie = transform_cd_2_ce(cie)
    cse = transform_cid_2_cse(cse)

    cas_jacobians = {
                     'cse': cse,
                     'csd': csd,
                     'cie': cie,
                     'cid': cid,
                    }
    return cas_jacobians


def get_block_type(block_number, nr_blocks):
    if block_number <= 3:
        block_type = block_number
    elif block_number <= nr_blocks-5:
        block_type = (block_number-4) % 4 + 4
    else:
        cont = -1*(nr_blocks-block_number-4)
        if cont % 2 != 0:
            block_type = nr_blocks-block_number-1
        elif cont == 0:
            block_type = 8
        else:
            block_type = 9
    return block_type


def create_jacobian(cas_jacobians, delta, rx, rz, z_range=600):
    cas_names = get_cassettes_names(delta)
    nr_pts_jac = cas_jacobians[cas_names[0]].shape[0]
    z_jac = np.linspace(-z_range/2, z_range/2, nr_pts_jac)
    nr_blocks = delta.cassettes_ref[cas_names[0]].nr_blocks

    jacobian = np.zeros((len(rz), len(rx), 3, 4*nr_blocks*5))

    for i, cas_name in enumerate(cas_names):
        cas_shift = delta.cassettes_ref[cas_name].center_point[-1]
        for j, block in enumerate(delta.cassettes_ref[cas_name].blocks):
            block_shift = block.center_point[-1]
            bl_type = get_block_type(j, nr_blocks)

            sel = (np.abs(rz-(block_shift + cas_shift)) <= z_range/2)
            z_pts = (rz-(block_shift + cas_shift))[sel]

            for fcomp in np.arange(3):
                for x, _ in enumerate(rx):
                    for mag in np.arange(5):
                        prejac = np.interp(z_pts, z_jac,
                                           cas_jacobians[cas_name][:, x, mag, bl_type, fcomp])

                        column = 5*i*nr_blocks + 5*j + mag
                        jacobian[sel, x, fcomp, column] = prejac
    jacobian = jacobian.reshape((len(rz)*len(rx)*3, 4*nr_blocks*5), order='F')
    return jacobian


def calc_inverse_jacobian(jacobian, tikhonovregconst=1e-3):
    u, s, vt = np.linalg.svd(jacobian, full_matrices=False)
    ismat = s/(s*s + tikhonovregconst*tikhonovregconst)
    ismat = np.diag(ismat)
    invmat = np.dot(np.dot(vt.T, ismat), u.T)
    return invmat, u, s, vt


def normalize_jacobian(jac_elem):
    jac_elem = np.swapaxes(jac_elem, -1, 2)
    shape = jac_elem.shape

    aux_elem = jac_elem.reshape(
        shape[0]*shape[1]*shape[2]*shape[3], 5, order='F')

    std_mag = np.std(aux_elem[:, 0])
    std_mag += np.std(aux_elem[:, 1])
    std_mag += np.std(aux_elem[:, 2])
    std_mag /= 3

    convx = std_mag/np.std(aux_elem[:, 3])
    convy = std_mag/np.std(aux_elem[:, 4])
    convx = 1
    convy = 1
    aux_elem[:, 3] *= convx
    aux_elem[:, 4] *= convy

    aux_elem = aux_elem.reshape(shape, order='F')
    jac_elem = np.swapaxes(aux_elem, -1, 2)

    return  jac_elem, convx, convy


if __name__ == "__main__":
    """."""
    phase = 0
    dgv = 13.125
    delta = generate_radia_model(phase=0, dgv=0)
    cas_names = get_cassettes_names(delta)
    nr_blocks = delta.cassettes_ref[cas_names[0]].nr_blocks

    # Load fmap
    fmap = get_fieldmap(phase=phase, dgv=dgv)
    rx = fmap.rx
    ry = fmap.ry
    rz = fmap.rz
    full_field_meas = get_full_field(fmap)
    full_field_meas = np.swapaxes(full_field_meas, 0, 1)

    nr_points_z = len(rz)
    nr_points_x = len(rx)

    nr_pts_jac = 2401
    z_range_jac = 600
    jac_elem = calculate_jac_elements(delta, rx, cas='cid',
                                      z_range=z_range_jac,
                                      nr_pts=nr_pts_jac)

    jac_elem_n, convx, convy = normalize_jacobian(jac_elem.copy())
    cas_jacobians = create_cas_jacobian(jac_elem_n)

    delta = generate_radia_model(phase=0, dgv=0)
    jacobian = create_jacobian(cas_jacobians, delta,
                               rx, rz, z_range=z_range_jac)

    invmat, u, s, vt = calc_inverse_jacobian(jacobian, tikhonovregconst=0.022)

    #  create radia model
    imaids.utils.set_len_tol(5e-10, 5e-10)
    delta = generate_radia_model(phase=phase, dgv=dgv)
    cas_names = get_cassettes_names(delta)
    nr_blocks = delta.cassettes_ref['cid'].nr_blocks

    residue = list()
    for k in range(20):
        print('iteraction: ', k)
        field_diff = calc_field_difference(full_field_meas, delta, rx, rz)
        residue_ = np.std(field_diff)
        residue.append(residue_)
        print(residue_)

        deltas = 0.2*np.dot(invmat, field_diff)
        deltas = deltas.reshape((len(cas_names), nr_blocks, 5))
        mag_dict = dict()
        error_dict = dict()
        cas_names = get_cassettes_names(delta)
        for i, cas_name in enumerate(cas_names):
            mag_list = list()
            error_list = list()
            for j, block in enumerate(delta.cassettes_ref[cas_name].blocks):
                mag_list.append(np.array(block.magnetization) + deltas[i, j, 0:3])

                previous_error = np.array(delta.position_err_dict[cas_name][j])
                error_new = np.array([deltas[i, j, 3]*convx, deltas[i, j, 4]*convy, 0]) + previous_error

                error_list.append(list(error_new))

            mag_dict[cas_name] = mag_list
            error_dict[cas_name] = error_list
        delta.create_radia_object(magnetization_dict=mag_dict,
                                  position_err_dict=error_dict)
        delta.set_cassete_positions(dp=phase, dgv=dgv)

    data = dict()
    data['mag'] = mag_dict
    data['shift'] = error_dict
    save_pickle(data, 'data', overwrite=True)
