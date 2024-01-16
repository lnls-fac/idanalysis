"""IDs analysis."""

import matplotlib.pyplot as _plt
import numpy as _np
import pyaccel
import pymodels
from apsuite.dynap import DynapXY
from fieldmaptrack import Beam
from scipy.integrate import cumtrapz
from scipy.optimize import minimize
import mathphys

from idanalysis import (
    IDKickMap as _IDKickMap,
    optics as optics,
    orbcorr as orbcorr,
)


class Tools:
    """Class with common methods."""

    @staticmethod
    def create_si_idmodel(
        kmap_fname,
        subsec,
        fam_name,
        nr_steps=40,
        rescale_kicks=1,
        rescale_length=1,
    ):
        """Create si id model.

        Args:
            kmap_fname (str): file name of kickmap
            subsec (str): String with ID's subsection location
            fam_name (str): ID's family name on SIRIUS model
            nr_steps (int, optional): nr of kickmap segments. Defaults to 40.
            rescale_kicks (float, optional): multiplicative factor for kicks.
            Defaults to 1.
            rescale_length (float, optional): multiplicative factor for
            id's length. Defaults to 1.


        Returns:
            IDModel object: IDModel class's object
        """
        idmodel = pymodels.si.IDModel
        si_idmodel = idmodel(
            subsec=subsec,
            file_name=kmap_fname,
            fam_name=fam_name,
            nr_steps=nr_steps,
            rescale_kicks=rescale_kicks,
            rescale_length=rescale_length,
        )
        return [
            si_idmodel,
        ]

    @staticmethod
    def create_model_with_ids(ids):
        """Create SI model with ids.

        Args:
            ids (IDModel objects): List with IDModel objects

        Returns:
            Accelerator: SI model with ids
        """
        model = pymodels.si.create_accelerator(ids=ids)
        model.cavity_on = False
        model.radiation_on = 0
        return model


class FieldMapAnalysis:
    """Fieldmaps analysis class."""

    def __init__(self, fmap):
        """Class constructor.

        Args:
            fmap (Fieldmap object): Fieldmap
        """
        self.fmap = fmap

    def get_field_component_on_axis(self, field_component):
        """Get field on axis.

        Args:
            field_component (_str_): Chosen field component
            (it can be bx, by or bz)

        Returns:
            _1D numpy array_: Field along z axis
        """
        if field_component == "by":
            b = self.fmap.by[self.fmap.ry_zero][self.fmap.rx_zero][:]
        elif field_component == "bx":
            b = self.fmap.bx[self.fmap.ry_zero][self.fmap.rx_zero][:]
        elif field_component == "bz":
            b = self.fmap.bz[self.fmap.ry_zero][self.fmap.rx_zero][:]
        return b

    def find_field_peak(self, field_component, peak_search):
        """Find field peak.

        Args:
            field_component (_str_): Chosen field component
            (it can be bx, by or bz)
            peak_search (_float_): search zone for peak
            (1 for all rz values of the fieldmap)

        Returns:
            _int_: Indice of peak
            _float_: rz value at peak field
            _float_: Field peak value
        """
        b = self.get_field_component_on_axis(field_component)
        rz = self.fmap.rz
        idx0 = _np.argmin(_np.abs(rz))
        b_slice = b[
            idx0 - int(peak_search / 2 * len(rz)) : idx0
            + int(peak_search / 2 * len(rz))
        ]

        idxmax = _np.argmax(_np.abs(b_slice))
        idxmax = idxmax + _np.argwhere(b == b_slice[0]).ravel()[0]

        rzmax = rz[idxmax]
        print("rz peak: {} mm".format(rzmax))
        print("b peak: {} T".format(b[idxmax]))
        return idxmax, rzmax, b[idxmax]

    def get_fmap_transverse_dependence(
        self, field_component, peak_search=0.1, plane="x"
    ):
        """Get field transverse dependence.

        Args:
            field_component (_str_): Chosen field component
            (it can be bx, by or bz)
            peak_search (_float_): search zone for peak
            (1 for all rz values of the fieldmap). Defaults to 0.1
            plane (str, optional): _Chosen plane (it can be x or y)_.
            Defaults to "x".

        Returns:
            _1D Numpy array_: Transverse positions of fieldmap
            _1D Numpy array_: Field values on transverse positions
        """
        idxmax, *_ = self.find_field_peak(field_component, peak_search)
        if field_component == "by":
            b = _np.array(self.fmap.by)
        elif field_component == "bx":
            b = _np.array(self.fmap.bx)
        elif field_component == "bz":
            b = _np.array(self.fmap.bz)

        if plane == "x":
            b_transverse = b[self.fmap.ry_zero, :, idxmax]
        elif plane == "y":
            b_transverse = b[:, self.fmap.rx_zero, idxmax]

        r_transverse = self.fmap.rx if plane == "x" else self.fmap.ry
        return r_transverse, b_transverse

    def get_fmap_roll_off(self, field_component, peak_search=0.1, plane="x"):
        """Get fieldmap roll-off.

        Args:
            field_component (_str_): Chosen field component
            (it can be bx, by or bz)
            peak_search (_float_): search zone for peak
            (1 for all rz values of the fieldmap). Defaults to 0.1
            plane (str, optional): _Chosen plane (it can be x or y)_.
            Defaults to "x".

        Returns:
            _1D numpy array_: Transverse positions
            _1D numpy array_: Field values on transverse positions
            _1D numpy array_: Roll-off on transverse positions [%]
        """
        r_transverse, b_transverse = self.get_fmap_transverse_dependence(
            field_component, peak_search, plane
        )

        rt_interp = _np.linspace(r_transverse.min(), r_transverse.max(), 101)
        bt_interp = _np.interp(rt_interp, r_transverse, b_transverse)

        r0_idx = _np.argmin(_np.abs(rt_interp))
        roff = 100 * (bt_interp[r0_idx] - bt_interp) / bt_interp[r0_idx]
        return rt_interp, bt_interp, roff


class RadiaModelAnalysis:
    """RADIA models field analysis class."""

    def __init__(self, model):
        """Class constructor."""
        self.model = model

    @staticmethod
    def _get_field_component_idx(field_component):
        components = {"bx": 0, "by": 1, "bz": 2}
        return components[field_component]

    def find_field_peak(self, field_component):
        """Find field peak.

        Args:
            field_component (_str_): Chosen field component
            (it can be bx, by or bz)

        Returns:
            _int_: Indice of peak
            _float_: rz value at peak field
            _float_: Field peak value
        """
        period = self.model.period_length
        comp_idx = self._get_field_component_idx(field_component)
        rz = _np.linspace(-period, period, 201)
        field = self.model.get_field(0, 0, rz)
        b = field[:, comp_idx]
        idxmax = _np.argmax(_np.abs(b))
        rzmax = rz[idxmax]
        return idxmax, rzmax, _np.abs(b[idxmax])

    def calc_radia_transverse_dependence(
        self, field_component, r_transverse, plane="x"
    ):
        """Get field transverse dependence.

        Args:
            field_component (_str_): Chosen field component
            (it can be bx, by or bz)
            r_transverse (_1D numpy array_): Numpy array with positions
            to calculate field
            plane (str, optional): _Chosen plane (it can be x or y)_.
            Defaults to "x".

        Returns:
            _1D numpy array_: Transverse positions
            _1D numpy array_: Field values on transverse positions
        """
        _, rzmax, _ = self.find_field_peak(field_component)
        comp_idx = self._get_field_component_idx(field_component)

        if plane == "x":
            field = self.model.get_field(r_transverse, 0, rzmax)
        elif plane == "y":
            field = self.model.get_field(0, r_transverse, rzmax)
        b_transverse = field[:, comp_idx]

        return r_transverse, b_transverse

    def calc_radia_roll_off(self, field_component, r_transverse, plane="x"):
        """Get field roll-off.

        Args:
            field_component (_type_): _description_
            r_transverse (_1D numpy array_): Numpy array with positions
            to calculate field
            plane (str, optional): _Chosen plane (it can be x or y)_.
            Defaults to "x".

        Returns:
            _1D numpy array_: Transverse positions
            _1D numpy array_: Field values on transverse positions
            _1D numpy array_: Roll-off on transverse positions [%]
        """
        _, b_transverse = self.calc_radia_transverse_dependence(
            field_component, r_transverse, plane
        )

        r0_idx = _np.argmin(_np.abs(r_transverse))
        roff = 100 * (b_transverse / b_transverse[r0_idx] - 1)
        return r_transverse, b_transverse, roff


class TrajectoryAnalysis:
    """Electron Trajectory analysis class."""

    def __init__(self, fieldsource):
        """Class constructor.

        Args:
            fieldsource: RADIA object or Fieldmap object
        """
        self._fieldsource = fieldsource
        self._beam_energy = Beam(3).energy  # [GeV]
        self._idkickmap = None  # Radia model or fieldmap
        self._rk_s_step = None  # [mm]
        self._traj_init_rz = None  # [mm]
        self._traj_max_rz = None  # [mm]
        self._traj_init_rx = 0  # [m]
        self._traj_init_ry = 0  # [m]
        self._traj_init_px = 0  # [rad]
        self._traj_init_py = 0  # [rad]
        self._kmap_idlen = None  # [m]
        self._kmap_fname = None
        self.traj = None

    @property
    def rk_s_step(self):
        """RK s step.

        Returns:
            float: RK trajectory step in [mm]
        """
        return self._rk_s_step

    @property
    def traj_init_rz(self):
        """Initial longitudinal point of trajectory.

        Returns:
            float: Initial point to start traj calculation in [mm]
        """
        return self._traj_init_rz

    @property
    def traj_max_rz(self):
        """Final longitudinal point of trajectory.

        Returns:
            _float : Final longitudinal point to calculate traj in [mm]
        """
        return self._traj_max_rz

    @property
    def traj_init_rx(self):
        """Initial rx point of trajectory.

        Returns:
            float: Initial horizontal point of traj [m]
        """
        return self._traj_init_rx

    @property
    def traj_init_ry(self):
        """Initial ry point of trajectory.

        Returns:
            float: Initial vertical point of traj [m]
        """
        return self._traj_init_ry

    @property
    def traj_init_px(self):
        """Initial px of trajectory.

        Returns:
            float: Initial horizontal angle of traj [rad]
        """
        return self._traj_init_px

    @property
    def traj_init_py(self):
        """Initial py of trajectory.

        Returns:
            float: Initial vertical angle of traj [rad]
        """
        return self._traj_init_py

    @property
    def kmap_idlen(self):
        """Kickmap length for SIRIUS model.

        Returns:
            float: Kickmap length for SIRIUS model
        """
        return self._kmap_idlen

    @property
    def beam_energy(self):
        """Beam energy.

        Returns:
            float: Beam energy in [GeV]
        """
        return self._beam_energy

    @property
    def fieldsource(self):
        """Field source.

        Returns:
            RADIA object or Fieldmap object: Fieldsource to calc
            trajectory
        """
        return self._fieldsource

    @property
    def kmap_fname(self):
        """Kickmap file name.

        Returns:
            str: File name to save kickmap
        """
        return self._kmap_fname

    @property
    def idkickmap(self):
        """ID Kickmap object.

        Returns:
            IDKickmap object: IDKickmap object
        """
        return self._idkickmap

    @rk_s_step.setter
    def rk_s_step(self, step):
        self._rk_s_step = step

    @traj_init_rz.setter
    def traj_init_rz(self, value):
        self._traj_init_rz = value

    @traj_max_rz.setter
    def traj_max_rz(self, value):
        self._traj_max_rz = value

    @traj_init_rx.setter
    def traj_init_rx(self, value):
        self._traj_init_rx = value

    @traj_init_ry.setter
    def traj_init_ry(self, value):
        self._traj_init_ry = value

    @traj_init_px.setter
    def traj_init_px(self, value):
        self._traj_init_px = value

    @traj_init_py.setter
    def traj_init_py(self, value):
        self._traj_init_py = value

    @kmap_idlen.setter
    def kmap_idlen(self, value):
        self._kmap_id_len = value

    @beam_energy.setter
    def beam_energy(self, value):
        self._beam_energy = value

    @fieldsource.setter
    def fieldsource(self, value):
        self._fieldsource = value

    @kmap_fname.setter
    def kmap_fname(self, value):
        self._kmap_fname = value

    def _is_fiedsource_fieldmap(self):
        typ = str(self._fieldsource.__class__)
        if "imaids" in typ:
            return False
        elif "fieldmap" in typ:
            return True
        else:
            raise ValueError("Field source must be RADIA model or fieldmap.")

    def set_traj_configs(self):
        """Set trajectory configurations."""
        self._idkickmap = _IDKickMap()

        if self._is_fiedsource_fieldmap():
            print("Fieldmap setted as fieldsource")
            self._idkickmap.fmap_fname = self.fieldsource.filename
        else:
            print("RADIA model setted as fieldsource")
            self._idkickmap.radia_model = self.fieldsource

        self._idkickmap.beam_energy = self.beam_energy
        self._idkickmap.rk_s_step = self.rk_s_step
        self._idkickmap.kmap_idlen = self.kmap_idlen

    def calculate_traj(self):
        """Calculate RK-4 trajectory.

        Returns:
            Fieldmaptrack object: Trajectory object
        """
        print("Calculating trajectory...")
        self._idkickmap.fmap_calc_trajectory(
            traj_init_rx=self.traj_init_rx,
            traj_init_ry=self.traj_init_ry,
            traj_init_px=self.traj_init_px,
            traj_init_py=self.traj_init_py,
            traj_init_rz=self.traj_init_rz,
            traj_rk_min_rz=self.traj_max_rz,
        )
        self.traj = self._idkickmap.traj
        return self.traj

    def generate_kickmap(self, gridx, gridy):
        """Generate kickmap.

        Args:
            gridx (1D numpy array): Values to calculate kickmap
            gridy (1D numpy array): Values to calculate kickmap
        """
        self._idkickmap.fmap_calc_kickmap(posx=gridx, posy=gridy)
        self._idkickmap.save_kickmap_file(kickmap_filename=self._kmap_fname)


class KickmapAnalysis(Tools):
    """Kickmap analysis class."""

    def __init__(self, kmap_fname):
        """Class constructor."""
        self.kmap_fname = kmap_fname
        self._idkickmap = _IDKickMap(kmap_fname=self.kmap_fname)

    def run_shift_kickmap(self):
        """Generate kickmap without dipolar term."""
        fname = self.kmap_fname
        fname = fname.replace(".txt", "-shifted_on_axis.txt")
        self._idkickmap.save_kickmap_file(fname)

    def _calc_idkmap_kicks(self, plane_idx=0, indep_var="X"):
        beam = Beam(energy=3)
        brho = beam.brho
        rx0 = self._idkickmap.posx
        ry0 = self._idkickmap.posy
        fposx = self._idkickmap.fposx
        fposy = self._idkickmap.fposy
        kickx = self._idkickmap.kickx
        kicky = self._idkickmap.kicky
        if indep_var.lower() == "x":
            rxf = fposx[plane_idx, :]
            ryf = fposy[plane_idx, :]
            pxf = kickx[plane_idx, :] / brho**2
            pyf = kicky[plane_idx, :] / brho**2
        elif indep_var.lower() == "y":
            rxf = fposx[:, plane_idx]
            ryf = fposy[:, plane_idx]
            pxf = kickx[:, plane_idx] / brho**2
            pyf = kicky[:, plane_idx] / brho**2

        return rx0, ry0, pxf, pyf, rxf, ryf

    def get_kicks_at_plane(self, indep_var="X", plane=0):
        """Get kicks at plane.

        Args:
            indep_var (str, optional): Variable to get kicks as a function of
            (it can be "x" or "y"). Defaults to "X".
            plane (float, optional): Value of the chosen plane
        Returns:
            1D numpy array: rx positions of kickmap
            1D numpy array: ry positions of kickmap
            1D numpy array: px kicks of kickmap
            1D numpy array: py kicks of kickmap
            1D numpy array: rx final positions of kickmap
            1D numpy array: ry final positions of kickmap
        """
        if indep_var.lower() == "x":
            pos_zero_idx = list(self._idkickmap.posy).index(plane)
        elif indep_var.lower() == "y":
            pos_zero_idx = list(self._idkickmap.posx).index(plane)

        rx0, ry0, pxf, pyf, rxf, ryf = self._calc_idkmap_kicks(
            plane_idx=pos_zero_idx, indep_var=indep_var
        )

        return rx0, ry0, pxf, pyf, rxf, ryf

    def get_kicks_all_planes(self, indep_var="X"):
        """Get kicks at all planes.

        Args:
            indep_var (str, optional): Variable to get kicks as a function of
            (it can be "x" or "y"). Defaults to "X".

        Returns:
            1D numpy array: rx positions of kickmap
            1D numpy array: ry positions of kickmap
            1D numpy array: px kicks of kickmap
            1D numpy array: py kicks of kickmap
            1D numpy array: rx final positions of kickmap
            1D numpy array: ry final positions of kickmap
        """
        if indep_var.lower() == "x":
            kmappos = self._idkickmap.posy
            nr_pts = len(self._idkickmap.posx)
        elif indep_var.lower() == "y":
            kmappos = self._idkickmap.posx
            nr_pts = len(self._idkickmap.posy)

        pxf, pyf, rxf, ryf = (
            _np.zeros((len(kmappos), nr_pts)),
            _np.zeros((len(kmappos), nr_pts)),
            _np.zeros((len(kmappos), nr_pts)),
            _np.zeros((len(kmappos), nr_pts)),
        )
        for plane_idx, pos in enumerate(kmappos):
            rx0, ry0, pxf_, pyf_, rxf_, ryf_ = self.get_kicks_at_plane(
                indep_var, pos
            )
            pxf[plane_idx, :] = pxf_
            pyf[plane_idx, :] = pyf_
            rxf[plane_idx, :] = rxf_
            ryf[plane_idx, :] = ryf_

        return rx0, ry0, pxf, pyf, rxf, ryf

    def check_tracking_at_plane(
        self,
        kmap_fname,
        subsec,
        fam_name,
        nr_steps=40,
        rescale_kicks=1,
        rescale_length=1,
        indep_var="X",
        plane=0,
    ):
        """Check kicks using tracking.

        Args:
            kmap_fname (str): file name of kickmap
            subsec (str): String with ID's subsection location
            fam_name (str): ID's family name on SIRIUS model
            nr_steps (int, optional): nr of kickmap segments. Defaults to 40.
            rescale_kicks (float, optional): multiplicative factor for kicks.
            Defaults to 1.
            rescale_length (float, optional): multiplicative factor for
            id's length. Defaults to 1.
            indep_var (str, optional): Variable to get kicks as a function of
            (it can be "x" or "y"). Defaults to "X".
            plane (float, optional): Value of the chosen plane

        Returns:
            1D numpy array: px kicks of tracking
            1D numpy array: py kicks of tracking
            1D numpy array: rx final positions of tracking
            1D numpy array: ry final positions of tracking
        """
        ids = self.create_si_idmodel(
            kmap_fname,
            subsec,
            fam_name,
            nr_steps,
            rescale_kicks,
            rescale_length,
        )
        model = self.create_model_with_ids(ids)

        famdata = pymodels.si.get_family_data(model)
        mia = pyaccel.lattice.find_indices(model, "fam_name", "mia")
        mib = pyaccel.lattice.find_indices(model, "fam_name", "mib")
        mip = pyaccel.lattice.find_indices(model, "fam_name", "mip")
        mid_subsections = _np.sort(_np.array(mia + mib + mip))
        idx = mid_subsections[int(subsec[2:4]) - 1]
        idcs = _np.array(famdata[fam_name]["index"])
        idcs = idcs[_np.isclose(idcs.mean(axis=1), idx)].ravel()
        idx_begin = idcs[0]
        idx_end = idcs[-1]
        idx_dif = idx_end - idx_begin

        model = pyaccel.lattice.shift(model, start=idx_begin)

        out = self.get_kicks_at_plane(indep_var=indep_var, plane=plane)
        rx0, ry0 = out[0], out[1]
        r0 = rx0 if indep_var.lower() == "x" else ry0
        rxf_trk, ryf_trk = _np.ones(len(r0)), _np.ones(len(r0))
        pxf_trk, pyf_trk = _np.ones(len(r0)), _np.ones(len(r0))
        for i, pos0 in enumerate(r0):
            coord_ini = (
                _np.array([pos0, 0, 0, 0, 0, 0])
                if indep_var.lower() == "x"
                else _np.array([0, 0, pos0, 0, 0, 0])
            )
            coord_fin, *_ = pyaccel.tracking.line_pass(
                model, coord_ini, indices="open"
            )
            rxf_trk[i] = coord_fin[0, idx_dif + 1]
            ryf_trk[i] = coord_fin[2, idx_dif + 1]
            pxf_trk[i] = coord_fin[1, idx_dif + 1]
            pyf_trk[i] = coord_fin[3, idx_dif + 1]

        return pxf_trk, pyf_trk, rxf_trk, ryf_trk


class StorageRingAnalysis(Tools):
    """Class to insert IDs on SI model and do analysis."""

    class CalcTypes:
        """Sub class to define which analysis will be done."""

        nominal = 0
        symmetrized = 1
        nonsymmetrized = 2

    def __init__(self):
        """Class constructor."""
        self.ids = []
        self.model_ids = None
        self.nom_model = None

        self._orbcorr_system = "SOFB"
        self._plot_orbcorr = False
        self._calc_type = self.CalcTypes.symmetrized
        self._figs_fpath = None

    @property
    def orbcorr_system(self):
        """Orbit correction system.

        Returns:
            string: Orbit correction system chosen.
        """
        return self._orbcorr_system

    @property
    def plot_orbcorr(self):
        """Plot orbit correction results.

        Returns:
            Bool: If True orbit correction results will be ploted.
        """
        return self._plot_orbcorr

    @property
    def calc_type(self):
        """Calculation type.

        Returns:
            CalcTypes options: It can be symmetrized, nominal or nonsymmetrized
        """
        return self._calc_type

    @property
    def figures_path(self):
        """Path to save figures.

        Returns:
            str: File path to save figures.
        """
        return self._figs_path

    @orbcorr_system.setter
    def orbcorr_system(self, value):
        self._orbcorr_system = value

    @plot_orbcorr.setter
    def plot_orbcorr(self, value):
        self._plot_orbcorr = value

    @calc_type.setter
    def calc_type(self, value):
        self._calc_type = value

    @figures_path.setter
    def figures_path(self, value):
        self._figs_fpath = value

    def add_id_to_model(
        self,
        kmap_fname,
        subsec,
        fam_name,
        nr_steps=40,
        rescale_kicks=1,
        rescale_length=1,
    ):
        """Add ID to SI model.

        Args:
            kmap_fname (str): file name of kickmap
            subsec (str): String with ID's subsection location
            fam_name (str): ID's family name on SIRIUS model
            nr_steps (int, optional): nr of kickmap segments. Defaults to 40.
            rescale_kicks (float, optional): multiplicative factor for kicks.
            Defaults to 1.
            rescale_length (float, optional): multiplicative factor for
            id's length. Defaults to 1.

        Returns:
            Bool: True
        """
        idmodel = self.create_si_idmodel(
            kmap_fname,
            subsec,
            fam_name,
            nr_steps,
            rescale_kicks,
            rescale_length,
        )
        self.ids.extend(idmodel)
        return True

    def set_model_ids(self):
        """Set model with IDS.

        Returns:
            Accelerator_: SI model with ids
        """
        model = self.create_model_with_ids(self.ids)
        self.model_ids = model
        return model

    def _create_model_nominal(self):
        model0 = pymodels.si.create_accelerator()
        model0.cavity_on = False
        model0.radiation_on = 0
        return model0

    def do_orbit_correction(self, corr_system, plot_flag):
        """Do orbit correction using feedback systems.

        Args:
            corr_system (str): "SOFB" or "FOFB"
            plot_flag (Bool): True to plot orbit correction results.

        Returns:
            1D numpy arrays: calculated kicks
            1D numpy arrays: position of bpms
            1D numpy arrays: Horizontal orbit distortion after correction
            1D numpy arrays: Vertical orbit distortion after correction
            1D numpy arrays: Horizontal orbit distortion before correction
            1D numpy arrays: Vertical orbit distortion before correction
            1D numpy arrays: BPMS indices
        """
        results = orbcorr.correct_orbit_fb(
            self.nom_model,
            self.model_ids,
            corr_system=corr_system,
            plot_flag=plot_flag,
        )
        return results

    def get_symm_knobs_locs(self):
        """Get symmetrization knobs.

        Returns:
            list: Indices of knobs
            list: Indices of beta
            list: straight section numbers
        """
        straight_nr = dict()
        knobs = dict()
        locs_beta = dict()
        for id_ in self.ids:
            straight_nr_ = int(id_.subsec[2:4])

            # get knobs and beta locations
            if straight_nr_ is not None:
                _, knobs_, _ = optics.symm_get_knobs(
                    self.model_ids, straight_nr_
                )
                locs_beta_ = optics.symm_get_locs_beta(knobs_)
            else:
                knobs_, locs_beta_ = None, None

            straight_nr[id_.subsec] = straight_nr_
            knobs[id_.subsec] = knobs_
            locs_beta[id_.subsec] = locs_beta_

        return knobs, locs_beta, straight_nr

    def calc_action_ratio(self, x0, nturns=1000):
        """Calc action ratio.

        Args:
            x0 (float): Initial position of particle
            nturns (int, optional): Number of turns. Defaults to 1000.

        Returns:
            float: Ratio of action
        """
        coord_ini = _np.array([x0, 0, 0, 0, 0, 0])
        coord_fin, *_ = pyaccel.tracking.ring_pass(
            self.model_ids,
            coord_ini,
            nr_turns=nturns,
            turn_by_turn=True,
            parallel=True,
        )
        rx = coord_fin[0, :]
        ry = coord_fin[2, :]
        twiss, *_ = pyaccel.optics.calc_twiss(self.model_ids)
        betax, betay = twiss.betax, twiss.betay  # Beta functions
        jx = 1 / (betax[0] * nturns) * (_np.sum(rx**2))
        jy = 1 / (betay[0] * nturns) * (_np.sum(ry**2))

        return jy / jx

    def calc_coupling(self):
        """Calc coupling using Edwards-Teng.

        Returns:
            float: Minimum tune separation
        """
        ed_tang, *_ = pyaccel.optics.calc_edwards_teng(self.model_ids)
        min_tunesep, _ = pyaccel.optics.estimate_coupling_parameters(ed_tang)
        return min_tunesep

    def calc_dtune(self, twiss=None, twiss_nom=None):
        """Calc detune.

        Args:
            twiss (twiss, optional): twiss parameters of perturbed model.
            Defaults to None.
            twiss_nom (twiss, optional): twiss parameters of nominal model.

        Returns:
            float: horizontal detune
            float: vertical detune
        """
        if twiss is None:
            twiss, *_ = pyaccel.optics.calc_twiss(
                self.model_ids, indices="closed"
            )
        if twiss_nom is None:
            twiss, *_ = pyaccel.optics.calc_twiss(
                self.nom_model, indices="closed"
            )
        dtunex = (twiss.mux[-1] - twiss_nom.mux[-1]) / 2 / _np.pi
        dtuney = (twiss.muy[-1] - twiss_nom.muy[-1]) / 2 / _np.pi
        return (dtunex, dtuney)

    def calc_beta_beating(self, twiss=None, twiss_nom=None):
        """Calc beta beating.

        Args:
            twiss (twiss, optional): twiss parameters of perturbed model.
            Defaults to None.
            twiss_nom (twiss, optional): twiss parameters of nominal model.
            Defaults to None.

        Returns:
            1D numpy array: Horizontal beta beating
            1D numpy array: Vertical beta beating
            float: Horizontal beta beating rms
            float: Vertical beta beating rms
            float: Horizontal beta beating absolute maximum
            float: Vertical beta beating absolute maximum
        """
        if twiss is None:
            twiss, *_ = pyaccel.optics.calc_twiss(
                self.model_ids, indices="closed"
            )
        if twiss_nom is None:
            twiss_nom, *_ = pyaccel.optics.calc_twiss(
                self.nom_model, indices="closed"
            )
        bbeatx = 100 * (twiss.betax - twiss_nom.betax) / twiss_nom.betax
        bbeaty = 100 * (twiss.betay - twiss_nom.betay) / twiss_nom.betay
        bbeatx_rms = _np.std(bbeatx)
        bbeaty_rms = _np.std(bbeaty)
        bbeatx_absmax = _np.max(_np.abs(bbeatx))
        bbeaty_absmax = _np.max(_np.abs(bbeaty))
        return (
            bbeatx,
            bbeaty,
            bbeatx_rms,
            bbeaty_rms,
            bbeatx_absmax,
            bbeaty_absmax,
        )

    def correct_beta(
        self, straight_nr, knobs, goal_beta, goal_alpha, verbose=True
    ):
        """Correct beta functions.

        Args:
            straight_nr (int): Straight section number
            knobs (list): Knobs indices used in correction.
            goal_beta (list): List of goal betas
            goal_alpha (list): List f goal alphas
            verbose (bool, optional): If True prints will be executed.
            Defaults to True.

        Returns:
            twiss: Twiss after beta correction
            string: String containing information about correction
        """
        dk_tot = _np.zeros(len(knobs))
        for i in range(7):
            dk, k = optics.correct_symmetry_withbeta(
                self.model_ids, straight_nr, goal_beta, goal_alpha
            )
            if i == 0:
                k0 = k
            if verbose:
                print("iteration #{}, \u0394K: {}".format(i + 1, dk))
            dk_tot += dk
        delta = dk_tot / k0
        stg = str()
        for i, fam in enumerate(knobs):
            stg += "{:<9s} \u0394K: {:+9.3f} % \n".format(fam, 100 * delta[i])
        twiss, *_ = pyaccel.optics.calc_twiss(self.model_ids, indices="closed")
        if verbose:
            print(stg)
            print()
        return twiss, stg

    def correct_tunes(self, goal_tunes, verbose=True, nr_iter=2):
        """Correct tunes.

        Args:
            goal_tunes (list): Goal tunes.
            verbose (bool, optional): If True prints will be executed.
            Defaults to True.
            nr_iter (int, optional): Number of iteractions. Defaults to 2.

        Returns:
            Twiss: Twiss after tunes corretion
        """
        twiss, *_ = pyaccel.optics.calc_twiss(self.model_ids, indices="closed")
        tunes = (
            twiss.mux[-1] / _np.pi / 2,
            twiss.muy[-1] / _np.pi / 2,
        )
        if verbose:
            print("init    tunes: {:.9f} {:.9f}".format(tunes[0], tunes[1]))
        for i in range(nr_iter):
            optics.correct_tunes_twoknobs(self.model_ids, goal_tunes)
            twiss, *_ = pyaccel.optics.calc_twiss(
                self.model_ids, indices="closed"
            )
            tunes = twiss.mux[-1] / _np.pi / 2, twiss.muy[-1] / _np.pi / 2
            if verbose:
                print(
                    "iter #{} tunes: {:.9f} {:.9f}".format(
                        i + 1, tunes[0], tunes[1]
                    )
                )
        if verbose:
            print(
                "goal    tunes: {:.9f} {:.9f}".format(
                    goal_tunes[0], goal_tunes[1]
                )
            )
            print()
        return twiss

    def do_optics_corrections(self):
        """Do optics correction.

        Returns:
            Twiss: Twiss without any correction
            Twiss: Twiss after beta correction
            Twiss: Twiss after tune corretion
            string: String containg informations about beta correction.
        """
        twiss_no_corr, *_ = pyaccel.optics.calc_twiss(
            self.model_ids, indices="closed"
        )
        knobs, locs_beta, straight_nr = self.get_symm_knobs_locs()

        print("element indices for straight section begin and end:")
        for idsubsec, locs_beta_ in locs_beta.items():
            print(idsubsec, locs_beta_)

        print("local quadrupole fams: ")
        for idsubsec, knobs_ in knobs.items():
            print(idsubsec, knobs_)

        # get list of ID model indices and set rescale_kicks to zero
        ids_ind_all = orbcorr.get_ids_indices(self.model_ids)
        rescale_kicks_orig = list()
        for idx in range(len(ids_ind_all) // 2):
            ind_id = ids_ind_all[2 * idx : 2 * (idx + 1)]
            rescale_kicks_orig.append(self.model_ids[ind_id[0]].rescale_kicks)
            self.model_ids[ind_id[0]].rescale_kicks = 0
            self.model_ids[ind_id[1]].rescale_kicks = 0

        # loop over IDs turning rescale_kicks on, one by one.
        for idx in range(len(ids_ind_all) // 2):
            # turn rescale_kicks on for ID index idx
            ind_id = ids_ind_all[2 * idx : 2 * (idx + 1)]
            self.model_ids[ind_id[0]].rescale_kicks = rescale_kicks_orig[idx]
            self.model_ids[ind_id[1]].rescale_kicks = rescale_kicks_orig[idx]
            fam_name = self.model_ids[ind_id[0]].fam_name

            # search knob and straight_nr for ID index idx
            for subsec in knobs:
                straight_nr_ = straight_nr[subsec]
                knobs_ = knobs[subsec]
                locs_beta_ = locs_beta[subsec]
                if min(locs_beta_) < ind_id[0] and ind_id[1] < max(locs_beta_):
                    break
            print("symmetrizing ID {} in subsec {}".format(fam_name, subsec))

            # calculate nominal twiss
            twiss0, *_ = pyaccel.optics.calc_twiss(self.nom_model)
            goal_tunes = _np.array(
                [
                    twiss0.mux[-1] / 2 / _np.pi,
                    twiss0.muy[-1] / 2 / _np.pi,
                ]
            )
            goal_beta = _np.array(
                [
                    twiss0.betax[locs_beta_],
                    twiss0.betay[locs_beta_],
                ]
            )
            goal_alpha = _np.array(
                [
                    twiss0.alphax[locs_beta_],
                    twiss0.alphay[locs_beta_],
                ]
            )

            # Symmetrize optics (local quad fam knobs)
            twiss_beta_corr, stg = self.correct_beta(
                straight_nr_, knobs_, goal_beta, goal_alpha
            )

            # Correct tunes
            twiss_tune_corr = self.correct_tunes(goal_tunes)

        return twiss_no_corr, twiss_beta_corr, twiss_tune_corr, stg

    def plot_optics_corr_results(
        self,
        twiss_nom,
        twiss_no_corr,
        twiss_beta_corr,
        twiss_tune_corr,
        stg_beta=None,
        fpath=None,
    ):
        """Plot optics correction results.

        Args:
            twiss_nom (Twiss): Twiss of nominal model
            twiss_no_corr (Twiss): Twiss without any correction
            twiss_beta_corr (Twiss): Twiss after beta correction
            twiss_tune_corr (Twiss): Twiss after tune corretion
            stg_beta (string, optional): String containg informations about
             beta correction. Defaults to None.
            fpath (string, optional): _description_. Defaults to None.

        Returns:
            Bool: True
        """
        # Compare optics between nominal value and uncorrect optics due ID
        dtunex, dtuney = self.calc_dtune(
            twiss=twiss_no_corr, twiss_nom=twiss_nom
        )
        bbeats = self.calc_beta_beating(
            twiss=twiss_no_corr, twiss_nom=twiss_nom
        )
        bbeatx, bbeaty = bbeats[0], bbeats[1]
        bbeatx_rms, bbeaty_rms = bbeats[2], bbeats[3]
        bbeatx_absmax, bbeaty_absmax = bbeats[4], bbeats[5]
        bbmax = _np.max(_np.concatenate((bbeatx, bbeaty)))
        bbmin = _np.min(_np.concatenate((bbeatx, bbeaty)))
        ylim = (1.1 * bbmin, 1.1 * bbmax)
        print("Not symmetrized optics :")
        print(f"dtunex: {dtunex:+.2e}")
        print(f"dtuney: {dtuney:+.2e}")
        print(
            f"bbetax: {bbeatx_rms:04.3f} % rms,"
            + f" {bbeatx_absmax:04.3f} % absmax"
        )
        print(
            f"bbetay: {bbeaty_rms:04.3f} % rms,"
            + f" {bbeaty_absmax:04.3f} % absmax"
        )
        print()

        stg_tune = f"\u0394\u03BDx: {dtunex:+0.04f}\n"
        stg_tune += f"\u0394\u03BDy: {dtuney:+0.04f}"
        labelx = f"X ({bbeatx_rms:.3f} % rms)"
        labely = f"Y ({bbeaty_rms:.3f} % rms)"

        _plt.figure()
        _plt.plot(twiss_nom.spos, bbeatx, color="b", alpha=1.0, label=labelx)
        _plt.plot(twiss_nom.spos, bbeaty, color="r", alpha=0.8, label=labely)
        _plt.ylim(ylim)
        _plt.xlabel("spos [m]")
        _plt.ylabel("Beta Beating [%]")
        _plt.title("Tune shift:" + "\n" + stg_tune)
        _plt.suptitle("Non-symmetrized optics")
        _plt.tight_layout()
        _plt.legend()
        _plt.grid()
        pyaccel.graphics.draw_lattice(
            self.model_ids, offset=bbmin, height=bbmax / 8, gca=True
        )
        if fpath is not None:
            _plt.savefig(fpath + "Non-symmetrized", dpi=300, format="png")

        # Compare optics between nominal value and symmetrized optics
        dtunex, dtuney = self.calc_dtune(
            twiss=twiss_beta_corr, twiss_nom=twiss_nom
        )
        bbeats = self.calc_beta_beating(
            twiss=twiss_beta_corr, twiss_nom=twiss_nom
        )
        bbeatx, bbeaty = bbeats[0], bbeats[1]
        bbeatx_rms, bbeaty_rms = bbeats[2], bbeats[3]
        bbeatx_absmax, bbeaty_absmax = bbeats[4], bbeats[5]
        bbmax = _np.max(_np.concatenate((bbeatx, bbeaty)))
        bbmin = _np.min(_np.concatenate((bbeatx, bbeaty)))
        ylim = (1.1 * bbmin, 1.1 * bbmax)
        print("symmetrized optics but uncorrect tunes:")
        print(f"dtunex: {dtunex:+.0e}")
        print(f"dtuney: {dtuney:+.0e}")
        print(
            f"bbetax: {bbeatx_rms:04.3f} % rms,"
            + f" {bbeatx_absmax:04.3f} % absmax"
        )
        print(
            f"bbetay: {bbeaty_rms:04.3f} % rms,"
            + f" {bbeaty_absmax:04.3f} % absmax"
        )
        print()

        labelx = f"X ({bbeatx_rms:.3f} % rms)"
        labely = f"Y ({bbeaty_rms:.3f} % rms)"

        _plt.figure()
        _plt.plot(twiss_nom.spos, bbeatx, color="b", alpha=1.0, label=labelx)
        _plt.plot(twiss_nom.spos, bbeaty, color="r", alpha=0.8, label=labely)
        _plt.ylim(ylim)
        # bbmax = _np.max(_np.concatenate((bbeatx, bbeaty)))
        _plt.xlabel("spos [m]")
        _plt.ylabel("Beta Beating [%]")
        _plt.title("Beta Beating")
        _plt.suptitle("Symmetrized optics and uncorrected tunes")
        _plt.legend()
        _plt.grid()
        _plt.tight_layout()
        pyaccel.graphics.draw_lattice(
            self.model_ids, offset=bbmin, height=bbmax / 8, gca=True
        )
        if fpath is not None:
            _plt.savefig(fpath + "Symmetrized", dpi=300, format="png")

        # Compare optics between nominal value and all corrected
        dtunex, dtuney = self.calc_dtune(
            twiss=twiss_tune_corr, twiss_nom=twiss_nom
        )
        bbeats = self.calc_beta_beating(
            twiss=twiss_tune_corr, twiss_nom=twiss_nom
        )
        bbeatx, bbeaty = bbeats[0], bbeats[1]
        bbeatx_rms, bbeaty_rms = bbeats[2], bbeats[3]
        bbeatx_absmax, bbeaty_absmax = bbeats[4], bbeats[5]
        bbmax = _np.max(_np.concatenate((bbeatx, bbeaty)))
        bbmin = _np.min(_np.concatenate((bbeatx, bbeaty)))
        ylim = (1.1 * bbmin, 1.1 * bbmax)

        print("symmetrized optics and corrected tunes:")
        print(f"dtunex: {dtunex:+.0e}")
        print(f"dtuney: {dtuney:+.0e}")
        print(
            f"bbetax: {bbeatx_rms:04.3f} % rms,"
            + f" {bbeatx_absmax:04.3f} % absmax"
        )
        print(
            f"bbetay: {bbeaty_rms:04.3f} % rms,"
            + f" {bbeaty_absmax:04.3f} % absmax"
        )

        labelx = f"X ({bbeatx_rms:.3f} % rms)"
        labely = f"Y ({bbeaty_rms:.3f} % rms)"

        _plt.figure()
        _plt.plot(twiss_nom.spos, bbeatx, color="b", alpha=1.0, label=labelx)
        _plt.plot(twiss_nom.spos, bbeaty, color="r", alpha=0.8, label=labely)

        _plt.ylim(ylim)
        # bbmax = _np.max(_np.concatenate((bbeatx, bbeaty)))
        _plt.xlabel("spos [m]")
        _plt.ylabel("Beta Beating [%]")
        _plt.title("Corrections: {}".format(stg_beta))
        _plt.suptitle("Symmetrized optics and corrected tunes")
        _plt.legend()
        _plt.grid()
        _plt.tight_layout()
        pyaccel.graphics.draw_lattice(
            self.model_ids, offset=bbmin, height=bbmax / 8, gca=True
        )
        if fpath is not None:
            _plt.savefig(fpath + "Symmetrized_TuneCorr", dpi=300, format="png")
        else:
            _plt.show()
        return True

    def run_correction_algorithms(self):
        """Run correction algorithms.

        Raises:
            ValueError: Error if calc type is invalid.

        Returns:
            Bool: True
        """
        self.nom_model = self._create_model_nominal()
        if self.calc_type == self.CalcTypes.nominal:
            self.model_ids = self._create_model_nominal()
        elif self.calc_type in (
            self.CalcTypes.symmetrized,
            self.CalcTypes.nonsymmetrized,
        ):
            self.set_model_ids()
            _ = self.do_orbit_correction(
                self.orbcorr_system, self._plot_orbcorr
            )
            if self.calc_type == self.CalcTypes.symmetrized:
                opt_corr = self.do_optics_corrections()
                twiss_no_corr, twiss_beta_corr, twiss_tune_corr, stg = opt_corr
                twiss_nom, *_ = pyaccel.optics.calc_twiss(
                    self.nom_model, indices="closed"
                )
                self.plot_optics_corr_results(
                    twiss_nom,
                    twiss_no_corr,
                    twiss_beta_corr,
                    twiss_tune_corr,
                    stg,
                    self._figs_fpath,
                )
        else:
            raise ValueError("Invalid calc_type")
        return True

    def analysis_dynapt(
        self,
        x_nrpts=40,
        y_nrpts=20,
        nr_turns=2048,
        nux_bounds=(49.05, 49.50),
        nuy_bounds=(14.12, 14.45),
        fpath=None,
        sufix=None,
    ):
        """Analysis dynamic aperture.

        Args:
            x_nrpts (int, optional): Horizontal nr of points. Defaults to 40.
            y_nrpts (int, optional): Vertical nr of points. Defaults to 20.
            nr_turns (int, optional): Number of turns. Defaults to 2048.
            nux_bounds (tuple, optional): Tune x bounds on plot.
            Defaults to (49.05, 49.50).
            nuy_bounds (tuple, optional): Tune y bounds on plot.
            Defaults to (14.12, 14.45).
            fpath (Str, optional): Figure path. Defaults to None.
            sufix (Str, optional): Figure name's sufix. Defaults to None.
        """
        self.model_ids.radiation_on = 0
        self.model_ids.cavity_on = False
        self.model_ids.vchamber_on = True

        dynapxy = DynapXY(self.model_ids)
        dynapxy.params.x_nrpts = x_nrpts
        dynapxy.params.y_nrpts = y_nrpts
        dynapxy.params.nrturns = nr_turns
        print(dynapxy)
        dynapxy.do_tracking()
        dynapxy.process_data()
        fig, _, _ = dynapxy.make_figure_diffusion(
            orders=(1, 2, 3, 4),
            nuy_bounds=nuy_bounds,
            nux_bounds=nux_bounds,
        )

        if fpath is not None:
            figname = fpath + "Dynapt" + sufix
            fig.savefig(figname, dpi=300, format="png")
        _plt.show()


class EqParamAnalysis:
    """Class to calculate beam equilibrium parameters."""

    class InsertionParams:
        """Class to specify insertion parameters."""

        def __init__(self):
            """Class constructor."""
            self._field_profile = None
            self._idname = None
            self._bx_peak = None
            self._by_peak = None
            self._kx = None
            self._ky = None
            self._period = None
            self._length = None
            self._nr_periods = None
            self._straight_section = None
            self._etax = None
            self._etaxdot = None
            self._etay = None
            self._etaydot = None
            self._delta_i1x = None
            self._delta_i2x = None
            self._delta_i3x = None
            self._delta_i4x = None
            self._delta_i5x = None
            self._delta_i1y = None
            self._delta_i2y = None
            self._delta_i3y = None
            self._delta_i4y = None
            self._delta_i5y = None
            self._u0 = None
            self.brho = Beam(3).brho
            self._e = mathphys.constants.elementary_charge
            self._m = mathphys.constants.electron_mass
            self._c = mathphys.constants.light_speed

        @property
        def idname(self):
            """Insertion device's name.

            Returns:
                string: ID's name
            """
            return self._idname

        @idname.setter
        def idname(self, value):
            self._idname = value

        @property
        def field_profile(self):
            """Field profile.

            Returns:
                numpy array: First column contains longitudinalspatial
                coordinate (z) [mm, second column contais vertical field
                [T], and third column constais horizontal field [T].
            """
            return self._field_profile

        @field_profile.setter
        def field_profile(self, value):
            self._field_profile = value

        @property
        def bx_peak(self):
            """Insertion device horizontal peak field [T].

            Returns:
                float: bx peak field [T]
            """
            return self._bx_peak

        @bx_peak.setter
        def bx_peak(self, value):
            self._bx_peak = value
            if self.period is not None:
                self._kx = (
                    1e-3
                    * self._e
                    / (2 * _np.pi * self._m * self._c)
                    * self._bx_peak
                    * self.period
                )

        @property
        def by_peak(self):
            """Insertion device vertical peak field [T].

            Returns:
                float: by peak field [T]
            """
            return self._by_peak

        @by_peak.setter
        def by_peak(self, value):
            self._by_peak = value
            if self.period is not None:
                self._ky = (
                    1e-3
                    * self._e
                    / (2 * _np.pi * self._m * self._c)
                    * self._by_peak
                    * self.period
                )

        @property
        def kx(self):
            """Horizontal deflection parameter (Kx).

            Returns:
                float: Horizontal deflection parameter
            """
            return self._kx

        @kx.setter
        def kx(self, value):
            self._kx = value
            if self.period is not None:
                self._bx_peak = (
                    2
                    * _np.pi
                    * self._m
                    * self._c
                    * self._kx
                    / (self._e * 1e-3 * self.period)
                )

        @property
        def ky(self):
            """Vertical deflection parameter (Ky).

            Returns:
                float: Vertical deflection parameter
            """
            return self._ky

        @ky.setter
        def ky(self, value):
            self._ky = value
            if self.period is not None:
                self._by_peak = (
                    2
                    * _np.pi
                    * self._m
                    * self._c
                    * self._ky
                    / (self._e * 1e-3 * self.period)
                )

        @property
        def period(self):
            """Insertion device period [mm].

            Returns:
                float: ID's period [mm]
            """
            return self._period

        @period.setter
        def period(self, value):
            self._period = value
            if self._bx_peak is not None:
                self._kx = (
                    1e-3
                    * self._e
                    / (2 * _np.pi * self._m * self._c)
                    * self._bx_peak
                    * self.period
                )
            if self._by_peak is not None:
                self._ky = (
                    1e-3
                    * self._e
                    / (2 * _np.pi * self._m * self._c)
                    * self._by_peak
                    * self.period
                )
            if self._kx is not None:
                self._bx_peak = (
                    2
                    * _np.pi
                    * self._m
                    * self._c
                    * self._kx
                    / (self._e * 1e-3 * self.period)
                )
            if self._ky is not None:
                self._by_peak = (
                    2
                    * _np.pi
                    * self._m
                    * self._c
                    * self._ky
                    / (self._e * 1e-3 * self.period)
                )

        @property
        def length(self):
            """Insertion device length [m].

            Returns:
                float: ID's length [m]
            """
            return self._length

        @length.setter
        def length(self, value):
            self._length = value
            if self.period is not None:
                self._nr_periods = int(self._length / (1e-3 * self.period)) - 2

        @property
        def nr_periods(self):
            """Insertion device number of periods.

            Returns:
                int: Number of ID's periods.
            """
            return self._nr_periods

        @nr_periods.setter
        def nr_periods(self, value):
            self._nr_periods = value
            if self.period is not None:
                self._length = 1e-3 * self.period * (self._nr_periods + 2)

        @property
        def straight_section(self):
            """Straight section where the ID will be inserted.

            Returns:
                int: Straight section number
            """
            return self._straight_section

        @straight_section.setter
        def straight_section(self, value):
            self._straight_section = value

        @property
        def etax(self):
            """Horizontal dispersion generated by ID.

            Returns:
                Numpy 1d array: Horizontal dispersion function
            """
            return self._etax

        @property
        def etaxdot(self):
            """d(etax)/ds generated by ID.

            Returns:
                Numpy 1d array: Derivative of horizontal dispersion function
            """
            return self._etaxdot

        @property
        def etay(self):
            """Vertical dispersion generated by ID.

            Returns:
                Numpy 1d array: Vertical dispersion function
            """
            return self._etay

        @property
        def etaydot(self):
            """d(etay)/ds generated by ID.

            Returns:
                Numpy 1d array: Derivative of vertical dispersion function
            """
            return self._etaydot

        @property
        def delta_i1x(self):
            """Contribution of ID to first rad integral.

            Returns:
                Numpy 1d array: Delta i1x
            """
            return self._delta_i1x

        @property
        def delta_i2x(self):
            """Contribution of ID to second rad integral.

            Returns:
                Numpy 1d array: Delta i2x
            """
            return self._delta_i2x

        @property
        def delta_i3x(self):
            """Contribution of ID to third rad integral.

            Returns:
                Numpy 1d array: Delta i3x
            """
            return self._delta_i3x

        @property
        def delta_i4x(self):
            """Contribution of ID to fourth rad integral.

            Returns:
                Numpy 1d array: Delta i4x
            """
            return self._delta_i4x

        @property
        def delta_i5x(self):
            """Contribution of ID to fifth rad integral.

            Returns:
                Numpy 1d array: Delta i5x
            """
            return self._delta_i5x

        @property
        def delta_i1y(self):
            """Contribution of ID to first rad integral.

            Returns:
                Numpy 1d array: Delta i1y
            """
            return self._delta_i1y

        @property
        def delta_i2y(self):
            """Contribution of ID to second rad integral.

            Returns:
                Numpy 1d array: Delta i2y
            """
            return self._delta_i2y

        @property
        def delta_i3y(self):
            """Contribution of ID to third rad integral.

            Returns:
                Numpy 1d array: Delta i3y
            """
            return self._delta_i3y

        @property
        def delta_i4y(self):
            """Contribution of ID to fourth rad integral.

            Returns:
                Numpy 1d array: Delta i4y
            """
            return self._delta_i4y

        @property
        def delta_i5y(self):
            """Contribution of ID to fifth rad integral.

            Returns:
                Numpy 1d array: Delta i5y
            """
            return self._delta_i5y

        @property
        def u0(self):
            """Contribution of ID to energy loss.

            Returns:
                float: Energy loss per turn in ID.
            """
            return self._u0

    def __init__(self):
        """Class constructor."""
        self.ids = list()
        self._eq_params_nominal = None
        self.brho = Beam(3).brho
        self.gamma = Beam(3).gamma
        self._emitsx = None
        self._emitsy = None
        self._espreads = None
        self._tausx = None
        self._tausy = None
        self._tause = None
        self._u0 = None

    @property
    def eq_params_nominal(self):
        """Equilibrium parameters from nominal model.

        Returns:
            EqParamFromRadIntegrals object
        """
        return self._eq_params_nominal

    @property
    def emitsx(self):
        """Cumulative horizontal emittances.

        Returns:
            numpy array: Horizontal emittance [m rad]
        """
        return self._emitsx

    @property
    def emitsy(self):
        """Cumulative vertical emittances.

        Returns:
            numpy array: Vertical emittance [m rad]
        """
        return self._emitsx

    @property
    def espreads(self):
        """Cumulative energy spread.

        Returns:
            numpy array: Energy spread
        """
        return self._espreads

    @property
    def tausx(self):
        """Cumulative horizontal damping time.

        Returns:
            numpy array: Horizontal damping time [s]
        """
        return self._tausx

    @property
    def tausy(self):
        """Cumulative vertical damping time.

        Returns:
            numpy array: Vertical damping time [s]
        """
        return self._tausy

    @property
    def tause(self):
        """Cumulative longitudinal damping time.

        Returns:
            numpy array: longitudinal damping time [s]
        """
        return self._tause

    @property
    def u0(self):
        """Cumulative energy loss per turn.

        Returns:
            numpy array: Energy loss [keV]
        """
        return self._u0

    @staticmethod
    def _generate_field(a, peak, period, nr_periods, pts_period):
        x_1period = _np.linspace(-period / 2, period / 2, pts_period)
        y = peak * _np.sin(2 * _np.pi / period * x_1period)

        x_nperiods = _np.linspace(
            -nr_periods * period / 2,
            nr_periods * period / 2,
            nr_periods * pts_period,
        )
        mid = peak * _np.sin(2 * _np.pi / period * x_nperiods)

        if nr_periods % 2 == 1:
            term0 = (_np.tanh(a * 2 * _np.pi / period * x_1period) + 1) / 2
            term1 = (-_np.tanh(a * 2 * _np.pi / period * x_1period) + 1) / 2
        else:
            term0 = -(_np.tanh(a * 2 * _np.pi / period * x_1period) + 1) / 2
            term1 = -(-_np.tanh(a * 2 * _np.pi / period * x_1period) + 1) / 2

        out = _np.concatenate((term0 * y, mid, term1 * y))
        x_out = _np.linspace(
            -(2 + nr_periods) * period / 2,
            (2 + nr_periods) * period / 2,
            1 * (2 + nr_periods) * len(x_1period),
        )

        return x_out, out

    @staticmethod
    def _calc_field_integral(a, *args):
        peak = args[0]
        period = args[1]
        nr_periods = args[2]
        pts_period = args[3]
        x_out, out = EqParamAnalysis._generate_field(
            a, peak, period, nr_periods, pts_period
        )
        dx = _np.diff(x_out)[0]
        i1 = cumtrapz(dx=dx, y=-out)
        i2 = cumtrapz(dx=dx, y=i1)
        return _np.abs(i2[-1])

    def create_field_profile(self, insertion_device, pts_period=1001):
        """Create a sinusoidal field with first and second integrals zero.

        Args:
            insertion_device (InsertionParams object): Object from class
             InsertionParams.
            pts_period (int, optional): Number of points per period. Defaults
             to 1001.

        Returns:
            numpy array: first row is the spatial coordinate and second row
            the field amplitude.
        """
        by_peak = insertion_device.by_peak
        bx_peak = insertion_device.bx_peak
        nr_periods = insertion_device.nr_periods
        period = insertion_device.period
        field = _np.zeros(((nr_periods + 2) * pts_period, 3))

        if by_peak is not None:
            result = minimize(
                EqParamAnalysis._calc_field_integral,
                0.4,
                args=(by_peak, period, nr_periods, pts_period),
            )
            x, by = self._generate_field(
                result.x, by_peak, period, nr_periods, pts_period
            )
            field[:, 1] = by

        if bx_peak is not None:
            result = minimize(
                EqParamAnalysis._calc_field_integral,
                0.4,
                args=(bx_peak, period, nr_periods, pts_period),
            )
            x, bx = self._generate_field(
                result.x, bx_peak, period, nr_periods, pts_period
            )
            field[:, 2] = bx

        field[:, 0] = x
        insertion_device.field_profile = field
        return field

    def calc_dispersion(self, insertion_device):
        """Calculate dispersion generated by one ID.

        Args:
            insertion_device (InsertionParams object): Object from class
             InsertionParams.

        Returns:
            numpy array: Horizontal dispersion on the ID
            numpy array: Vertical dispersion on the ID
        """
        x = 1e-3 * insertion_device.field_profile[:, 0]
        by = insertion_device.field_profile[:, 1]
        bx = insertion_device.field_profile[:, 2]
        dx = _np.diff(x)[0]
        i1x = cumtrapz(dx=dx, y=by)
        i2x = cumtrapz(dx=dx, y=i1x)
        i1y = cumtrapz(dx=dx, y=-bx)
        i2y = cumtrapz(dx=dx, y=i1y)
        etax = -1 * i2x / self.brho
        etay = -1 * i2y / self.brho
        insertion_device._etax = etax
        insertion_device._etay = etay
        return etax, etay

    def calc_dispersion_dot(self, insertion_device):
        """Calculate the derivative of dispersion generated by one ID.

        Args:
            insertion_device (InsertionParams object): Object from class
             InsertionParams.

        Returns:
            numpy array: d(etax)/ds on the ID
            numpy array: d(etay)/ds on the ID
        """
        x = 1e-3 * insertion_device.field_profile[:, 0]
        by = insertion_device.field_profile[:, 1]
        bx = insertion_device.field_profile[:, 2]
        dx = _np.diff(x)[0]
        i1x = cumtrapz(dx=dx, y=by)
        i1y = cumtrapz(dx=dx, y=-bx)
        etadotx = -1 * i1x / self.brho
        etadoty = -1 * i1y / self.brho
        insertion_device._etadotx = etadotx
        insertion_device._etadoty = etadoty
        return etadotx, etadoty

    def get_curly_h(self, insertion_device):
        """Calculate curly H with ID.

        Args:
            insertion_device (InsertionParams object): Object from class
             InsertionParams.

        Returns:
            numpy 1d array: Curly Hx in the ID region.
            numpy 1d array: Curly Hy in the ID region.
        """
        x = 1e-3 * insertion_device.field_profile[:, 0]
        length = x[-1] - x[0]
        subsec = insertion_device.straight_section
        si = pymodels.si.create_accelerator()
        si = pyaccel.lattice.refine_lattice(
            si, max_length=0.01, fam_names=["BC", "B1", "B2", "QN"]
        )
        mia = pyaccel.lattice.find_indices(si, "fam_name", "mia")
        mib = pyaccel.lattice.find_indices(si, "fam_name", "mib")
        mip = pyaccel.lattice.find_indices(si, "fam_name", "mip")
        mid_subsections = _np.sort(_np.array(mia + mib + mip))
        idx = mid_subsections[int(subsec[2:4]) - 1]
        twiss, *_ = pyaccel.optics.calc_twiss(si, indices="open")
        spos = twiss.spos
        s0 = spos[idx]
        idx_end = _np.argmin(_np.abs(spos - s0 - length / 2)) + 1
        idx_begin = _np.argmin(_np.abs(spos - s0 + length / 2))

        betax = twiss.betax[idx_begin : idx_end + 1]
        alphax = twiss.alphax[idx_begin : idx_end + 1]

        betay = twiss.betay[idx_begin : idx_end + 1]
        alphay = twiss.alphay[idx_begin : idx_end + 1]

        pos = twiss.spos[idx_begin : idx_end + 1]
        pos = pos - pos[0]

        betax_interp = _np.interp(
            _np.linspace(pos[0], pos[-1], len(x)), pos, betax
        )
        alphax_interp = _np.interp(
            _np.linspace(pos[0], pos[-1], len(x)), pos, alphax
        )

        betay_interp = _np.interp(
            _np.linspace(pos[0], pos[-1], len(x)), pos, betay
        )
        alphay_interp = _np.interp(
            _np.linspace(pos[0], pos[-1], len(x)), pos, alphay
        )

        etax, etay = self.calc_dispersion(insertion_device)
        etaxdot, etaydot = self.calc_dispersion_dot(insertion_device)

        hx = pyaccel.optics.get_curlyh(
            betax_interp[1:-1], alphax_interp[1:-1], etax, etaxdot[:-1]
        )

        hy = pyaccel.optics.get_curlyh(
            betay_interp[1:-1], alphay_interp[1:-1], etay, etaydot[:-1]
        )

        return hx, hy

    def calc_delta_i1(self, insertion_device):
        """Calculate first radiation integral contribution from one ID.

        Args:
            insertion_device (InsertionParams object): Object from class
             InsertionParams.

        Returns:
            float: delta I1x
            float: delta I1y
        """
        by = insertion_device.field_profile[:, 1]
        bx = insertion_device.field_profile[:, 2]
        x = 1e-3 * insertion_device.field_profile[:, 0]
        dx = _np.diff(x)[0]
        etax, etay = self.calc_dispersion(insertion_device)
        delta_i1x = cumtrapz(dx=dx, y=etax * by[1:-1] / self.brho)[-1]
        delta_i1y = cumtrapz(dx=dx, y=etay * bx[1:-1] / self.brho)[-1]
        insertion_device._delta_i1x = delta_i1x
        insertion_device._delta_i1y = delta_i1y
        return delta_i1x, delta_i1y

    def calc_delta_i2(self, insertion_device):
        """Calculate second radiation integral contribution from one ID.

        Args:
            insertion_device (InsertionParams object): Object from class
             InsertionParams.

        Returns:
            float: delta I2x
            float: delta I2y
        """
        by = insertion_device.field_profile[:, 1]
        bx = insertion_device.field_profile[:, 2]
        x = 1e-3 * insertion_device.field_profile[:, 0]
        dx = _np.diff(x)[0]
        delta_i2x = cumtrapz(dx=dx, y=(by / self.brho) ** 2)[-1]
        delta_i2y = cumtrapz(dx=dx, y=(bx / self.brho) ** 2)[-1]
        insertion_device._delta_i2x = delta_i2x
        insertion_device._delta_i2y = delta_i2y
        return delta_i2x, delta_i2y

    def calc_delta_i3(self, insertion_device):
        """Calculate third radiation integral contribution from one ID.

        Args:
            insertion_device (InsertionParams object): Object from class
             InsertionParams.

        Returns:
            float: delta I3x
            float: delta I3y
        """
        by = insertion_device.field_profile[:, 1]
        bx = insertion_device.field_profile[:, 2]
        x = 1e-3 * insertion_device.field_profile[:, 0]
        dx = _np.diff(x)[0]
        delta_i3x = cumtrapz(dx=dx, y=_np.abs((by / self.brho) ** 3))[-1]
        delta_i3y = cumtrapz(dx=dx, y=_np.abs((bx / self.brho) ** 3))[-1]
        insertion_device._delta_i3x = delta_i3x
        insertion_device._delta_i3y = delta_i3y
        return delta_i3x, delta_i3y

    def calc_delta_i4(self, insertion_device):
        """Calculate fourth radiation integral contribution from one ID.

        Args:
            insertion_device (InsertionParams object): Object from class
             InsertionParams.

        Returns:
            float: delta I4x
            float: delta I4y
        """
        by = insertion_device.field_profile[:, 1]
        bx = insertion_device.field_profile[:, 2]
        x = 1e-3 * insertion_device.field_profile[:, 0]
        dx = _np.diff(x)[0]
        dby = _np.diff(by)
        dbx = _np.diff(bx)
        px = cumtrapz(dx=dx, y=by) / self.brho
        py = cumtrapz(dx=dx, y=-bx) / self.brho
        kx = -(dby / dx) * px / self.brho
        ky = -(dbx / dx) * py / self.brho
        etax, etay = self.calc_dispersion(insertion_device)
        delta_i4x = cumtrapz(
            dx=dx,
            y=(
                etax * (by[1:-1] / self.brho) ** 3
                - 2 * ky[:-1] * etax * (by[1:-1] / self.brho)
            ),
        )[-1]
        delta_i4y = cumtrapz(
            dx=dx,
            y=(
                etay * (bx[1:-1] / self.brho) ** 3
                - 2 * kx[:-1] * etay * (bx[1:-1] / self.brho)
            ),
        )[-1]
        insertion_device._delta_i4x = delta_i4x
        insertion_device._delta_i4y = delta_i4y
        return delta_i4x, delta_i4y

    def calc_delta_i5(self, insertion_device):
        """Calculate fifth radiation integral contribution from one ID.

        Args:
            insertion_device (InsertionParams object): Object from class
             InsertionParams.

        Returns:
            float: delta I5x
            float: delta I5y
        """
        by = insertion_device.field_profile[:, 1]
        bx = insertion_device.field_profile[:, 2]
        x = 1e-3 * insertion_device.field_profile[:, 0]
        dx = _np.diff(x)[0]
        hx, hy = self.get_curly_h(insertion_device)
        delta_i5x = cumtrapz(
            dx=dx, y=hx * _np.abs((by / self.brho) ** 3)[1:-1]
        )[-1]
        delta_i5y = cumtrapz(
            dx=dx, y=hy * _np.abs((bx / self.brho) ** 3)[1:-1]
        )[-1]
        insertion_device._delta_i5x = delta_i5x
        insertion_device._delta_i5y = delta_i5y
        return delta_i5x, delta_i5y

    def calc_u0(self, insertion_device):
        """Calculate energy loss in ID.

        Args:
            insertion_device (InsertionParams object): Object from class
             InsertionParams.

        Returns:
            float: Energy loss in [keV]
        """
        x = 1e-3 * insertion_device.field_profile[:, 0]
        dx = _np.diff(x)[0]
        by = insertion_device.field_profile[:, 1]
        bx = insertion_device.field_profile[:, 2]
        b = _np.sqrt(bx**2 + by**2)
        gamma = Beam(3).gamma
        brho = Beam(3).brho
        e = mathphys.constants.elementary_charge
        e0 = mathphys.constants.vacuum_permitticity
        u0 = (
            1e-3
            * (
                (e * gamma**4)
                * cumtrapz(dx=dx, y=(b / brho) ** 2)
                / (6 * _np.pi * e0)
            )[-1]
        )
        insertion_device._u0 = u0
        return u0

    def calc_power(self, insertion_device, current=0.1):
        """Calculate radiated power in ID.

        Args:
            insertion_device (InsertionParams object): Object from class
             InsertionParams.
            current (float, optional): Current in [A]. Defaults to 0.1.

        Returns:
            float: Power in [kW]
        """
        p = current * self.calc_u0(insertion_device)
        return p

    def get_nominal_model_eqparams(self, model=None):
        """Get nominal model equilibrium parameters.

        Args:
            model (Pymodels object, optional): Sirius model. Defaults to None.

        Returns:
            EqParamsFromRadIntegrals object: Object containing Eqparams
        """
        if model is None:
            model = pymodels.si.create_accelerator()
        model = pyaccel.lattice.refine_lattice(
            model, max_length=0.01, fam_names=["BC", "B1", "B2", "QN"]
        )
        eqparam_nom = pyaccel.optics.EqParamsFromRadIntegrals(model)
        self._eq_params_nominal = eqparam_nom
        return eqparam_nom

    def calc_id_effect_on_emittance(
        self, insertion_device=None, model=None, *args
    ):
        """Calculate ID's effect on emittances.

        Args:
            insertion_device (InsertionParams object): Object from class
             InsertionParams. Defaults to None
            model (Pymodels object, optional): Sirius model. Defaults to None.
            *args (floats): Delta in radiation integrals.
            The order of args must be the following:
            args[0] = delta_i2x, args[1] = delta_i2y
            args[2] = delta_i4x, args[3] = delta_i4y
            args[4] = delta_i5x, args[5] = delta_i5y


        Returns:
            Float: Horizontal emittance [m rad]
            Float: Vertical emittance [m rad]
        """
        if (
            insertion_device is not None
            and insertion_device.field_profile is None
        ):
            self.create_field_profile(insertion_device)
        if self.eq_params_nominal is None:
            self.get_nominal_model_eqparams(model)

        i5x = self.eq_params_nominal.I5x
        i5y = self.eq_params_nominal.I5y

        i4x = self.eq_params_nominal.I4x
        i4y = self.eq_params_nominal.I4y

        i2x = self.eq_params_nominal.I2
        i2y = self.eq_params_nominal.I2

        if len(args) == 0:
            di2x, di2y = self.calc_delta_i2(insertion_device)
            di4x, di4y = self.calc_delta_i4(insertion_device)
            di5x, di5y = self.calc_delta_i5(insertion_device)
        else:
            di2x, di2y = args[0], args[1]
            di4x, di4y = args[2], args[3]
            di5x, di5y = args[4], args[5]

        i5x += di5x
        i2x += di2x
        i4x += di4x

        i5y += di5y
        i2y += di2y
        i4y += di4y

        emitx = (
            mathphys.constants.Cq * Beam(3).gamma ** 2 * (i5x / (i2x - i4x))
        )
        emity = (
            mathphys.constants.Cq * Beam(3).gamma ** 2 * (i5y / (i2y - i4y))
        )

        return emitx, emity

    def calc_id_effect_on_espread(
        self, insertion_device=None, model=None, *args
    ):
        """Calculate ID's effect on energy spread.

        Args:
            insertion_device (InsertionParams object): Object from class
             InsertionParams. Defaults to None.
            model (Pymodels object, optional): Sirius model. Defaults to None.
            *args (floats): Delta in radiation integrals.
            The order of args must be the following:
            args[0] = delta_i2
            args[1] = delta_i3
            args[2] = delta_i4

        Returns:
            float: energy spread
        """
        if (
            insertion_device is not None
            and insertion_device.field_profile is None
        ):
            self.create_field_profile(insertion_device)
        if self.eq_params_nominal is None:
            self.get_nominal_model_eqparams(model)

        i2 = self.eq_params_nominal.I2
        i3 = self.eq_params_nominal.I3
        i4 = self.eq_params_nominal.I4x

        if len(args) == 0:
            di2, _ = self.calc_delta_i2(insertion_device)
            di3, _ = self.calc_delta_i3(insertion_device)
            di4, _ = self.calc_delta_i4(insertion_device)
        else:
            di2 = args[0]
            di3 = args[1]
            di4 = args[2]

        i2 += di2
        i3 += di3
        i4 += di4

        espread = _np.sqrt(
            mathphys.constants.Cq * Beam(3).gamma ** 2 * (i3 / (2 * i2 + i4))
        )

        return espread

    def calc_id_effect_on_damping_times(
        self, insertion_device=None, model=None, *args
    ):
        """Calculate ID's effect on damping times.

        Args:
            insertion_device (InsertionParams object): Object from class
             InsertionParams. Defaults to None.
            model (Pymodels object, optional): Sirius model. Defaults to None.
            *args (floats): Delta in radiation integrals.
            The order of args must be the following:
            args[0] = delta_i2
            args[1] = delta_i4
            args[2] = delta_u0

        Returns:
            float: taux [s]
            float: tauy [s]
            float: taue [s]
        """
        if (
            insertion_device is not None
            and insertion_device.field_profile is None
        ):
            self.create_field_profile(insertion_device)
        if self.eq_params_nominal is None:
            self.get_nominal_model_eqparams(model)
        if model is None:
            si = pymodels.si.create_accelerator()

        i2 = self.eq_params_nominal.I2
        i4 = self.eq_params_nominal.I4x
        u0 = self.eq_params_nominal.U0 * mathphys.constants.elementary_charge

        if len(args) == 0:
            di2, _ = self.calc_delta_i2(insertion_device)
            di4, _ = self.calc_delta_i4(insertion_device)
            du0 = (
                1e-3
                * self.calc_u0(insertion_device)
                * mathphys.constants.elementary_charge
            )
        else:
            di2 = args[0]
            di4 = args[1]
            du0 = args[2]

        i2 += di2
        i4 += di4
        u0 += du0

        jx = 1 - i4 / i2
        jy = 1
        jz = 2 + i4 / i2

        e = Beam(3).energy * 1e9 * mathphys.constants.elementary_charge
        t0 = si.length / mathphys.constants.light_speed

        taux = 2 * e * t0 / (u0 * jx)
        tauy = 2 * e * t0 / (u0 * jy)
        taue = 2 * e * t0 / (u0 * jz)

        return taux, tauy, taue

    def add_id_to_lattice(self, insertion_device):
        """Add ID to lattice.

        Args:
            insertion_device (InsertionParams object): Object from class
             InsertionParams.
        """
        self.ids.append(insertion_device)

    def calc_total_effect_on_emittance(self, model=None):
        """Calculate cumulative effect on emittances.

        Args:
            model (Pymodels object, optional): Sirius model. Defaults to None.

        Returns:
            numpy 1d array: Horizontal emittance [m rad]
            numpy 1d array: Vertical emittance [m rad]
        """
        emitx = _np.zeros(len(self.ids) + 1)
        emity = _np.zeros(len(self.ids) + 1)
        if self.eq_params_nominal is None:
            self.get_nominal_model_eqparams(model)
        emitx[0] = self.eq_params_nominal.emitx
        emity[0] = self.eq_params_nominal.emity
        di2x, di2y = 0, 0
        di4x, di4y = 0, 0
        di5x, di5y = 0, 0
        for i, insertion_device in enumerate(self.ids):
            if insertion_device.field_profile is None:
                self.create_field_profile(insertion_device)
            di2x_, di2y_ = self.calc_delta_i2(insertion_device)
            di4x_, di4y_ = self.calc_delta_i4(insertion_device)
            di5x_, di5y_ = self.calc_delta_i5(insertion_device)
            di2x += di2x_
            di2y += di2y_
            di4x += di4x_
            di4y += di4y_
            di5x += di5x_
            di5y += di5y_
            emitx[i + 1], emity[i + 1] = self.calc_id_effect_on_emittance(
                None, model, di2x, di2y, di4x, di4y, di5x, di5y
            )
        self._emitsx = emitx
        self._emitsy = emity
        return emitx, emity

    def calc_total_effect_on_espread(self, model=None):
        """Calculate cumulative effect on energy spread.

        Args:
            model (Pymodels object, optional): Sirius model. Defaults to None.

        Returns:
            numpy 1d array: Energy spread
        """
        espread = _np.zeros(len(self.ids) + 1)
        if self.eq_params_nominal is None:
            self.get_nominal_model_eqparams(model)
        espread[0] = self.eq_params_nominal.espread0
        di2, di3, di4 = 0, 0, 0
        for i, insertion_device in enumerate(self.ids):
            if insertion_device.field_profile is None:
                self.create_field_profile(insertion_device)
            di2_, _ = self.calc_delta_i2(insertion_device)
            di3_, _ = self.calc_delta_i3(insertion_device)
            di4_, _ = self.calc_delta_i4(insertion_device)
            di2 += di2_
            di3 += di3_
            di4 += di4_
            espread[i + 1] = self.calc_id_effect_on_espread(
                None, model, di2, di3, di4
            )
        self._espreads = espread
        return espread

    def calc_total_effect_on_damping_times(self, model=None):
        """Calculate cumulative effect on damping_times.

        Args:
            model (Pymodels object, optional): Sirius model. Defaults to None.

        Returns:
            numpy 1d array: Horizontal damping time [s]
            numpy 1d array: Vertical damping time [s]
            numpy 1d array: Longitudinal damping time [s]
        """
        taux = _np.zeros(len(self.ids) + 1)
        tauy = _np.zeros(len(self.ids) + 1)
        taue = _np.zeros(len(self.ids) + 1)
        if self.eq_params_nominal is None:
            self.get_nominal_model_eqparams(model)
        taux[0] = self.eq_params_nominal.taux
        tauy[0] = self.eq_params_nominal.tauy
        taue[0] = self.eq_params_nominal.taue
        di2, di4, du0 = 0, 0, 0
        for i, insertion_device in enumerate(self.ids):
            if insertion_device.field_profile is None:
                self.create_field_profile(insertion_device)
            di2_, _ = self.calc_delta_i2(insertion_device)
            di4_, _ = self.calc_delta_i4(insertion_device)
            du0_ = (
                1e-3
                * self.calc_u0(insertion_device)
                * mathphys.constants.elementary_charge
            )
            di2 += di2_
            di4 += di4_
            du0 += du0_
            (
                taux[i + 1],
                tauy[i + 1],
                taue[i + 1],
            ) = self.calc_id_effect_on_damping_times(
                None, model, di2, di4, du0
            )
        self._tausx = taux
        self._tausy = tauy
        self._tause = taue
        return taux, tauy, taue

    def calc_total_energy_loss(self, model=None):
        """Calculate cumulative effect on damping_times.

        Args:
            model (Pymodels object, optional): Sirius model. Defaults to None.

        Returns:
            numpy 1d array: Energy loss [keV]
        """
        u0 = _np.zeros(len(self.ids) + 1)
        if self.eq_params_nominal is None:
            self.get_nominal_model_eqparams(model)
        u0[0] = 1e-3 * self.eq_params_nominal.U0
        for i, insertion_device in enumerate(self.ids):
            if insertion_device.field_profile is None:
                self.create_field_profile(insertion_device)
            du0 = self.calc_u0(insertion_device)
            u0[i + 1] = u0[i] + du0
        self._u0 = u0
        return u0

    def plot_ids_effects_emit_espread(self):
        """Plot ID's effects on horizontal emittance and energy spread."""
        emitx, _ = self.calc_total_effect_on_emittance()
        energy_spread = self.calc_total_effect_on_espread()

        fig, ax1 = _plt.subplots(1, sharey=True)
        fig.set_figwidth(9)
        fig.set_figheight(5)

        ax1.tick_params(axis="x", rotation=65, labelsize=12)
        ax1.tick_params(axis="y", labelsize=13.5, labelcolor="C0")

        ids_list = list()
        ids_list.append("BARE")

        for insertion_device in self.ids:
            id_name = insertion_device.idname
            ss_nr = insertion_device.straight_section[2:4]
            id_name += "-"
            id_name += ss_nr
            ids_list.append(id_name)

        ax1.plot(ids_list, 1e9 * emitx, color="C0", marker="o")

        # Set the title and labels for the plot
        ax1.set_title("Sirius IDs effects on Eq. Params.", fontsize=14.5)
        ax1.set_ylabel("Emittance [nm.rad]", fontsize=13.5, color="C0")

        ax1_twin = ax1.twinx()
        ax1_twin.plot(ids_list, 100 * energy_spread, color="red", marker="d")

        # ax1_twin.yaxis.set_ticklabels([])
        ax1_twin.tick_params(axis="y", labelsize=13.5, labelcolor="#B22222")
        ax1_twin.set_ylabel(
            "Energy spread x100 [%]", fontsize=13.5, color="#B22222"
        )
        fig.tight_layout(pad=1.0)
        ax1.grid()
