"""IDs analysis."""

import matplotlib.pyplot as _plt
import numpy as _np
import pyaccel
import pymodels
import utils
from apsuite.dynap import DynapXY
from fieldmaptrack import Beam
from mathphys.functions import load_pickle as _load_pickle, \
    save_pickle as _save_pickle
from scipy.optimize import curve_fit as _curve_fit

from idanalysis import IDKickMap as _IDKickMap, optics as optics, \
    orbcorr as orbcorr


class FieldMapAnalysis():
    """Class for fieldmaps analysis."""
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
            idx0 - int(peak_search/2 * len(rz)) : idx0
            + int(peak_search/2 * len(rz))
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
            _1D numpy array_: Roll-off on transverse positions
        """
        r_transverse, b_transverse = self.get_fmap_transverse_dependence(
            field_component, peak_search, plane
        )

        rt_interp = _np.linspace(r_transverse.min(), r_transverse.max(), 101)
        bt_interp = _np.interp(rt_interp, r_transverse, b_transverse)

        r0_idx = _np.argmin(_np.abs(rt_interp))
        roff = 100 * (bt_interp[r0_idx] - bt_interp) / bt_interp[r0_idx]
        return rt_interp, bt_interp, roff


class RadiaModelAnalysis():
    """Class for RADIA models field analysis."""
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
        return idxmax, rzmax, b[idxmax]

    def calc_radia_transverse_dependence(self, field_component,
                                         r_transverse, plane="x"):
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
            _1D numpy array_: Roll-off on transverse positions
        """
        _, b_transverse = self.calc_radia_transverse_dependence(
            field_component, r_transverse, plane)

        r0_idx = _np.argmin(_np.abs(r_transverse))
        roff = b_transverse / b_transverse[r0_idx] - 1
        return r_transverse, b_transverse, roff


class TrajectoryAnalysis():
    """Class for Electron Trajectory analysis."""
    def __init__(self, fieldsource):
        """Class constructor.

        Args:
            fieldsource (_Fielmap or RADIA object_): _description_
        """
        self.fieldsource = fieldsource
        self.beam_energy = Beam(3).energy  # [GeV]
        self._idkickmap = None  # Radia model or fieldmap
        self._rk_s_step = None  # [mm]
        self._traj_init_rz = None  # [mm]
        self._traj_max_rz = None  # [mm]
        self._kmap_idlen = None  # [m]

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
    def kmap_idlen(self):
        """Kickmap length for SIRIUS model.

        Returns:
            float: Kickmap length for SIRIUS model
        """
        return self._kmap_idlen

    @property
    def idkickmap(self):
        """ID Kickmap object.

        Returns:
            IDKickmap object: IDKickmap object
        """
        return self._idkickmap


class Tools:
    @staticmethod
    def create_model_ids(
        fname,
        rescale_kicks,
        rescale_length,
        fitted_model,
        linear,
        create_ids,
        fit_path,
    ):
        if linear:
            rescale_kicks = 1
            rescale_length = 1
        if create_ids is None:
            raise ValueError
        ids = create_ids(
            fname, rescale_kicks=rescale_kicks, rescale_length=rescale_length
        )
        model = pymodels.si.create_accelerator(ids=ids)
        if fitted_model:
            famdata = pymodels.si.families.get_family_data(model)
            idcs_qn = _np.array(famdata["QN"]["index"]).ravel()
            idcs_qs = _np.array(famdata["QS"]["index"]).ravel()
            data = _load_pickle(fit_path)
            kl = data["KL"]
            ksl = data["KsL"]
            pyaccel.lattice.set_attribute(model, "KL", idcs_qn, kl)
            pyaccel.lattice.set_attribute(model, "KsL", idcs_qs, ksl)
        model.cavity_on = False
        model.radiation_on = 0
        twiss, *_ = pyaccel.optics.calc_twiss(model, indices="closed")
        print("\nModel with ID:")
        print("length : {:.4f} m".format(model.length))
        print("tunex  : {:.6f}".format(twiss.mux[-1] / 2 / _np.pi))
        print("tuney  : {:.6f}".format(twiss.muy[-1] / 2 / _np.pi))
        return model, ids



# class FieldAnalysisFromFieldmap(Tools):
    # def __init__(self):
    #     """."""
    #     # fmap attributes
    #     self._fmaps = None
    #     self._fmaps_names = None
    #     self.data = dict()

    #     # Trajectory attributes
    #     self._idkickmap = None
    #     self.beam_energy = utils.BEAM_ENERGY  # [Gev]
    #     self.rk_s_step = utils.DEF_RK_S_STEP  # [mm]
    #     self.traj_init_rz = None  # [mm]
    #     self.traj_max_rz = None  # [mm]
    #     self.kmap_idlen = None  # [m]
    #     self.gridx = None
    #     self.gridy = None

    #     self.FOLDER_DATA = "./results/measurements/data/"



    # @idkickmap.setter
    # def idkickmap(self, fmap_fname):
    #     """Set idkickmap config for trajectory."""
    #     self._idkickmap = _IDKickMap()
    #     self._idkickmap.fmap_fname = fmap_fname
    #     self._idkickmap.beam_energy = self.beam_energy
    #     self._idkickmap.rk_s_step = self.rk_s_step
    #     self._idkickmap.kmap_idlen = self.kmap_idlen


    # def _generate_kickmap(self, fmap_fname, **kwargs):
    #     self.idkickmap = fmap_fname
    #     self.idkickmap.fmap_calc_kickmap(posx=self.gridx, posy=self.gridy)
    #     fname = self.get_kmap_filename(
    #         folder_data=utils.FOLDER_DATA, meas_flag=True, **kwargs
    #     )
    #     self.idkickmap.save_kickmap_file(kickmap_filename=fname)

    # def run_generate_kickmap(self):
    #     kwargs = dict()
    #     self._get_fmaps()
    #     param_names = self.fmaps_names["params"]
    #     for configs, value in self.fmaps_names["configs"].items():
    #         stg = "calc kickmap for "
    #         for i, config in enumerate(configs):
    #             stg += param_names[i]
    #             stg += f" {config}, "
    #             kwargs[param_names[i]] = config
    #         print(stg)
    #         self._generate_kickmap(fmap_fname=value, **kwargs)

    # def _get_field_on_trajectory(self):
    #     print("Calculating field on trajectory...")
    #     for config, fmap_name in self.fmaps_names["configs"].items():
    #         # create IDKickMap and calc trajectory
    #         self.idkickmap = fmap_name
    #         self.idkickmap.fmap_calc_trajectory(
    #             traj_init_rx=0, traj_init_ry=0, traj_init_px=0, traj_init_py=0
    #         )
    #         traj = self.idkickmap.traj

    #         self.data[config].update({"ontraj_bx": traj.bx})
    #         self.data[config].update({"ontraj_by": traj.by})
    #         self.data[config].update({"ontraj_bz": traj.bz})
    #         self.data[config].update({"ontraj_rx": traj.rx})
    #         self.data[config].update({"ontraj_ry": traj.ry})
    #         self.data[config].update({"ontraj_rz": traj.rz})
    #         self.data[config].update({"ontraj_px": traj.px})
    #         self.data[config].update({"ontraj_py": traj.py})
    #         self.data[config].update({"ontraj_pz": traj.pz})
    #         self.data[config].update({"ontraj_s": traj.s})


class FieldAnalysisFromRadia(Tools):
    def __init__(self):
        # Model attributes
        self.rt_field_max = None  # [mm]
        self.rt_field_nrpts = None
        self.rz_field_max = None  # [mm]
        self.rz_field_nrpts = None
        self.roll_off_rt = utils.ROLL_OFF_POS  # [mm]
        self.data = dict()
        self._models = dict()

        # Trajectory attributes
        self._idkickmap = None
        self.beam_energy = utils.BEAM_ENERGY  # [Gev]
        self.rk_s_step = utils.DEF_RK_S_STEP  # [mm]
        self.traj_init_rz = None  # [mm]
        self.traj_max_rz = None  # [mm]
        self.kmap_idlen = None  # [m]
        self.gridx = None
        self.gridy = None

        # fmap attributes
        self.roff_deltas = _np.linspace(1, 1, 1)

        self.FOLDER_DATA = "./results/model/data/"

    @property
    def idkickmap(self):
        """Return an object of IDKickMap class.

        Returns:
            IDKickMap object:
        """
        return self._idkickmap

    @property
    def models(self):
        """Return a dictionary with all ID models.

        Returns:
            Dictionary: A dictionary in which keys are
                the variable parameters and the values are the models.
                example: {(2, 40): model1, (2, 45): model2}, the keys could be
                gap and width for example.
        """
        return self._models

    @models.setter
    def models(self, ids):
        """Set models attribute.

        Args:
            models (dictionary): A dictionary in which keys are
                the variable parameters and the values are the models.
                example: {(2, 40): model1, (2, 45): model2}, the keys could be
                gap and width for example.
        """
        self._models = ids

    @idkickmap.setter
    def idkickmap(self, id):
        """Set idkickmap config for trajectory."""
        self._idkickmap = _IDKickMap()
        self._idkickmap.radia_model = id
        self._idkickmap.beam_energy = self.beam_energy
        self._idkickmap.rk_s_step = self.rk_s_step
        self._idkickmap.traj_init_rz = self.traj_init_rz
        self._idkickmap.traj_rk_min_rz = self.traj_max_rz
        self._idkickmap.kmap_idlen = self.kmap_idlen

    def _create_models(self, **kwargs):
        models_ = {"params": list(kwargs.keys()), "configs": dict()}
        param_names = models_["params"]
        config_list = list()
        for values in kwargs.values():
            config_list.append(_np.array(values))
        config_grid = _np.meshgrid(*config_list)
        flat_grid = list()
        for grid in config_grid:
            flat_list = list(_np.concatenate(grid).flat)
            flat_grid.append(flat_list)
        matrix = _np.array(flat_grid)
        configs = list()
        for j in _np.arange(_np.shape(matrix)[1]):
            configs.append(tuple(matrix[:, j]))
        for config in configs:
            stg = "Generating model for: \n"
            for j, param in enumerate(config):
                stg += param_names[j]
                stg += f" = {param} \n"
                kwargs[param_names[j]] = param
            print(stg)
            id = utils.generate_radia_model(solve=utils.SOLVE_FLAG, **kwargs)
            models_["configs"][config] = id
        self.models = models_

    def _get_field_roll_off(
        self, rt, peak_idx=0, field_component="by", plane="x"
    ):
        """Calculate the roll-off of a field component.

        Args:
            rt (numpy 1D array): array with positions where the field will be
                calculated
            peak_idx (int): Peak index where the roll-off will be calculated.
                Defaults to 0.
        """
        roff_pos = utils.ROLL_OFF_POS
        print("Calculating field roll off...")
        for config, model in self.models["configs"].items():
            b, roff, _ = self.calc_radia_roll_off(
                field_component, model, rt, roff_pos, plane=plane, peak_idx=0
            )
            self.data[config] = dict()
            self.data[config].update({"rolloff_rt": rt})
            self.data[config].update({"rolloff_{}".format(field_component): b})
            self.data[config].update({"rolloff_value": roff})

    def _get_field_on_axis(self, rz):
        """Get the field along z axis.

        Args:
            rz (numpy 1D array): array with longitudinal positions where the
                field will be calculated
        """
        print("Calculating field on axis...")
        for config, model in self.models["configs"].items():
            field = model.get_field(0, 0, rz)
            bx = field[:, 0]
            by = field[:, 1]
            bz = field[:, 2]
            self.data[config].update({"onaxis_bx": bx})
            self.data[config].update({"onaxis_by": by})
            self.data[config].update({"onaxis_bz": bz})
            self.data[config].update({"onaxis_rz": rz})







    def run_plot_data(self, sulfix=None, **kwargs):
        data_plot = self.get_data_plot(**kwargs)
        self._plot_field_on_axis(data=data_plot, sulfix=sulfix)
        self._plot_rk_traj(data=data_plot, sulfix=sulfix)
        self._plot_field_roll_off(data=data_plot, sulfix=sulfix)

    def _plot_field_on_axis(self, data, sulfix=None):
        colors = ["C0", "b", "g", "y", "C1", "r", "k"]
        fig, axs = _plt.subplots(3, 1, sharex=True, figsize=(12, 8))
        output_dir = self.FOLDER_DATA + "general"
        filename = output_dir + "/field-profile"
        if sulfix is not None:
            filename += sulfix
        self.mkdir_function(output_dir)
        var_parameter_name = list(data.keys())[0]
        for i, value in enumerate(data[var_parameter_name].keys()):
            label = var_parameter_name + " {}".format(value)
            bx = data[var_parameter_name][value]["onaxis_bx"]
            by = data[var_parameter_name][value]["onaxis_by"]
            bz = data[var_parameter_name][value]["onaxis_bz"]
            rz = data[var_parameter_name][value]["onaxis_rz"]
            axs[0].plot(rz, bx, label=label, color=colors[i])
            axs[0].set_ylabel("bx [T]")
            axs[1].plot(rz, by, label=label, color=colors[i])
            axs[1].set_ylabel("by [T]")
            axs[2].plot(rz, bz, label=label, color=colors[i])
            axs[2].set_ylabel("bz [T]")
        for j in _np.arange(3):
            axs[j].set_xlabel("z [mm]")
            axs[j].legend()
            axs[j].grid()
        _plt.savefig(filename, dpi=300)
        _plt.show()

    def _plot_rk_traj(self, data, sulfix=None):
        colors = ["C0", "b", "g", "y", "C1", "r", "k"]
        output_dir = self.FOLDER_DATA + "general"
        var_parameter_name = list(data.keys())[0]
        for i, value in enumerate(data[var_parameter_name].keys()):
            s = data[var_parameter_name][value]["ontraj_s"]
            rx = data[var_parameter_name][value]["ontraj_rx"]
            ry = data[var_parameter_name][value]["ontraj_ry"]
            px = 1e6 * data[var_parameter_name][value]["ontraj_px"]
            py = 1e6 * data[var_parameter_name][value]["ontraj_py"]
            label = var_parameter_name + " {}".format(value)

            _plt.figure(1)
            _plt.plot(s, 1e3 * rx, color=colors[i], label=label)
            _plt.xlabel("s [mm]")
            _plt.ylabel("x [um]")

            _plt.figure(2)
            _plt.plot(s, 1e3 * ry, color=colors[i], label=label)
            _plt.xlabel("s [mm]")
            _plt.ylabel("y [um]")
            _plt.legend()

            _plt.figure(3)
            _plt.plot(s, px, color=colors[i], label=label)
            _plt.xlabel("s [mm]")
            _plt.ylabel("px [urad]")

            _plt.figure(4)
            _plt.plot(s, py, color=colors[i], label=label)
            _plt.xlabel("s [mm]")
            _plt.ylabel("py [urad]")
        sulfix_ = ["/traj-rx", "/traj-ry", "/traj-px", "/traj-py"]
        for i in [1, 2, 3, 4]:
            _plt.figure(i)
            _plt.legend()
            _plt.grid()
            filename = output_dir + sulfix_[i - 1]
            if sulfix is not None:
                filename += sulfix
            _plt.savefig(filename, dpi=300)
        _plt.show()

    def _read_data_roll_off(self, data, value, field_component="by"):
        if "rolloff_{}".format(field_component) in data[value]:
            b = data[value]["rolloff_{}".format(field_component)]
            if "rolloff_rt" in data[value]:
                rt = data[value]["rolloff_rt"]
            elif "rolloff_rx" in data[value]:
                rt = data[value]["rolloff_rx"]
            elif "rolloff_ry" in data[value]:
                rt = data[value]["rolloff_ry"]
            rtp_idx = _np.argmin(_np.abs(rt - utils.ROLL_OFF_POS))
            rt0_idx = _np.argmin(_np.abs(rt))
            roff = _np.abs(b[rtp_idx] / b[rt0_idx] - 1)
            b0 = b[rt0_idx]
            roll_off = 100 * (b / b0 - 1)
            return rt, b, roll_off, roff

    def _plot_field_roll_off(self, data, sulfix=None, field_component="by"):
        _plt.figure(1)
        output_dir = self.FOLDER_DATA + "general"
        filename = output_dir + "/field-rolloff"
        if sulfix is not None:
            filename += sulfix
        colors = ["C0", "b", "g", "y", "C1", "r", "k"]
        var_parameter_name = list(data.keys())[0]
        for i, value in enumerate(data[var_parameter_name].keys()):
            roff_data = self._read_data_roll_off(
                data[var_parameter_name], value
            )
            if roff_data is None:
                continue
            else:
                rt, b, roll_off, roff = roff_data
            label = var_parameter_name + " {} mm, roll-off = {:.2f} %".format(
                value, 100 * roff
            )
            _plt.plot(rt, roll_off, ".-", label=label, color=colors[i])
        if field_component == "by":
            _plt.xlabel("x [mm]")
        else:
            _plt.xlabel("y [mm]")
        _plt.ylabel("Field roll off [%]")
        _plt.xlim(-utils.ROLL_OFF_POS, utils.ROLL_OFF_POS)
        _plt.ylim(-101 * roff, 20 * roff)
        if field_component == "by":
            _plt.title(
                "Field roll-off at x = {} mm".format(utils.ROLL_OFF_POS)
            )
        elif field_component == "bx":
            _plt.title(
                "Field roll-off at y = {} mm".format(utils.ROLL_OFF_POS)
            )
        _plt.legend()
        _plt.grid()
        _plt.savefig(filename, dpi=300)
        _plt.show()

    def _generate_kickmap(self, model, **kwargs):
        self.idkickmap = model
        self.idkickmap.fmap_calc_kickmap(posx=self.gridx, posy=self.gridy)
        fname = self.get_kmap_filename(
            folder_data=utils.FOLDER_DATA, meas_flag=True, **kwargs
        )
        self.idkickmap.save_kickmap_file(kickmap_filename=fname)

    def run_generate_kickmap(self, **kwargs):
        if self.models:
            print("models ready")
        else:
            self._create_models(**kwargs)
            print("models ready")

        param_names = self.models["params"]
        for configs, value in self.models["configs"].items():
            stg = "calc kickmap for "
            for i, config in enumerate(configs):
                stg += param_names[i]
                stg += f" = {config} \n, "
                kwargs[param_names[i]] = config
            print(stg)
            self._generate_kickmap(model=value, **kwargs)

    def _get_planar_id_features(
        self, id_period, plot_flag=False, field_component="by", **kwargs
    ):
        def cos(x, b0, kx):
            b = b0 * _np.cos(kx * x)
            return b

        data = self.get_data_plot(**kwargs)
        var_parameter_name = list(data.keys())[0]

        rt, b, *_ = self._read_data_roll_off(
            data[var_parameter_name], kwargs[var_parameter_name]
        )

        idxi = _np.argmin(_np.abs(rt + 1))
        idxf = _np.argmin(_np.abs(rt - 1))
        rt = rt[idxi:idxf]
        b = b[idxi:idxf]
        opt = _curve_fit(cos, rt, b)[0]
        k = opt[1]
        b0 = opt[0]
        print("field amplitude = {:.3f} T".format(b0))
        if plot_flag:
            _plt.plot(rt, b, label="model")
            b_fitted = cos(rt, opt[0], opt[1])
            _plt.plot(rt, b_fitted, label="fitting")
            if field_component == "by":
                _plt.xlabel("x [mm]")
            elif field_component == "bx":
                _plt.xlabel("y [mm]")
            _plt.ylabel("B [T]")
            _plt.title("Field Roll-off fitting")
            _plt.grid()
            _plt.legend()
            _plt.show()

        kz = 2 * _np.pi / (id_period * 1e-3)
        kx = k * 1e3
        ky = _np.sqrt(kx**2 + kz**2)
        if field_component == "by":
            print("kx = {:.2f}".format(kx))
            print("ky = {:.2f}".format(ky))
        elif field_component == "bx":
            print("kx = {:.2f}".format(ky))
            print("ky = {:.2f}".format(kx))
        print("kz = {:.2f}".format(kz))

        return b0, kx, ky, kz

    def _get_id_quad_coef(
        self, id_period, plot_flag=False, field_component="by", **kwargs
    ):
        id_len = utils.SIMODEL_ID_LEN
        b0, kx, ky, kz = self._get_planar_id_features(
            plot_flag=plot_flag, id_period=id_period, **kwargs
        )
        beam = Beam(energy=3)
        brho = beam.brho
        phase = phase * 1e-3
        factor = 1 + _np.cos(kz * phase)
        factor -= (kx**2) / (kz**2 + kx**2) * (1 - _np.cos(kz * phase))
        a = (1 / (kz**2)) * id_len * (b0**2) / (4 * brho**2)
        if field_component == "by":
            coefx = a * factor * kx**2
            coefy = -a * 2 * ky**2
        elif field_component == "bx":
            coefx = -a * factor * ky**2
            coefy = a * 2 * kx**2

        return coefx, coefy

    def get_id_estimated_focusing(
        self,
        betax,
        betay,
        id_period,
        plot_flag=False,
        field_component="by",
        **kwargs,
    ):
        coefx, coefy = self._get_id_quad_coef(
            plot_flag=plot_flag,
            id_period=id_period,
            field_component=field_component,
            **kwargs,
        )
        dtunex = -coefx * betax / (4 * _np.pi)
        dtuney = -coefy * betay / (4 * _np.pi)

        print("horizontal quadrupolar term KL: {:.5f}".format(coefx))
        print("vertical quadrupolar term KL: {:.5f}".format(coefy))
        print("horizontal delta tune: {:.5f}".format(dtunex))
        print("vertical delta tune: {:.5f}".format(dtuney))

        return coefx, coefy, dtunex, dtuney

    def generate_linear_kickmap(self, cxy=0, cyx=0, **kwargs):
        beam = Beam(energy=3)
        brho = beam.brho
        idkickmap = _IDKickMap()
        cxx, cyy = self._get_id_quad_coef(**kwargs)

        idkickmap.generate_linear_kickmap(
            brho=brho,
            posx=self.gridx,
            posy=self.gridy,
            cxx=cxx,
            cyy=cyy,
            cxy=cxy,
            cyx=cyx,
        )
        fname = self.get_kmap_filename(folder_data=utils.FOLDER_DATA, **kwargs)
        fname = fname.replace(".txt", "-linear.txt")
        idkickmap.kmap_idlen = utils.ID_KMAP_LEN
        idkickmap.save_kickmap_file(kickmap_filename=fname)


class AnalysisKickmap(Tools):
    def __init__(self):
        self._idkickmap = None
        self.save_flag = False
        self.plot_flag = False
        self.filter_flag = False
        self.shift_flag = False
        self.linear = False
        self.meas_flag = False
        self.FOLDER_DATA = utils.FOLDER_DATA

    def _get_figname_plane(
        self, kick_plane, var_param_name, var="X", **kwargs
    ):
        fpath = utils.FOLDER_DATA
        if self.meas_flag:
            fpath = fpath.replace("model/", "measurements/")
        fpath = fpath.replace("data/", "data/general/")

        fname = fpath + "kick{}-vs-{}".format(kick_plane, var.lower())
        forbidden_list = [var_param_name]
        items = list(kwargs.items())
        items_ = [elem for elem in items if elem[0] not in forbidden_list]
        for key, value in items_:
            fname += "_{}{}".format(key, value)
        fname += ".png"

        if self.linear:
            fname = fname.replace(".png", "-linear.png")

        return fname

    def _get_figname_allplanes(self, kick_plane, var="X", **kwargs):
        fname = utils.FOLDER_DATA
        if self.meas_flag:
            fname = fname.replace("model/", "measurements/")
        sulfix = "kick{}-vs-{}-all-planes".format(
            kick_plane.lower(), var.lower()
        )

        if "width" in kwargs.keys():
            fname += "widths/width_{}/".format(kwargs.get("width"))
        if "phase" in kwargs.keys():
            phase_str = Tools.get_phase_str(kwargs.get("phase"))
            fname += "phases/phase_{}/".format(phase_str)
        if "gap" in kwargs.keys():
            gap_str = Tools.get_gap_str(kwargs.get("gap"))
            fname += "gap_{}/".format(gap_str)

        forbidden_list = ["width", "phase", "gap"]
        items = list(kwargs.items())
        items_ = [elem for elem in items if elem[0] not in forbidden_list]
        for key, value in items_:
            fname += "{}{}/".format(key, value)

        Tools.mkdir_function(fname)
        fname += sulfix
        if self.linear:
            fname += "-linear"

        return fname

    def run_shift_kickmap(self, **kwargs):
        fname = self.get_kmap_filename(
            folder_data=utils.FOLDER_DATA, meas_flag=self.meas_flag, **kwargs
        )
        self._idkickmap = _IDKickMap(kmap_fname=fname, shift_on_axis=True)
        fname = fname.replace(".txt", "-shifted_on_axis.txt")
        self._idkickmap.save_kickmap_file(fname)

    def run_filter_kickmap(
        self, rx, ry, filter_order=5, is_shifted=True, **kwargs
    ):
        fname = self.get_kmap_filename(
            folder_data=utils.FOLDER_DATA,
            shift_flag=is_shifted,
            meas_flag=self.meas_flag,
            **kwargs,
        )
        self._idkickmap = _IDKickMap(fname)
        self._idkickmap.filter_kmap(
            posx=rx, posy=ry, order=filter_order, plot_flag=True
        )
        self._idkickmap.kmap_idlen = utils.ID_KMAP_LEN
        fname = fname.replace(".txt", "-filtered.txt")
        self._idkickmap.save_kickmap_file(fname)

    def _calc_idkmap_kicks(self, plane_idx=0, var="X"):
        beam = Beam(energy=3)
        brho = beam.brho
        rx0 = self._idkickmap.posx
        ry0 = self._idkickmap.posy
        fposx = self._idkickmap.fposx
        fposy = self._idkickmap.fposy
        kickx = self._idkickmap.kickx
        kicky = self._idkickmap.kicky
        if var.lower() == "x":
            rxf = fposx[plane_idx, :]
            ryf = fposy[plane_idx, :]
            pxf = kickx[plane_idx, :] / brho**2
            pyf = kicky[plane_idx, :] / brho**2
        elif var.lower() == "y":
            rxf = fposx[:, plane_idx]
            ryf = fposy[:, plane_idx]
            pxf = kickx[:, plane_idx] / brho**2
            pyf = kicky[:, plane_idx] / brho**2

        return rx0, ry0, pxf, pyf, rxf, ryf

    def check_kick_at_plane(
        self, planes=["X", "Y"], kick_planes=["X", "Y"], **kwargs
    ):
        var_param = [item for item in kwargs.items() if type(item[1]) == list]
        var_param = var_param[0]
        var_param_name = var_param[0]
        var_param_values = var_param[1]

        for var in planes:
            for kick_plane in kick_planes:
                for i, var_param_value in enumerate(var_param_values):
                    kwargs[var_param_name] = var_param_value

                    fname = self.get_kmap_filename(
                        folder_data=utils.FOLDER_DATA,
                        shift_flag=self.shift_flag,
                        filter_flag=self.filter_flag,
                        linear=self.linear,
                        meas_flag=self.meas_flag,
                        **kwargs,
                    )
                    fname_fig = self._get_figname_plane(
                        kick_plane=kick_plane,
                        var=var,
                        var_param_name=var_param_name,
                        **kwargs,
                    )

                    self._idkickmap = _IDKickMap(fname)
                    if var.lower() == "x":
                        pos_zero_idx = list(self._idkickmap.posy).index(0)
                    elif var.lower() == "y":
                        pos_zero_idx = list(self._idkickmap.posx).index(0)
                    rx0, ry0, pxf, pyf, *_ = self._calc_idkmap_kicks(
                        plane_idx=pos_zero_idx, var=var
                    )
                    if not self.linear:
                        pxf *= utils.RESCALE_KICKS
                        pyf *= utils.RESCALE_KICKS

                    pf, klabel = (
                        (pxf, "px")
                        if kick_plane.lower() == "x"
                        else (pyf, "py")
                    )

                    if var.lower() == "x":
                        r0, xlabel, rvar = (rx0, "x0 [mm]", "y")
                        pfit = _np.polyfit(r0, pf, len(r0) - 1)
                    else:
                        r0, xlabel, rvar = (ry0, "y0 [mm]", "x")
                        pfit = _np.polyfit(r0, pf, len(r0) - 1)
                    pf_fit = _np.polyval(pfit, r0)

                    label = var_param_name + " = {} mm".format(var_param_value)
                    _plt.figure(1)
                    _plt.plot(
                        1e3 * r0,
                        1e6 * pf,
                        ".-",
                        color=_plt.cm.jet(i / len(var_param_values)),
                        label=label,
                    )
                    _plt.plot(
                        1e3 * r0,
                        1e6 * pf_fit,
                        "-",
                        color=_plt.cm.jet(i / len(var_param_values)),
                        alpha=0.6,
                    )
                    print("kick plane: ", kick_plane)
                    print("plane: ", var)
                    print("Fitting:")
                    print(pfit[::-1])

                _plt.figure(1)
                _plt.xlabel(xlabel)
                _plt.ylabel("final {} [urad]".format(klabel))
                _plt.title(
                    "Kick{}, at pos{} {:+.3f} mm".format(
                        kick_plane.upper(), rvar, 0
                    )
                )
                _plt.legend()
                _plt.grid()
                _plt.tight_layout()
                if self.save_flag:
                    _plt.savefig(fname_fig, dpi=300)
                if self.plot_flag:
                    _plt.show()
                _plt.close()

    def check_kick_all_planes(
        self, planes=["X", "Y"], kick_planes=["X", "Y"], **kwargs
    ):
        for var in planes:
            for kick_plane in kick_planes:
                fname = self.get_kmap_filename(
                    folder_data=utils.FOLDER_DATA,
                    linear=self.linear,
                    meas_flag=self.meas_flag,
                    shift_flag=self.shift_flag,
                    filter_flag=self.filter_flag,
                    **kwargs,
                )
                self._idkickmap = _IDKickMap(fname)
                fname_fig = self._get_figname_allplanes(
                    var=var, kick_plane=kick_plane, **kwargs
                )

                if var.lower() == "x":
                    kmappos = self._idkickmap.posy
                else:
                    kmappos = self._idkickmap.posx

                for plane_idx, pos in enumerate(kmappos):
                    # if pos > 0:
                    # continue
                    rx0, ry0, pxf, pyf, *_ = self._calc_idkmap_kicks(
                        var=var, plane_idx=plane_idx
                    )
                    if not self.linear:
                        pxf *= utils.RESCALE_KICKS
                        pyf *= utils.RESCALE_KICKS
                    pf, klabel = (
                        (pxf, "px")
                        if kick_plane.lower() == "x"
                        else (pyf, "py")
                    )
                    if var.lower() == "x":
                        r0, xlabel, rvar = (rx0, "x0 [mm]", "y")
                    else:
                        r0, xlabel, rvar = (ry0, "y0 [mm]", "x")
                    rvar = "y" if var.lower() == "x" else "x"
                    label = "pos{} = {:+.3f} mm".format(rvar, 1e3 * pos)
                    _plt.plot(
                        1e3 * r0,
                        1e6 * pf,
                        ".-",
                        color=_plt.cm.jet(plane_idx / len(kmappos)),
                        label=label,
                    )
                    _plt.xlabel(xlabel)
                    _plt.ylabel("final {} [urad]".format(klabel))
                    # _plt.title(
                    #     'Kicks for gap {} mm, width {} mm'.format(gap, width))
                _plt.legend()
                _plt.grid()
                _plt.tight_layout()
                if self.save_flag:
                    _plt.savefig(fname_fig, dpi=300)
                if self.plot_flag:
                    _plt.show()
                _plt.close()

    def check_kick_at_plane_trk(self, **kwargs):
        fname = self.get_kmap_filename(
            folder_data=utils.FOLDER_DATA,
            linear=self.linear,
            meas_flag=self.meas_flag,
            shift_flag=self.shift_flag,
            filter_flag=self.filter_flag,
            **kwargs,
        )
        self._idkickmap = _IDKickMap(fname)
        plane_idx = list(self._idkickmap.posy).index(0)
        out = self._calc_idkmap_kicks(plane_idx=plane_idx)
        rx0, ry0 = out[0], out[1]
        pxf, pyf = out[2], out[3]
        rxf, ryf = out[4], out[5]
        if not self.linear:
            pxf *= utils.RESCALE_KICKS
            pyf *= utils.RESCALE_KICKS

        # lattice with IDs
        model, _ = self.create_model_ids(
            fname=fname,
            linear=self.linear,
            create_ids=utils.create_ids,
            rescale_kicks=utils.RESCALE_KICKS,
            rescale_length=utils.RESCALE_LENGTH,
            fitted_model=False,
            fit_path=utils.FIT_PATH,
        )

        famdata = pymodels.si.get_family_data(model)

        # shift model
        idx = famdata[utils.ID_FAMNAME]["index"]
        idx_begin = idx[0][0]
        idx_end = idx[0][-1]
        idx_dif = idx_end - idx_begin

        model = pyaccel.lattice.shift(model, start=idx_begin)

        rxf_trk, ryf_trk = _np.ones(len(rx0)), _np.ones(len(rx0))
        pxf_trk, pyf_trk = _np.ones(len(rx0)), _np.ones(len(rx0))
        for i, x0 in enumerate(rx0):
            coord_ini = _np.array([x0, 0, 0, 0, 0, 0])
            coord_fin, *_ = pyaccel.tracking.line_pass(
                model, coord_ini, indices="open"
            )

            rxf_trk[i] = coord_fin[0, idx_dif + 1]
            ryf_trk[i] = coord_fin[2, idx_dif + 1]
            pxf_trk[i] = coord_fin[1, idx_dif + 1]
            pyf_trk[i] = coord_fin[3, idx_dif + 1]

        _plt.plot(
            1e3 * rx0,
            1e6 * (rxf - rx0),
            ".-",
            color="C1",
            label="Pos X  kickmap",
        )
        _plt.plot(
            1e3 * rx0, 1e6 * ryf, ".-", color="b", label="Pos Y  kickmap"
        )
        _plt.plot(
            1e3 * rx0,
            1e6 * (rxf_trk - rx0),
            "o",
            color="C1",
            label="Pos X  tracking",
        )
        _plt.plot(
            1e3 * rx0, 1e6 * ryf_trk, "o", color="b", label="Pos Y  tracking"
        )
        _plt.xlabel("x0 [mm]")
        _plt.ylabel("final dpos [um]")
        _plt.title("dPos")
        _plt.legend()
        _plt.grid()
        _plt.show()

        _plt.plot(
            1e3 * rx0, 1e6 * pxf, ".-", color="C1", label="Kick X  kickmap"
        )
        _plt.plot(
            1e3 * rx0, 1e6 * pyf, ".-", color="b", label="Kick Y  kickmap"
        )
        _plt.plot(
            1e3 * rx0, 1e6 * pxf_trk, "o", color="C1", label="Kick X  tracking"
        )
        _plt.plot(
            1e3 * rx0, 1e6 * pyf_trk, "o", color="b", label="Kick Y  tracking"
        )
        _plt.xlabel("x0 [mm]")
        _plt.ylabel("final px [urad]")
        _plt.title("Kicks")
        _plt.legend()
        _plt.grid()
        _plt.show()


class AnalysisEffects(Tools):
    def __init__(self):
        self._idkickmap = None
        self._model_id = None
        self._ids = None
        self._twiss0 = None
        self._twiss1 = None
        self._twiss2 = None
        self._twiss3 = None
        self._beta_flag = None
        self._stg = None
        self.id_famname = utils.ID_FAMNAME
        self.fitted_model = False
        self.calc_type = 0
        self.corr_system = "SOFB"
        self.filter_flag = False
        self.shift_flag = True
        self.orbcorr_plot_flag = False
        self.bb_plot_flag = False
        self.meas_flag = False
        self.linear = False

    def _create_model_nominal(self):
        model0 = pymodels.si.create_accelerator()
        if self.fitted_model:
            famdata0 = pymodels.si.families.get_family_data(model0)
            idcs_qn0 = _np.array(famdata0["QN"]["index"]).ravel()
            idcs_qs0 = _np.array(famdata0["QS"]["index"]).ravel()
            data = _load_pickle(utils.FIT_PATH)
            kl = data["KL"]
            ksl = data["KsL"]
            pyaccel.lattice.set_attribute(model0, "KL", idcs_qn0, kl)
            pyaccel.lattice.set_attribute(model0, "KsL", idcs_qs0, ksl)

        model0.cavity_on = False
        model0.radiation_on = 0
        return model0

    def _get_knobs_locs(self):
        straight_nr = dict()
        knobs = dict()
        locs_beta = dict()
        for id_ in self._ids:
            straight_nr_ = int(id_.subsec[2:4])

            # get knobs and beta locations
            if straight_nr_ is not None:
                _, knobs_, _ = optics.symm_get_knobs(
                    self._model_id, straight_nr_
                )
                locs_beta_ = optics.symm_get_locs_beta(knobs_)
            else:
                knobs_, locs_beta_ = None, None

            straight_nr[id_.subsec] = straight_nr_
            knobs[id_.subsec] = knobs_
            locs_beta[id_.subsec] = locs_beta_

        return knobs, locs_beta, straight_nr

    def _calc_action_ratio(self, x0, nturns=1000):
        coord_ini = _np.array([x0, 0, 0, 0, 0, 0])
        coord_fin, *_ = pyaccel.tracking.ring_pass(
            self._model_id,
            coord_ini,
            nr_turns=nturns,
            turn_by_turn=True,
            parallel=True,
        )
        rx = coord_fin[0, :]
        ry = coord_fin[2, :]
        twiss, *_ = pyaccel.optics.calc_twiss(self._model_id)
        betax, betay = twiss.betax, twiss.betay  # Beta functions
        jx = 1 / (betax[0] * nturns) * (_np.sum(rx**2))
        jy = 1 / (betay[0] * nturns) * (_np.sum(ry**2))

        print("action ratio = {:.3f}".format(jy / jx))
        return jy / jx

    def _calc_coupling(self):
        ed_tang, *_ = pyaccel.optics.calc_edwards_teng(self._model_id)
        min_tunesep, ratio = pyaccel.optics.estimate_coupling_parameters(
            ed_tang
        )
        print("Minimum tune separation = {:.3f}".format(min_tunesep))

        return min_tunesep

    def _calc_dtune_betabeat(self, twiss1):
        dtunex = (twiss1.mux[-1] - self._twiss0.mux[-1]) / 2 / _np.pi
        dtuney = (twiss1.muy[-1] - self._twiss0.muy[-1]) / 2 / _np.pi
        bbeatx = (twiss1.betax - self._twiss0.betax) / self._twiss0.betax
        bbeaty = (twiss1.betay - self._twiss0.betay) / self._twiss0.betay
        bbeatx *= 100
        bbeaty *= 100
        bbeatx_rms = _np.std(bbeatx)
        bbeaty_rms = _np.std(bbeaty)
        bbeatx_absmax = _np.max(_np.abs(bbeatx))
        bbeaty_absmax = _np.max(_np.abs(bbeaty))
        return (
            dtunex,
            dtuney,
            bbeatx,
            bbeaty,
            bbeatx_rms,
            bbeaty_rms,
            bbeatx_absmax,
            bbeaty_absmax,
        )

    def _analysis_uncorrected_perturbation(self, plot_flag=True):
        self._twiss1, *_ = pyaccel.optics.calc_twiss(
            self._model_id, indices="closed"
        )

        results = self._calc_dtune_betabeat(twiss1=self._twiss1)
        dtunex, dtuney = results[0], results[1]
        bbeatx, bbeaty = results[2], results[3]
        bbeatx_rms, bbeaty_rms = results[4], results[5]
        bbeatx_absmax, bbeaty_absmax = results[6], results[7]

        if plot_flag:
            print(f"dtunex: {dtunex:+.6f}")
            print(f"dtunex: {dtuney:+.6f}")
            txt = f"bbetax: {bbeatx_rms:04.1f} % rms, "
            txt += f"{bbeatx_absmax:04.1f} % absmax"
            print(txt)
            txt = f"bbetay: {bbeaty_rms:04.1f} % rms, "
            txt += f"{bbeaty_absmax:04.1f} % absmax"
            print(txt)

            labelx = f"X ({bbeatx_rms:.1f} % rms)"
            labely = f"Y ({bbeaty_rms:.1f} % rms)"
            _plt.plot(
                self._twiss1.spos, bbeatx, color="b", alpha=1, label=labelx
            )
            _plt.plot(
                self._twiss1.spos, bbeaty, color="r", alpha=0.8, label=labely
            )
            _plt.xlabel("spos [m]")
            _plt.ylabel("Beta Beat [%]")
            _plt.title("Beta Beating from " + self.id_famname)
            _plt.legend()
            _plt.grid()
            _plt.show()
            _plt.clf()

    def _plot_beta_beating(self, xlim=None, **kwargs):
        fpath = self.get_data_path(
            folder_data=utils.FOLDER_DATA, meas_flag=self.meas_flag, **kwargs
        )
        # Compare optics between nominal value and uncorrect optics due ID
        results = self._calc_dtune_betabeat(twiss1=self._twiss1)
        dtunex, dtuney = results[0], results[1]
        bbeatx, bbeaty = results[2], results[3]
        bbeatx_rms, bbeaty_rms = results[4], results[5]
        bbeatx_absmax, bbeaty_absmax = results[6], results[7]
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

        _plt.clf()

        label1 = {False: "-nominal", True: "-fittedmodel"}[self.fitted_model]

        _plt.figure(1)
        stg_tune = f"\u0394\u03BDx: {dtunex:+0.04f}\n"
        stg_tune += f"\u0394\u03BDy: {dtuney:+0.04f}"
        labelx = f"X ({bbeatx_rms:.3f} % rms)"
        labely = f"Y ({bbeaty_rms:.3f} % rms)"
        figname = fpath + "opt{}-ids-nonsymm".format(label1)
        Tools.mkdir_function(fpath)
        if self.linear:
            figname += "-linear"
        _plt.plot(
            self._twiss0.spos, bbeatx, color="b", alpha=1.0, label=labelx
        )
        _plt.plot(
            self._twiss0.spos, bbeaty, color="r", alpha=0.8, label=labely
        )
        bbmax = _np.max(_np.concatenate((bbeatx, bbeaty)))
        bbmin = _np.min(_np.concatenate((bbeatx, bbeaty)))
        ylim = (1.1 * bbmin, 1.1 * bbmax)
        if xlim is None:
            xlim = (0, self._twiss0.spos[-1])
        pyaccel.graphics.draw_lattice(
            self._model_id, offset=bbmin, height=bbmax / 8, gca=True
        )
        _plt.xlim(xlim)
        _plt.ylim(ylim)
        _plt.xlabel("spos [m]")
        _plt.ylabel("Beta Beating [%]")
        _plt.title("Tune shift:" + "\n" + stg_tune)
        _plt.suptitle(self.id_famname + " - Non-symmetrized optics")
        _plt.tight_layout()
        _plt.legend()
        _plt.grid()
        _plt.savefig(figname, dpi=300)
        _plt.clf()

        # Compare optics between nominal value and symmetrized optics
        results = self._calc_dtune_betabeat(twiss1=self._twiss2)
        dtunex, dtuney = results[0], results[1]
        bbeatx, bbeaty = results[2], results[3]
        bbeatx_rms, bbeaty_rms = results[4], results[5]
        bbeatx_absmax, bbeaty_absmax = results[6], results[7]
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

        _plt.figure(2)
        labelx = f"X ({bbeatx_rms:.3f} % rms)"
        labely = f"Y ({bbeaty_rms:.3f} % rms)"
        figname = fpath + "opt{}-ids-symm".format(label1)
        if self.linear:
            figname += "-linear"
        _plt.plot(
            self._twiss0.spos, bbeatx, color="b", alpha=1.0, label=labelx
        )
        _plt.plot(
            self._twiss0.spos, bbeaty, color="r", alpha=0.8, label=labely
        )
        bbmax = _np.max(_np.concatenate((bbeatx, bbeaty)))
        bbmin = _np.min(_np.concatenate((bbeatx, bbeaty)))
        ylim = (1.1 * bbmin, 1.1 * bbmax)
        _plt.xlim(xlim)
        _plt.ylim(ylim)
        # bbmax = _np.max(_np.concatenate((bbeatx, bbeaty)))
        pyaccel.graphics.draw_lattice(
            self._model_id, offset=bbmin, height=bbmax / 8, gca=True
        )
        _plt.xlabel("spos [m]")
        _plt.ylabel("Beta Beating [%]")
        _plt.title("Beta Beating")
        _plt.suptitle(
            self.id_famname + "- Symmetrized optics and uncorrected tunes"
        )
        _plt.legend()
        _plt.grid()
        _plt.tight_layout()
        _plt.savefig(figname, dpi=300)
        _plt.clf()

        # Compare optics between nominal value and all corrected
        results = self._calc_dtune_betabeat(twiss1=self._twiss3)
        dtunex, dtuney = results[0], results[1]
        bbeatx, bbeaty = results[2], results[3]
        bbeatx_rms, bbeaty_rms = results[4], results[5]
        bbeatx_absmax, bbeaty_absmax = results[6], results[7]
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

        _plt.figure(3)
        labelx = f"X ({bbeatx_rms:.3f} % rms)"
        labely = f"Y ({bbeaty_rms:.3f} % rms)"
        figname = fpath + "opt{}-ids-symm-tunes".format(label1)
        if self.linear:
            figname += "-linear"
        _plt.plot(
            self._twiss0.spos, bbeatx, color="b", alpha=1.0, label=labelx
        )
        _plt.plot(
            self._twiss0.spos, bbeaty, color="r", alpha=0.8, label=labely
        )
        bbmax = _np.max(_np.concatenate((bbeatx, bbeaty)))
        bbmin = _np.min(_np.concatenate((bbeatx, bbeaty)))
        ylim = (1.1 * bbmin, 1.1 * bbmax)
        _plt.xlim(xlim)
        _plt.ylim(ylim)
        # bbmax = _np.max(_np.concatenate((bbeatx, bbeaty)))
        pyaccel.graphics.draw_lattice(
            self._model_id, offset=bbmin, height=bbmax / 8, gca=True
        )
        _plt.xlabel("spos [m]")
        _plt.ylabel("Beta Beating [%]")
        _plt.title("Corrections:" + "\n" + self._stg)
        _plt.suptitle(
            self.id_famname + "- Symmetrized optics and corrected tunes"
        )
        _plt.legend()
        _plt.grid()
        _plt.tight_layout()
        _plt.savefig(figname, dpi=300)
        # _plt.show()
        _plt.clf()

    def _correct_beta(self, straight_nr, knobs, goal_beta, goal_alpha):
        dk_tot = _np.zeros(len(knobs))
        for i in range(7):
            dk, k = optics.correct_symmetry_withbeta(
                self._model_id, straight_nr, goal_beta, goal_alpha
            )
            if i == 0:
                k0 = k
            print("iteration #{}, \u0394K: {}".format(i + 1, dk))
            dk_tot += dk
        delta = dk_tot / k0
        self._stg = str()
        for i, fam in enumerate(knobs):
            self._stg += "{:<9s} \u0394K: {:+9.3f} % \n".format(
                fam, 100 * delta[i]
            )
        print(self._stg)
        self._twiss2, *_ = pyaccel.optics.calc_twiss(
            self._model_id, indices="closed"
        )
        print()

    def _correct_tunes(self, goal_tunes, knobs_out=None):
        tunes = (
            self._twiss1.mux[-1] / _np.pi / 2,
            self._twiss1.muy[-1] / _np.pi / 2,
        )
        print("init    tunes: {:.9f} {:.9f}".format(tunes[0], tunes[1]))
        for i in range(2):
            optics.correct_tunes_twoknobs(
                self._model_id, goal_tunes, knobs_out
            )
            twiss, *_ = pyaccel.optics.calc_twiss(self._model_id)
            tunes = twiss.mux[-1] / _np.pi / 2, twiss.muy[-1] / _np.pi / 2
            print(
                "iter #{} tunes: {:.9f} {:.9f}".format(
                    i + 1, tunes[0], tunes[1]
                )
            )
        print(
            "goal    tunes: {:.9f} {:.9f}".format(goal_tunes[0], goal_tunes[1])
        )
        self._twiss3, *_ = pyaccel.optics.calc_twiss(
            self._model_id, indices="closed"
        )
        print()

    def _execute_correction_algorithm(self, **kwargs):
        # create unperturbed model for reference
        model0 = self._create_model_nominal()
        self._twiss0, *_ = pyaccel.optics.calc_twiss(model0, indices="closed")
        print("Model without ID:")
        print("length : {:.4f} m".format(model0.length))
        print("tunex  : {:.6f}".format(self._twiss0.mux[-1] / 2 / _np.pi))
        print("tuney  : {:.6f}".format(self._twiss0.muy[-1] / 2 / _np.pi))
        # create model with ID
        fname = self.get_kmap_filename(
            folder_data=utils.FOLDER_DATA,
            shift_flag=self.shift_flag,
            filter_flag=self.filter_flag,
            linear=self.linear,
            meas_flag=self.meas_flag,
            **kwargs,
        )
        self._model_id, self._ids = self.create_model_ids(
            fname=fname,
            linear=self.linear,
            create_ids=utils.create_ids,
            rescale_kicks=utils.RESCALE_KICKS,
            rescale_length=utils.RESCALE_LENGTH,
            fitted_model=self.fitted_model,
            fit_path=utils.FIT_PATH,
        )
        knobs, locs_beta, straight_nr = self._get_knobs_locs()

        print("element indices for straight section begin and end:")
        for idsubsec, locs_beta_ in locs_beta.items():
            print(idsubsec, locs_beta_)

        print("local quadrupole fams: ")
        for idsubsec, knobs_ in knobs.items():
            print(idsubsec, knobs_)

        # correct orbit
        (
            kicks,
            spos_bpms,
            codx_c,
            cody_c,
            codx_u,
            cody_u,
            bpms,
        ) = orbcorr.correct_orbit_fb(
            model0,
            self._model_id,
            corr_system=self.corr_system,
            nr_steps=1,
            plot_flag=self.orbcorr_plot_flag,
        )

        # calculate beta beating and delta tunes
        self._analysis_uncorrected_perturbation(plot_flag=False)

        # get list of ID model indices and set rescale_kicks to zero
        ids_ind_all = orbcorr.get_ids_indices(self._model_id)
        rescale_kicks_orig = list()
        for idx in range(len(ids_ind_all) // 2):
            ind_id = ids_ind_all[2 * idx : 2 * (idx + 1)]
            rescale_kicks_orig.append(self._model_id[ind_id[0]].rescale_kicks)
            self._model_id[ind_id[0]].rescale_kicks = 0
            self._model_id[ind_id[1]].rescale_kicks = 0

        # loop over IDs turning rescale_kicks on, one by one.
        for idx in range(len(ids_ind_all) // 2):
            # turn rescale_kicks on for ID index idx
            ind_id = ids_ind_all[2 * idx : 2 * (idx + 1)]
            self._model_id[ind_id[0]].rescale_kicks = rescale_kicks_orig[idx]
            self._model_id[ind_id[1]].rescale_kicks = rescale_kicks_orig[idx]
            fam_name = self._model_id[ind_id[0]].fam_name
            # print(idx, ind_id)
            # continue

            # search knob and straight_nr for ID index idx
            for subsec in knobs:
                straight_nr_ = straight_nr[subsec]
                knobs_ = knobs[subsec]
                locs_beta_ = locs_beta[subsec]
                if min(locs_beta_) < ind_id[0] and ind_id[1] < max(locs_beta_):
                    break
            k = self._calc_coupling()
            print()
            print("symmetrizing ID {} in subsec {}".format(fam_name, subsec))

            # calculate nominal twiss
            goal_tunes = _np.array(
                [
                    self._twiss0.mux[-1] / 2 / _np.pi,
                    self._twiss0.muy[-1] / 2 / _np.pi,
                ]
            )
            goal_beta = _np.array(
                [
                    self._twiss0.betax[locs_beta_],
                    self._twiss0.betay[locs_beta_],
                ]
            )
            goal_alpha = _np.array(
                [
                    self._twiss0.alphax[locs_beta_],
                    self._twiss0.alphay[locs_beta_],
                ]
            )
            print("goal_beta:")
            print(goal_beta)

            # symmetrize optics (local quad fam knobs)
            if self._beta_flag:
                self._correct_beta(straight_nr_, knobs_, goal_beta, goal_alpha)

                self._correct_tunes(goal_tunes)

                if self.bb_plot_flag:
                    # plot results
                    self._plot_beta_beating(**kwargs)

        return self._model_id

    def _analysis_dynapt(self, model, **kwargs):
        model.radiation_on = 0
        model.cavity_on = False
        model.vchamber_on = True

        dynapxy = DynapXY(model)
        dynapxy.params.x_nrpts = 40
        dynapxy.params.y_nrpts = 20
        dynapxy.params.nrturns = 2 * 1024
        print(dynapxy)
        dynapxy.do_tracking()
        dynapxy.process_data()
        fig, axx, ayy = dynapxy.make_figure_diffusion(
            orders=(1, 2, 3, 4),
            nuy_bounds=(14.12, 14.45),
            nux_bounds=(49.05, 49.50),
        )

        fpath = self.get_data_path(
            folder_data=utils.FOLDER_DATA, meas_flag=self.meas_flag, **kwargs
        )
        Tools.mkdir_function(fpath)
        print(fpath)
        label1 = ["", "-ids-nonsymm", "-ids-symm"][self.calc_type]
        label2 = {False: "-nominal", True: "-fittedmodel"}[self.fitted_model]
        fig_name = fpath + "dynapt{}{}.png".format(label2, label1)
        if self.linear:
            fig_name = fig_name.replace(".png", "-linear.png")
        fig.savefig(fig_name, dpi=300, format="png")
        # fig.clf()
        # _plt.show()

    def run_analysis_dynapt(self, **kwargs):
        if self.calc_type == self.CALC_TYPES.nominal:
            model = self._create_model_nominal()
        elif self.calc_type in (
            self.CALC_TYPES.symmetrized,
            self.CALC_TYPES.nonsymmetrized,
        ):
            self._beta_flag = self.calc_type == self.CALC_TYPES.symmetrized
            model = self._execute_correction_algorithm(**kwargs)
        else:
            raise ValueError("Invalid calc_type")

        self._analysis_dynapt(model=model, **kwargs)
