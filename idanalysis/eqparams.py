"""IDs analysis."""

import mathphys
import matplotlib.pyplot as _plt
import numpy as _np
import pyaccel
import pymodels
from fieldmaptrack import Beam
from scipy.integrate import cumtrapz
from scipy.optimize import minimize

ECHARGE = mathphys.constants.elementary_charge
EMASS = mathphys.constants.electron_mass
LSPEED = mathphys.constants.light_speed
VCPERM = mathphys.constants.vacuum_permitticity
CQ = mathphys.constants.Cq
ECHARGE_MC = ECHARGE / (2 * _np.pi * EMASS * LSPEED)

_model = pymodels.si.create_accelerator()
_mid_subsections = None
_twiss = None


class InsertionParams:
    """Class to specify insertion parameters.

     All equations used here are from the book: Clarke, James A.,
    'The Science and Technology of Undulators and Wigglers'.
    Chapter 10: 'The Effect of Insertion Devices on the Electron Beam'.
    """

    def __init__(self, beam_energy=3.0):
        """Class constructor.

        Args:
            beam_energy (Float, optional): Beam energy [GeV]. Defaults to 3.0.
        """
        self._field_profile = None
        self._fam_name = None
        self._bx_peak = None
        self._by_peak = None
        self._kx = None
        self._ky = None
        self._period = None
        self._length = None
        self._nr_periods = None
        self._straight_section = None
        self._etax = None
        self._etapx = None
        self._etay = None
        self._etapy = None
        self._i1x = None
        self._i2x = None
        self._i3x = None
        self._i4x = None
        self._i5x = None
        self._i1y = None
        self._i2y = None
        self._i3y = None
        self._i4y = None
        self._i5y = None
        self._u0 = None
        self._mid_subsections = None
        self.beam = Beam(beam_energy)

        global _twiss
        global _mid_subsections
        if _twiss is None:
            mia = pyaccel.lattice.find_indices(_model, "fam_name", "mia")
            mib = pyaccel.lattice.find_indices(_model, "fam_name", "mib")
            mip = pyaccel.lattice.find_indices(_model, "fam_name", "mip")
            _mid_subsections = _np.sort(_np.array(mia + mib + mip))
            _twiss, *_ = pyaccel.optics.calc_twiss(_model, indices="open")

    @staticmethod
    def set_model(model):
        """Set Sirius model.

        Args:
            model (Pymodels object): Sirius model
        """
        global _model
        global _mid_subsections
        global _twiss
        _model = model
        mia = pyaccel.lattice.find_indices(_model, "fam_name", "mia")
        mib = pyaccel.lattice.find_indices(_model, "fam_name", "mib")
        mip = pyaccel.lattice.find_indices(_model, "fam_name", "mip")
        _mid_subsections = _np.sort(_np.array(mia + mib + mip))
        _twiss, *_ = pyaccel.optics.calc_twiss(_model, indices="open")

    @property
    def fam_name(self):
        """Insertion device's name.

        Returns:
            string: ID's name
        """
        return self._fam_name

    @fam_name.setter
    def fam_name(self, value):
        self._fam_name = value

    @property
    def field_profile(self):
        """Field profile.

        Returns:
            numpy array: First column contains longitudinal spatial
            coordinate (z) [mm], second column contais vertical field
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
            self._kx = 1e-3 * ECHARGE_MC * self._bx_peak * self.period

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
            self._ky = 1e-3 * ECHARGE_MC * self._by_peak * self.period

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
            self._bx_peak = self._kx / (ECHARGE_MC * 1e-3 * self.period)

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
            self._by_peak = self._ky / (ECHARGE_MC * 1e-3 * self.period)

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
            self._kx = 1e-3 * ECHARGE_MC * self._bx_peak * self.period
        if self._by_peak is not None:
            self._ky = 1e-3 * ECHARGE_MC * self._by_peak * self.period
        if self._kx is not None:
            self._bx_peak = self._kx / (ECHARGE_MC * 1e-3 * self.period)
        if self._ky is not None:
            self._by_peak = self._ky / (ECHARGE_MC * 1e-3 * self.period)

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
    def etapx(self):
        """d(etax)/ds generated by ID.

        Returns:
            Numpy 1d array: Derivative of horizontal dispersion function
        """
        return self._etapx

    @property
    def etay(self):
        """Vertical dispersion generated by ID.

        Returns:
            Numpy 1d array: Vertical dispersion function
        """
        return self._etay

    @property
    def etapy(self):
        """d(etay)/ds generated by ID.

        Returns:
            Numpy 1d array: Derivative of vertical dispersion function
        """
        return self._etapy

    @property
    def i1x(self):
        """Contribution of ID to first rad integral.

        Returns:
            Numpy 1d array: Delta i1x
        """
        return self._i1x

    @property
    def i2x(self):
        """Contribution of ID to second rad integral.

        Returns:
            Numpy 1d array: Delta i2x
        """
        return self._i2x

    @property
    def i3x(self):
        """Contribution of ID to third rad integral.

        Returns:
            Numpy 1d array: Delta i3x
        """
        return self._i3x

    @property
    def i4x(self):
        """Contribution of ID to fourth rad integral.

        Returns:
            Numpy 1d array: Delta i4x
        """
        return self._i4x

    @property
    def i5x(self):
        """Contribution of ID to fifth rad integral.

        Returns:
            Numpy 1d array: Delta i5x
        """
        return self._i5x

    @property
    def i1y(self):
        """Contribution of ID to first rad integral.

        Returns:
            Numpy 1d array: Delta i1y
        """
        return self._i1y

    @property
    def i2y(self):
        """Contribution of ID to second rad integral.

        Returns:
            Numpy 1d array: Delta i2y
        """
        return self._i2y

    @property
    def i3y(self):
        """Contribution of ID to third rad integral.

        Returns:
            Numpy 1d array: Delta i3y
        """
        return self._i3y

    @property
    def i4y(self):
        """Contribution of ID to fourth rad integral.

        Returns:
            Numpy 1d array: Delta i4y
        """
        return self._i4y

    @property
    def i5y(self):
        """Contribution of ID to fifth rad integral.

        Returns:
            Numpy 1d array: Delta i5y
        """
        return self._i5y

    @property
    def u0(self):
        """Contribution of ID to energy loss.

        Returns:
            float: Energy loss per turn in ID.
        """
        return self._u0

    def create_field_profile(self, pts_period=1001):
        """Create a sinusoidal field with first and second integrals zero.

        Args:
            pts_period (int, optional): Number of points per period. Defaults
             to 1001.

        Returns:
            numpy array: First column contains longitudinal spatial
            coordinate (z) [mm], second column contais vertical field
            [T], and third column constais horizontal field [T].
        """
        by_peak = self.by_peak
        bx_peak = self.bx_peak
        nr_periods = self.nr_periods
        period = self.period
        field = _np.zeros(((nr_periods + 2) * pts_period, 3))

        if by_peak is not None:
            result = minimize(
                self._calc_field_integral,
                0.4,
                args=(by_peak, period, nr_periods, pts_period),
            )
            s, by = self._generate_field(
                result.x, by_peak, period, nr_periods, pts_period
            )
            field[:, 1] = by

        if bx_peak is not None:
            result = minimize(
                self._calc_field_integral,
                0.4,
                args=(bx_peak, period, nr_periods, pts_period),
            )
            s, bx = self._generate_field(
                result.x, bx_peak, period, nr_periods, pts_period
            )
            field[:, 2] = bx

        field[:, 0] = s
        self.field_profile = field
        return field

    def calc_dispersion(self):
        """Calculate dispersion (and its derivative) generated by the ID.

        Returns:
            numpy array: Horizontal dispersion on the ID
            numpy array: Vertical dispersion on the ID
            numpy array: d(etax)/ds on the ID
            numpy array: d(etay)/ds on the ID
        """
        brho = self.beam.brho
        s = 1e-3 * self.field_profile[:, 0]
        by = self.field_profile[:, 1]
        bx = self.field_profile[:, 2]
        ds = s[1] - s[0]

        i1x = cumtrapz(dx=ds, y=by, initial=0)
        i1y = cumtrapz(dx=ds, y=-bx, initial=0)
        etapx = -1 * i1x / brho
        etapy = -1 * i1y / brho

        i2x = cumtrapz(dx=ds, y=i1x, initial=0)
        i2y = cumtrapz(dx=ds, y=i1y, initial=0)
        etax = -1 * i2x / brho
        etay = -1 * i2y / brho

        self._etapx = etapx
        self._etapy = etapy
        self._etax = etax
        self._etay = etay
        return etax, etay, etapx, etapy

    def get_curly_h(self):
        """Calculate curly H in ID section.

        Returns:
            numpy 1d array: Curly Hx in the ID region.
            numpy 1d array: Curly Hy in the ID region.
        """
        global _twiss
        global _mid_subsections
        s = 1e-3 * self.field_profile[:, 0]
        if len(s) % 2 == 0:
            length = (s - s[0])[: int(len(s) / 2)]
        else:
            length = (s - s[0])[: int((len(s) + 1) / 2)]
        subsec = self.straight_section
        idx = _mid_subsections[int(subsec[2:4]) - 1]

        etax0 = _twiss.etax[idx]
        etay0 = _twiss.etay[idx]

        etapx0 = _twiss.etapx[idx]
        etapy0 = _twiss.etapy[idx]

        betax0 = _twiss.betax[idx]
        betay0 = _twiss.betay[idx]

        alphax0 = _twiss.alphax[idx]
        alphay0 = _twiss.alphay[idx]

        gammax0 = _twiss.gammax[idx]
        gammay0 = _twiss.gammay[idx]

        etapx_acc = etapx0 * _np.ones(len(s))
        etax_acc_after = etax0 + etapx_acc[0] * length
        etax_acc_before = etax0 - etapx_acc[0] * length

        etapy_acc = etapy0 * _np.ones(len(s))
        etay_acc_after = etay0 + etapx_acc[0] * length
        etay_acc_before = etay0 - etapx_acc[0] * length

        betax_after = betax0 - 2 * length * alphax0 + length**2 * gammax0
        betax_before = betax0 + 2 * length * alphax0 + length**2 * gammax0

        alphax_after = alphax0 - length * gammax0
        alphax_before = alphax0 + length * gammax0

        betay_after = betay0 - 2 * length * alphay0 + length**2 * gammay0
        betay_before = betay0 + 2 * length * alphay0 + length**2 * gammay0

        alphay_after = alphay0 - length * gammay0
        alphay_before = alphay0 + length * gammay0
        if len(s) % 2 != 0:
            betax_after = betax_after[1:]
            alphax_after = alphax_after[1:]
            etax_acc_after = etax_acc_after[1:]

            betay_after = betay_after[1:]
            alphay_after = alphay_after[1:]
            etay_acc_after = etay_acc_after[1:]

        betax = _np.concatenate((betax_before[::-1], betax_after))
        alphax = _np.concatenate((alphax_before[::-1], alphax_after))
        etax_acc = _np.concatenate((etax_acc_before[::-1], etax_acc_after))

        betay = _np.concatenate((betay_before[::-1], betay_after))
        alphay = _np.concatenate((alphay_before[::-1], alphay_after))
        etay_acc = _np.concatenate((etay_acc_before[::-1], etay_acc_after))

        etax, etay, etapx, etapy = self.calc_dispersion()

        etax += etax_acc
        etay += etay_acc
        etapx += etapx_acc
        etapx += etapy_acc

        hx = pyaccel.optics.get_curlyh(betax, alphax, etax, etapx)
        hy = pyaccel.optics.get_curlyh(betay, alphay, etay, etapy)

        return hx, hy

    def calc_u0(self):
        """Calculate energy loss in ID.

        Returns:
            float: Energy loss in [keV]
        """
        brho = self.beam.brho
        gamma = self.beam.gamma
        s = 1e-3 * self.field_profile[:, 0]
        ds = s[1] - s[0]
        by = self.field_profile[:, 1]
        bx = self.field_profile[:, 2]
        b = _np.sqrt(bx**2 + by**2)
        u0 = 1e-3 * (
            (ECHARGE * gamma**4)
            * _np.trapz(dx=ds, y=(b / brho) ** 2)
            / (6 * _np.pi * VCPERM)
        )
        self._u0 = u0
        return u0

    def calc_power(self, current=0.1):
        """Calculate radiated power in ID.

        Args:
            current (float, optional): Current in [A]. Defaults to 0.1.

        Returns:
            float: Power in [kW]
        """
        p = current * self.calc_u0()
        return p

    def calc_i1(self):
        """Calculate first radiation integral contribution from one ID.

        Returns:
            float: delta I1x
            float: delta I1y
        """
        brho = self.beam.brho
        by = self.field_profile[:, 1]
        bx = self.field_profile[:, 2]
        s = 1e-3 * self.field_profile[:, 0]
        ds = s[1] - s[0]
        etax, etay, *_ = self.calc_dispersion()
        i1x = _np.trapz(dx=ds, y=etax * by / brho)
        i1y = _np.trapz(dx=ds, y=etay * bx / brho)
        self._i1x = i1x
        self._i1y = i1y
        return i1x, i1y

    def calc_i2(self):
        """Calculate second radiation integral contribution from one ID.

        Returns:
            float: delta I2x
            float: delta I2y
        """
        brho = self.beam.brho
        by = self.field_profile[:, 1]
        bx = self.field_profile[:, 2]
        s = 1e-3 * self.field_profile[:, 0]
        ds = s[1] - s[0]
        i2x = _np.trapz(dx=ds, y=(by / brho) ** 2)
        i2y = _np.trapz(dx=ds, y=(bx / brho) ** 2)
        self._i2x = i2x
        self._i2y = i2y
        return i2x, i2y

    def calc_i3(self):
        """Calculate third radiation integral contribution from one ID.

        Returns:
            float: delta I3x
            float: delta I3y
        """
        brho = self.beam.brho
        by = self.field_profile[:, 1]
        bx = self.field_profile[:, 2]
        s = 1e-3 * self.field_profile[:, 0]
        ds = s[1] - s[0]
        i3x = _np.trapz(dx=ds, y=_np.abs((by / brho) ** 3))
        i3y = _np.trapz(dx=ds, y=_np.abs((bx / brho) ** 3))
        self._i3x = i3x
        self._i3y = i3y
        return i3x, i3y

    def calc_i4(self):
        """Calculate fourth radiation integral contribution from one ID.

        Returns:
            float: delta I4x
            float: delta I4y
        """
        brho = self.beam.brho
        by = self.field_profile[:, 1]
        bx = self.field_profile[:, 2]
        s = 1e-3 * self.field_profile[:, 0]
        ds = s[1] - s[0]
        dby = _np.diff(by)
        dbx = _np.diff(bx)
        dby_ds = _np.append(dby, dby[-1]) / ds
        dbx_ds = _np.append(dbx, dbx[-1]) / ds
        kx = -cumtrapz(dx=ds, y=by * dby_ds, initial=0) / (brho) ** 2
        ky = -cumtrapz(dx=ds, y=-bx * dbx_ds, initial=0) / (brho) ** 2
        etax, etay, *_ = self.calc_dispersion()
        i4x = _np.trapz(
            dx=ds,
            y=etax * (by / brho) ** 3 - 2 * ky * etax * (by / brho),
        )
        i4y = _np.trapz(
            dx=ds,
            y=etay * (bx / brho) ** 3 - 2 * kx * etay * (bx / brho),
        )
        self._i4x = i4x
        self._i4y = i4y
        return i4x, i4y

    def calc_i5(self):
        """Calculate fifth radiation integral contribution from one ID.

        Returns:
            float: delta I5x
            float: delta I5y
        """
        brho = self.beam.brho
        by = self.field_profile[:, 1]
        bx = self.field_profile[:, 2]
        s = 1e-3 * self.field_profile[:, 0]
        ds = s[1] - s[0]
        hx, hy = self.get_curly_h()
        i5x = _np.trapz(dx=ds, y=hx * _np.abs((by / brho) ** 3))
        i5y = _np.trapz(dx=ds, y=hy * _np.abs((bx / brho) ** 3))
        self._i5x = i5x
        self._i5y = i5y
        return i5x, i5y

    def _calc_field_integral(self, a, *args):
        peak = args[0]
        period = args[1]
        nr_periods = args[2]
        pts_period = args[3]
        x_out, out = self._generate_field(
            a, peak, period, nr_periods, pts_period
        )
        ds = _np.diff(x_out)[0]
        i1 = cumtrapz(dx=ds, y=-out)
        i2 = _np.trapz(dx=ds, y=i1)
        return _np.abs(i2)

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

        tanh = _np.tanh(a * 2 * _np.pi / period * x_1period)
        if nr_periods % 2 == 1:
            term0 = (tanh + 1) / 2
            term1 = (-tanh + 1) / 2
        else:
            term0 = -(tanh + 1) / 2
            term1 = (tanh - 1) / 2

        out = _np.concatenate((term0 * y, mid, term1 * y))
        x_out = _np.linspace(
            -(2 + nr_periods) * period / 2,
            (2 + nr_periods) * period / 2,
            1 * (2 + nr_periods) * len(x_1period),
        )

        return x_out, out


class EqParamAnalysis:
    """Class to calculate beam equilibrium parameters.

    All equations used here are from the book: Clarke, James A.,
    'The Science and Technology of Undulators and Wigglers'.
    Chapter 10: 'The Effect of Insertion Devices on the Electron Beam'.
    """

    def __init__(self, beam_energy=3.0):
        """Class constructor.

        Args:
            beam_energy (Float, optional): Beam energy [GeV]. Defaults to 3.0.
        """
        self.ids = list()
        self.beam = Beam(beam_energy)
        self._eq_params_nominal = None
        self._emitx = None
        self._emity = None
        self._espread = None
        self._taux = None
        self._tauy = None
        self._taue = None
        self._u0 = None

        global _twiss
        global _mid_subsections
        if _twiss is None:
            mia = pyaccel.lattice.find_indices(_model, "fam_name", "mia")
            mib = pyaccel.lattice.find_indices(_model, "fam_name", "mib")
            mip = pyaccel.lattice.find_indices(_model, "fam_name", "mip")
            _mid_subsections = _np.sort(_np.array(mia + mib + mip))
            _twiss, *_ = pyaccel.optics.calc_twiss(_model, indices="open")

    @staticmethod
    def set_model(model):
        """Set Sirius model.

        Args:
            model (Pymodels object): Sirius model
        """
        global _model
        global _mid_subsections
        global _twiss
        _model = model
        mia = pyaccel.lattice.find_indices(_model, "fam_name", "mia")
        mib = pyaccel.lattice.find_indices(_model, "fam_name", "mib")
        mip = pyaccel.lattice.find_indices(_model, "fam_name", "mip")
        _mid_subsections = _np.sort(_np.array(mia + mib + mip))
        _twiss, *_ = pyaccel.optics.calc_twiss(_model, indices="open")

    @property
    def eq_params_nominal(self):
        """Equilibrium parameters from nominal model.

        Returns:
            EqParamFromRadIntegrals object
        """
        return self._eq_params_nominal

    @property
    def emitx(self):
        """Cumulative horizontal emittances.

        Returns:
            numpy array: Horizontal emittance [m rad]
        """
        return self._emitx

    @property
    def emity(self):
        """Cumulative vertical emittances.

        Returns:
            numpy array: Vertical emittance [m rad]
        """
        return self._emity

    @property
    def espread(self):
        """Cumulative energy spread.

        Returns:
            numpy array: Energy spread
        """
        return self._espread

    @property
    def taux(self):
        """Cumulative horizontal damping time.

        Returns:
            numpy array: Horizontal damping time [s]
        """
        return self._taux

    @property
    def tauy(self):
        """Cumulative vertical damping time.

        Returns:
            numpy array: Vertical damping time [s]
        """
        return self._tauy

    @property
    def taue(self):
        """Cumulative longitudinal damping time.

        Returns:
            numpy array: longitudinal damping time [s]
        """
        return self._taue

    @property
    def u0(self):
        """Cumulative energy loss per turn.

        Returns:
            numpy array: Energy loss [keV]
        """
        return self._u0

    def get_nominal_model_eqparams(self):
        """Get nominal model equilibrium parameters.

        Returns:
            EqParamsFromRadIntegrals object: Object containing Eqparams
        """
        model = pyaccel.lattice.refine_lattice(
            _model, max_length=0.01, fam_names=["BC", "B1", "B2", "QN"]
        )
        eqparam_nom = pyaccel.optics.EqParamsFromRadIntegrals(model)
        self._eq_params_nominal = eqparam_nom
        return eqparam_nom

    def calc_id_effect_on_emittance(self, insertion_device=None, *args):
        """Calculate ID's effect on emittances.

        Args:
            insertion_device (InsertionParams object): Object from class
             InsertionParams. Defaults to None
            *args (floats): Delta in radiation integrals.
            The order of args must be the following:
            args[0] = i2x, args[1] = i2y
            args[2] = i4x, args[3] = i4y
            args[4] = i5x, args[5] = i5y


        Returns:
            Float: Horizontal emittance [m rad]
            Float: Vertical emittance [m rad]
        """
        gamma = self.beam.gamma
        if (
            insertion_device is not None
            and insertion_device.field_profile is None
        ):
            insertion_device.create_field_profile()
        if self.eq_params_nominal is None:
            self.get_nominal_model_eqparams()

        i5x = self.eq_params_nominal.I5x
        i5y = self.eq_params_nominal.I5y

        i4x = self.eq_params_nominal.I4x
        i4y = self.eq_params_nominal.I4y

        i2x = self.eq_params_nominal.I2
        i2y = self.eq_params_nominal.I2

        if len(args) == 0:
            di2x, di2y = insertion_device.calc_i2()
            di4x, di4y = insertion_device.calc_i4()
            di5x, di5y = insertion_device.calc_i5()
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

        emitx = CQ * gamma**2 * (i5x / (i2x - i4x))
        emity = CQ * gamma**2 * (i5y / (i2y - i4y))

        return emitx, emity

    def calc_id_effect_on_espread(self, insertion_device=None, *args):
        """Calculate ID's effect on energy spread.

        Args:
            insertion_device (InsertionParams object): Object from class
             InsertionParams. Defaults to None.
            *args (floats): Delta in radiation integrals.
            The order of args must be the following:
            args[0] = i2
            args[1] = i3
            args[2] = i4

        Returns:
            float: energy spread
        """
        gamma = self.beam.gamma
        if (
            insertion_device is not None
            and insertion_device.field_profile is None
        ):
            insertion_device.create_field_profile()
        if self.eq_params_nominal is None:
            self.get_nominal_model_eqparams()

        i2 = self.eq_params_nominal.I2
        i3 = self.eq_params_nominal.I3
        i4 = self.eq_params_nominal.I4x

        if len(args) == 0:
            di2, _ = insertion_device.calc_i2()
            di3, _ = insertion_device.calc_i3()
            di4, _ = insertion_device.calc_i4()
        else:
            di2 = args[0]
            di3 = args[1]
            di4 = args[2]

        i2 += di2
        i3 += di3
        i4 += di4

        espread = _np.sqrt(CQ * gamma**2 * (i3 / (2 * i2 + i4)))

        return espread

    def calc_id_effect_on_damping_times(self, insertion_device=None, *args):
        """Calculate ID's effect on damping times.

        Args:
            insertion_device (InsertionParams object): Object from class
             InsertionParams. Defaults to None.
            *args (floats): Delta in radiation integrals.
            The order of args must be the following:
            args[0] = i2
            args[1] = i4
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
            insertion_device.create_field_profile()
        if self.eq_params_nominal is None:
            self.get_nominal_model_eqparams()

        i2 = self.eq_params_nominal.I2
        i4 = self.eq_params_nominal.I4x
        u0 = self.eq_params_nominal.U0 * ECHARGE

        if len(args) == 0:
            di2, _ = insertion_device.calc_i2()
            di4, _ = insertion_device.calc_i4()
            du0 = 1e-3 * self.calc_u0(insertion_device) * ECHARGE
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

        energy = self.beam.energy * 1e9 * ECHARGE
        t0 = _model.length / LSPEED

        taux = 2 * energy * t0 / (u0 * jx)
        tauy = 2 * energy * t0 / (u0 * jy)
        taue = 2 * energy * t0 / (u0 * jz)

        return taux, tauy, taue

    def add_id_to_lattice(self, insertion_device):
        """Add ID to lattice.

        Args:
            insertion_device (InsertionParams object): Object from class
             InsertionParams.
        """
        self.ids.append(insertion_device)

    def calc_total_effect_on_eqparams(self):
        """Calculate cumulative effect on equilibrium parameters.

        Returns:
            numpy 1d array: Horizontal emittance [m rad]
            numpy 1d array: Vertical emittance [m rad]
            numpy 1d array: Energy spread
            numpy 1d array: Horizontal damping time [s]
            numpy 1d array: Vertical damping time [s]
            numpy 1d array: Longitudinal damping time [s]
        """
        emitx = _np.zeros(len(self.ids) + 1)
        emity = _np.zeros(len(self.ids) + 1)
        espread = _np.zeros(len(self.ids) + 1)
        taux = _np.zeros(len(self.ids) + 1)
        tauy = _np.zeros(len(self.ids) + 1)
        taue = _np.zeros(len(self.ids) + 1)
        u0 = _np.zeros(len(self.ids) + 1)

        if self.eq_params_nominal is None:
            self.get_nominal_model_eqparams()
        emitx[0] = self.eq_params_nominal.emitx
        emity[0] = self.eq_params_nominal.emity
        espread[0] = self.eq_params_nominal.espread0
        taux[0] = self.eq_params_nominal.taux
        tauy[0] = self.eq_params_nominal.tauy
        taue[0] = self.eq_params_nominal.taue
        u0[0] = 1e-3 * self.eq_params_nominal.U0

        di2x, di2y = 0, 0
        di4x, di4y = 0, 0
        di5x, di5y = 0, 0
        di3, du0 = 0, 0
        for i, insertion_device in enumerate(self.ids):
            if insertion_device.field_profile is None:
                insertion_device.create_field_profile()
            di2x_, di2y_ = insertion_device.calc_i2()
            di4x_, di4y_ = insertion_device.calc_i4()
            di5x_, di5y_ = insertion_device.calc_i5()
            di3_, _ = insertion_device.calc_i3()
            du0_ = 1e-3 * insertion_device.calc_u0() * ECHARGE
            di2x += di2x_
            di2y += di2y_
            di4x += di4x_
            di4y += di4y_
            di5x += di5x_
            di5y += di5y_
            di3 += di3_
            du0 += du0_
            emitx[i + 1], emity[i + 1] = self.calc_id_effect_on_emittance(
                None, di2x, di2y, di4x, di4y, di5x, di5y
            )
            espread[i + 1] = self.calc_id_effect_on_espread(
                None, di2x, di3, di4x
            )
            (
                taux[i + 1],
                tauy[i + 1],
                taue[i + 1],
            ) = self.calc_id_effect_on_damping_times(None, di2x, di4x, du0)
            u0[i + 1] = u0[i] + 1e3 * du0_ / ECHARGE
        self._emitx = emitx
        self._emity = emity
        self._espread = espread
        self._taux = taux
        self._tauy = tauy
        self._taue = taue
        self._u0 = u0
        return

    def plot_ids_effects_emit_espread(self):
        """Plot ID's effects on horizontal emittance and energy spread."""
        self.calc_total_effect_on_eqparams()
        emitx = self.emitx
        energy_spread = self.espread

        fig, ax1 = _plt.subplots(1, sharey=True)
        fig.set_figwidth(9)
        fig.set_figheight(5)

        ax1.tick_params(axis="x", rotation=65, labelsize=12)
        ax1.tick_params(axis="y", labelsize=13.5, labelcolor="C0")

        ids_list = list()
        ids_list.append("BARE")

        for insertion_device in self.ids:
            id_name = insertion_device.fam_name
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


class SiriusIDS:
    """Class with some Sirius configurations regarding the IDs."""

    def __init__(self):
        """Class constructor."""
        self.ids = list()

    def set_current_ids(self):
        """Create all current Sirius IDs (01/02/2024)."""
        ids = list()

        id1 = InsertionParams()
        id1.fam_name = "APU22"
        id1.period = 22
        id1.by_peak = 0.70
        id1.nr_periods = 51
        id1.straight_section = "ID06SB"
        ids.append(id1)

        id2 = InsertionParams()
        id2.fam_name = "APU22"
        id2.period = 22
        id2.by_peak = 0.70
        id2.nr_periods = 51
        id2.straight_section = "ID07SP"
        ids.append(id2)

        id3 = InsertionParams()
        id3.fam_name = "APU22"
        id3.period = 22
        id3.by_peak = 0.70
        id3.nr_periods = 51
        id3.straight_section = "ID08SB"
        ids.append(id3)

        id4 = InsertionParams()
        id4.fam_name = "APU22"
        id4.period = 22
        id4.by_peak = 0.70
        id4.nr_periods = 51
        id4.straight_section = "ID09SA"
        ids.append(id4)

        id5 = InsertionParams()
        id5.fam_name = "APU58"
        id5.period = 58
        id5.by_peak = 0.95
        id5.nr_periods = 18
        id5.straight_section = "ID11SB"
        ids.append(id5)

        id6 = InsertionParams()
        id6.fam_name = "PAPU50"
        id6.period = 50
        id6.by_peak = 0.42
        id6.nr_periods = 18
        id6.straight_section = "ID17SA"
        ids.append(id6)

        id7 = InsertionParams()
        id7.fam_name = "WIG180"
        id7.period = 180
        id7.by_peak = 1.0
        id7.nr_periods = 13
        id7.straight_section = "ID14SB"
        ids.append(id7)

        id8 = InsertionParams()
        id8.fam_name = "DELTA52"
        id8.period = 52.5
        id8.by_peak = 1.25
        id8.bx_peak = 1.25
        id8.nr_periods = 21
        id8.straight_section = "ID10SB"
        ids.append(id8)

        self.ids = ids
        return ids

    def set_phase1_ids(self):
        """Create all Sirius Phase I Ids."""
        ids = list()

        id1 = InsertionParams()
        id1.fam_name = "VPU29"
        id1.period = 29
        id1.bx_peak = 0.82
        id1.nr_periods = 51
        id1.straight_section = "ID06SB"
        ids.append(id1)

        id2 = InsertionParams()
        id2.fam_name = "VPU29"
        id2.period = 29
        id2.bx_peak = 0.82
        id2.nr_periods = 51
        id2.straight_section = "ID07SP"
        ids.append(id2)

        id3 = InsertionParams()
        id3.fam_name = "IVU18"
        id3.period = 18.5
        id3.by_peak = 1.22
        id3.nr_periods = 108
        id3.straight_section = "ID08SB"
        ids.append(id3)

        id4 = InsertionParams()
        id4.fam_name = "APU22"
        id4.period = 22
        id4.by_peak = 0.70
        id4.nr_periods = 2 * 51
        id4.straight_section = "ID09SA"
        ids.append(id4)

        # This ID can be changed by an APPLE-II, but there is no
        # specification by the moment.
        id5 = InsertionParams()
        id5.fam_name = "APU58"
        id5.period = 58
        id5.by_peak = 0.95
        id5.nr_periods = 18
        id5.straight_section = "ID11SB"
        ids.append(id5)

        id6 = InsertionParams()
        id6.fam_name = "APU22"
        id6.period = 22
        id6.by_peak = 0.70
        id6.nr_periods = 2 * 51
        id6.straight_section = "ID17SA"
        ids.append(id6)

        id7 = InsertionParams()
        id7.fam_name = "IVU18"
        id7.period = 18.5
        id7.by_peak = 1.22
        id7.nr_periods = 108
        id7.straight_section = "ID14SB"
        ids.append(id7)

        id8 = InsertionParams()
        id8.fam_name = "DELTA52"
        id8.period = 52.5
        id8.by_peak = 1.25
        id8.bx_peak = 1.25
        id8.nr_periods = 21
        id8.straight_section = "ID10SB"
        ids.append(id8)

        self.ids = ids
        return ids
