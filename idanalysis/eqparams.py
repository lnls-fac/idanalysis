"""IDs analysis."""

import mathphys
import matplotlib.pyplot as _plt
import numpy as _np
import pyaccel
import pymodels
from fieldmaptrack import Beam
from scipy.integrate import cumtrapz
from scipy.optimize import minimize


class InsertionParams:
    """Class to specify insertion parameters.

     All equations used here are from the book: Clarke, James A.,
    'The Science and Technology of Undulators and Wigglers'.
    Chapter 10: 'The Effect of Insertion Devices on the Electron Beam'.
    """

    def __init__(self):
        """Class constructor."""
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
        self.brho = Beam(3).brho
        self._e = mathphys.constants.elementary_charge
        self._m = mathphys.constants.electron_mass
        self._c = mathphys.constants.light_speed

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
        x_out, out = InsertionParams._generate_field(
            a, peak, period, nr_periods, pts_period
        )
        dx = _np.diff(x_out)[0]
        i1 = cumtrapz(dx=dx, y=-out)
        i2 = cumtrapz(dx=dx, y=i1)
        return _np.abs(i2[-1])

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
            x, by = self._generate_field(
                result.x, by_peak, period, nr_periods, pts_period
            )
            field[:, 1] = by

        if bx_peak is not None:
            result = minimize(
                self._calc_field_integral,
                0.4,
                args=(bx_peak, period, nr_periods, pts_period),
            )
            x, bx = self._generate_field(
                result.x, bx_peak, period, nr_periods, pts_period
            )
            field[:, 2] = bx

        field[:, 0] = x
        self.field_profile = field
        return field

    def calc_dispersion(self):
        """Calculate dispersion generated by the ID.

        Returns:
            numpy array: Horizontal dispersion on the ID
            numpy array: Vertical dispersion on the ID
        """
        x = 1e-3 * self.field_profile[:, 0]
        by = self.field_profile[:, 1]
        bx = self.field_profile[:, 2]
        dx = _np.diff(x)[0]
        i1x = cumtrapz(dx=dx, y=by)
        i2x = cumtrapz(dx=dx, y=i1x)
        i1y = cumtrapz(dx=dx, y=-bx)
        i2y = cumtrapz(dx=dx, y=i1y)
        etax = -1 * i2x / self.brho
        etay = -1 * i2y / self.brho
        self._etax = etax
        self._etay = etay
        return etax, etay

    def calc_dispersion_derivative(self):
        """Calculate the derivative of dispersion generated by the ID.

        Returns:
            numpy array: d(etax)/ds on the ID
            numpy array: d(etay)/ds on the ID
        """
        x = 1e-3 * self.field_profile[:, 0]
        by = self.field_profile[:, 1]
        bx = self.field_profile[:, 2]
        dx = _np.diff(x)[0]
        i1x = cumtrapz(dx=dx, y=by)
        i1y = cumtrapz(dx=dx, y=-bx)
        etapx = -1 * i1x / self.brho
        etapy = -1 * i1y / self.brho
        self._etapx = etapx
        self._etapy = etapy
        return etapx, etapy

    def get_curly_h(self):
        """Calculate curly H in ID section.

        Returns:
            numpy 1d array: Curly Hx in the ID region.
            numpy 1d array: Curly Hy in the ID region.
        """
        x = 1e-3 * self.field_profile[:, 0]
        if len(x) % 2 == 0:
            length = (x - x[0])[:int(len(x)/2)]
        else:
            length = (x - x[0])[:int((len(x)+1)/2)]
        subsec = self.straight_section
        si = pymodels.si.create_accelerator()
        mia = pyaccel.lattice.find_indices(si, "fam_name", "mia")
        mib = pyaccel.lattice.find_indices(si, "fam_name", "mib")
        mip = pyaccel.lattice.find_indices(si, "fam_name", "mip")
        mid_subsections = _np.sort(_np.array(mia + mib + mip))
        idx = mid_subsections[int(subsec[2:4]) - 1]
        twiss, *_ = pyaccel.optics.calc_twiss(si, indices="open")

        betax0 = twiss.betax[idx]
        betay0 = twiss.betay[idx]

        alphax0 = twiss.alphax[idx]
        alphay0 = twiss.alphay[idx]

        gammax0 = twiss.gammax[idx]
        gammay0 = twiss.gammay[idx]

        betax_after = betax0 - 2*length*alphax0 + length**2*gammax0
        betax_before = betax0 + 2*length*alphax0 + length**2*gammax0

        alphax_after = alphax0 - length*gammax0
        alphax_before = alphax0 + length*gammax0

        betay_after = betay0 - 2*length*alphay0 + length**2*gammay0
        betay_before = betay0 + 2*length*alphay0 + length**2*gammay0

        alphay_after = alphay0 - length*gammay0
        alphay_before = alphay0 + length*gammay0
        if len(x) % 2 != 0 :
            betax_after = betax_after[1:]
            alphax_after = alphax_after[1:]
            betay_after = betay_after[1:]
            alphay_after = alphay_after[1:]

        betax = _np.concatenate((betax_before[::-1], betax_after))
        alphax = _np.concatenate((alphax_before[::-1], alphax_after))
        betay = _np.concatenate((betay_before[::-1], betay_after))
        alphay = _np.concatenate((alphay_before[::-1], alphay_after))

        etax, etay = self.calc_dispersion()
        etapx, etapy = self.calc_dispersion_derivative()

        hx = pyaccel.optics.get_curlyh(
                betax[1:-1], alphax[1:-1], etax, etapx[:-1]
            )

        hy = pyaccel.optics.get_curlyh(
                betay[1:-1], alphay[1:-1], etay, etapy[:-1]
            )

        return hx, hy

    def calc_u0(self):
        """Calculate energy loss in ID.

        Returns:
            float: Energy loss in [keV]
        """
        x = 1e-3 * self.field_profile[:, 0]
        dx = _np.diff(x)[0]
        by = self.field_profile[:, 1]
        bx = self.field_profile[:, 2]
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
        by = self.field_profile[:, 1]
        bx = self.field_profile[:, 2]
        x = 1e-3 * self.field_profile[:, 0]
        dx = _np.diff(x)[0]
        etax, etay = self.calc_dispersion()
        i1x = cumtrapz(dx=dx, y=etax * by[1:-1] / self.brho)[-1]
        i1y = cumtrapz(dx=dx, y=etay * bx[1:-1] / self.brho)[-1]
        self._i1x = i1x
        self._i1y = i1y
        return i1x, i1y

    def calc_i2(self):
        """Calculate second radiation integral contribution from one ID.

        Returns:
            float: delta I2x
            float: delta I2y
        """
        by = self.field_profile[:, 1]
        bx = self.field_profile[:, 2]
        x = 1e-3 * self.field_profile[:, 0]
        dx = _np.diff(x)[0]
        i2x = cumtrapz(dx=dx, y=(by / self.brho) ** 2)[-1]
        i2y = cumtrapz(dx=dx, y=(bx / self.brho) ** 2)[-1]
        self._i2x = i2x
        self._i2y = i2y
        return i2x, i2y

    def calc_i3(self):
        """Calculate third radiation integral contribution from one ID.

        Returns:
            float: delta I3x
            float: delta I3y
        """
        by = self.field_profile[:, 1]
        bx = self.field_profile[:, 2]
        x = 1e-3 * self.field_profile[:, 0]
        dx = _np.diff(x)[0]
        i3x = cumtrapz(dx=dx, y=_np.abs((by / self.brho) ** 3))[-1]
        i3y = cumtrapz(dx=dx, y=_np.abs((bx / self.brho) ** 3))[-1]
        self._i3x = i3x
        self._i3y = i3y
        return i3x, i3y

    def calc_i4(self):
        """Calculate fourth radiation integral contribution from one ID.

        Returns:
            float: delta I4x
            float: delta I4y
        """
        by = self.field_profile[:, 1]
        bx = self.field_profile[:, 2]
        x = 1e-3 * self.field_profile[:, 0]
        dx = _np.diff(x)[0]
        dby = _np.diff(by)
        dbx = _np.diff(bx)
        px = cumtrapz(dx=dx, y=by) / self.brho
        py = cumtrapz(dx=dx, y=-bx) / self.brho
        kx = -(dby / dx) * px / self.brho
        ky = -(dbx / dx) * py / self.brho
        etax, etay = self.calc_dispersion()
        i4x = cumtrapz(
            dx=dx,
            y=(
                etax * (by[1:-1] / self.brho) ** 3
                - 2 * ky[:-1] * etax * (by[1:-1] / self.brho)
            ),
        )[-1]
        i4y = cumtrapz(
            dx=dx,
            y=(
                etay * (bx[1:-1] / self.brho) ** 3
                - 2 * kx[:-1] * etay * (bx[1:-1] / self.brho)
            ),
        )[-1]
        self._i4x = i4x
        self._i4y = i4y
        return i4x, i4y

    def calc_i5(self):
        """Calculate fifth radiation integral contribution from one ID.

        Returns:
            float: delta I5x
            float: delta I5y
        """
        by = self.field_profile[:, 1]
        bx = self.field_profile[:, 2]
        x = 1e-3 * self.field_profile[:, 0]
        dx = _np.diff(x)[0]
        hx, hy = self.get_curly_h()
        i5x = cumtrapz(
            dx=dx, y=hx * _np.abs((by / self.brho) ** 3)[1:-1]
        )[-1]
        i5y = cumtrapz(
            dx=dx, y=hy * _np.abs((bx / self.brho) ** 3)[1:-1]
        )[-1]
        self._i5x = i5x
        self._i5y = i5y
        return i5x, i5y


class EqParamAnalysis:
    """Class to calculate beam equilibrium parameters.

    All equations used here are from the book: Clarke, James A.,
    'The Science and Technology of Undulators and Wigglers'.
    Chapter 10: 'The Effect of Insertion Devices on the Electron Beam'.
    """

    def __init__(self):
        """Class constructor."""
        self.ids = list()
        self._eq_params_nominal = None
        self._beam = Beam(3)
        self.brho = self._beam.brho
        self.gamma = self._beam.gamma
        self._emitx = None
        self._emity = None
        self._espread = None
        self._taux = None
        self._tauy = None
        self._taue = None
        self._u0 = None

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
        return self._emitx

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
            args[0] = i2x, args[1] = i2y
            args[2] = i4x, args[3] = i4y
            args[4] = i5x, args[5] = i5y


        Returns:
            Float: Horizontal emittance [m rad]
            Float: Vertical emittance [m rad]
        """
        if (
            insertion_device is not None
            and insertion_device.field_profile is None
        ):
            insertion_device.create_field_profile()
        if self.eq_params_nominal is None:
            self.get_nominal_model_eqparams(model)

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

        emitx = (
            mathphys.constants.Cq * self.gamma ** 2 * (i5x / (i2x - i4x))
        )
        emity = (
            mathphys.constants.Cq * self.gamma ** 2 * (i5y / (i2y - i4y))
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
            args[0] = i2
            args[1] = i3
            args[2] = i4

        Returns:
            float: energy spread
        """
        if (
            insertion_device is not None
            and insertion_device.field_profile is None
        ):
            insertion_device.create_field_profile()
        if self.eq_params_nominal is None:
            self.get_nominal_model_eqparams(model)

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

        espread = _np.sqrt(
            mathphys.constants.Cq * self.gamma ** 2 * (i3 / (2 * i2 + i4))
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
            self.get_nominal_model_eqparams(model)
        if model is None:
            si = pymodels.si.create_accelerator()

        i2 = self.eq_params_nominal.I2
        i4 = self.eq_params_nominal.I4x
        u0 = self.eq_params_nominal.U0 * mathphys.constants.elementary_charge

        if len(args) == 0:
            di2, _ = insertion_device.calc_i2()
            di4, _ = insertion_device.calc_i4()
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
                insertion_device.create_field_profile()
            di2x_, di2y_ = insertion_device.calc_i2()
            di4x_, di4y_ = insertion_device.calc_i4()
            di5x_, di5y_ = insertion_device.calc_i5()
            di2x += di2x_
            di2y += di2y_
            di4x += di4x_
            di4y += di4y_
            di5x += di5x_
            di5y += di5y_
            emitx[i + 1], emity[i + 1] = self.calc_id_effect_on_emittance(
                None, model, di2x, di2y, di4x, di4y, di5x, di5y
            )
        self._emitx = emitx
        self._emity = emity
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
                insertion_device.create_field_profile()
            di2_, _ = insertion_device.calc_i2()
            di3_, _ = insertion_device.calc_i3()
            di4_, _ = insertion_device.calc_i4()
            di2 += di2_
            di3 += di3_
            di4 += di4_
            espread[i + 1] = self.calc_id_effect_on_espread(
                None, model, di2, di3, di4
            )
        self._espread = espread
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
                insertion_device.create_field_profile()
            di2_, _ = insertion_device.calc_i2()
            di4_, _ = insertion_device.calc_i4()
            du0_ = (
                1e-3
                * insertion_device.calc_u0()
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
        self._taux = taux
        self._tauy = tauy
        self._taue = taue
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
                insertion_device.create_field_profile()
            du0 = insertion_device.calc_u0()
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
