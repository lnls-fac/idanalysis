"""IDs RADIA models."""

import numpy as np
from imaids.models import (
    AppleIISabia,
    DeltaSabia,
    HybridPlanar,
    Kyma22,
    Kyma58,
    MiniPlanarSabia,
    PAPU,
    AppleII,
)

from idanalysis.analysis import RadiaModelAnalysis


class DELTA52(RadiaModelAnalysis):
    """DELTA52 RADIA model creation and manipulation."""

    def __init__(self):
        """Class constructor."""
        self.nr_periods = 21
        self.period_length = 52.5
        self.gap = 13.6
        self.mr = 1.39
        self.longitudinal_distance = 0.125
        self.model = None

    def generate_radia_model(self, solve=False, dp=0, dgv=0, nr_periods=None):
        """Generate DELTA52 RADIA model.

        Args:
            solve (bool, optional): Solve ID using relaxation method.
            Defaults to False.
            dp (float, optional): ID Pparam. Defaults to 0.
            dgv (float, optional): ID Kparam. Defaults to 0.
            nr_periods (int, optional): nr_periods. Defaults to None.

        Returns:
            _RADIA object_: DELTA52 RADIA model
        """
        nr_periods = nr_periods if nr_periods is not None else self.nr_periods
        period_length = self.period_length
        gap = self.gap
        mr = self.mr
        longitudinal_distance = self.longitudinal_distance
        delta = DeltaSabia(
            nr_periods=nr_periods,
            period_length=period_length,
            gap=gap,
            mr=mr,
            longitudinal_distance=longitudinal_distance,
        )

        delta.set_cassete_positions(dp=dp, dgv=dgv)

        if solve:
            delta.solve()

        self.model = delta
        return delta


class EPU50(RadiaModelAnalysis):
    """EPU50 RADIA model creation and manipulation."""

    def __init__(self):
        """Class constructor."""
        self.nr_periods = 54
        self.period_length = 50
        self.gap = 22
        self.mr = 1.25
        self.longitudinal_distance = 0.2
        self.block_width = 40
        self.cassettes_gap = 0.1
        self.model = None

    def generate_radia_model(
        self, solve=False, phase=0, gap=None, nr_periods=None
    ):
        """Generate EPU50 RADIA model.

        Args:
            solve (bool, optional): Solve ID using relaxation method.
            Defaults to False.
            phase (float, optional): ID Pparam. Defaults to 0.
            gap (float, optional): ID Kparam. Defaults to None.
            nr_periods (int, optional): nr_periods. Defaults to None.

        Returns:
            _RADIA object_: EPU50 RADIA model
        """
        gap = gap if gap is not None else self.gap
        nr_periods = nr_periods if nr_periods is not None else self.nr_periods
        period_length = self.period_length
        cassettes_gap = self.cassettes_gap
        width = self.block_width
        mr = self.mr
        longitudinal_distance = self.longitudinal_distance
        block_shape = [
            [
                [cassettes_gap, 0],
                [width + cassettes_gap, 0],
                [width + cassettes_gap, -width],
                [cassettes_gap, -width],
            ]
        ]
        block_len = period_length / 4 - longitudinal_distance
        start_lengths = [
            block_len / 4,
            block_len / 2,
            3 * block_len / 4,
            block_len,
        ]
        start_distances = [
            block_len / 2,
            block_len / 4,
            0,
            longitudinal_distance,
        ]
        end_lenghts = start_lengths[-2::-1]
        end_distances = start_distances[-2::-1]
        epu = AppleIISabia(
            gap=gap,
            nr_periods=nr_periods,
            period_length=period_length,
            mr=mr,
            block_shape=block_shape,
            start_blocks_length=start_lengths,
            start_blocks_distance=start_distances,
            end_blocks_length=end_lenghts,
            end_blocks_distance=end_distances,
        )

        epu.dp = phase
        if solve:
            epu.solve()

        self.model = epu
        return epu


class APU22(RadiaModelAnalysis):
    """KYMA22 RADIA model creation and manipulation."""

    def __init__(self):
        """Class constructor."""
        self.nr_periods = 10
        self.period_length = 22
        self.gap = 8.0
        self.mr = 1.32
        self.longitudinal_distance = 0.1
        self.model = None

    def generate_radia_model(self, solve=False, phase=0, nr_periods=None):
        """Generate Kyma22 RADIA model.

        Args:
            solve (bool, optional): Solve ID using relaxation method.
            Defaults to False.
            phase (float, optional): ID Kparam. Defaults to 0.
            nr_periods (int, optional): nr_periods. Defaults to None.

        Returns:
            _RADIA object_: KYMA22 RADIA model
        """
        nr_periods = nr_periods if nr_periods is not None else self.nr_periods
        kyma = Kyma22(nr_periods=nr_periods)
        kyma.dg = phase

        if solve:
            kyma.solve()

        self.model = kyma
        return kyma


class APU58(RadiaModelAnalysis):
    """KYMA58 RADIA model creation and manipulation."""

    def __init__(self):
        """Class constructor."""
        self.nr_periods = 18
        self.period_length = 58
        self.gap = 15.8
        self.mr = 1.32
        self.longitudinal_distance = 0.1
        self.model = None

    def generate_radia_model(self, solve=False, phase=0, nr_periods=None):
        """Generate Kyma58 RADIA model.

        Args:
            solve (bool, optional): Solve ID using relaxation method.
            Defaults to False.
            phase (float, optional): ID Kparam. Defaults to 0.
            nr_periods (int, optional): nr_periods. Defaults to None.

        Returns:
            _RADIA object_: KYMA58 RADIA model
        """
        nr_periods = nr_periods if nr_periods is not None else self.nr_periods
        kyma = Kyma58(nr_periods=nr_periods)
        kyma.dg = phase

        if solve:
            kyma.solve()

        self.model = kyma
        return kyma


class PAPU50(RadiaModelAnalysis):
    """PAPU50 RADIA model creation and manipulation."""

    def __init__(self):
        """Class constructor."""
        self.nr_periods = 18
        self.period_length = 50
        self.gap = 24.0
        self.mr = 1.22
        self.longitudinal_distance = 0.2
        self.model = None

    def generate_radia_model(self, solve=False, phase=0, nr_periods=None):
        """Generate PAPU50 RADIA model.

        Args:
            solve (bool, optional): Solve ID using relaxation method.
            Defaults to False.
            phase (float, optional): ID Kparam. Defaults to 0.
            nr_periods (int, optional): nr_periods. Defaults to None.

        Returns:
            _RADIA object_: PAPU50 RADIA model
        """
        nr_periods = nr_periods if nr_periods is not None else self.nr_periods
        papu = PAPU(nr_periods=nr_periods)
        papu.dg = phase

        if solve:
            papu.solve()

        self.model = papu
        return papu


class WIG180(RadiaModelAnalysis):
    """WIG180 RADIA model creation and manipulation."""

    def __init__(self):
        """Class constructor."""
        self.nr_periods = 13
        self.period_length = 180
        self.gap = 49.73
        self.mr = 1.25
        self.longitudinal_distance = 0.125
        self.block_height = 40
        self.block_width = 80
        self.model = None

    def generate_radia_model(self, solve=False, gap=None, nr_periods=None):
        """Generate Wiggler 180 RADIA model.

        Args:
            solve (bool, optional): Solve ID using relaxation method.
            Defaults to False.
            gap (float, optional): ID Kparam. Defaults to None.
            nr_periods (int, optional): nr_periods. Defaults to None.

        Returns:
            _RADIA object_: Wiggler180 RADIA model
        """
        gap = gap if gap is not None else self.gap
        nr_periods = nr_periods if nr_periods is not None else self.nr_periods
        period_length = self.period_length
        longitudinal_distance = self.longitudinal_distance
        height = self.block_height
        width = self.block_width
        mr = self.mr

        block_shape = [
            [-width / 2, 0],
            [-width / 2, -height],
            [width / 2, -height],
            [width / 2, 0],
        ]
        block_subdivision = [3, 3, 3]

        block_len = period_length / 4 - longitudinal_distance
        start_lengths = [
            block_len / 4,
            block_len / 2,
            3 * block_len / 4,
            block_len,
        ]
        start_distances = [
            block_len / 2,
            block_len / 4,
            0,
            longitudinal_distance,
        ]
        end_lenghts = start_lengths[-2::-1]
        end_distances = start_distances[-2::-1]
        wig = MiniPlanarSabia(
            gap=gap,
            nr_periods=nr_periods,
            period_length=period_length,
            mr=mr,
            block_shape=block_shape,
            block_subdivision=block_subdivision,
            start_blocks_length=start_lengths,
            start_blocks_distance=start_distances,
            end_blocks_length=end_lenghts,
            end_blocks_distance=end_distances,
        )

        if solve:
            wig.solve()

        self.model = wig
        return wig


class IVU18(RadiaModelAnalysis):
    """IVU18 RADIA model creation and manipulation."""

    def __init__(self):
        """Class constructor."""
        self.nr_periods = 11
        self.period_length = 18.5
        self.gap = 4.3
        self.mr = 1.24
        self.longitudinal_distance = 0
        self.block_width = 49
        self.block_height = 26
        self.pole_width = 34
        self.pole_height = 21
        self.pole_length = 2.8
        self.block_chamfer = 4
        self.pole_chamfer = 3
        self.model = None

    def generate_radia_model(self, solve=False, gap=None, nr_periods=None):
        """Generate IVU18 RADIA model.

        Args:
            solve (bool, optional): Solve ID using relaxation method.
            Defaults to False.
            gap (float, optional): ID Kparam. Defaults to None.
            nr_periods (int, optional): nr_periods. Defaults to None.

        Returns:
            _RADIA object_: IVU18 RADIA model
        """
        gap = gap if gap is not None else self.gap
        nr_periods = nr_periods if nr_periods is not None else self.nr_periods
        period_length = self.period_length
        mr = self.mr
        longitudinal_distance = self.longitudinal_distance
        b_width = self.block_width
        b_height = self.block_height
        p_width = self.pole_width
        p_height = self.pole_height
        chamfer_b = self.block_chamfer
        chamfer_p = self.pole_chamfer
        pole_length = self.pole_length

        block_shape = [
            [-b_width / 2, -chamfer_b],
            [-b_width / 2, -b_height + chamfer_b],
            [-b_width / 2 + chamfer_b, -b_height],
            [b_width / 2 - chamfer_b, -b_height],
            [b_width / 2, -b_height + chamfer_b],
            [b_width / 2, -chamfer_b],
            [b_width / 2 - chamfer_b, 0],
            [-b_width / 2 + chamfer_b, 0],
        ]

        pole_shape = [
            [-p_width / 2, -chamfer_p],
            [-p_width / 2, -p_height],
            [p_width / 2, -p_height],
            [p_width / 2, -chamfer_p],
            [p_width / 2 - chamfer_p, 0],
            [-p_width / 2 + chamfer_p, 0],
        ]

        block_subdivision = [8, 4, 4]
        pole_subdivision = [8, 5, 5]

        start_blocks_length = [1.9, 1.29, 5]
        start_blocks_distance = [2.4, 2.4, 0]

        end_blocks_length = [2.8, 6.45, 2.8, 5, 1.29, 1.9]
        end_blocks_distance = [0, 0, 0, 0, 2.4, 2.4]

        ivu = HybridPlanar(
            gap=gap,
            period_length=period_length,
            mr=mr,
            nr_periods=nr_periods,
            longitudinal_distance=longitudinal_distance,
            block_shape=block_shape,
            pole_shape=pole_shape,
            block_subdivision=block_subdivision,
            pole_subdivision=pole_subdivision,
            pole_length=pole_length,
            start_blocks_length=start_blocks_length,
            start_blocks_distance=start_blocks_distance,
            end_blocks_length=end_blocks_length,
            end_blocks_distance=end_blocks_distance,
            trf_on_blocks=True,
        )

        ivu.magnetization_dict["cs"]
        cs_mag = ivu.magnetization_dict["cs"]
        cs_mag[0] = [0.0, 0.0, 1.24]
        cs_mag[1] = [0.0, 0.0, 0.0]
        cs_mag[2] = [0.0, 0.0, -1.24]

        cs_mag[-4] = [0.0, 0.0, 0.0]
        cs_mag[-3] = [0.0, 0.0, -1.24]
        cs_mag[-2] = [0.0, 0.0, 0.0]
        cs_mag[-1] = [0.0, 0.0, +1.24]

        mags = np.array(cs_mag)
        cs_mag = mags.tolist()

        ci_mag = ivu.magnetization_dict["ci"]
        ci_mag[0] = [0.0, 0.0, -1.24]
        ci_mag[1] = [0.0, 0.0, 0.0]
        ci_mag[2] = [0.0, 0.0, 1.24]

        ci_mag[-4] = [0.0, 0.0, 0.0]
        ci_mag[-3] = [0.0, 0.0, +1.24]
        ci_mag[-2] = [0.0, 0.0, 0.0]
        ci_mag[-1] = [0.0, 0.0, -1.24]

        mags = np.array(ci_mag)
        ci_mag = mags.tolist()

        magnetization_dict = dict()
        magnetization_dict["cs"] = cs_mag
        magnetization_dict["ci"] = ci_mag

        start_blocks_pole_list = [False, True, False]
        end_blocks_pole_list = [True, False, True, False, True, False]
        core_blocks_pole_list = (
            2 * [True, False] * int(ivu.cassettes_ref["cs"].nr_core_blocks / 2)
        )
        is_pole_list = (
            start_blocks_pole_list
            + core_blocks_pole_list
            + end_blocks_pole_list
        )

        ivu.create_radia_object(
            magnetization_dict=magnetization_dict, is_pole_list=is_pole_list
        )

        if solve:
            ivu.solve()
        self.model = ivu
        return ivu


class VPU29(RadiaModelAnalysis):
    """VPU29 RADIA model creation and manipulation."""

    def __init__(self):
        """Class constructor."""
        self.nr_periods = 9
        self.period_length = 29
        self.gap = 9.7
        self.mr = 1.26
        self.longitudinal_distance = 0.1
        self.block_width = 55
        self.block_height = 55
        self.pole_width = 40
        self.pole_height_1 = 15.5
        self.pole_height_2 = 27.5
        self.pole_length = 4
        self.block_chamfer = 4
        self.pole_chamfer = 4
        self.model = None

    def generate_radia_model(self, solve=False, gap=None, nr_periods=None):
        """Generate VPU29 RADIA model.

        Args:
            solve (bool, optional): Solve ID using relaxation method.
            Defaults to False.
            gap (float, optional): ID Kparam. Defaults to None.
            nr_periods (int, optional): nr_periods. Defaults to None.

        Returns:
            _RADIA object_: VPU29 RADIA model
        """
        gap = gap if gap is not None else self.gap
        nr_periods = nr_periods if nr_periods is not None else self.nr_periods
        period_length = self.period_length
        mr = self.mr

        b_width = self.block_width
        b_height = self.block_height
        b_chamfer = self.block_chamfer

        p_width = self.pole_width
        p_height1 = self.pole_height_1
        p_height2 = self.pole_height_2
        p_chamfer = self.pole_chamfer
        pole_length = self.pole_length

        longitudinal_distance = self.longitudinal_distance

        block_shape = [
            [
                [-b_width / 2, -b_chamfer],
                [-b_width / 2, -b_height / 2 + b_chamfer],
                [-b_width / 2 + b_chamfer, -b_height / 2],
                [b_width / 2 - b_chamfer, -b_height / 2],
                [b_width / 2, -b_height / 2 + b_chamfer],
                [b_width / 2, -b_chamfer],
                [b_width / 2 - b_chamfer, 0],
                [-b_width / 2 + b_chamfer, 0],
            ],
            [
                [-b_width / 2, -b_chamfer - b_height / 2],
                [-b_width / 2, -b_height + b_chamfer],
                [-b_width / 2 + b_chamfer, -b_height],
                [b_width / 2 - b_chamfer, -b_height],
                [b_width / 2, -b_height + b_chamfer],
                [b_width / 2, -b_chamfer - b_height / 2],
                [b_width / 2 - b_chamfer, -b_height / 2],
                [-b_width / 2 + b_chamfer, -b_height / 2],
            ],
        ]

        pole_shape = [
            [
                [-p_width / 2, 0],
                [-p_width / 2, -p_height2 + p_chamfer],
                [-p_width / 2 + p_chamfer, -p_height2],
                [p_width / 2 - p_chamfer, -p_height2],
                [p_width / 2, -p_height2 + p_chamfer],
                [p_width / 2, 0],
            ],
            [
                [-p_width / 2, -p_chamfer - p_height2],
                [-p_width / 2, -p_height1 + p_chamfer - p_height2],
                [-p_width / 2 + p_chamfer, -p_height1 - p_height2],
                [p_width / 2 - p_chamfer, -p_height1 - p_height2],
                [p_width / 2, p_chamfer - p_height1 - p_height2],
                [p_width / 2, -p_chamfer - p_height2],
                [p_width / 2 - p_chamfer, -p_height2],
                [-p_width / 2 + p_chamfer, -p_height2],
            ],
        ]

        # block_subdivision = [[1, 1, 1], [1, 1, 1]]
        # pole_subdivision = [[22, 4, 4], [6, 3, 4]]

        block_subdivision = [[1, 1, 1], [1, 1, 1]]
        pole_subdivision = [[22, 4, 4], [6, 4, 4]]

        start_blocks_length = [3.80, 2.07, 9.10]
        start_blocks_distance = [3.25, 3.25, 0]

        end_blocks_length = [4, 9.10, 2.07, 3.80]
        end_blocks_distance = [0.1, 0, 3.25, 3.25]

        vpu = HybridPlanar(
            gap=gap,
            period_length=period_length,
            mr=mr,
            nr_periods=nr_periods,
            longitudinal_distance=longitudinal_distance,
            block_shape=block_shape,
            pole_shape=pole_shape,
            block_subdivision=block_subdivision,
            pole_subdivision=pole_subdivision,
            pole_length=pole_length,
            start_blocks_length=start_blocks_length,
            start_blocks_distance=start_blocks_distance,
            end_blocks_length=end_blocks_length,
            end_blocks_distance=end_blocks_distance,
        )

        cs_mag = vpu.magnetization_dict["cs"]
        cs_mag[0] = [0.0, 0.0, 1.26]
        cs_mag[1] = [0.0, 0.0, 0.0]
        cs_mag[2] = [0.0, 0.0, -1.26]

        cs_mag[-1] = [0.0, 0.0, -1.26]
        cs_mag[-2] = [0.0, 0.0, 0.0]
        cs_mag[-3] = [0.0, 0.0, 1.26]
        cs_mag[-4] = [0.0, 0.0, 0.0]

        mags = np.array(cs_mag)
        mags = -1 * mags
        cs_mag = mags.tolist()

        ci_mag = vpu.magnetization_dict["ci"]
        ci_mag[0] = [0.0, 0.0, -1.26]
        ci_mag[1] = [0.0, 0.0, 0.0]
        ci_mag[2] = [0.0, 0.0, 1.26]

        ci_mag[-1] = [0.0, 0.0, 1.26]
        ci_mag[-2] = [0.0, 0.0, 0.0]
        ci_mag[-3] = [0.0, 0.0, -1.26]
        ci_mag[-4] = [0.0, 0.0, 0.0]

        mags = np.array(ci_mag)
        mags = -1 * mags
        ci_mag = mags.tolist()

        magnetization_dict = dict()
        magnetization_dict["cs"] = cs_mag
        magnetization_dict["ci"] = ci_mag

        start_blocks_pole_list = [False, True, False]
        end_blocks_pole_list = [True, False, True, False]
        core_blocks_pole_list = (
            2 * [True, False] * int(vpu.cassettes_ref["cs"].nr_core_blocks / 2)
        )

        is_pole_list = (
            start_blocks_pole_list
            + core_blocks_pole_list
            + end_blocks_pole_list
        )

        vpu.create_radia_object(
            magnetization_dict=magnetization_dict, is_pole_list=is_pole_list
        )

        vpu.cassettes["ci"].rotate([0, 0, 0], [0, 0, 1], -np.pi / 2)
        vpu.cassettes["cs"].rotate([0, 0, 0], [0, 0, 1], -np.pi / 2)

        if solve:
            vpu.solve()

        self.model = vpu
        return vpu


class UE44(RadiaModelAnalysis):
    """UE44 RADIA model creation and manipulation."""

    def __init__(self):
        """Class constructor."""
        self.nr_periods = 10
        self.period_length = 44
        self.gap = 11
        self.mr = 1.08
        self.slit = 0.47
        self.longitudinal_distance = 0
        self.block_width = 30
        self.block_chamfer = 5
        self.model = None

    def generate_radia_model(
        self, solve=False, gap=None, nr_periods=None, phase=0, de=0
    ):
        """Generate UE44 RADIA model.

        Args:
            solve (bool, optional): Solve ID using relaxation method.
            Defaults to False.
            gap (float, optional): ID gap. Defaults to None.
            nr_periods (int, optional): nr_periods. Defaults to None.
            phase (float, optional): Id Pparam. Defaults to 0.
            de (float, optional): ID Kparam. Defaults to 0.

        Returns:
            _RADIA object_: UE44 RADIA model
        """
        gap = gap if gap is not None else self.gap
        nr_periods = nr_periods if nr_periods is not None else self.nr_periods
        period_length = self.period_length
        mr = self.mr
        slit = self.slit
        b_width = self.block_width
        b_chamfer = self.block_chamfer

        longitudinal_distance = self.longitudinal_distance

        block_shape = [
            [
                [0, 0],
                [b_width - b_chamfer, 0],
                [b_width - b_chamfer, b_chamfer],
                [b_chamfer, b_width - b_chamfer],
                [0, b_width - b_chamfer],
            ],
            [
                [b_width - b_chamfer, b_chamfer],
                [b_width, b_chamfer],
                [b_width, b_width],
                [b_chamfer, b_width],
                [b_chamfer, b_width - b_chamfer],
            ],
        ]

        for i in np.arange(2):
            for j in np.arange(5):
                block_shape[i][j][1] -= b_width
                block_shape[i][j][0] -= b_width + slit / 2

        l1 = 1.44671233
        l2 = 4.55528286
        l3 = 6.93386859
        d1 = 4.0
        d2 = 4.0
        d3 = 4.0

        start_blocks_length = [l1, l2, l3, 11]
        start_blocks_distance = [d1, d2, d3, 0]

        end_blocks_length = [11, 11, l3, l2, l1]
        end_blocks_distance = [0, 0, d3, d2, d1]

        apple = AppleII(
            block_shape=block_shape,
            nr_periods=nr_periods,
            period_length=period_length,
            gap=gap,
            mr=mr,
            longitudinal_distance=longitudinal_distance,
            start_blocks_distance=start_blocks_distance,
            start_blocks_length=start_blocks_length,
            end_blocks_distance=end_blocks_distance,
            end_blocks_length=end_blocks_length,
        )

        csd = apple.cassettes_ref['csd']
        cse = apple.cassettes_ref['cse']
        cid = apple.cassettes_ref['cid']
        cie = apple.cassettes_ref['cie']

        csd_z0 = csd.center_point[2]
        cse_z0 = cse.center_point[2]
        cie_z0 = cie.center_point[2]

        diff_csd = -csd_z0 - phase - de
        diff_cse = -cse_z0 - de
        diff_cie = -cie_z0 - phase

        csd.shift([0, 0, diff_csd])
        cse.shift([0, 0, diff_cse])
        cie.shift([0, 0, diff_cie])
        cid.shift([0, 0, 0])

        if solve:
            apple.solve()

        self.model = apple

        return apple
