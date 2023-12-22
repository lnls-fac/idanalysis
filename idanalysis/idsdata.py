"""IDs data."""

from idanalysis import IDKickMap as _IDKickMap

DATA_REPOS_PATH = "/media/gabriel/Dados/"  # Put your data repository path here
REPOS_PATH = "/home/gabriel/repos/idanalysis/"  # Put your repository path here


class Tools:
    """Common tools for IDs data manipulation."""

    def __init__(self):
        """Class constructor."""
        self.fmap = None

    @staticmethod
    def get_gap_str(gap):
        """Construct gap string.

        Args:
            gap (float): gap value

        Returns:
            str: gap string
        """
        gap_str = "{:04.1f}".format(gap).replace(".", "p")
        return gap_str

    @staticmethod
    def get_phase_str(phase):
        """Construct phase string.

        Args:
            phase (float): phase value

        Returns:
            str: phase string
        """
        phase_str = "{:+07.3f}".format(phase).replace(".", "p")
        phase_str = phase_str.replace("+", "pos").replace("-", "neg")
        return phase_str

    @staticmethod
    def _mkdir_function(mypath):
        """Create directory.

        Args:
            mypath (str): directory path
        """
        from errno import EEXIST
        from os import makedirs, path

        try:
            makedirs(mypath)
        except OSError as exc:
            if exc.errno == EEXIST and path.isdir(mypath):
                pass
            else:
                raise

    def get_kickmap_filename(self, meas_flag=False, **kwargs):
        """Get kickmap file name.

        Args:
            meas_flag (bool, optional): Get kickmap from field measurements.
            Defaults to False.
            **kwargs (): Specialized argument for each ID type

        Returns:
            str: Kickmap file name
        """
        fpath = self.folder_base_kickmaps
        if meas_flag:
            fpath += "measurements/"
        else:
            fpath += "model/"
        fname = self._get_kmap_config_name(**kwargs)
        fname = fpath + fname + ".txt"

        return fname

    def get_linear_kickmap_filename(self, meas_flag=False, **kwargs):
        """Get linear kickmap file name.

        Args:
            meas_flag (bool, optional): Get kickmap from field measurements.
            Defaults to False.
            **kwargs (): Specialized argument for each ID type

        Returns:
            str: Kickmap file name
        """
        fname = self.get_kickmap_filename(meas_flag=meas_flag, **kwargs)
        fname = fname.replace(".txt", "-linear.txt")
        return fname

    def get_shifted_kickmap_filename(self, meas_flag=False, **kwargs):
        """Get kickmap with no dipolar term file name.

        Args:
            meas_flag (bool, optional): Get kickmap from field measurements.
            Defaults to False.
            **kwargs (): Specialized argument for each ID type

        Returns:
            str: Kickmap file name
        """
        fname = self.get_kickmap_filename(meas_flag=meas_flag, **kwargs)
        fname = fname.replace(".txt", "-shifted_on_axis.txt")
        return fname

    def get_filtered_kickmap_filename(self, meas_flag=False, **kwargs):
        """Ger filtered kickmap file name.

        Args:
            meas_flag (bool, optional): Get kickmap from field measurements.
            Defaults to False.
            **kwargs (): Specialized argument for each ID type

        Returns:
            str: Kickmap file name
        """
        fname = self.get_kickmap_filename(meas_flag=meas_flag, **kwargs)
        fname = fname.replace(".txt", "-filtered.txt")
        return fname

    def get_data_output_path(self, meas_flag=False, **kwargs):
        """Get output path for data.

        Args:
            meas_flag (bool, optional): Is this data from field measurements?
            **kwargs (): Specialized argument for each ID type

        Returns:
            str: Data output path
        """
        fpath = self.folder_base_output
        if meas_flag:
            fpath += "measurements/"
        else:
            fpath += "model/"
        config = self._get_config_path(**kwargs)
        fpath += config
        return fpath

    def get_meas_data_output_path(self, **kwargs):
        """Get output path for measurements's results.

        Returns:
            str: Data output path
        """
        fpath = self.get_data_output_path(meas_flag=True, **kwargs)
        return fpath

    def get_model_data_output_path(self, **kwargs):
        """Get output dath for model's results.

        Returns:
            str: Data output path
        """
        fpath = self.get_data_output_path(meas_flag=False, **kwargs)
        return fpath

    def get_fmap(self, **kwargs):
        """Get fieldmap.

        Returns:
            _Fieldmap object_: Fieldmap object
        """
        fmap_fname = self.get_fmap_fname(**kwargs)
        print(fmap_fname)
        idkickmap = _IDKickMap()
        idkickmap.fmap_fname = fmap_fname
        fmap = idkickmap.fmap_config.fmap
        self.fmap = fmap
        return fmap

    @property
    def period_length(self):
        """ID period length.

        Returns:
            float: ID period length
        """
        return self._params.PERIOD_LEN

    @property
    def id_length(self):
        """ID length.

        Returns:
            float: ID length
        """
        return self._params.ID_LEN

    @property
    def nr_periods(self):
        """Number of ID's periods.

        Returns:
            int: Number of periods
        """
        return self._params.NR_PERIODS

    @property
    def pparam_name(self):
        """ID Pparam name.

        Returns:
            str: Pparam name
        """
        return self._params.PPARAMETER_NAME

    @property
    def kparam_name(self):
        """ID Kparam name.

        Returns:
            str: Kparam name
        """
        return self._params.KPARAMETER_NAME

    @property
    def default_pparam(self):
        """Default ID Pparam.

        Returns:
            float: Default value of ID's pparam (if it exists)
        """
        return self._params.DEFAULT_PPARAMETER

    @property
    def default_kparam(self):
        """Default ID Kparam.

        Returns:
            float: Default value of ID's kparam (if it exists)
        """
        return self._params.DEFAULT_KPARAMETER

    @property
    def id_famname(self):
        """ID's family name on SIRIUS model.

        Returns:
            str: ID's family name.
        """
        return self._params.ID_FAMNAME

    @property
    def subsecs(self):
        """ID's subsection on SIRIUS model.

        Returns:
            list of str: ID's subsections.
        """
        return self._params.SUBSECS

    @property
    def folder_base_output(self):
        """Base folder for output data.

        Returns:
            str: Path of base folder for output data
        """
        return self._params.FOLDER_BASE_OUTPUT

    @property
    def folder_base_kickmaps(self):
        """Base folder to get kickmaps.

        Returns:
            str: Path of base folder for kickmaps
        """
        return self._params.KICKMAPS_DATA_PATH

    @property
    def folder_base_fieldmaps(self):
        """Base folder to get fieldmaps.

        Returns:
            str: Path of base folder for fieldmaps
        """
        return self._params.FIELDMAPS_DATA_PATH


class _PARAMS:
    """."""

    # --- ID parameters ---
    PERIOD_LEN = None
    ID_LEN = None
    NR_PERIODS = None
    PPARAMETER_NAME = None
    KPARAMETER_NAME = None
    DEFAULT_PPARAMETER = None
    DEFAULT_KPARAMETER = None
    ID_FAMNAME = None
    SUBSECS = None

    # --- data parameters
    FIELDMAPS_DATA_PATH = None
    KICKMAPS_DATA_PATH = None
    FOLDER_BASE_OUTPUT = None


class DELTA52Data(Tools):
    """DELTA data access and manipulation class."""

    PARAMS = _PARAMS()
    PARAMS.PERIOD_LEN = 52.5  # [mm]
    PARAMS.ID_LEN = 1.200  # [m]
    PARAMS.NR_PERIODS = 21
    PARAMS.PPARAMETER_NAME = "dp"
    PARAMS.KPARAMETER_NAME = "dgv"
    PARAMS.ID_FAMNAME = "DELTA52"
    PARAMS.SUBSECS = ["ID10SB"]

    PARAMS.KICKMAPS_DATA_PATH = REPOS_PATH + "scripts/delta52/kickmaps/"
    PARAMS.FIELDMAPS_DATA_PATH = (
        DATA_REPOS_PATH
        + "lnls-ima/delta52/id-sabia/model-03/measurement/magnetic/hallprobe/"
    )  # noqa: E501
    PARAMS.FOLDER_BASE_OUTPUT = REPOS_PATH + "scripts/delta52/results/data/"

    FIELMAPS_CONFIGS = {
        # dp = 0 mm
        "ID4862": "delta_sabia_final/LinH/2023-10-25_DeltaSabia_Phase="
        + "0mm_GV=0mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_"
        + "ID=4862.dat",
        "ID4863": "delta_sabia_final/LinH/2023-10-25_DeltaSabia_Phase="
        + "0mm_GV=6.5625mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_"
        + "ID=4863.dat",
        "ID4864": "delta_sabia_final/LinH/2023-10-25_DeltaSabia_Phase="
        + "0mm_GV=13.125mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_"
        + "ID=4864.dat",
        "ID4865": "delta_sabia_final/LinH/2023-10-25_DeltaSabia_Phase="
        + "0mm_GV=19.6875mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_"
        + "ID=4865.dat",
        "ID4866": "delta_sabia_final/LinH/2023-10-25_DeltaSabia_Phase="
        + "0mm_GV=26.25mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_"
        + "ID=4866.dat",
        # dp = -26.25 mm
        "ID4872": "delta_sabia_final/LinV/2023-10-26_DeltaSabia_Phase="
        + "-26.25mm_GV=0mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_"
        + "ID=4872.dat",
        "ID4873": "delta_sabia_final/LinV/2023-10-26_DeltaSabia_Phase="
        + "-26.25mm_GV=6.5625mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_"
        + "ID=4873.dat",
        "ID4874": "delta_sabia_final/LinV/2023-10-26_DeltaSabia_Phase="
        + "-26.25mm_GV=13.125mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_"
        + "ID=4874.dat",
        "ID4875": "delta_sabia_final/LinV/2023-10-26_DeltaSabia_Phase="
        + "-26.25mm_GV=19.6875mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_"
        + "ID=4875.dat",
        "ID4876": "delta_sabia_final/LinV/2023-10-27_DeltaSabia_Phase="
        + "-26.25mm_GV=26.25mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_"
        + "ID=4876.dat",
        # dp =  -13.125 mm
        "ID4867": "delta_sabia_final/CircN/2023-10-25_DeltaSabia_Phase="
        + "-13.125mm_GV=0mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_"
        + "ID=4867.dat",
        "ID4868": "delta_sabia_final/CircN/2023-10-26_DeltaSabia_Phase="
        + "-13.125mm_GV=6.5625mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_"
        + "ID=4868.dat",
        "ID4869": "delta_sabia_final/CircN/2023-10-26_DeltaSabia_Phase="
        + "-13.125mm_GV=13.125mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_"
        + "ID=4869.dat",
        "ID4870": "delta_sabia_final/CircN/2023-10-26_DeltaSabia_Phase="
        + "-13.125mm_GV=19.6875mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_"
        + "ID=4870.dat",
        "ID4871": "delta_sabia_final/CircN/2023-10-26_DeltaSabia_Phase="
        + "-13.125mm_GV=26.25mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_"
        + "ID=4871.dat",
        # dp =  13.125 mm
        "ID4877": "delta_sabia_final/CircP/2023-10-27_DeltaSabia_Phase="
        + "13.125mm_GV=0mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_"
        + "ID=4877.dat",
        "ID4878": "delta_sabia_final/CircP/2023-10-27_DeltaSabia_Phase="
        + "13.125mm_GV=-6.5625mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_"
        + "ID=4878.dat",
        "ID4879": "delta_sabia_final/CircP/2023-10-27_DeltaSabia_Phase="
        + "13.125mm_GV=-13.125mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_"
        + "ID=4879.dat",
        "ID4880": "delta_sabia_final/CircP/2023-10-27_DeltaSabia_Phase="
        + "13.125mm_GV=-19.6875mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_"
        + "ID=4880.dat",
        "ID4881": "delta_sabia_final/CircP/2023-10-27_DeltaSabia_Phase="
        + "13.125mm_GV=-26.25mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_"
        + "ID=4881.dat",
        # zero K
        "ID4884": "delta_sabia_final/zeroK/2023-10-27_DeltaSabia_Phase="
        + "6.5625mm_GV=0mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_"
        + "ID=4884.dat",
        "ID4882": "delta_sabia_final/zeroK/2023-10-27_DeltaSabia_Phase="
        + "-6.5625mm_GV=0mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_"
        + "ID=4882.dat",
        "ID4883": "delta_sabia_final/zeroK/2023-10-27_DeltaSabia_Phase="
        + "-19.6875mm_GV=0mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_"
        + "ID=4883.dat",
    }

    dp_dgv_dict = {
        (0, 0): "ID4862",
        (0, 6.5625): "ID4863",
        (0, 13.125): "ID4864",
        (0, 19.6875): "ID4865",
        (0, 26.25): "ID4866",
        (-13.125, 0): "ID4867",
        (-13.125, 6.5625): "ID4868",
        (-13.125, 13.125): "ID4869",
        (-13.125, 19.6875): "ID4870",
        (-13.125, 26.25): "ID4871",
        (13.125, 0): "ID4877",
        (13.125, 6.5625): "ID4878",
        (13.125, 13.125): "ID4879",
        (13.125, 19.6875): "ID4880",
        (13.125, 26.25): "ID4881",
        (-26.25, 0): "ID4872",
        (-26.25, 6.5625): "ID4873",
        (-26.25, 13.125): "ID4874",
        (-26.25, 19.6875): "ID4875",
        (-26.25, 26.25): "ID4876",
        (6.5625, 0): "ID4884",
        (-6.5625, 0): "ID4882",
        (-19.6875, 0): "ID4883",
    }

    def __init__(self):
        """Class constructor."""
        self._params = DELTA52Data.PARAMS
        self.si_idmodel = None

    @staticmethod
    def _get_config_path(dp, dgv):
        path = ""
        dp_str = Tools.get_phase_str(dp)
        path += "dps/dp_{}/".format(dp_str)
        dgv_str = Tools.get_phase_str(dgv)
        path += "dgvs/dgv_{}/".format(dgv_str)
        return path

    @staticmethod
    def _get_kmap_config_name(dp, dgv):
        fname = "kickmap-ID"
        dp_str = Tools.get_phase_str(dp)
        fname += "_dp_{}".format(dp_str)
        dgv_str = Tools.get_phase_str(dgv)
        fname += "_dgv_{}".format(dgv_str)
        return fname

    def get_fmap_fname(self, dp, dgv):
        """Get fmap fname.

        Args:
            dp (float): dp value
            dgv (float): dgv value

        Returns:
            str: fielmap name
        """
        fpath = self.folder_base_fieldmaps
        idconfig = self.dp_dgv_dict[(dp, dgv)]
        fname = self.FIELMAPS_CONFIGS[idconfig]
        fmap_fname = fpath + fname
        return fmap_fname


class APU22Data(Tools):
    """APU22 data access and manipulation class."""

    PARAMS = _PARAMS()
    PARAMS.PERIOD_LEN = 22  # [mm]
    PARAMS.ID_LEN = 1.300  # [m]
    PARAMS.NR_PERIODS = 51
    PARAMS.KPARAMETER_NAME = "phase"
    PARAMS.ID_FAMNAME = "APU22"
    PARAMS.SUBSECS = ["ID06SB", "ID07SP", "ID08SB", "ID09SA"]

    PARAMS.KICKMAPS_DATA_PATH = REPOS_PATH + "scripts/apu22/kickmaps/"
    PARAMS.FIELDMAPS_DATA_PATH = (
        DATA_REPOS_PATH
        + "lnls-ima/kyma22/id-manaca/commissioning_id/measurement/magnetic/"
        + "lnls/hallprobe/vertical_position_0mm/"
    )  # noqa: E501
    PARAMS.FOLDER_BASE_OUTPUT = REPOS_PATH + "scripts/apu22/results/data/"

    FIELMAPS_CONFIGS = {
        "ID2828": "2020-05-26_1991b_Fieldmap_X=-12_12mm_Z=-740_740mm_"
        + "Y=0mm_ID=2828_Phase=0mm.dat",
        "ID2829": "2020-05-26_1991b_Fieldmap_X=-12_12mm_Z=-740_740mm_"
        + "Y=0mm_ID=2829_Phase=1mm.dat",
        "ID2830": "2020-05-26_1991b_Fieldmap_X=-12_12mm_Z=-740_740mm_"
        + "Y=0mm_ID=2830_Phase=2mm.dat",
        "ID2831": "2020-05-26_1991b_Fieldmap_X=-12_12mm_Z=-740_740mm_"
        + "Y=0mm_ID=2831_Phase=3mm.dat",
        "ID2832": "2020-05-26_1991b_Fieldmap_X=-12_12mm_Z=-740_740mm_"
        + "Y=0mm_ID=2832_Phase=4mm.dat",
        "ID2833": "2020-05-26_1991b_Fieldmap_X=-12_12mm_Z=-740_740mm_"
        + "Y=0mm_ID=2833_Phase=5mm.dat",
        "ID2834": "2020-05-26_1991b_Fieldmap_X=-12_12mm_Z=-740_740mm_"
        + "Y=0mm_ID=2834_Phase=6mm.dat",
        "ID2835": "2020-05-26_1991b_Fieldmap_X=-12_12mm_Z=-740_740mm_"
        + "Y=0mm_ID=2835_Phase=7mm.dat",
        "ID2836": "2020-05-26_1991b_Fieldmap_X=-12_12mm_Z=-740_740mm_"
        + "Y=0mm_ID=2836_Phase=8mm.dat",
        "ID2837": "2020-05-26_1991b_Fieldmap_X=-12_12mm_Z=-740_740mm_"
        + "Y=0mm_ID=2837_Phase=9mm.dat",
        "ID2838": "2020-05-26_1991b_Fieldmap_X=-12_12mm_Z=-740_740mm_"
        + "Y=0mm_ID=2838_Phase=10mm.dat",
        "ID2839": "2020-05-26_1991b_Fieldmap_X=-12_12mm_Z=-740_740mm_"
        + "Y=0mm_ID=2839_Phase=11mm.dat",
    }

    phase_dict = {
        (0): "ID2828",
        (1): "ID2829",
        (2): "ID2830",
        (3): "ID2831",
        (4): "ID2832",
        (5): "ID2833",
        (6): "ID2834",
        (7): "ID2835",
        (8): "ID2836",
        (9): "ID2837",
        (10): "ID2838",
        (11): "ID2839",
    }

    def __init__(self):
        """Class constructor."""
        self._params = APU22Data.PARAMS
        self.si_idmodel = None

    @staticmethod
    def _get_config_path(phase):
        path = ""
        phase_str = Tools.get_phase_str(phase)
        path += "phases/phase_{}/".format(phase_str)
        return path

    @staticmethod
    def _get_kmap_config_name(phase):
        fname = "kickmap-ID"
        phase_str = Tools.get_phase_str(phase)
        fname += "_phase_{}".format(phase_str)
        return fname

    def get_fmap_fname(self, phase):
        """Get fmap fname.

        Args:
            phase (float): phase value

        Returns:
            str: fielmap name
        """
        fpath = self.folder_base_fieldmaps
        idconfig = self.phase_dict[(phase)]
        fname = self.FIELMAPS_CONFIGS[idconfig]
        fmap_fname = fpath + fname
        return fmap_fname


class APU58Data(Tools):
    """APU58 data access and manipulation class."""

    PARAMS = _PARAMS()
    PARAMS.PERIOD_LEN = 58  # [mm]
    PARAMS.ID_LEN = 1.300  # [m]
    PARAMS.NR_PERIODS = 18
    PARAMS.KPARAMETER_NAME = "phase"
    PARAMS.ID_FAMNAME = "APU58"
    PARAMS.SUBSECS = ["ID11SP"]

    PARAMS.KICKMAPS_DATA_PATH = REPOS_PATH + "scripts/apu58/kickmaps/"
    PARAMS.FIELDMAPS_DATA_PATH = (
        DATA_REPOS_PATH
        + "lnls-ima/kyma58/id-ipe/commissioning_id/measurement/magnetic/lnls/"
        + "hallprobe/vertical_position_0mm/"
    )  # noqa: E501
    PARAMS.FOLDER_BASE_OUTPUT = REPOS_PATH + "scripts/apu58/results/data/"

    FIELMAPS_CONFIGS = {
        "ID2945": "2020-10-23_1995_Fieldmap_X=-10_10mm_Z=-750_750mm_"
        + "Y=0mm_ID=2945_Phase=0.0mm.dat",
        "ID2946": "2020-10-23_1995_Fieldmap_X=-10_10mm_Z=-750_750mm_"
        + "Y=0mm_ID=2946_Phase=1.812mm.dat",
        "ID2947": "2020-10-23_1995_Fieldmap_X=-10_10mm_Z=-750_750mm_"
        + "Y=0mm_ID=2947_Phase=3.625mm.dat",
        "ID2948": "2020-10-23_1995_Fieldmap_X=-10_10mm_Z=-750_750mm_"
        + "Y=0mm_ID=2948_Phase=5.438mm.dat",
        "ID2949": "2020-10-23_1995_Fieldmap_X=-10_10mm_Z=-750_750mm_"
        + "Y=0mm_ID=2949_Phase=7.25mm.dat",
        "ID2950": "2020-10-23_1995_Fieldmap_X=-10_10mm_Z=-750_750mm_"
        + "Y=0mm_ID=2950_Phase=9.062mm.dat",
        "ID2951": "2020-10-23_1995_Fieldmap_X=-10_10mm_Z=-750_750mm_"
        + "Y=0mm_ID=2951_Phase=10.875mm.dat",
        "ID2952": "2020-10-23_1995_Fieldmap_X=-10_10mm_Z=-750_750mm_"
        + "Y=0mm_ID=2952_Phase=12.688mm.dat",
        "ID2953": "2020-10-23_1995_Fieldmap_X=-10_10mm_Z=-750_750mm_"
        + "Y=0mm_ID=2953_Phase=14.5mm.dat",
        "ID2954": "2020-10-23_1995_Fieldmap_X=-10_10mm_Z=-750_750mm_"
        + "Y=0mm_ID=2954_Phase=16.312mm.dat",
        "ID2955": "2020-10-23_1995_Fieldmap_X=-10_10mm_Z=-750_750mm_"
        + "Y=0mm_ID=2955_Phase=18.125mm.dat",
        "ID2956": "2020-10-23_1995_Fieldmap_X=-10_10mm_Z=-750_750mm_"
        + "Y=0mm_ID=2956_Phase=19.938mm.dat",
        "ID2957": "2020-10-23_1995_Fieldmap_X=-10_10mm_Z=-750_750mm_"
        + "Y=0mm_ID=2957_Phase=21.75mm.dat",
        "ID2958": "2020-10-23_1995_Fieldmap_X=-10_10mm_Z=-750_750mm_"
        + "Y=0mm_ID=2958_Phase=23.562mm.dat",
        "ID2959": "2020-10-23_1995_Fieldmap_X=-10_10mm_Z=-750_750mm_"
        + "Y=0mm_ID=2959_Phase=25.375mm.dat",
        "ID2960": "2020-10-23_1995_Fieldmap_X=-10_10mm_Z=-750_750mm_"
        + "Y=0mm_ID=2960_Phase=27.188mm.dat",
        "ID2961": "2020-10-24_1995_Fieldmap_X=-10_10mm_Z=-750_750mm_"
        + "Y=0mm_ID=2961_Phase=29.0mm.dat",
    }

    phase_dict = {
        (0.0): "ID2945",
        (1.812): "ID2946",
        (3.625): "ID2947",
        (5.438): "ID2948",
        (7.25): "ID2949",
        (9.062): "ID2950",
        (10.875): "ID2951",
        (12.688): "ID2952",
        (14.5): "ID2953",
        (16.312): "ID2954",
        (18.125): "ID2955",
        (19.938): "ID2956",
        (21.75): "ID2957",
        (23.562): "ID2958",
        (25.375): "ID2959",
        (27.188): "ID2960",
        (29.0): "ID2961",
    }

    def __init__(self):
        """Class constructor."""
        self._params = APU58Data.PARAMS
        self.si_idmodel = None

    @staticmethod
    def _get_config_path(phase):
        path = ""
        phase_str = Tools.get_phase_str(phase)
        path += "phases/phase_{}/".format(phase_str)
        return path

    @staticmethod
    def _get_kmap_config_name(phase):
        fname = "kickmap-ID"
        phase_str = Tools.get_phase_str(phase)
        fname += "_phase_{}".format(phase_str)
        return fname

    def get_fmap_fname(self, phase):
        """Get fmap fname.

        Args:
            phase (float): phase value

        Returns:
            str: fielmap name
        """
        fpath = self.folder_base_fieldmaps
        idconfig = self.phase_dict[(phase)]
        fname = self.FIELMAPS_CONFIGS[idconfig]
        fmap_fname = fpath + fname
        return fmap_fname


class PAPU50Data(Tools):
    """PAPU50 data access and manipulation class."""

    PARAMS = _PARAMS()
    PARAMS.PERIOD_LEN = 50  # [mm]
    PARAMS.ID_LEN = 0.984  # [m]
    PARAMS.NR_PERIODS = 18
    PARAMS.KPARAMETER_NAME = "phase"
    PARAMS.ID_FAMNAME = "PAPU50"
    PARAMS.SUBSECS = ["ID09SA"]

    PARAMS.KICKMAPS_DATA_PATH = REPOS_PATH + "scripts/papu50/kickmaps/"
    PARAMS.FIELDMAPS_DATA_PATH = (
        DATA_REPOS_PATH
        + "lnls-ima/papu50/id-papu/model-01/measurement/magnetic/hallprobe/"
    )  # noqa: E501
    PARAMS.FOLDER_BASE_OUTPUT = REPOS_PATH + "scripts/papu50/results/data/"

    FIELMAPS_CONFIGS = {
        "ID4647": "2023-05-30_PAPU_Fieldmap_Corrected_X=-10_10mm_"
        + "Z=-900_900mm_Y=0mm_ID=4647_Phase=4.33mm.dat",
        "ID4648": "2023-05-30_PAPU_Fieldmap_Corrected_X=-10_10mm_"
        + "Z=-900_900mm_Y=0mm_ID=4648_Phase=0mm.dat",
        "ID4649": "2023-05-30_PAPU_Fieldmap_Corrected_X=-10_10mm_"
        + "Z=-900_900mm_Y=0mm_ID=4649_Phase=6.25mm.dat",
        "ID4650": "2023-05-30_PAPU_Fieldmap_Corrected_X=-10_10mm_"
        + "Z=-900_900mm_Y=0mm_ID=4650_Phase=12.5mm.dat",
        "ID4651": "2023-05-31_PAPU_Fieldmap_Corrected_X=-10_10mm_"
        + "Z=-900_900mm_Y=0mm_ID=4651_Phase=18.75mm.dat",
        "ID4652": "2023-05-31_PAPU_Fieldmap_Corrected_X=-10_10mm_"
        + "Z=-900_900mm_Y=0mm_ID=4652_Phase=25mm.dat",
        "ID4653": "2023-05-31_PAPU_Fieldmap_Corrected_X=-10_10mm_"
        + "Z=-900_900mm_Y=0mm_ID=4653_Phase=9.24mm.dat",
        "ID4654": "2023-05-31_PAPU_Fieldmap_Corrected_X=-10_10mm_"
        + "Z=-900_900mm_Y=0mm_ID=4654_Phase=12.82mm.dat",
        "ID4655": "2023-05-31_PAPU_Fieldmap_Corrected_X=-10_10mm_"
        + "Z=-900_900mm_Y=0mm_ID=4655_Phase=16.48mm.dat",
    }

    phase_dict = {
        (0): "ID4648",
        (4.33): "ID4647",
        (6.25): "ID4649",
        (9.24): "ID4653",
        (12.5): "ID4650",
        (12.82): "ID4654",
        (16.48): "ID4655",
        (18.75): "ID4651",
        (25): "ID4652",
    }

    def __init__(self):
        """Class constructor."""
        self._params = PAPU50Data.PARAMS
        self.si_idmodel = None

    @staticmethod
    def _get_config_path(phase):
        path = ""
        phase_str = Tools.get_phase_str(phase)
        path += "phases/phase_{}/".format(phase_str)
        return path

    @staticmethod
    def _get_kmap_config_name(phase):
        fname = "kickmap-ID"
        phase_str = Tools.get_phase_str(phase)
        fname += "_phase_{}".format(phase_str)
        return fname

    def get_fmap_fname(self, phase):
        """Get fmap fname.

        Args:
            phase (float): phase value

        Returns:
            str: fielmap name
        """
        fpath = self.folder_base_fieldmaps
        idconfig = self.phase_dict[(phase)]
        fname = self.FIELMAPS_CONFIGS[idconfig]
        fmap_fname = fpath + fname
        return fmap_fname


class WIG180Data(Tools):
    """Wiggler data access and manipulation class."""

    PARAMS = _PARAMS()
    PARAMS.PERIOD_LEN = 180  # [mm]
    PARAMS.ID_LEN = 2.654  # [m]
    PARAMS.NR_PERIODS = 13
    PARAMS.KPARAMETER_NAME = "gap"
    PARAMS.ID_FAMNAME = "WIG180"
    PARAMS.SUBSECS = ["ID14SB"]

    PARAMS.KICKMAPS_DATA_PATH = REPOS_PATH + "scripts/wig180/kickmaps/"
    PARAMS.FIELDMAPS_DATA_PATH = (
        DATA_REPOS_PATH
        + "lnls-ima/wig180/wiggler-2T-STI/commissioning_id/measurement/"
        + "magnetic/lnls/hallprobe/vertical_position_0mm/"
    )  # noqa: E501
    PARAMS.FOLDER_BASE_OUTPUT = REPOS_PATH + "scripts/wig180/results/data/"

    FIELMAPS_CONFIGS = {
        # wiggler with correctors - gap 022.00mm
        "ID3977": "gap 022.00mm/2022-08-25_WigglerSTI_022.00mm_"
        + "U+0.00_D+0.00_Fieldmap_X=-20_20mm_Z=-1650_1650mm_ID=3977.dat",
        # wiggler with correctors - gap 023.00mm
        "ID3986": "gap 023.00mm/2022-08-26_WigglerSTI_023.00mm_"
        + "U+0.00_D+0.00_Fieldmap_Z=-1650_1650mm_ID=3986.dat",
        # wiggler with correctors - gap 023.95mm
        "ID3987": "gap 023.95mm/2022-08-26_WigglerSTI_023.95mm_"
        + "U+0.00_D-0.00_Fieldmap_Z=-1650_1650mm_ID=3987.dat",
        # wiggler with correctors - gap 026.90mm
        "ID3988": "gap 026.90mm/2022-08-26_WigglerSTI_026.90mm_"
        + "U+0.00_D-0.00_Fieldmap_Z=-1650_1650mm_ID=3988.dat",
        # wiggler with correctors - gap 034.80mm
        "ID3989": "gap 034.80mm/2022-08-26_WigglerSTI_034.80mm_"
        + "U+0.00_D-0.00_Fieldmap_Z=-1650_1650mm_ID=3989.dat",
        # wiggler with correctors - gap 042.70mm
        "ID3990": "gap 042.70mm/2022-08-26_WigglerSTI_042.70mm_"
        + "U+0.00_D+0.00_Fieldmap_Z=-1650_1650mm_ID=3990.dat",
        # wiggler with correctors - gap 045.00mm
        "ID4020": "gap 045.00mm/2022-09-01_WigglerSTI_45mm_"
        + "U+0.00_D+0.00_Fieldmap_X=-20_20mm_Z=-1650_1650mm_ID=4020.dat",
        # wiggler with correctors - gap 049.73mm
        "ID4019": "gap 049.73mm/2022-09-01_WigglerSTI_49.73mm_"
        + "U+0.00_D+0.00_Fieldmap_X=-20_20mm_Z=-1650_1650mm_ID=4019.dat",
        # wiggler with correctors - gap 051.65mm
        "ID3991": "gap 051.65mm/2022-08-26_WigglerSTI_051.65mm_"
        + "U+0.00_D+0.00_Fieldmap_Z=-1650_1650mm_ID=3991.dat",
        # wiggler with correctors - gap 059.60mm
        "ID3979": "gap 059.60mm/2022-08-25_WigglerSTI_059.60mm_"
        + "U+0.00_D+0.00_Fieldmap_X=-20_20mm_Z=-1650_1650mm_ID=3979.dat",
        # wiggler with correctors - gap 099.50mm
        "ID3992": "gap 099.50mm/2022-08-26_WigglerSTI_099.50mm_"
        + "U+0.00_D+0.00_Fieldmap_Z=-1650_1650mm_ID=3992.dat",
        # wiggler with correctors - gap 199.50mm
        "ID3993": "gap 199.50mm/2022-08-26_WigglerSTI_199.50mm_"
        + "U+0.00_D+0.00_Fieldmap_Z=-1650_1650mm_ID=3993.dat",
        # wiggler with correctors - gap 300.00mm
        "ID3994": "gap 300.00mm/2022-08-26_WigglerSTI_300.00mm_"
        + "U+0.00_D+0.00_Fieldmap_Z=-1650_1650mm_ID=3994.dat",
    }

    gap_dict = {
        (022.00): "ID3977",
        (023.00): "ID3986",
        (023.95): "ID3987",
        (026.90): "ID3988",
        (034.80): "ID3989",
        (042.70): "ID3990",
        (45): "ID4020",
        (49.73): "ID4019",
        (051.65): "ID3991",
        (059.60): "ID3979",
        (099.50): "ID3992",
        (199.50): "ID3993",
        (300.00): "ID3994",
    }

    def __init__(self):
        """Class constructor."""
        self._params = WIG180Data.PARAMS
        self.si_idmodel = None

    @staticmethod
    def _get_config_path(gap):
        path = ""
        gap_str = Tools.get_phase_str(gap)
        path += "gaps/gap_{}/".format(gap_str)
        return path

    @staticmethod
    def _get_kmap_config_name(gap):
        fname = "kickmap-ID"
        gap_str = Tools.get_phase_str(gap)
        fname += "_gap_{}".format(gap_str)
        return fname

    def get_fmap_fname(self, gap):
        """Get fmap fname.

        Args:
            gap (float): gap value

        Returns:
            str: fielmap name
        """
        fpath = self.folder_base_fieldmaps
        idconfig = self.gap_dict[(gap)]
        fname = self.FIELMAPS_CONFIGS[idconfig]
        fmap_fname = fpath + fname
        return fmap_fname


class EPU50Data(Tools):
    """EPU50 data access and manipulation class."""

    PARAMS = _PARAMS()
    PARAMS.PERIOD_LEN = 50  # [mm]
    PARAMS.ID_LEN = 2.770  # [m]
    PARAMS.NR_PERIODS = 54
    PARAMS.PPARAMETER_NAME = "phase"
    PARAMS.KPARAMETER_NAME = "gap"
    PARAMS.ID_FAMNAME = "EPU50"
    PARAMS.SUBSECS = ["ID10SB"]

    PARAMS.KICKMAPS_DATA_PATH = REPOS_PATH + "scripts/epu50/kickmaps/"
    PARAMS.FIELDMAPS_DATA_PATH = (
        DATA_REPOS_PATH
        + "lnls-ima/epu-uvx/measurement/magnetic/hallprobe/probes 133-14/"
    )  # noqa: E501
    PARAMS.FOLDER_BASE_OUTPUT = REPOS_PATH + "scripts/epu50/results/data/"

    FIELMAPS_CONFIGS = {
        # sensor 133-14 (without crosstalk) - gap 22.0 mm
        "ID4079": "gap 22.0mm/2022-10-19_"
        + "EPU_gap22.0_fase00.00_X=-20_20mm_Z=-1800_1800mm_ID=4079.dat",
        "ID4080": "gap 22.0mm/2022-10-20_"
        + "EPU_gap22.0_fase16.39_X=-20_20mm_Z=-1800_1800mm_ID=4080.dat",
        "ID4082": "gap 22.0mm/2022-10-20_"
        + "EPU_gap22.0_fase25.00_X=-20_20mm_Z=-1800_1800mm_ID=4082.dat",
        "ID4081": "gap 22.0mm/2022-10-20_"
        + "EPU_gap22.0_fase-16.39_X=-20_20mm_Z=-1800_1800mm_ID=4081.dat",
        "ID4083": "gap 22.0mm/2022-10-20_"
        + "EPU_gap22.0_fase-25.00_X=-20_20mm_Z=-1800_1800mm_ID=4083.dat",
        # sensor 133-14 (without crosstalk) - gap 23.3 mm
        "ID4099": "probes 133-14/gap 23.3mm/2022-10-24_"
        + "EPU_gap23.3_fase00.00_X=-20_20mm_Z=-1800_1800mm_ID=4099.dat",
        "ID4100": "probes 133-14/gap 23.3mm/2022-10-24_"
        + "EPU_gap23.3_fase16.39_X=-20_20mm_Z=-1800_1800mm_ID=4100.dat",
        "ID4102": "probes 133-14/gap 23.3mm/2022-10-24_"
        + "EPU_gap23.3_fase25.00_X=-20_20mm_Z=-1800_1800mm_ID=4102.dat",
        "ID4101": "probes 133-14/gap 23.3mm/2022-10-24_"
        + "EPU_gap23.3_fase-16.39_X=-20_20mm_Z=-1800_1800mm_ID=4101.dat",
        "ID4103": "probes 133-14/gap 23.3mm/2022-10-24_"
        + "EPU_gap23.3_fase-25.00_X=-20_20mm_Z=-1800_1800mm_ID=4103.dat",
        # sensor 133-14 (without crosstalk) - gap 25.7 mm
        "ID4084": "probes 133-14/gap 25.7mm/2022-10-20_"
        + "EPU_gap25.7_fase00.00_X=-20_20mm_Z=-1800_1800mm_ID=4084.dat",
        "ID4085": "probes 133-14/gap 25.7mm/2022-10-20_"
        + "EPU_gap25.7_fase16.39_X=-20_20mm_Z=-1800_1800mm_ID=4085.dat",
        "ID4087": "probes 133-14/gap 25.7mm/2022-10-20_"
        + "EPU_gap25.7_fase25.00_X=-20_20mm_Z=-1800_1800mm_ID=4087.dat",
        "ID4086": "probes 133-14/gap 25.7mm/2022-10-20_"
        + "EPU_gap25.7_fase-16.39_X=-20_20mm_Z=-1800_1800mm_ID=4086.dat",
        "ID4088": "probes 133-14/gap 25.7mm/2022-10-21_"
        + "EPU_gap25.7_fase-25.00_X=-20_20mm_Z=-1800_1800mm_ID=4088.dat",
        # sensor 133-14 (without crosstalk) - gap 29.3 mm
        "ID4089": "probes 133-14/gap 29.3mm/2022-10-21_"
        + "EPU_gap29.3_fase00.00_X=-20_20mm_Z=-1800_1800mm_ID=4089.dat",
        "ID4090": "probes 133-14/gap 29.3mm/2022-10-21_"
        + "EPU_gap29.3_fase16.39_X=-20_20mm_Z=-1800_1800mm_ID=4090.dat",
        "ID4092": "probes 133-14/gap 29.3mm/2022-10-21_"
        + "EPU_gap29.3_fase25.00_X=-20_20mm_Z=-1800_1800mm_ID=4092.dat",
        "ID4091": "probes 133-14/gap 29.3mm/2022-10-21_"
        + "EPU_gap29.3_fase-16.39_X=-20_20mm_Z=-1800_1800mm_ID=4091.dat",
        "ID4093": "probes 133-14/gap 29.3mm/2022-10-21_"
        + "EPU_gap29.3_fase-25.00_X=-20_20mm_Z=-1800_1800mm_ID=4093.dat",
        # sensor 133-14 (without crosstalk) - gap 32.5 mm
        "ID4104": "probes 133-14/gap 32.5mm/2022-10-25_"
        + "EPU_gap32.5_fase00.00_X=-20_20mm_Z=-1800_1800mm_ID=4104.dat",
        "ID4105": "probes 133-14/gap 32.5mm/2022-10-25_"
        + "EPU_gap32.5_fase16.39_X=-20_20mm_Z=-1800_1800mm_ID=4105.dat",
        "ID4107": "probes 133-14/gap 32.5mm/2022-10-25_"
        + "EPU_gap32.5_fase25.00_X=-20_20mm_Z=-1800_1800mm_ID=4107.dat",
        "ID4106": "probes 133-14/gap 32.5mm/2022-10-25_"
        + "EPU_gap32.5_fase-16.39_X=-20_20mm_Z=-1800_1800mm_ID=4106.dat",
        "ID4108": "probes 133-14/gap 32.5mm/2022-10-25_"
        + "EPU_gap32.5_fase-25.00_X=-20_20mm_Z=-1800_1800mm_ID=4108.dat",
        # sensor 133-14 (without crosstalk) - gap 40.9 mm
        "ID4094": "probes 133-14/gap 40.9mm/2022-10-21_"
        + "EPU_gap40.9_fase00.00_X=-20_20mm_Z=-1800_1800mm_ID=4094.dat",
        "ID4095": "probes 133-14/gap 40.9mm/2022-10-21_"
        + "EPU_gap40.9_fase16.39_X=-20_20mm_Z=-1800_1800mm_ID=4095.dat",
        "ID4097": "probes 133-14/gap 40.9mm/2022-10-24_"
        + "EPU_gap40.9_fase25.00_X=-20_20mm_Z=-1800_1800mm_ID=4097.dat",
        "ID4096": "probes 133-14/gap 40.9mm/2022-10-24_"
        + "EPU_gap40.9_fase-16.39_X=-20_20mm_Z=-1800_1800mm_ID=4096.dat",
        "ID4098": "probes 133-14/gap 40.9mm/2022-10-24_"
        + "EPU_gap40.9_fase-25.00_X=-20_20mm_Z=-1800_1800mm_ID=4098.dat",
    }

    phase_gap_dict = {
        (0.00, 22.0): "ID4079",
        (16.39, 22.0): "ID4080",
        (25.00, 22.0): "ID4082",
        (-16.39, 22.0): "ID4081",
        (-25.00, 22.0): "ID4083",
        (0.00, 23.3): "ID4099",
        (16.39, 23.3): "ID4100",
        (25.00, 23.3): "ID4102",
        (-16.39, 23.3): "ID4101",
        (-25.00, 23.3): "ID4103",
        (0.00, 25.7): "ID4084",
        (16.39, 25.7): "ID4085",
        (25.00, 25.7): "ID4087",
        (-16.39, 25.7): "ID4086",
        (-25.00, 25.7): "ID4088",
        (0.00, 29.3): "ID4089",
        (16.39, 29.3): "ID4090",
        (25.00, 29.3): "ID4092",
        (-16.39, 29.3): "ID4091",
        (-25.00, 29.3): "ID4093",
        (0.00, 32.5): "ID4104",
        (16.39, 32.5): "ID4105",
        (25.00, 32.5): "ID4107",
        (-16.39, 32.5): "ID4106",
        (-25.00, 32.5): "ID4108",
        (0.00, 40.9): "ID4094",
        (16.39, 40.9): "ID4095",
        (25.00, 40.9): "ID4097",
        (-16.39, 40.9): "ID4096",
        (-25.00, 40.9): "ID4098",
    }

    def __init__(self):
        """Class constructor."""
        self._params = EPU50Data.PARAMS
        self.si_idmodel = None

    @staticmethod
    def _get_config_path(phase, gap):
        path = ""
        phase_str = Tools.get_phase_str(phase)
        path += "phases/phase_{}/".format(phase_str)
        gap_str = Tools.get_phase_str(gap)
        path += "gaps/gap_{}/".format(gap_str)
        return path

    @staticmethod
    def _get_kmap_config_name(phase, gap):
        fname = "kickmap-ID"
        phase_str = Tools.get_phase_str(phase)
        fname += "_phase_{}".format(phase_str)
        gap_str = Tools.get_phase_str(gap)
        fname += "_gap_{}".format(gap_str)
        return fname

    def get_fmap_fname(self, phase, gap):
        """Get fmap fname.

        Args:
            phase (float): phase value
            gap (float): gap value

        Returns:
            str: fielmap name
        """
        fpath = self.folder_base_fieldmaps
        idconfig = self.phase_gap_dict[(phase, gap)]
        fname = self.FIELMAPS_CONFIGS[idconfig]
        fmap_fname = fpath + fname
        return fmap_fname


class IVU18Data(Tools):
    """IVU18 data access and manipulation class."""

    PARAMS = _PARAMS()
    PARAMS.PERIOD_LEN = 18.5  # [mm]
    PARAMS.ID_LEN = 2.500  # [m]
    PARAMS.NR_PERIODS = 108
    PARAMS.KPARAMETER_NAME = "gap"
    PARAMS.ID_FAMNAME = "IVU18"
    PARAMS.SUBSECS = ["ID08SB", "ID14SB"]

    PARAMS.KICKMAPS_DATA_PATH = REPOS_PATH + "scripts/ivu18/kickmaps/"
    PARAMS.FIELDMAPS_DATA_PATH = None  # noqa: E501
    PARAMS.FOLDER_BASE_OUTPUT = REPOS_PATH + "scripts/ivu18/results/data/"

    FIELMAPS_CONFIGS = {None}

    gap_dict = {None}

    def __init__(self):
        """Class constructor."""
        self._params = IVU18Data.PARAMS
        self.si_idmodel = None

    @staticmethod
    def _get_config_path(gap):
        path = ""
        gap_str = Tools.get_gap_str(gap)
        path += "gaps/gap_{}/".format(gap_str)
        return path

    @staticmethod
    def _get_kmap_config_name(gap):
        fname = "kickmap-ID"
        gap_str = Tools.get_phase_str(gap)
        fname += "_gap_{}".format(gap_str)
        return fname

    def get_fmap_fname(self, gap):
        """Get fmap fname.

        Args:
            gap (float): gap value

        Returns:
            str: fielmap name
        """
        fpath = self.folder_base_fieldmaps
        idconfig = self.gap_dict[(gap)]
        fname = self.FIELMAPS_CONFIGS[idconfig]
        fmap_fname = fpath + fname
        return fmap_fname


class VPU29Data(Tools):
    """VPU29 data access and manipulation class."""

    PARAMS = _PARAMS()
    PARAMS.PERIOD_LEN = 29  # [mm]
    PARAMS.ID_LEN = 1.700  # [m]
    PARAMS.NR_PERIODS = 51
    PARAMS.KPARAMETER_NAME = "gap"
    PARAMS.ID_FAMNAME = "VPU29"
    PARAMS.SUBSECS = ["ID06SB", "ID07SP"]

    PARAMS.KICKMAPS_DATA_PATH = REPOS_PATH + "scripts/vpu29/kickmaps/"
    PARAMS.FIELDMAPS_DATA_PATH = None  # noqa: E501
    PARAMS.FOLDER_BASE_OUTPUT = REPOS_PATH + "scripts/vpu29/results/data/"

    FIELMAPS_CONFIGS = {None}

    gap_dict = {None}

    def __init__(self):
        """Class constructor."""
        self._params = VPU29Data.PARAMS
        self.si_idmodel = None

    @staticmethod
    def _get_config_path(gap):
        path = ""
        gap_str = Tools.get_gap_str(gap)
        path += "gaps/gap_{}/".format(gap_str)
        return path

    @staticmethod
    def _get_kmap_config_name(gap):
        fname = "kickmap-ID"
        gap_str = Tools.get_phase_str(gap)
        fname += "_gap_{}".format(gap_str)
        return fname

    def get_fmap_fname(self, gap):
        """Get fmap fname.

        Args:
            gap (float): gap value

        Returns:
            str: fielmap name
        """
        fpath = self.folder_base_fieldmaps
        idconfig = self.gap_dict[(gap)]
        fname = self.FIELMAPS_CONFIGS[idconfig]
        fmap_fname = fpath + fname
        return fmap_fname
