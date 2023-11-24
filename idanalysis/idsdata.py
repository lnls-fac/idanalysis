"""IDs configurations."""

from idanalysis import IDKickMap as _IDKickMap

class Tools:

    @staticmethod
    def get_gap_str(gap):
        gap_str = '{:04.1f}'.format(gap).replace('.', 'p')
        return gap_str

    @staticmethod
    def get_phase_str(phase):
        phase_str = '{:+07.3f}'.format(phase).replace('.', 'p')
        phase_str = phase_str.replace('+', 'pos').replace('-', 'neg')
        return phase_str

    @staticmethod
    def mkdir_function(mypath):
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
        fpath = self.folder_base_kickmaps
        if meas_flag:
            fpath += 'measurements/'
        else:
            fpath += 'model/'
        fname = self._get_kmap_config_name(**kwargs)
        fname = fpath + fname

        return fname

    def get_linear_kickmap_filename(self, meas_flag=False, **kwargs):
        fname = self.get_kickmap_filename(meas_flag=meas_flag, **kwargs)
        fname = fname.replace('.txt', '-linear.txt')
        return fname

    def get_shifted_kickmap_filename(self, meas_flag=False, **kwargs):
        fname = self.get_kickmap_filename(meas_flag=meas_flag, **kwargs)
        fname = fname.replace('.txt', '-shifted_on_axis.txt')
        return fname

    def get_filtered_kickmap_filename(self, meas_flag=False, **kwargs):
        fname = self.get_kickmap_filename(meas_flag=meas_flag, **kwargs)
        fname = fname.replace('.txt', '-filtered.txt')
        return fname

    def get_data_output_path(self, meas_flag=False, **kwargs):
        fpath = self.folder_base_output
        if meas_flag:
            fpath += 'measurements/'
        else:
            fpath += 'model/'
        config = self._get_config_path(**kwargs)
        fpath += config
        return fpath

    def get_meas_data_output_path(self, **kwargs):
        fpath = self.get_data_output_path(meas_flag=True, **kwargs)
        return fpath

    def get_model_data_output_path(self, **kwargs):
        fpath = self.get_data_output_path(meas_flag=False, **kwargs)
        return fpath

    def get_fmap(self, **kwargs):
        fmap_fname = self.get_fmap_fname(**kwargs)
        print(fmap_fname)
        idkickmap = _IDKickMap()
        idkickmap.fmap_fname = fmap_fname
        fmap = idkickmap.fmap_config.fmap
        return fmap


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

    # --- data parameters
    FIELDMAPS_DATA_PATH = None
    KICKMAPS_DATA_PATH = None
    FOLDER_BASE_OUTPUT = None


class DELTAData(Tools):
    """DELTA data access and manipulation class."""

    PARAMS = _PARAMS()
    PARAMS.PERIOD_LEN = 52.5  # [mm]
    PARAMS.ID_LEN = 1.200  # [m]
    PARAMS.NR_PERIODS = 21
    PARAMS.PPARAMETER_NAME = 'dp'
    PARAMS.KPARAMETER_NAME = 'dgv'
    PARAMS.ID_FAMNAME = 'DELTA52'

    PARAMS.KICKMAPS_DATA_PATH = '../scripts/delta52/kickmaps/'
    PARAMS.FIELDMAPS_DATA_PATH = '/opt/lnls-ima/delta52/id-sabia/model-03/measurement/magnetic/hallprobe/'
    PARAMS.FOLDER_BASE_OUTPUT = '../scripts/delta52/results/data/'

    FIELMAPS_CONFIGS = {

        # dp = 0 mm
        'ID4862': 'delta_sabia_final/LinH/2023-10-25_DeltaSabia_Phase=' +
        '0mm_GV=0mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_' +
        'ID=4862.dat',

        'ID4863': 'delta_sabia_final/LinH/2023-10-25_DeltaSabia_Phase=' +
        '0mm_GV=6.5625mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_' +
        'ID=4863.dat',

        'ID4864': 'delta_sabia_final/LinH/2023-10-25_DeltaSabia_Phase=' +
        '0mm_GV=13.125mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_' +
        'ID=4864.dat',

        'ID4865': 'delta_sabia_final/LinH/2023-10-25_DeltaSabia_Phase=' +
        '0mm_GV=19.6875mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_' +
        'ID=4865.dat',

        'ID4866': 'delta_sabia_final/LinH/2023-10-25_DeltaSabia_Phase=' +
        '0mm_GV=26.25mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_' +
        'ID=4866.dat',

        # dp = -26.25 mm
        'ID4872': 'delta_sabia_final/LinV/2023-10-26_DeltaSabia_Phase=' +
        '-26.25mm_GV=0mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_ID=4872.dat',

        'ID4873': 'delta_sabia_final/LinV/2023-10-26_DeltaSabia_Phase=' +
        '-26.25mm_GV=6.5625mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_' +
        'ID=4873.dat',

        'ID4874': 'delta_sabia_final/LinV/2023-10-26_DeltaSabia_Phase=' +
        '-26.25mm_GV=13.125mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_' +
        'ID=4874.dat',

        'ID4875': 'delta_sabia_final/LinV/2023-10-26_DeltaSabia_Phase=' +
        '-26.25mm_GV=19.6875mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_' +
        'ID=4875.dat',

        'ID4876': 'delta_sabia_final/LinV/2023-10-27_DeltaSabia_Phase=' +
        '-26.25mm_GV=26.25mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_' +
        'ID=4876.dat',

        # dp =  -13.125 mm
        'ID4867': 'delta_sabia_final/CircN/2023-10-25_DeltaSabia_Phase=' +
        '-13.125mm_GV=0mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_' +
        'ID=4867.dat',

        'ID4868': 'delta_sabia_final/CircN/2023-10-26_DeltaSabia_Phase=' +
        '-13.125mm_GV=6.5625mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_' +
        'ID=4868.dat',

        'ID4869': 'delta_sabia_final/CircN/2023-10-26_DeltaSabia_Phase=' +
        '-13.125mm_GV=13.125mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_' +
        'ID=4869.dat',

        'ID4870': 'delta_sabia_final/CircN/2023-10-26_DeltaSabia_Phase=' +
        '-13.125mm_GV=19.6875mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_' +
        'ID=4870.dat',

        'ID4871': 'delta_sabia_final/CircN/2023-10-26_DeltaSabia_Phase=' +
        '-13.125mm_GV=26.25mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_' +
        'ID=4871.dat',

        # dp =  13.125 mm
        'ID4877': 'delta_sabia_final/CircP/2023-10-27_DeltaSabia_Phase=' +
        '13.125mm_GV=0mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_' +
        'ID=4877.dat',

        'ID4878': 'delta_sabia_final/CircP/2023-10-27_DeltaSabia_Phase=' +
        '13.125mm_GV=-6.5625mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_' +
        'ID=4878.dat',

        'ID4879': 'delta_sabia_final/CircP/2023-10-27_DeltaSabia_Phase=' +
        '13.125mm_GV=-13.125mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_' +
        'ID=4879.dat',

        'ID4880': 'delta_sabia_final/CircP/2023-10-27_DeltaSabia_Phase=' +
        '13.125mm_GV=-19.6875mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_' +
        'ID=4880.dat',

        'ID4881': 'delta_sabia_final/CircP/2023-10-27_DeltaSabia_Phase=' +
        '13.125mm_GV=-26.25mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_' +
        'ID=4881.dat',

        # zero K
        'ID4884': 'delta_sabia_final/zeroK/2023-10-27_DeltaSabia_Phase=' +
        '6.5625mm_GV=0mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_' +
        'ID=4884.dat',

        'ID4882': 'delta_sabia_final/zeroK/2023-10-27_DeltaSabia_Phase=' +
        '-6.5625mm_GV=0mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_' +
        'ID=4882.dat',

        'ID4883': 'delta_sabia_final/zeroK/2023-10-27_DeltaSabia_Phase=' +
        '-19.6875mm_GV=0mm_Fieldmap_Corrected_X=-5_5mm_Z=-900_900mm_' +
        'ID=4883.dat',

    }

    dp_dgv_dict = {
               (0, 0): 'ID4862',
               (0, 6.5625): 'ID4863',
               (0, 13.125): 'ID4864',
               (0, 19.6875): 'ID4865',
               (0, 26.25): 'ID4866',

               (-13.125, 0): 'ID4867',
               (-13.125, 6.5625): 'ID4868',
               (-13.125, 13.125): 'ID4869',
               (-13.125, 19.6875): 'ID4870',
               (-13.125, 26.25): 'ID4871',

               (13.125, 0): 'ID4877',
               (13.125, 6.5625): 'ID4878',
               (13.125, 13.125): 'ID4879',
               (13.125, 19.6875): 'ID4880',
               (13.125, 26.25): 'ID4881',

               (-26.25, 0): 'ID4872',
               (-26.25, 6.5625): 'ID4873',
               (-26.25, 13.125): 'ID4874',
               (-26.25, 19.6875): 'ID4875',
               (-26.25, 26.25): 'ID4876',

               (6.5625, 0): 'ID4884',
               (-6.5625, 0): 'ID4882',
               (-19.6875, 0): 'ID4883'}

    def __init__(self):
        self._params = DELTAData.PARAMS

    @property
    def period_length(self):
        return self._params.PERIOD_LEN

    @property
    def id_length(self):
        return self._params.ID_LEN

    @property
    def nr_periods(self):
        return self._params.NR_PERIODS

    @property
    def id_famname(self):
        return self._params.ID_FAMNAME

    @property
    def folder_base_output(self):
        return self._params.FOLDER_BASE_OUTPUT

    @property
    def folder_base_kickmaps(self):
        return self._params.KICKMAPS_DATA_PATH

    @property
    def folder_base_fieldmaps(self):
        return self._params.FIELDMAPS_DATA_PATH

    @staticmethod
    def _get_config_path(dp, dgv):
        path = ''
        dp_str = Tools.get_phase_str(dp)
        path += 'dps/dp_{}/'.format(dp_str)
        dgv_str = Tools.get_phase_str(dgv)
        path += 'dgvs/dgv_{}/'.format(dgv_str)
        return path

    @staticmethod
    def _get_kmap_config_name(dp, dgv):
        fname = 'kickmap-ID'
        dp_str = Tools.get_phase_str(dp)
        fname += '_dp_{}'.format(dp_str)
        dgv_str = Tools.get_phase_str(dgv)
        fname += '_dgv_{}'.format(dgv_str)
        return fname

    def get_fmap_fname(self, dp, dgv):
        fpath = self.folder_base_fieldmaps
        idconfig = self.dp_dgv_dict[(dp, dgv)]
        fname = self.FIELMAPS_CONFIGS[idconfig]
        fmap_fname = fpath + fname
        return fmap_fname
