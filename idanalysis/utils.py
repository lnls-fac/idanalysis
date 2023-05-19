import numpy as _np

from idanalysis import DeltaData, EPUData

DEFAULT_RANDOM_IDS = True
FOLDER_BASE = None  # Need to be defined by user!


def create_deltadata(random_ids=DEFAULT_RANDOM_IDS):
    if random_ids:
        folder = FOLDER_BASE + DeltaData.FOLDER_SABIA_MAPS_ERR
        configs = DeltaData.SABIA_CONFIGS_ERR
    else:
        folder = FOLDER_BASE + DeltaData.FOLDER_SABIA_MAPS
        configs = DeltaData.SABIA_CONFIGS
    deltadata = DeltaData(folder=folder, configs=configs)
    return deltadata


def create_epudata():

    folder = FOLDER_BASE + EPUData.FOLDER_EPU_MAPS
    configs = EPUData.EPU_CONFIGS
    epudata = EPUData(folder=folder, configs=configs)
    return epudata


def get_config_names(data):
    names = []
    for config in data:
        names.append(config)
    return names


def calc_rz_of_field_center(rz, bx, by, bz):
    """."""
    b2 = bx**2 + by**2 + bz**2
    sum_field2 = _np.sum(b2)
    rz_avg = _np.sum(rz*b2) / sum_field2
    return rz_avg
