"""IDs configurations."""

import numpy as _np
import matplotlib.pyplot as _plt

import pyaccel
from fieldmaptrack.fieldmap import FieldMap as _FieldMap


class DeltaData:
    """Delta fieldmap and kickmap data access class."""
    class _Aux:
        """Auxilliary consts."""

        of, nf = '', '2021-08-11/'
        lioH_ = '_DeltaSabia_LinearPolarization_' # old name
        lioV_ = '_DeltaSabia_LinearPolarization_' # old name
        lioHZ = '_DeltaSabia_LinearPolarization_' # old_name
        linH_ = '_DeltaSabia_HorizontalPolarization_'
        linVZ = '_DeltaSabia_VerticalPolarizationZero_'
        cirNZ = '_DeltaSabia_LHCircularPolarizationZero_'
        cirN_ = '_DeltaSabia_LHCircularPolarization_'
        cirPZ = '_DeltaSabia_RHCircularPolarizationZero_'
        ellip = '_DeltaSabia_EllipticalPolarization_'
        lincp = '_DeltaSabia_LinearPolarization_'
        su = '_X=-7_7mm_Y=-4_4mm_Z=-1000_1000mm'
        d1 = '2021-03-13'
        d2 = '2021-08-10'
        d3 = '2021-03-15'
        d4 = '2021-08-09'
        d1 = '2021-03-13'
        d3 = '2021-03-15'
        d2 = '2021-08-10'
        d5 = '2021-03-17'
        d1 = '2021-03-13'
        d4 = '2021-08-09'
        d4 = '2021-08-09'
        d2 = '2021-08-10'
        d1 = '2021-03-13'
        d6 = '2021-03-14'
        d1 = '2021-03-13'
        d6 = '2021-03-14'
        d6 = '2021-03-14'
        d6 = '2021-03-14'
        d5 = '2021-03-17'
        d6 = '2021-03-14'
        d1 = '2021-03-13'
        d6 = '2021-03-14'

        of1 = 'circular_polarization/kzero/'

        su2 = '_X=-7_7mm_Y=-4_4mm_Z=-800_800mm.kck'
        linVZ = '_DeltaSabia_VerticalPolarizationZero_'
        cirN = '_DeltaSabia_LHCircularPolarization_'

        delta_configs_err = [
            'circular_polarization/kzero/2021-08-13_DeltaSabia_RandomModelXX_LHCircularPolarization_Kh=0.0_Kv=0.0_dGV' + su2,
            'circular_polarization/kmid/2021-08-13_DeltaSabia_RandomModelXX_LHCircularPolarization_Kh=3.0_Kv=3.1_dGV' + su2,
            'circular_polarization/kmax/2021-08-12_DeltaSabia_RandomModelXX_LHCircularPolarization_Kh=4.3_Kv=4.3' + su2,
            
            'horizontal_polarization/kzero/2021-08-14_DeltaSabia_RandomModelXX_HorizontalPolarization_Kh=0.0_Kv=0.0_dGV' + su2,
            'horizontal_polarization/kmid/2021-08-14_DeltaSabia_RandomModelXX_HorizontalPolarization_Kh=4.4_Kv=0.0_dGV' + su2,
            'horizontal_polarization/kmax/2021-08-14_DeltaSabia_RandomModelXX_HorizontalPolarization_Kh=6.1_Kv=0.0' + su2,

            'vertical_polarization/kzero/2021-08-15_DeltaSabia_RandomModelXX_VerticalPolarization_Kh=0.0_Kv=0.0_dGV' + su2,
            'vertical_polarization/kmid/2021-08-15_DeltaSabia_RandomModelXX_VerticalPolarization_Kh=0.0_Kv=4.3_dGV' + su2,
            'vertical_polarization/kmax/2021-08-15_DeltaSabia_RandomModelXX_VerticalPolarization_Kh=0.0_Kv=6.1' + su2,
        ]

    FOLDER_SABIA = 'id-sabia/model-03/simulation/magnetic/'
    FOLDER_SABIA_MAPS = ( FOLDER_SABIA + 'fieldmaps_and_kickmaps/')
    FOLDER_SABIA_MAPS_ERR = ( FOLDER_SABIA + 'fieldmaps_and_kickmaps_with_errors/')

    SABIA_CONFIGS = (
        #                                                                            polarization variation
        #                                                                         |-------------------------|
        #                                                                     energy (K) variation (dGV)
        #                                                                    |-------------------------|
        _Aux.of + _Aux.d1 + _Aux.lioHZ + 'Kh=0.0_Kv=0.0_+dGV'   + _Aux.su,  # CSD: +26.250 CSE: +26.250 CID: +00.000 CIE: +00.000
        _Aux.nf + _Aux.d2 + _Aux.linH_ + 'Kh=4.33_Kv=0.0_-dGV'  + _Aux.su,  # CSD: -13.125 CSE: -13.125 CID: +00.000 CIE: +00.000
        _Aux.of + _Aux.d3 + _Aux.lioH_ + 'Kh=5.99_Kv=0.0'       + _Aux.su,  # CSD: +00.000 CSE: +00.000 CID: +00.000 CIE: +00.000
        _Aux.nf + _Aux.d4 + _Aux.linVZ + 'Kh=0.0_Kv=0.0_-dGV'   + _Aux.su,  # CSD: -26.250 CSE: +00.000 CID: +26.250 CIE: +00.000
        _Aux.of + _Aux.d1 + _Aux.lioV_ + 'Kh=0.0_Kv=4.25_-dGV'  + _Aux.su,  # CSD: -13.125 CSE: +13.125 CID: +26.250 CIE: +00.000
        _Aux.of + _Aux.d3 + _Aux.lioV_ + 'Kh=0.0_Kv=5.99'       + _Aux.su,  # CSD: +00.000 CSE: +26.250 CID: +26.250 CIE: +00.000
        _Aux.nf + _Aux.d2 + _Aux.cirPZ + 'Kh=0.0_Kv=0.0_-dGV'   + _Aux.su,  # CSD: -26.250 CSE: -13.125 CID: +13.125 CIE: +00.000
        _Aux.of + _Aux.d5 + _Aux.ellip + 'Kh=2.98_Kv=3.03_-dGV' + _Aux.su,  # CSD: -13.125 CSE: +00.000 CID: +13.125 CIE: +00.000
        _Aux.of + _Aux.d1 + _Aux.ellip + 'Kh=4.27_Kv=4.27'      + _Aux.su,  # CSD: +00.000 CSE: +13.125 CID: +13.125 CIE: +00.000
        _Aux.nf + _Aux.d4 + _Aux.cirNZ + 'Kh=0.0_Kv=0.0_+dGV'   + _Aux.su,  # CSD: +26.250 CSE: +13.125 CID: -13.125 CIE: +00.000
        _Aux.nf + _Aux.d4 + _Aux.cirN_ + 'Kh=2.98_Kv=3.03_+dGV' + _Aux.su,  # CSD: +13.125 CSE: +00.000 CID: -13.125 CIE: +00.000
        _Aux.nf + _Aux.d2 + _Aux.cirN_ + 'Kh=4.27_Kv=4.27'      + _Aux.su,  # CSD: +00.000 CSE: -13.125 CID: -13.125 CIE: +00.000
        _Aux.of + _Aux.d1 + _Aux.ellip + 'Kh=1.64_Kv=3.91_-dGV' + _Aux.su,  # CSD: -13.125 CSE: +06.562 CID: +19.688 CIE: +00.000
        _Aux.of + _Aux.d6 + _Aux.ellip + 'Kh=2.23_Kv=5.58'      + _Aux.su,  # CSD: +00.000 CSE: +19.688 CID: +19.688 CIE: +00.000
        _Aux.of + _Aux.d1 + _Aux.ellip + 'Kh=3.94_Kv=1.71_-dGV' + _Aux.su,  # CSD: -13.125 CSE: -06.562 CID: +06.562 CIE: +00.000
        _Aux.of + _Aux.d1 + _Aux.ellip + 'Kh=5.58_Kv=2.23'      + _Aux.su,  # CSD: +00.000 CSE: +06.562 CID: +06.562 CIE: +00.000
        _Aux.of + _Aux.d6 + _Aux.lioHZ + 'Kh=0.0_Kv=0.0_+dGH'   + _Aux.su,  # CSD: +00.000 CSE: +26.250 CID: +00.000 CIE: +26.250
        _Aux.of + _Aux.d6 + _Aux.lioV_ + 'Kh=0.0_Kv=4.33_-dGH'  + _Aux.su,  # CSD: +00.000 CSE: +13.125 CID: +26.250 CIE: -13.125
        _Aux.of + _Aux.d6 + _Aux.ellip + 'Kh=3.91_Kv=1.64_-dGH' + _Aux.su,  # CSD: +00.000 CSE: -06.562 CID: +06.562 CIE: -13.125
        _Aux.of + _Aux.d5 + _Aux.ellip + 'Kh=3.03_Kv=2.98_-dGH' + _Aux.su,  # CSD: +00.000 CSE: +00.000 CID: +13.125 CIE: -13.125
        _Aux.of + _Aux.d6 + _Aux.ellip + 'Kh=1.71_Kv=3.94_-dGH' + _Aux.su,  # CSD: +00.000 CSE: +06.562 CID: +19.688 CIE: -13.125
        _Aux.of + _Aux.d6 + _Aux.lincp + 'Kh=2.99_Kv=2.99_+dCP' + _Aux.su,  # CSD: +00.000 CSE: +13.125 CID: -13.125 CIE: +00.000
        )

    SABIA_CONFIGS_ERR = (
        'horizontal_polarization/kzero/2021-08-14_DeltaSabia_RandomModel01_HorizontalPolarization_Kh=0.0_Kv=0.0_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'horizontal_polarization/kzero/2021-08-14_DeltaSabia_RandomModel02_HorizontalPolarization_Kh=0.0_Kv=0.0_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'horizontal_polarization/kzero/2021-08-14_DeltaSabia_RandomModel03_HorizontalPolarization_Kh=0.0_Kv=0.0_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'horizontal_polarization/kzero/2021-08-14_DeltaSabia_RandomModel04_HorizontalPolarization_Kh=0.0_Kv=0.0_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'horizontal_polarization/kzero/2021-08-14_DeltaSabia_RandomModel05_HorizontalPolarization_Kh=0.0_Kv=0.0_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'horizontal_polarization/kzero/2021-08-14_DeltaSabia_RandomModel06_HorizontalPolarization_Kh=0.0_Kv=0.0_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'horizontal_polarization/kzero/2021-08-14_DeltaSabia_RandomModel07_HorizontalPolarization_Kh=0.0_Kv=0.0_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'horizontal_polarization/kzero/2021-08-14_DeltaSabia_RandomModel08_HorizontalPolarization_Kh=0.0_Kv=0.0_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'horizontal_polarization/kzero/2021-08-14_DeltaSabia_RandomModel09_HorizontalPolarization_Kh=0.0_Kv=0.0_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'horizontal_polarization/kzero/2021-08-14_DeltaSabia_RandomModel10_HorizontalPolarization_Kh=0.0_Kv=0.0_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'horizontal_polarization/kmid/2021-08-14_DeltaSabia_RandomModel01_HorizontalPolarization_Kh=4.4_Kv=0.0_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'horizontal_polarization/kmid/2021-08-14_DeltaSabia_RandomModel02_HorizontalPolarization_Kh=4.4_Kv=0.0_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'horizontal_polarization/kmid/2021-08-14_DeltaSabia_RandomModel03_HorizontalPolarization_Kh=4.4_Kv=0.0_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'horizontal_polarization/kmid/2021-08-14_DeltaSabia_RandomModel04_HorizontalPolarization_Kh=4.4_Kv=0.0_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'horizontal_polarization/kmid/2021-08-14_DeltaSabia_RandomModel05_HorizontalPolarization_Kh=4.4_Kv=0.0_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'horizontal_polarization/kmid/2021-08-14_DeltaSabia_RandomModel06_HorizontalPolarization_Kh=4.4_Kv=0.0_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'horizontal_polarization/kmid/2021-08-14_DeltaSabia_RandomModel07_HorizontalPolarization_Kh=4.4_Kv=0.0_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'horizontal_polarization/kmid/2021-08-14_DeltaSabia_RandomModel08_HorizontalPolarization_Kh=4.4_Kv=0.0_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'horizontal_polarization/kmid/2021-08-14_DeltaSabia_RandomModel09_HorizontalPolarization_Kh=4.4_Kv=0.0_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'horizontal_polarization/kmid/2021-08-14_DeltaSabia_RandomModel10_HorizontalPolarization_Kh=4.4_Kv=0.0_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'horizontal_polarization/kmax/2021-08-14_DeltaSabia_RandomModel01_HorizontalPolarization_Kh=6.1_Kv=0.0_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'horizontal_polarization/kmax/2021-08-14_DeltaSabia_RandomModel02_HorizontalPolarization_Kh=6.1_Kv=0.0_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'horizontal_polarization/kmax/2021-08-14_DeltaSabia_RandomModel03_HorizontalPolarization_Kh=6.1_Kv=0.0_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'horizontal_polarization/kmax/2021-08-14_DeltaSabia_RandomModel04_HorizontalPolarization_Kh=6.1_Kv=0.0_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'horizontal_polarization/kmax/2021-08-14_DeltaSabia_RandomModel05_HorizontalPolarization_Kh=6.1_Kv=0.0_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'horizontal_polarization/kmax/2021-08-14_DeltaSabia_RandomModel06_HorizontalPolarization_Kh=6.1_Kv=0.0_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'horizontal_polarization/kmax/2021-08-14_DeltaSabia_RandomModel07_HorizontalPolarization_Kh=6.1_Kv=0.0_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'horizontal_polarization/kmax/2021-08-14_DeltaSabia_RandomModel08_HorizontalPolarization_Kh=6.1_Kv=0.0_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'horizontal_polarization/kmax/2021-08-14_DeltaSabia_RandomModel09_HorizontalPolarization_Kh=6.1_Kv=0.0_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'horizontal_polarization/kmax/2021-08-14_DeltaSabia_RandomModel10_HorizontalPolarization_Kh=6.1_Kv=0.0_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'circular_polarization/kzero/2021-08-13_DeltaSabia_RandomModel01_LHCircularPolarization_Kh=0.0_Kv=0.0_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'circular_polarization/kzero/2021-08-13_DeltaSabia_RandomModel02_LHCircularPolarization_Kh=0.0_Kv=0.0_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'circular_polarization/kzero/2021-08-13_DeltaSabia_RandomModel03_LHCircularPolarization_Kh=0.0_Kv=0.0_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'circular_polarization/kzero/2021-08-13_DeltaSabia_RandomModel04_LHCircularPolarization_Kh=0.0_Kv=0.0_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'circular_polarization/kzero/2021-08-13_DeltaSabia_RandomModel05_LHCircularPolarization_Kh=0.0_Kv=0.0_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'circular_polarization/kzero/2021-08-13_DeltaSabia_RandomModel06_LHCircularPolarization_Kh=0.0_Kv=0.0_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'circular_polarization/kzero/2021-08-13_DeltaSabia_RandomModel07_LHCircularPolarization_Kh=0.0_Kv=0.0_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'circular_polarization/kzero/2021-08-13_DeltaSabia_RandomModel08_LHCircularPolarization_Kh=0.0_Kv=0.0_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'circular_polarization/kzero/2021-08-13_DeltaSabia_RandomModel09_LHCircularPolarization_Kh=0.0_Kv=0.0_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'circular_polarization/kzero/2021-08-13_DeltaSabia_RandomModel10_LHCircularPolarization_Kh=0.0_Kv=0.0_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',        
        'circular_polarization/kmid/2021-08-13_DeltaSabia_RandomModel01_LHCircularPolarization_Kh=3.0_Kv=3.1_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'circular_polarization/kmid/2021-08-13_DeltaSabia_RandomModel02_LHCircularPolarization_Kh=3.0_Kv=3.1_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'circular_polarization/kmid/2021-08-13_DeltaSabia_RandomModel03_LHCircularPolarization_Kh=3.0_Kv=3.1_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'circular_polarization/kmid/2021-08-13_DeltaSabia_RandomModel04_LHCircularPolarization_Kh=3.0_Kv=3.1_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'circular_polarization/kmid/2021-08-13_DeltaSabia_RandomModel05_LHCircularPolarization_Kh=3.0_Kv=3.1_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'circular_polarization/kmid/2021-08-13_DeltaSabia_RandomModel06_LHCircularPolarization_Kh=3.0_Kv=3.1_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'circular_polarization/kmid/2021-08-13_DeltaSabia_RandomModel07_LHCircularPolarization_Kh=3.0_Kv=3.1_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'circular_polarization/kmid/2021-08-13_DeltaSabia_RandomModel08_LHCircularPolarization_Kh=3.0_Kv=3.1_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'circular_polarization/kmid/2021-08-13_DeltaSabia_RandomModel09_LHCircularPolarization_Kh=3.0_Kv=3.1_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'circular_polarization/kmid/2021-08-13_DeltaSabia_RandomModel10_LHCircularPolarization_Kh=3.0_Kv=3.1_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'circular_polarization/kmax/2021-08-12_DeltaSabia_RandomModel01_LHCircularPolarization_Kh=4.3_Kv=4.3_X=-7_7mm_Y=-4_4mm_Z=-800_800mm', 
        'circular_polarization/kmax/2021-08-12_DeltaSabia_RandomModel02_LHCircularPolarization_Kh=4.3_Kv=4.3_X=-7_7mm_Y=-4_4mm_Z=-800_800mm', 
        'circular_polarization/kmax/2021-08-12_DeltaSabia_RandomModel03_LHCircularPolarization_Kh=4.3_Kv=4.3_X=-7_7mm_Y=-4_4mm_Z=-800_800mm', 
        'circular_polarization/kmax/2021-08-12_DeltaSabia_RandomModel04_LHCircularPolarization_Kh=4.3_Kv=4.3_X=-7_7mm_Y=-4_4mm_Z=-800_800mm', 
        'circular_polarization/kmax/2021-08-12_DeltaSabia_RandomModel05_LHCircularPolarization_Kh=4.3_Kv=4.3_X=-7_7mm_Y=-4_4mm_Z=-800_800mm', 
        'circular_polarization/kmax/2021-08-12_DeltaSabia_RandomModel06_LHCircularPolarization_Kh=4.3_Kv=4.3_X=-7_7mm_Y=-4_4mm_Z=-800_800mm', 
        'circular_polarization/kmax/2021-08-12_DeltaSabia_RandomModel07_LHCircularPolarization_Kh=4.3_Kv=4.3_X=-7_7mm_Y=-4_4mm_Z=-800_800mm', 
        'circular_polarization/kmax/2021-08-12_DeltaSabia_RandomModel08_LHCircularPolarization_Kh=4.3_Kv=4.3_X=-7_7mm_Y=-4_4mm_Z=-800_800mm', 
        'circular_polarization/kmax/2021-08-12_DeltaSabia_RandomModel09_LHCircularPolarization_Kh=4.3_Kv=4.3_X=-7_7mm_Y=-4_4mm_Z=-800_800mm', 
        'circular_polarization/kmax/2021-08-12_DeltaSabia_RandomModel10_LHCircularPolarization_Kh=4.3_Kv=4.3_X=-7_7mm_Y=-4_4mm_Z=-800_800mm', 
        'vertical_polarization/kzero/2021-08-15_DeltaSabia_RandomModel01_VerticalPolarization_Kh=0.0_Kv=0.0_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'vertical_polarization/kzero/2021-08-15_DeltaSabia_RandomModel02_VerticalPolarization_Kh=0.0_Kv=0.0_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'vertical_polarization/kzero/2021-08-15_DeltaSabia_RandomModel03_VerticalPolarization_Kh=0.0_Kv=0.0_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'vertical_polarization/kzero/2021-08-15_DeltaSabia_RandomModel04_VerticalPolarization_Kh=0.0_Kv=0.0_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'vertical_polarization/kzero/2021-08-15_DeltaSabia_RandomModel05_VerticalPolarization_Kh=0.0_Kv=0.0_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'vertical_polarization/kzero/2021-08-15_DeltaSabia_RandomModel06_VerticalPolarization_Kh=0.0_Kv=0.0_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'vertical_polarization/kzero/2021-08-15_DeltaSabia_RandomModel07_VerticalPolarization_Kh=0.0_Kv=0.0_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'vertical_polarization/kzero/2021-08-15_DeltaSabia_RandomModel08_VerticalPolarization_Kh=0.0_Kv=0.0_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'vertical_polarization/kzero/2021-08-15_DeltaSabia_RandomModel09_VerticalPolarization_Kh=0.0_Kv=0.0_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'vertical_polarization/kzero/2021-08-15_DeltaSabia_RandomModel10_VerticalPolarization_Kh=0.0_Kv=0.0_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'vertical_polarization/kmid/2021-08-15_DeltaSabia_RandomModel01_VerticalPolarization_Kh=0.0_Kv=4.3_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'vertical_polarization/kmid/2021-08-15_DeltaSabia_RandomModel02_VerticalPolarization_Kh=0.0_Kv=4.3_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'vertical_polarization/kmid/2021-08-15_DeltaSabia_RandomModel03_VerticalPolarization_Kh=0.0_Kv=4.3_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'vertical_polarization/kmid/2021-08-15_DeltaSabia_RandomModel04_VerticalPolarization_Kh=0.0_Kv=4.3_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'vertical_polarization/kmid/2021-08-15_DeltaSabia_RandomModel05_VerticalPolarization_Kh=0.0_Kv=4.3_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'vertical_polarization/kmid/2021-08-15_DeltaSabia_RandomModel06_VerticalPolarization_Kh=0.0_Kv=4.3_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'vertical_polarization/kmid/2021-08-15_DeltaSabia_RandomModel07_VerticalPolarization_Kh=0.0_Kv=4.3_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'vertical_polarization/kmid/2021-08-15_DeltaSabia_RandomModel08_VerticalPolarization_Kh=0.0_Kv=4.3_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'vertical_polarization/kmid/2021-08-15_DeltaSabia_RandomModel09_VerticalPolarization_Kh=0.0_Kv=4.3_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'vertical_polarization/kmid/2021-08-15_DeltaSabia_RandomModel10_VerticalPolarization_Kh=0.0_Kv=4.3_dGV_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'vertical_polarization/kmax/2021-08-15_DeltaSabia_RandomModel01_VerticalPolarization_Kh=0.0_Kv=6.1_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'vertical_polarization/kmax/2021-08-15_DeltaSabia_RandomModel02_VerticalPolarization_Kh=0.0_Kv=6.1_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'vertical_polarization/kmax/2021-08-15_DeltaSabia_RandomModel03_VerticalPolarization_Kh=0.0_Kv=6.1_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'vertical_polarization/kmax/2021-08-15_DeltaSabia_RandomModel04_VerticalPolarization_Kh=0.0_Kv=6.1_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'vertical_polarization/kmax/2021-08-15_DeltaSabia_RandomModel05_VerticalPolarization_Kh=0.0_Kv=6.1_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'vertical_polarization/kmax/2021-08-15_DeltaSabia_RandomModel06_VerticalPolarization_Kh=0.0_Kv=6.1_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'vertical_polarization/kmax/2021-08-15_DeltaSabia_RandomModel07_VerticalPolarization_Kh=0.0_Kv=6.1_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'vertical_polarization/kmax/2021-08-15_DeltaSabia_RandomModel08_VerticalPolarization_Kh=0.0_Kv=6.1_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'vertical_polarization/kmax/2021-08-15_DeltaSabia_RandomModel09_VerticalPolarization_Kh=0.0_Kv=6.1_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',
        'vertical_polarization/kmax/2021-08-15_DeltaSabia_RandomModel10_VerticalPolarization_Kh=0.0_Kv=6.1_X=-7_7mm_Y=-4_4mm_Z=-800_800mm',)

    def __init__(self, folder, configs=None):
        """."""
        self._folder = folder
        if configs is None:
            configs = DeltaData.CONFIGS
        self._configs = configs
        self._header = dict()
        self._output_folder = ''

    @property
    def output_folder(self):
        """."""
        return self._output_folder

    @output_folder.setter
    def output_folder(self, value):
        """."""
        value = value if value[-1] == '/' else value + '/'
        self._output_folder = value

    def get_header(self, config):
        """."""
        return self._header[config]
        
    def get_fieldmap(self, config):
        """Return Trajectory object for a given config."""
        fname = self.get_fieldmap_filename(config)
        fmap = _FieldMap(fname=fname)
        label = self.get_config_label(config)
        return fmap, label

    def check_is_dGV(self, config):
        """Return True if config is a dGV configuration."""
        dGH = self.get_dGH(config)
        dCP = self.get_dCP(config)
        return dGH == 0 and dCP == 0

    def check_is_dGH(self, config):
        """Return True if config is a dGH configuration."""
        dGV = self.get_dGV(config)
        dCP = self.get_dCP(config)
        return dGV == 0 and dCP == 0

    def check_is_dCP(self, config):
        """Return True if config is a dCP configuration."""
        dCP = self.get_dCP(config)
        return dCP != 0

    def get_config_label(self, config):
        dP = self.get_dP(config)
        dGV = self.get_dGV(config)
        dCP = self.get_dCP(config)
        dGH = self.get_dGH(config)
        fstr = 'dP={:+7.4f}_dGV={:+7.4f}_dGH={:+7.4f}_dCP={:+7.4f}'
        return fstr.format(dP, dGV, dGH, dCP)

    def get_kickmap_filename(self, config):
        """Return kickmap filename of a config."""
        return self._folder + config + '.kck'

    def get_fieldmap_filename(self, config):
        """Return fieldmap filename of a config."""
        return self._folder + config + '.fld'

    def get_period_length(self, config=None):
        if config is None:
            config = self._configs[0]
        if config not in self._header:
            self._read_fieldmap_header(config)
        return self._header[config]['period_length']

    def get_dP(self, config, norm=None):
        """Return dP [mm] position of ID configuration."""
        if config not in self._header:
            self._read_fieldmap_header(config)
        data = self._header[config]
        if norm is None:
            return data['dP']/(self.get_period_length()/2)
        else:
            return data['dP']/norm

    def get_dGV(self, config, norm=None):
        """Return dGV [mm] position of ID configuration."""
        if config not in self._header:
            self._read_fieldmap_header(config)
        data = self._header[config]
        if norm is None:
            return data['dGV']/(self.get_period_length()/2)
        else:
            return data['dGV']/norm

    def get_dCP(self, config, norm=None):
        """Return dCP [mm] position of ID configuration."""
        if config not in self._header:
            self._read_fieldmap_header(config)
        data = self._header[config]
        if norm is None:
            return data['dCP']/(self.get_period_length()/2)
        else:
            return data['dCP']/norm

    def get_dGH(self, config, norm=None):
        """Return dGH [mm] position of ID configuration."""
        if config not in self._header:
            self._read_fieldmap_header(config)
        data = self._header[config]
        if norm is None:
            return data['dGH']/(self.get_period_length()/2)
        else:
            return data['dGH']/norm

    def get_dCSD(self, config, norm=None):
        """Return dCSD [mm] position of ID configuration."""
        if config not in self._header:
            self._read_fieldmap_header(config)
        data = self._header[config]
        if norm is None:
            return data['posCSD']/(self.get_period_length()/2)
        else:
            return data['posCSD']/norm

    def get_dCSE(self, config, norm=None):
        """Return dCSE [mm] position of ID configuration."""
        if config not in self._header:
            self._read_fieldmap_header(config)
        data = self._header[config]
        if norm is None:
            return data['posCSE']/(self.get_period_length()/2)
        else:
            return data['posCSE']/norm

    def get_dCID(self, config, norm=None):
        """Return dCID [mm] position of ID configuration."""
        if config not in self._header:
            self._read_fieldmap_header(config)
        data = self._header[config]
        if norm is None:
            return data['posCID']/(self.get_period_length()/2)
        else:
            return data['posCID']/norm

    def get_dCIE(self, config, norm=None):
        """Return dCIE [mm] position of ID configuration."""
        if config not in self._header:
            self._read_fieldmap_header(config)
        data = self._header[config]
        if norm is None:
            return data['posCIE']/(self.get_period_length()/2)
        else:
            return data['posCIE']/norm
    
    def get_config_names(self):
        names = []
        for config in self:
            names.append(config)
        return names

    # --- analysis ---

    def plot_dGV_config_space(self, save=False, plot=True):
        """Plot configs in dP x dGV ID space."""
        cse = _np.array(((1, 1, 0, -1, -1, 0, 1), (0, -1, -1, 0, +1, +1, 0)))
        lims = (3/2) * _np.array((-1, +1))
        linv = _np.array((lims, (+1, +1)))
        linh = _np.array((lims, (0, 0)))
        cirp = _np.array((lims, (+1/2, +1/2)))
        kzer = _np.array(((+1, +1), lims))
        kmax = _np.array(((0, 0), lims))

        ticks_labels = ['-1', '-3/4', '-2/4', '-1/4', '0', '+1/4', '+2/4', '+3/4', '+1']
        ticks = _np.array([eval(v) for v in ticks_labels])
        red, blu, mag, gre = (1,0,0), (0,0,1), (1,0,1), (0.4,0.4,0.4)
        
        dP = _np.array([self.get_dP(c) for c in self._configs])
        dGV = _np.array([self.get_dGV(c) for c in self._configs])
        
        # filter only dVG-type configs
        sel = _np.array(
            [self.check_is_dGV(c) for c in self._configs]).nonzero()[0]
        dP = dP[sel]
        dGV = dGV[sel]

        _plt.gcf()

        _plt.plot(linv[0], +linv[1], '--', color=red, label='LinV')
        _plt.plot(cirp[0], +cirp[1], '--', color=mag, label='cirP')
        _plt.plot(linh[0], +linh[1], '--', color=blu, label='LinH')
        _plt.plot(cirp[0], -cirp[1], '--', color=mag, label='cirN')
        _plt.plot(linv[0], -linv[1], '--', color=red)
        _plt.plot(cse[0], cse[1], color=(0,0,0), label='CSE')
        _plt.plot(+kzer[0], kzer[1], '-.', color=gre, label='Kzero')
        _plt.plot(-kzer[0], kzer[1], '-.', color=gre)
        _plt.plot(+kmax[0], kmax[1], '--', color=gre, label='Kmax')
        _plt.plot(dGV, dP, 'o', color=(0,0.7,0))

        _plt.xticks(ticks, ticks_labels)
        _plt.yticks(ticks, ticks_labels)
        _plt.xlim(lims[0], lims[1])
        _plt.ylim(lims[0], lims[1])
        _plt.xlabel(r'$dGV \; / \; (\lambda/2)$')
        _plt.ylabel(r'$dP \; / \; (\lambda/2)$')
        _plt.grid()
        _plt.legend()
        _plt.suptitle('Delta dGV Configuration Space')
        _plt.title('CSD = dGV; CSE = dGV + dP; CID = dP; CIE = 0')
        if save:
            _plt.savefig(self.output_folder + 'delta-dGVxdP.svg')
        if plot:
            _plt.show()
        
    def __getitem__(self, idx):
        """."""
        return self._configs[idx]

    def __len__(self):
        """."""
        return len(self._configs)

    def __iter__(self):
        """."""
        self._idx = 0
        return self

    def __next__(self):
        """."""
        if self._idx < len(self._configs):
            config = self._configs[self._idx]
            self._idx += 1
            return config
        else:
            raise StopIteration

    def _read_fieldmap_header(self, config):
        """Read fieldmap file header data."""
        data = dict()
        fname = self.get_fieldmap_filename(config)
        with open(fname, 'r') as fp:
            while True:
                line = fp.readline()
                if 'X[mm]' in line:
                    break
                words = line.split()
                if len(words) == 2:
                    wd = words[0]
                    wd = wd.replace(':', '')
                    wd = wd.replace('[mm]', '')
                    try:
                        data[wd] = float(words[1])
                    except ValueError:
                        data[wd] = words[1]
        self._header[config] = data

    def _read_kickmap(self, config):
        """."""
        fname = self.get_kickmap_filename(config)
        return pyaccel.elements.Kicktable(filename=fname)


