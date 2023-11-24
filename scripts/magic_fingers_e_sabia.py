import numpy as np
import radia as rad
from imaids.models import DeltaSabia
from imaids.magicfingers import MagicFingersSabia
from imaids.fieldsource import FieldModel

def create_d52_with_mf(move_type = 'k0'):

    undulator_movement_dict = {
        'k0':{},
        'hp_kmax':{'dgv':-26.25, 'dp':0},
        'vp_kmax':{'dgv':-26.25, 'dp':26.25},
        'cp_kmax':{'dgv':-26.25, 'dp':13.125},
        'hp_kmed':{'dgv':-13.125, 'dp':0},
        'vp_kmed':{'dgv':-13.125, 'dp':26.25},
        'cp_kmed':{'dgv':-13.125, 'dp':13.125}
        }

    # --------------- Define shifts list --------------- #

    shifts = [-3.17898653, -1.57540257,  3.09682008,  5.        ,  0.69483866,
               1.52490602, -3.70685746,  4.95300059,  2.2465917 ,  4.34806921,
              -4.93890742, -4.42491604,  0.53782199, -1.99751295, -3.15610805,
               2.981559  ]

    block_shift_list = [[0,y,0] for y in shifts]

    # ----------- Define magnetizations list ----------- #

    magnetizations = ['u', 'f', 'b', 'd', 'f', 'f', 'd', 'b',
                    'u', 'b', 'u', 'b', 'd', 'u', 'u', 'u']

    def label_mag(label, mr):
        if label == 'u':
            return [0, mr, 0] # "up"
        if label == 'd':
            return [0, -1*mr, 0] # "down"
        if label == 'f':
            return [0, 0, mr] # "forward"
        if label == 'b':
            return [0, 0, -1*mr] # "backward"

    magnetization_init_list = [label_mag(m, 1.37) for m in magnetizations]
    magnetization_init_list = np.array(magnetization_init_list)
    magnetization_init_list = magnetization_init_list.reshape(16,3).tolist()

    # ------------ Create undulator object ------------- #

    und = DeltaSabia()
    undulator_movement_kwargs = undulator_movement_dict[move_type]
    und.set_cassete_positions(**undulator_movement_kwargs)

    # ---------- Create magic fingers objects ---------- #

    dist_cas = 21.5
    magicpos = (und.cassettes['cse'].longitudinal_position_list[-1] +
                11.2662 + dist_cas)
                # Magic fingers position to coordinates origin:
                # = position of last block
                # + half width of the last block
                # + distance between magic fingers block center
                #   and face of cassette (end face of last block).

    magic1 = MagicFingersSabia(magnetization_init_list=magnetization_init_list,
                            block_shift_list=block_shift_list,
                            device_position=-magicpos)
    magic2= MagicFingersSabia(magnetization_init_list=magnetization_init_list,
                            block_shift_list=block_shift_list,
                            device_position=magicpos)

    dp = undulator_movement_kwargs.get('dp', 0)
    dgv = undulator_movement_kwargs.get('dgv', 0)
    # Movement of four magic fingers block sets from dp and dgv.
    magic_movement_list = [0, dp+dgv, dgv, dp]

    magic1.group_shift_list = magic_movement_list
    magic2.group_shift_list = magic_movement_list

    # ---------------- Create container ---------------- #

    # Display colors differentiating magic fingers and undulator.
    rad.ObjDrwAtr(und.radia_object, [0.7, 0.7, 0.7], 0.001)
    rad.ObjDrwAtr(magic1.radia_object, [0, 0.5, 1], 0.001)
    rad.ObjDrwAtr(magic2.radia_object, [0, 0.5, 1], 0.001)

    container = rad.ObjCnt([magic1.radia_object,
                            magic2.radia_object,
                            und.radia_object])

    return container

if __name__ == '__main__':

    d52wmf = create_d52_with_mf(move_type='hp_kmax')
    rad.ObjDrwOpenGL(d52wmf)
    print('Object display created (1/2)')

    # For using the lnls-ima/insertion-devices package:
    device = FieldModel(d52wmf)
    rad.ObjDrwOpenGL(device.radia_object) # Note: device.draw() does not
                                          # support different colors.
    print('Object display created (2/2)')

    input('Press any key to finish script')
