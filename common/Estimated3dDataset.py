from common.h36m_dataset import Human36mDataset
from common.mocap_dataset import MocapDataset
from common.skeleton import Skeleton
from common.camera import normalize_screen_coordinates, image_coordinates
import numpy as np


def mod_h36m_skeleton():

    joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23]
    joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31]

    parents=[-1,  0,  1,  2,  3,  4,  0,  6,  7,  8,  9,  0, 11, 12, 13, 14, 12,
           16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30]

    ids = list(range(32))
    for i in [4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31]:
        ids.remove(i)

    for i in range(len(joints_left)):
        try:
            joints_left[i] = ids.index(joints_left[i])
        except:
            joints_left[i] = -5

    for i in range(len(joints_right)):
        try:
            joints_right[i] = ids.index(joints_right[i])
        except:
            joints_right[i] = -5


    for i in range(len(parents)):
        if i in ids:
            if parents[i] == -1:
                continue
            try:
                parents[i] = ids.index(parents[i])
            except:
                parents[i] = 0
        else:
            parents[i] = -5



    joints_left = list(np.array(joints_left)[np.array(joints_left)!=-5])
    joints_right = list(np.array(joints_right)[np.array(joints_right)!=-5])
    parents = list(np.array(parents)[np.array(parents)!=-5])

    parents[11] = 8
    parents[14] = 8

    return Skeleton(parents=parents,joints_left=joints_left,joints_right=joints_right)


class Estimated3dDataset(Human36mDataset):
    def __init__(self,path):
        super(Estimated3dDataset, self).__init__(path, remove_static_joints=False, skeleton=mod_h36m_skeleton())

    def supports_semi_supervised(self):
        return False