from common.h36m_dataset import Human36mDataset
from common.mocap_dataset import MocapDataset
from common.skeleton import Skeleton
from common.camera import normalize_screen_coordinates, image_coordinates
import numpy as np

class Estimated3dDatasetTears(Human36mDataset):
    def __mod_h36m_skeleton(self):

        j = self.Joints()
        joints_left = [j[k] for k in "lear leye lshoulder lelbow lwrist lhip lknee lancle".split(" ")]
        joints_right = [j[k] for k in "rear reye rshoulder relbow rwrist rhip rknee rancle".split(" ")]
        '''
        "lear": 3,
                "leye": 1,
                "reye": 2,
                "rear": 4,
                "nose": 0,
                #"mshoulder": 8,
                "lshoulder": 5,
                "rshoulder": 6,
                "lelbow": 7,
                "relbow": 8,
                "lwrist": 9,
                "rwrist": 10,
                #"chest": 7,
                #"mhip": 0,
                "lhip": 11,
                "rhip": 12,
                "lknee": 13,
                "rknee": 14,
                "lancle": 15,
                "rancle": 16}'''
        parents = [-1] * 17

        parents[j["leye"]] = j["lear"]
        parents[j["lear"]] = j["nose"]
        parents[j["lshoulder"]] = j["nose"]
        parents[j["lelbow"]] = j["lshoulder"]
        parents[j["lwrist"]] = j["lelbow"]
        parents[j["lhip"]] = j["lshoulder"]
        parents[j["lknee"]] = j["lhip"]
        parents[j["lancle"]] = j["lknee"]

        parents[j["reye"]] = j["rear"]
        parents[j["rear"]] = j["nose"]
        parents[j["rshoulder"]] = j["nose"]
        parents[j["relbow"]] = j["rshoulder"]
        parents[j["rwrist"]] = j["relbow"]
        parents[j["rhip"]] = j["rshoulder"]
        parents[j["rknee"]] = j["rhip"]
        parents[j["rancle"]] = j["rknee"]

        return Skeleton(parents=parents, joints_left=joints_left, joints_right=joints_right)

    def __init__(self,path):
        super(Estimated3dDatasetTears, self).__init__(path, remove_static_joints=False, skeleton=self.__mod_h36m_skeleton())

    def supports_semi_supervised(self):
        return False

    @staticmethod
    def Joints():
        return {#"tophead": 10,
                "lear": 3,
                "leye": 1,
                "reye": 2,
                "rear": 4,
                "nose": 0,
                #"mshoulder": 8,
                "lshoulder": 5,
                "rshoulder": 6,
                "lelbow": 7,
                "relbow": 8,
                "lwrist": 9,
                "rwrist": 10,
                #"chest": 7,
                #"mhip": 0,
                "lhip": 11,
                "rhip": 12,
                "lknee": 13,
                "rknee": 14,
                "lancle": 15,
                "rancle": 16}