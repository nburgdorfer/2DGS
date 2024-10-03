import os
import random
import numpy as np
import torch
import torch.utils.data as data
import cv2
import sys

from cvt.io import read_pfm
from cvt.camera import compute_baselines, _intrinsic_pyramid, scale_cam, crop_cam
from cvt.common import _build_depth_pyramid, _normalize_image, crop_image

def build_dataset(cfg, scene):
    if cfg["dataset"] == 'TNT':
        from src.datasets.TNT import TNT as Dataset
    elif cfg["dataset"] == 'DTU':
        from src.datasets.DTU import DTU as Dataset
    else:
        raise Exception(f"Unknown Dataset {self.cfg['dataset']}")

    return Dataset(cfg, scene)

class BaseDataset():
    def __init__(self, cfg, scene):
        self.cfg = cfg
        self.data_path = self.cfg["data_path"]
        self.device = self.cfg["device"]
        self.scene = scene
        self.crop_h = self.cfg["camera"]["crop_h"]
        self.crop_w = self.cfg["camera"]["crop_w"]
        self.scale = self.cfg["inference"]["scale"]

        self.set_paths()

        self.samples = self.build_samples()
        print(self.samples)
        sys.exit()

    def build_samples(self, frame_spacing):
        raise NotImplementedError()

    def load_intrinsics(self):
        raise NotImplementedError()

    def get_pose(self, frame_id):
        raise NotImplementedError()

    def get_image(self, image_file, scale=True):
        raise NotImplementedError()

    def get_depth(self, depth_file, scale=True):
        raise NotImplementedError()
