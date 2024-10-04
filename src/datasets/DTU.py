import os,sys
import random
import numpy as np
import torch
import cv2
import sys
import json
from tqdm import tqdm
import yaml
import open3d as o3d

from cvt.common import _normalize_image
from cvt.camera import scale_cam
from cvt.io import read_single_cam_sfm, read_pfm

from src.datasets.BaseDataset import BaseDataset
from src.scene.gaussian_model import BasicPointCloud

class DTU(BaseDataset):
    def __init__(self, cfg, scene):
        super(DTU, self).__init__(cfg, scene)
        self.units = "mm"

        self.image_path = os.path.join(self.data_path, "Images", self.scene)
        self.depth_path = os.path.join(self.data_path, "Depths", self.scene)
        self.camera_path = os.path.join(self.data_path, "Cameras")
        self.points_file = os.path.join(self.data_path, "Sparse_Points", f"{self.scene}_sparse.ply")

    def get_cameras(self):
        cameras = []
        camera_files = os.listdir(self.camera_path)
        camera_files = [ f for f in camera_files if f[-8:] == "_cam.txt" ]
        camera_files.sort()
        for i, camera_file in tqdm(enumerate(camera_files), desc="Loading Camera Files", unit="file"):
            camera = read_single_cam_sfm(os.path.join(self.camera_path,camera_file))
            if(np.isnan(camera[0]).any()):
                print("ERROR: invalid camera extrinsics matrix '{camera_file}': {camera[0]}")
                sys.exit()

            camera[1,0,2] -= (self.crop_w//2)
            camera[1,1,2] -= (self.crop_h//2)

            if i==0:
                self.H = int(self.scale * (self.cfg["camera"]["height"] - self.crop_h))
                self.W = int(self.scale * (self.cfg["camera"]["width"]- self.crop_w))

            cameras.append(camera)
        return np.asarray(cameras, dtype=np.float32)

    def get_images(self):
        image_files = os.listdir(self.image_path)
        image_files = [ f for f in image_files if f[-3:] == "png" ]
        image_files.sort()

        images = []
        for image_file in tqdm(image_files, desc="Loading Image Files", unit="file"):
            images.append(self._get_image(os.path.join(self.image_path, image_file)))
        return np.asarray(images, dtype=np.float32)

    def get_depths(self):
        depth_files = os.listdir(self.depth_path)
        depth_files = [ f for f in depth_files if f[-3:] == "pfm" ]
        depth_files.sort()

        depths = []
        for depth_file in tqdm(depth_files, desc="Loading Depth Files", unit="file"):
            depths.append(self._get_depth(os.path.join(self.depth_path, depth_file)))
        return np.asarray(depths, dtype=np.float32)

    def get_points(self):
        cloud = o3d.io.read_point_cloud(self.points_file)
        #cloud = cloud.voxel_down_sample(voxel_size=self.voxel_size)
        positions = np.asarray(cloud.points).astype(np.float32)
        colors = np.asarray(cloud.colors).astype(np.float32)
        if (len(cloud.normals) == 0):
            cloud.estimate_normals()
        normals = np.asarray(cloud.normals).astype(np.float32)
        return BasicPointCloud(points=positions, colors=colors, normals=normals)

    def _get_image(self, image_file):
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # crop and resize image
        h,w,_ = image.shape
        image = image[(self.crop_h//2):h-(self.crop_h//2),(self.crop_w//2):w-(self.crop_w//2), :]
        image = cv2.resize(image, (self.W,self.H), interpolation=cv2.INTER_LINEAR)
        image = _normalize_image(image)
        #image = np.moveaxis(image, [0,1,2], [1,2,0]) # [3 x H x W]
        image = image[:,:,::-1] # [H x W x 3] [RGB]
        return image.astype(np.float32)

    def _get_depth(self, depth_file):
        depth = read_pfm(depth_file)

        # crop and resize depth
        h,w = depth.shape
        depth = depth[(self.crop_h//2):h-(self.crop_h//2),(self.crop_w//2):w-(self.crop_w//2)]
        depth = cv2.resize(depth, (self.W,self.H), interpolation=cv2.INTER_LINEAR)
        depth = depth.reshape(1, depth.shape[0], depth.shape[1]) # [1 x H x W]
        return depth.astype(np.float32)
