import os,sys
import random
import numpy as np
import torch
import cv2
import sys
import json
from tqdm import tqdm
import yaml
from cvt.camera import scale_cam
from cvt.io import read_single_cam_sfm, read_pfm, read_cluster_list

from src.datasets.BaseDataset import BaseDataset

class DTU(BaseDataset):
    def __init__(self, cfg, scene):
        super(DTU, self).__init__(cfg, scene)
        self.units = "mm"

    def set_paths(self):
        self.image_path = os.path.join(self.data_path, "Images", self.scene)
        self.depth_path = os.path.join(self.data_path, "Depths", self.scene)
        self.camera_path = os.path.join(self.data_path, "Cameras")
        self.point_cloud = os.path.join(self.data_path, "Sparse_Points", f"{self.scene}_sparse.ply")

    def get_frame_count(self):
        image_files = os.listdir(self.image_path)
        image_files = [img for img in image_files if img[-4:]==".png"]
        return len(image_files)

    def build_samples(self):
        samples = []
        frame_count = self.get_frame_count()

        for i in range(frame_count):
            image_file = os.path.join(self.image_path, f"{i:08d}.png")
            depth_file = os.path.join(self.depth_path, f"{i:08d}_depth.pfm")
            camera_file = os.path.join(self.camera_path, f"{i:08d}_cam.txt")

            samples.append({
                            "scene": self.scene,
                            "view_num": i,
                            "image_file": image_file,
                            "depth_file": depth_file,
                            "camera_file": camera_file
                            })

        return np.asarray(samples)

    def load_intrinsics(self, scene):
        cam_file = os.path.join(self.data_path, "Cameras/00000000_cam.txt")
        cam = read_single_cam_sfm(cam_file)
        self.K[scene] = cam[1,:3,:3].astype(np.float32)
        self.K[scene][0,2] -= (self.crop_w//2)
        self.K[scene][1,2] -= (self.crop_h//2)
        self.H = int(self.scale * (self.cfg["camera"]["height"] - self.crop_h))
        self.W = int(self.scale * (self.cfg["camera"]["width"]- self.crop_w))

    def get_pose(self, pose_file, frame_id=None):
        try:
            cam = read_single_cam_sfm(pose_file)
        except:
            print(f"Pose file {pose_file} does not exist.")
            sys.exit()
        pose = cam[0]
        if(np.isnan(pose).any()):
            print(pose, pose_file)
        return pose

    def get_image(self, image_file, scale=True):
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # crop and resize image
        h,w,_ = image.shape
        image = image[(self.crop_h//2):h-(self.crop_h//2),(self.crop_w//2):w-(self.crop_w//2), :]
        if scale:
            image = cv2.resize(image, (self.W,self.H), interpolation=cv2.INTER_LINEAR)
        image = _normalize_image(image, mean=0.5, std=0.5)
        image = np.moveaxis(image, [0,1,2], [1,2,0])
        return image.astype(np.float32)

    def get_depth(self, depth_file, scale=True):
        if (depth_file[-3:] == "pfm"):
            depth = read_pfm(depth_file)
        elif (depth_file[-3:] == "png"):
            depth = cv2.imread(depth_file, 2) / self.png_depth_scale

        # crop and resize depth
        h,w = depth.shape
        depth = depth[(self.crop_h//2):h-(self.crop_h//2),(self.crop_w//2):w-(self.crop_w//2)]
        if scale:
            depth = cv2.resize(depth, (self.W,self.H), interpolation=cv2.INTER_LINEAR)
        depth = depth.reshape(1, depth.shape[0], depth.shape[1])
        return depth.astype(np.float32)

    def get_all_poses(self, scene):
        poses = []
        pose_path = os.path.join(self.data_path, "Cameras")
        pose_files = os.listdir(pose_path)
        pose_files.sort()
        for pose_file in pose_files:
            if (pose_file[-8:] != "_cam.txt"):
                continue
            cam = read_single_cam_sfm(os.path.join(pose_path,pose_file))
            pose = cam[0]
            if(np.isnan(pose).any()):
                print(pose, pose_file)
            poses.append(pose)
        return poses

    def get_all_depths(self, scale):
        gt_depth_files = os.listdir(self.gt_depth_path)
        gt_depth_files = [os.path.join(self.gt_depth_path, gdf) for gdf in gt_depth_files if os.path.isfile(os.path.join(self.gt_depth_path, gdf))]
        gt_depth_files.sort()

        gt_depths = {}
        for gdf in gt_depth_files:
            ref_ind = int(gdf[-18:-10])
            gt_depth = self.get_depth(os.path.join(self.gt_depth_path, gdf), scale=scale)
            gt_depths[ref_ind] = gt_depth
        return gt_depths
