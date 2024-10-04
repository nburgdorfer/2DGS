# Python libraries
import cv2
import json
import numpy as np
import os
import random
from random import randint
import shutil
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import NamedTuple

from cvt.common import print_gpu_mem, to_gpu
from cvt.io import write_pfm, load_ckpt, save_ckpt

## Custom libraries
from src.config import save_config
from src.datasets.BaseDataset import build_dataset
from src.gaussian_renderer import render, network_gui
from src.scene import GaussianModel
from src.utils.camera_utils import cameraList_from_camInfos
from src.utils.graphics_utils import focal2fov, getNerfppNorm, BasicPointCloud
from src.utils.image_utils import psnr, render_net_image
from src.utils.loss_utils import l1_loss, ssim

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int 

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str  


class Pipeline():
    def __init__(self, cfg, config_path, scene):
        self.cfg = cfg
        self.device = self.cfg["device"]
        self.scene = scene
        self.iterations = cfg["optimization"]["iterations"]

        # set data paths
        self.data_path = os.path.join(self.cfg["data_path"], self.scene)
        self.output_path = os.path.join(self.cfg["output_path"], self.scene)
        self.ckpt_path = os.path.join(self.output_path, "ckpts")
        self.log_path = os.path.join(self.output_path, "log")
        self.points_path = os.path.join(self.output_path, "points")
        self.depth_path = os.path.join(self.output_path, "depth")
        self.normal_path = os.path.join(self.output_path, "normal")
        self.depth_normal_path = os.path.join(self.output_path, "depth_normal")
        self.opacity_path = os.path.join(self.output_path, "opacity")
        self.image_path = os.path.join(self.output_path, "image")
        self.video_path = os.path.join(self.output_path, "video")

        # create directories
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.ckpt_path, exist_ok=True)
        os.makedirs(self.points_path, exist_ok=True)
        os.makedirs(self.depth_path, exist_ok=True)
        os.makedirs(self.normal_path, exist_ok=True)
        os.makedirs(self.depth_normal_path, exist_ok=True)
        os.makedirs(self.opacity_path, exist_ok=True)
        os.makedirs(self.image_path, exist_ok=True)
        os.makedirs(self.video_path, exist_ok=True)

        # create logger
        self.logger = SummaryWriter(log_dir=self.log_path)

        # build all network components
        self.build_dataset()
        #self.build_optimizer()
        #self.build_scheduler()

        # log current configuration used
        save_config(self.log_path, self.cfg)

    def get_network(self):
        raise NotImplementedError()

    def compute_loss(self):
        raise NotImplementedError()

    def compute_stats(self):
        raise NotImplementedError()

    def save_output(self, data, output, ind):
        with torch.set_grad_enabled((torch.is_grad_enabled and not torch.is_inference_mode_enabled)):
            # save confidence map
            pass

    def build_dataset(self):
        self.dataset = build_dataset(self.cfg, self.scene)

    def build_optimizer(self):
        rate = self.cfg["learning_rate"]
        self.optimizer = optim.Adam(self.parameters_to_train, lr=rate, betas=(0.9, 0.999))

    def build_scheduler(self):
        lr_steps = [int(epoch_idx) for epoch_idx in self.cfg["training"]["lr_steps"].split(',')]
        lr_gamma = float(self.cfg["training"]["lr_gamma"])
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, lr_steps, gamma=lr_gamma)

    def build_2dgs_scene(self, cameras, images, pcd):
        cam_infos = []
        for i,camera in enumerate(cameras):
            height, width, _ = images[i].shape
            cam_infos.append(CameraInfo(
                                uid=1,
                                R=camera[0,:3,:3].T,
                                T=camera[0,:3,3],
                                FovY=focal2fov(camera[1,1,1], height),
                                FovX=focal2fov(camera[1,0,0], width),
                                image=images[i],
                                image_path=None,
                                image_name=None,
                                width=width,
                                height=height))
        nerf_normalization = getNerfppNorm(cam_infos)

        scene_info = SceneInfo(point_cloud=pcd,
                               train_cameras=cam_infos,
                               test_cameras=[],
                               nerf_normalization=nerf_normalization,
                               ply_path=self.dataset.points_file)
        with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.points_path, "input.ply") , 'wb') as dest_file:
            dest_file.write(src_file.read())

        camlist = scene_info.train_cameras

        random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
        cameras_extent = scene_info.nerf_normalization["radius"]
        train_cameras = cameraList_from_camInfos(scene_info.train_cameras, self.cfg)

        gaussians = GaussianModel(self.cfg["model"]["sh_degree"])
        gaussians.create_from_pcd(scene_info.point_cloud, cameras_extent)

        return train_cameras, cameras_extent, gaussians

    def run(self):
        # load data
        cameras = self.dataset.get_cameras()
        images = self.dataset.get_images()
        depths = self.dataset.get_depths()
        points = self.dataset.get_points()

        # build 2DGS scene
        train_cameras, cameras_extent, gaussians = self.build_2dgs_scene(cameras, images, points)
        gaussians.training_setup(self.cfg)

        # set background color
        bg_color = [1, 1, 1] if self.cfg["model"]["white_background"] else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=self.device)

        # enable timing
        iter_start = torch.cuda.Event(enable_timing = True)
        iter_end = torch.cuda.Event(enable_timing = True)

        viewpoint_stack = None
        ema_loss_for_log = 0.0
        ema_dist_for_log = 0.0
        ema_normal_for_log = 0.0

        progress_bar = tqdm(range(self.iterations), desc="Optimizing Surfels")
        for iteration in range(1, self.iterations + 1):        
            iter_start.record()
            gaussians.update_learning_rate(iteration)

            # Every 1000 iterations we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = train_cameras.copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            
            render_pkg = render(self.cfg, viewpoint_cam, gaussians, background)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - self.cfg["loss"]["dssim_weight"]) * Ll1 + self.cfg["loss"]["dssim_weight"] * (1.0 - ssim(image, gt_image))
            

            rend_dist = render_pkg["rend_dist"]
            rend_normal  = render_pkg['rend_normal']
            surf_normal = render_pkg['surf_normal']

            # regularization
            if iteration > self.cfg["loss"]["normal_iter"]:
                normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
                normal_loss = self.cfg["loss"]["normal_weight"] * (normal_error).mean()
            else:
                normal_loss = torch.tensor(0.0)
            if iteration > self.cfg["loss"]["dist_iter"]:
                dist_loss = self.cfg["loss"]["dist_weight"] * (rend_dist).mean()
            else:
                dist_loss = torch.tensor(0.0)

            # loss
            total_loss = loss + dist_loss + normal_loss
            total_loss.backward()
            iter_end.record()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
                ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log

                if iteration % 10 == 0:
                    loss_dict = {
                        "Loss": f"{ema_loss_for_log:.{5}f}",
                        "distort": f"{ema_dist_for_log:.{5}f}",
                        "normal": f"{ema_normal_for_log:.{5}f}",
                        "Points": f"{len(gaussians.get_xyz)}"
                    }
                    progress_bar.set_postfix(loss_dict)
                    progress_bar.update(10)

                if iteration == self.iterations:
                    progress_bar.close()
                    gaussians.save_ply(os.path.join(self.points_path, "points.ply"))

                # Densification
                if iteration < self.cfg["optimization"]["densify_until_iter"]:
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > self.cfg["optimization"]["densify_from_iter"] and iteration % self.cfg["optimization"]["densification_interval"] == 0:
                        size_threshold = 20 if iteration > self.cfg["optimization"]["opacity_reset_interval"] else None
                        gaussians.densify_and_prune(self.cfg["optimization"]["densify_grad_threshold"], self.cfg["optimization"]["opacity_th"], cameras_extent, size_threshold)
                    
                    if iteration % self.cfg["optimization"]["opacity_reset_interval"] == 0 or \
                        (self.cfg["model"]["white_background"] and iteration == self.cfg["optimization"]["densify_from_iter"]):
                        gaussians.reset_opacity()

                # Optimizer step
                if iteration < self.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

                if ((iteration % self.cfg["optimization"]["ckpt_freq"] == 0 and self.cfg["optimization"]["ckpt_freq"] != -1) or (iteration == self.iterations)):
                    torch.save((gaussians.capture(), iteration), os.path.join(self.ckpt_path, f"{iteration}.pt"))

            with torch.no_grad():
                if network_gui.conn == None:
                    network_gui.try_connect(self.cfg["rendering"]["maps"])
                while network_gui.conn != None:
                    try:
                        net_image_bytes = None
                        custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                        if custom_cam != None:
                            render_pkg = render(self.cfg, custom_cam, gaussians, background, scaling_modifer)   
                            net_image = render_net_image(render_pkg, self.cfg["rendering"]["maps"], render_mode, custom_cam)
                            net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                        metrics_dict = {
                            "#": gaussians.get_opacity.shape[0],
                            "loss": ema_loss_for_log
                            # Add more metrics as needed
                        }
                        # Send the data
                        network_gui.send(net_image_bytes, self.data_path, metrics_dict)
                        if do_training and ((iteration < int(self.iterations)) or not keep_alive):
                            break
                    except Exception as e:
                        # raise e
                        network_gui.conn = None
