# Python libraries
import cv2
import json
import numpy as np
import os
import random
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

## Custom libraries
from src.config import save_config
from src.datasets.BaseDataset import build_dataset
from cvt.common import print_gpu_mem, to_gpu
from cvt.io import write_pfm, load_ckpt, save_ckpt

class BasePipeline():
    def __init__(self, cfg, config_path, log_path, model_name, scene=None):
        self.cfg = cfg
        self.device = self.cfg["device"]
        self.scene = scene

        # set data paths
        self.data_path = os.path.join(self.cfg["data_path"], self.inference_scene[0])
        self.output_path = os.path.join(self.cfg["output_path"], self.inference_scene[0])
        self.ckpt_path = os.path.join(self.output_path, "ckpts")
        self.log_path = os.path.join(self.output_path, "log")
        self.depth_path = os.path.join(self.output_path, "depth")
        self.normal_path = os.path.join(self.output_path, "normal")
        self.depth_normal_path = os.path.join(self.output_path, "depth_normal")
        self.opacity_path = os.path.join(self.output_path, "opacity")
        self.image_path = os.path.join(self.output_path, "image")
        self.video_path = os.path.join(self.output_path, "video")

        # create directories
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.ckpt_path, exist_ok=True)
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
        self.build_model()
        self.build_optimizer()
        self.build_scheduler()

        # log current configuration used
        save_config(self.log_path, self.cfg)

    def get_network(self):
        raise NotImplementedError()

    def compute_loss(self):
        raise NotImplementedError()

    def compute_stats(self):
        raise NotImplementedError()

    def save_output(self, data, output, sample_ind):
        with torch.set_grad_enabled((torch.is_grad_enabled and not torch.is_inference_mode_enabled)):
            # save confidence map
            pass

    def build_dataset(self):
        self.dataset = build_dataset(self.cfg, self.scenes)

    def build_optimizer(self):
        rate = self.cfg["learning_rate"]
        self.optimizer = optim.Adam(self.parameters_to_train, lr=rate, betas=(0.9, 0.999))

    def build_scheduler(self):
        lr_steps = [int(epoch_idx) for epoch_idx in self.cfg["training"]["lr_steps"].split(',')]
        lr_gamma = float(self.cfg["training"]["lr_gamma"])
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, lr_steps, gamma=lr_gamma)

    def run(self):
        first_iter = 0
        tb_writer = prepare_output_and_logger(dataset)
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians)
        gaussians.training_setup(opt)
        if checkpoint:
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        iter_start = torch.cuda.Event(enable_timing = True)
        iter_end = torch.cuda.Event(enable_timing = True)

        viewpoint_stack = None
        ema_loss_for_log = 0.0
        ema_dist_for_log = 0.0
        ema_normal_for_log = 0.0

        progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
        first_iter += 1
        for iteration in range(first_iter, opt.iterations + 1):        

            iter_start.record()

            gaussians.update_learning_rate(iteration)

            # Every 1000 iterations we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            

            rend_dist = render_pkg["rend_dist"]
            rend_normal  = render_pkg['rend_normal']
            surf_normal = render_pkg['surf_normal']

            # regularization
            if iteration > 7000:
                normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
                normal_loss = opt.lambda_normal * (normal_error).mean()
            else:
                normal_loss = torch.tensor(0.0)
            if iteration > 3000:
                dist_loss = opt.lambda_dist * (rend_dist).mean()
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
                if iteration == opt.iterations:
                    progress_bar.close()

                ### Log and save
                #if tb_writer is not None:
                #    tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                #    tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

                #training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
                if (iteration == opt.iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)

                # Densification
                if iteration < opt.densify_until_iter:
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                    
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

                if (iteration in checkpoint_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            with torch.no_grad():        
                if network_gui.conn == None:
                    network_gui.try_connect(dataset.render_items)
                while network_gui.conn != None:
                    try:
                        net_image_bytes = None
                        custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                        if custom_cam != None:
                            render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                            net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                            net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                        metrics_dict = {
                            "#": gaussians.get_opacity.shape[0],
                            "loss": ema_loss_for_log
                            # Add more metrics as needed
                        }
                        # Send the data
                        network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                        if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                            break
                    except Exception as e:
                        # raise e
                        network_gui.conn = None
