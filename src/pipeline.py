# Python libraries
import os
from random import randint
import random
import sys
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import open3d as o3d
import mediapy as media

from cvt.io import write_pfm, load_ckpt, save_ckpt

## Custom libraries
from src.config import save_config
from src.datasets.BaseDataset import build_dataset

### 2DGS dependencies
from src.gs_comps.utils import *
from src.gaussian_model import GaussianModel
from src.gs_comps import network_gui

class Pipeline():
    def __init__(self, cfg, config_path, scene):
        self.cfg = cfg
        self.device = self.cfg["device"]
        self.scene = scene
        self.iterations = cfg["optimization"]["iterations"]
        self.near_depth = self.cfg["camera"]["near"]
        self.far_depth = self.cfg["camera"]["far"]

        # set data paths
        self.data_path = os.path.join(self.cfg["data_path"], self.scene)
        self.output_path = os.path.join(self.cfg["output_path"], self.scene)
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

        self.logger = SummaryWriter(log_dir=self.log_path)
        self.build_dataset()
        save_config(self.log_path, self.cfg)

    def build_dataset(self):
        self.dataset = build_dataset(self.cfg, self.scene)

    def save_output(self, output):
        with torch.set_grad_enabled((torch.is_grad_enabled and not torch.is_inference_mode_enabled)):
            for output_view in output:
                idx = output_view["idx"]
                rendered_image = output_view["rendered_image"][:,:,::-1]
                cv2.imwrite(os.path.join(self.image_path, f"{idx:08d}.png"), rendered_image*255)

                depth = output_view["depth"]
                write_pfm(os.path.join(self.depth_path, f"{idx:08d}.pfm"), depth)
                cv2.imwrite(os.path.join(self.depth_path, f"{idx:08d}.png"), 255*(depth-depth.min()) / (depth.max()-depth.min()+1e-10))

                normal = output_view["normal"]
                write_pfm(os.path.join(self.normal_path, f"{idx:08d}.pfm"), normal)
                cv2.imwrite(os.path.join(self.normal_path, f"{idx:08d}.png"), 255*(normal-normal.min()) / (normal.max()-normal.min()+1e-10))

                depth_normal = output_view["depth_normal"]
                write_pfm(os.path.join(self.depth_normal_path, f"{idx:08d}.pfm"), depth_normal)
                cv2.imwrite(os.path.join(self.depth_normal_path, f"{idx:08d}.png"), 255*(depth_normal-depth_normal.min()) / (depth_normal.max()-depth_normal.min()+1e-10))

                opacity = output_view["opacity"]
                write_pfm(os.path.join(self.opacity_path, f"{idx:08d}.pfm"), opacity)
                cv2.imwrite(os.path.join(self.opacity_path, f"{idx:08d}.png"), 255*(opacity-opacity.min()) / (opacity.max()-opacity.min()+1e-10))

    def save_output_video(self, output):
        """Creates videos out of the images saved to disk."""
        frames = self.cfg["rendering"]["frames"]
        zpad = max(5, len(str(frames - 1)))
        idx_to_str = lambda idx: str(idx).zfill(zpad)
        render_dist_curve_fn = np.log

        video_kwargs = {
            'shape': (self.dataset.H, self.dataset.W),
            'codec': 'h264',
            'fps': 60,
            'crf': 18,
        }
      
        for k in self.cfg["rendering"]["video_maps"]:
            video_file = os.path.join(self.video_path, f'{k}.mp4')
            input_format = "rgb" if (k=="rendered_image" or k=="normal") else "gray"
            with media.VideoWriter(
                video_file, **video_kwargs, input_format=input_format) as writer:
                for output_view in tqdm(output, desc=f"Rendering {k} video"):
                    idx = output_view["idx"]
                    image = output_view[k]
                    if k=="depth":
                        image = (np.clip(image, self.near_depth, self.far_depth) - self.near_depth) / (self.far_depth-self.near_depth+1e-10)
                    elif k=="rendered_image":
                        image = image[:,:,::-1]
                    frame = (np.clip(np.nan_to_num(image), 0., 1.) * 255.).astype(np.uint8)
                    writer.add_image(frame)

    def build_2dgs_scene(self, cameras, images, pcd, gaussians_file=None, shuffle=True):
        cam_infos = []
        for i,camera in enumerate(cameras):
            height, width, _ = images[i].shape
            info = {
                "uid": 1,
                "R": camera[0,:3,:3].T,
                "T": camera[0,:3,3],
                "FovY": focal2fov(camera[1,1,1], height),
                "FovX": focal2fov(camera[1,0,0], width),
                "image": images[i],
                "image_path": None,
                "image_name": f"{i:08d}",
                "width": width,
                "height": height
                }
            cam_infos.append(info)
        nerf_normalization = getNerfppNorm(cam_infos)

        scene_info = {
                "point_cloud": pcd,
                "train_cameras": cam_infos,
                "test_cameras": [],
                "nerf_normalization": nerf_normalization,
                "ply_path": self.dataset.points_file,
                "sparse_ply_path": self.dataset.sparse_points_file
                }
        with open(scene_info["ply_path"], 'rb') as src_file, open(os.path.join(self.output_path, "initial_points.ply") , 'wb') as dest_file:
            dest_file.write(src_file.read())
        # sparse points produced to be friendly to the SIBR viewer (too slow with dense points)
        with open(scene_info["sparse_ply_path"], 'rb') as src_file, open(os.path.join(self.output_path, "input.ply") , 'wb') as dest_file:
            dest_file.write(src_file.read())

        camlist = scene_info["train_cameras"]
        # store cameras as json
        json_cams = []
        for id, cam in enumerate(camlist):
            json_cams.append(camera_to_JSON(id, cam))
        with open(os.path.join(self.output_path, "cameras.json"), 'w') as file:
            json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info["train_cameras"])  # Multi-res consistent random shuffling
        cameras_extent = scene_info["nerf_normalization"]["radius"]
        train_cameras = cameraList_from_camInfos(scene_info["train_cameras"], self.cfg)

        gaussians = GaussianModel(self.cfg, self.cfg["model"]["sh_degree"])
        if gaussians_file==None:
            gaussians.create_from_pcd(scene_info["point_cloud"], cameras_extent)
        else:
            gaussians.load_ply(gaussians_file)


        return train_cameras, cameras_extent, gaussians

    def render(self):
        # load data
        cameras = self.dataset.get_cameras()
        images = self.dataset.get_images()
        depths = self.dataset.get_depths()
        points = self.dataset.get_points()

        # optimized gaussians file
        if self.cfg["rendering"]["iteration"] == -1:
            gaussians_file = os.path.join(self.output_path, "gaussians.ply")
        else:
            gaussians_file = os.path.join(self.ckpt_path, f"gaussians_{self.cfg['rendering']['iteration']:08d}.ply")

        # build 2DGS scene
        train_cameras, cameras_extent, gaussians = self.build_2dgs_scene(cameras, images, points, gaussians_file=gaussians_file, shuffle=False)

        # set background color
        bg_color = [1, 1, 1] if self.cfg["model"]["white_background"] else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=self.device)

        # load gaussian extractor
        gaussExtractor = GaussianExtractor(self.cfg, gaussians, render, bg_color=bg_color)    

        # export training images
        gaussExtractor.reconstruction(train_cameras)
        output = gaussExtractor.export_image()
        self.save_output(output)
        
        # render videos
        cam_traj = generate_path(train_cameras, self.cfg["rendering"]["frames"])
        gaussExtractor.reconstruction(cam_traj)
        output = gaussExtractor.export_image()
        self.save_output_video(output)

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
            if iteration % self.cfg["optimization"]["sh_increase_interval"] == 0:
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
                    gaussians.save_ply(os.path.join(self.output_path, f"gaussians.ply"))

                ### Densification
                #if iteration < self.cfg["optimization"]["densify_until_iter"]:
                #    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                #    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                #    if iteration > self.cfg["optimization"]["densify_from_iter"] and iteration % self.cfg["optimization"]["densification_interval"] == 0:
                #        size_threshold = 20 if iteration > self.cfg["optimization"]["opacity_reset_interval"] else None
                #        gaussians.densify_and_prune(self.cfg["optimization"]["densify_grad_threshold"], self.cfg["optimization"]["opacity_th"], cameras_extent, size_threshold)
                #    
                #    if iteration % self.cfg["optimization"]["opacity_reset_interval"] == 0 or \
                #        (self.cfg["model"]["white_background"] and iteration == self.cfg["optimization"]["densify_from_iter"]):
                #        gaussians.reset_opacity()

                # Optimizer step
                if iteration < self.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

                if ((iteration % self.cfg["optimization"]["ckpt_freq"] == 0 and self.cfg["optimization"]["ckpt_freq"] != -1) or (iteration == self.iterations)):
                    torch.save((gaussians.capture(), iteration), os.path.join(self.ckpt_path, f"{iteration}.pt"))
                    gaussians.save_ply(os.path.join(self.ckpt_path, f"gaussians_{iteration:08d}.ply"))

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
                        network_gui.send(net_image_bytes, self.output_path, metrics_dict)
                        if do_training and ((iteration < int(self.iterations)) or not keep_alive):
                            break
                    except Exception as e:
                        # raise e
                        network_gui.conn = None
