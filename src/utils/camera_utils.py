import numpy as np
import sys
import torch

from src.scene.cameras import Camera

def loadCam(cfg, ind, cam_info):
    orig_h, orig_w,_ = cam_info.image.shape
    global_down = 1
    resolution = (int(orig_w), int(orig_h))

    resized_image_rgb = torch.from_numpy(cam_info.image).permute(2,0,1)
    loaded_mask = None
    gt_image = resized_image_rgb

    return Camera(
                colmap_id=cam_info.uid,
                R=cam_info.R,
                T=cam_info.T,
                FoVx=cam_info.FovX,
                FoVy=cam_info.FovY,
                image=gt_image,
                gt_alpha_mask=loaded_mask,
                image_name=cam_info.image_name,
                uid=ind,
                data_device=cfg["device"])

def cameraList_from_camInfos(cam_infos, cfg):
    camera_list = []

    for ind, cam_info in enumerate(cam_infos):
        camera_list.append(loadCam(cfg, ind, cam_info))

    return camera_list

