import numpy as np
import sys
import os
import torch
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cvt.io import read_pfm
from cvt.metrics import MAE

def eval_2d(depth_path, conf_path, dataset, thresholds=[0.125, 0.25, 0.5, 1.0]):
    # load data
    target_depths = dataset.get_depths()

    # get depth map filenames
    depth_files = os.listdir(depth_path)
    depth_files = [os.path.join(depth_path, f) for f in depth_files if f[-3:] == "pfm" ]
    depth_files.sort()

    # get confidence map filenames
    conf_files = os.listdir(conf_path)
    conf_files = [os.path.join(conf_path, f) for f in conf_files if f[-3:] == "pfm" ]
    conf_files.sort()

    assert len(depth_files) == len(conf_files), f"Number of depth files ({len(depth_files)}) does not match number of confidence files ({len(conf_files)})."
    num_views = len(depth_files)

    mae = np.zeros((num_views), dtype=np.float32)
    auc = np.zeros((num_views), dtype=np.float32)
    percentages = np.zeros((num_views, len(thresholds)), dtype=np.float32)
    for i, (df, cf) in enumerate(zip(depth_files, conf_files)):
        ref_ind = int(df[-12:-4])
        depth = torch.tensor(read_pfm(df))
        conf = torch.tensor(read_pfm(cf))
        target_depth = torch.tensor(target_depths[ref_ind])[0]
        mask = torch.where(target_depth > 0, 1, 0)
        
        # compute MAE
        mae[i] = float(MAE(depth, target_depth, reduction_dims=(0,1), mask=mask).item())

        # compute AUC
        auc[i] = float(auc_score(depth, conf, target_depth, ref_ind, mask=mask))

        # compute pixel percentages
        percs = depth_percentages(depth, target_depth, thresholds=thresholds, mask=mask)
        percentages[i] = percs

    mae = mae.mean()
    auc = auc.mean()
    percentages = percentages.mean(axis=0)
    percentages *= 100
    print(f"MAE: {mae:6.3f}")
    print(f"AUC: {auc:6.3f}")
    print(f"Percentages: {percentages[0]:3.2f}%  |  {percentages[1]:3.2f}%  |  {percentages[2]:3.2f}%  |  {percentages[3]:3.2f}%")

    return mae, auc, percentages

def auc_score(est_depth, est_conf, target_depth, ref_ind, mask, vis_path=None):
    # grab valid reprojection error values
    pixel_count = int(torch.sum(mask).item())

    inds = torch.where(mask==1)
    target_depth = target_depth[inds].cpu().numpy()
    est_depth = est_depth[inds].cpu().numpy()
    est_conf = est_conf[inds].cpu().numpy()

    # flatten to 1D tensor
    target_depth = target_depth.flatten()
    est_depth = est_depth.flatten()
    est_conf = est_conf.flatten()

    # compute error
    error = np.abs(est_depth - target_depth)
    
    # sort orcale curves by error
    oracle_indices = np.argsort(error)
    oracle_error = np.take(error, indices=oracle_indices, axis=0)

    # sort all tensors by confidence value
    est_indices = np.argsort(est_conf)
    est_indices = est_indices[::-1]
    est_error = np.take(error, indices=est_indices, axis=0)

    # build density vector
    perc = np.array(list(range(5,105,5)))
    density = np.array((perc/100) * (pixel_count), dtype=np.int32)

    oracle_roc = np.zeros(density.shape)
    est_roc = np.zeros(density.shape)
    for i,k in enumerate(density):
        j=0
        #if i==0:
        #    j=0
        #else:
        #    j=density[i-1]
        oe = oracle_error[j:k]
        ee = est_error[j:k]

        if (oe.shape[0] == 0):
            oracle_roc[i] = 0.0
            est_roc[i] = 0.0
        else:
            oracle_roc[i] = np.mean(oe)
            est_roc[i] = np.mean(ee)

    # comput AUC
    oracle_auc = np.trapz(oracle_roc, dx=1)
    est_auc = np.trapz(est_roc, dx=1)

    if(vis_path!=None):
        # plot ROC density errors
        plt.plot(perc, oracle_roc, label="Oracle")
        plt.plot(perc, est_roc, label="Estimate")
        plt.title("ROC Error")
        plt.xlabel("density")
        plt.ylabel("absolte error")
        plt.legend()
        plt.savefig(os.path.join(vis_path,f"roc_{ref_ind:08d}.png"))
        plt.close()

    return est_auc


def depth_percentages(estimate, target, thresholds, mask=None, relative=False):
    assert(estimate.shape==target.shape)
    num_pixels = estimate.flatten().shape[0]

    error = estimate - target
    if relative:
        error /= target
    error = torch.abs(error)

    percs = []
    if mask != None:
        assert(error.shape==mask.shape)
        error *= mask

    for th in thresholds:
        inliers = torch.where(error <= th, 1, 0)
        if mask != None:
            inliers *= mask
            perc = (inliers.sum()) / (mask.sum()+1e-10)
        else:
            perc = inliers.sum() / num_pixels
        percs.append(perc)
    return np.asarray(percs)
    

def target_coverage(data, outputs):
    pass

    target_depth = data["target_depth"]
    loss = []
    near_depth = torch.ones((cfg["training"]["batch_size"])).to(cfg["device"]) * self.cfg["camera"]["near"]
    far_depth = torch.ones((cfg["training"]["batch_size"])).to(cfg["device"]) * self.cfg["camera"]["far"]

    hypos = outputs["hypos"]
    hypo_coords = outputs["hypo_coords"]
    intervals = outputs["intervals"]
    global_probs = outputs["global_probs"]
    prob_grids = outputs["prob_grids"]

    # Calculate edge mask
    down_gt = F.interpolate(target_depth.unsqueeze(1),scale_factor=0.5,mode='bilinear',align_corners=False,recompute_scale_factor=False)
    down_up_gt = F.interpolate(down_gt,scale_factor=2,mode='bilinear',align_corners=False,recompute_scale_factor=False)
    res = torch.abs(target_depth.unsqueeze(1)-down_up_gt)
    high_frequency_mask = res>(0.001*(far_depth-near_depth)[:,None,None,None])
    valid_gt_mask = (-F.max_pool2d(-target_depth.unsqueeze(1),kernel_size=5,stride=1,padding=2))>near_depth[:,None,None,None]
    high_frequency_mask = high_frequency_mask * valid_gt_mask

    for level in reversed(range(len(cfg["model"]["gwc_groups"]))):
        if level ==0:
            # Apply softargmax depth regression for subpixel depth estimation on final level.
            B,_,D,H,W = prob_grids[level].shape

            final_prob = prob_grids[level]
            final_hypo = hypos[level]
            regressed_depth = torch.sum(final_prob*final_hypo,dim=2)
            gt_depth = target_depth.unsqueeze(1)

            mask = (-F.max_pool2d(-target_depth.unsqueeze(1),kernel_size=5,stride=1,padding=2))>near_depth[:,None,None,None]
            tmp_loss = F.smooth_l1_loss(regressed_depth[mask], gt_depth[mask], reduction='none')

            tmp_high_frequency_mask = high_frequency_mask[mask]
            tmp_high_frequency_weight = tmp_high_frequency_mask.float().mean()
            weight = (1-tmp_high_frequency_weight)*tmp_high_frequency_mask + (tmp_high_frequency_weight)*(~tmp_high_frequency_mask)
            tmp_loss *= weight
            tmp_loss *= 0.1
            loss.append(tmp_loss.mean())
            continue

        B,_,D,H,W = prob_grids[level].shape

        # Create gt labels
        unfold_kernel_size = int(2**level)
        assert unfold_kernel_size%2 == 0 or unfold_kernel_size == 1
        unfolded_patch_depth = F.unfold(target_depth.unsqueeze(1),unfold_kernel_size,dilation=1,padding=0,stride=unfold_kernel_size)
        unfolded_patch_depth = unfolded_patch_depth.reshape(B,1,unfold_kernel_size**2,H,W)
        # valid gt depth mask
        mask = (unfolded_patch_depth>near_depth.view((B,1,1,1,1))).all(dim=2)
        mask *= (unfolded_patch_depth<far_depth.view((B,1,1,1,1))).all(dim=2)
        # Approximate depth distribution from depth observations
        gt_occ_grid = torch.zeros_like(hypos[level])
        if self.cfg["loss"]["gt_prob_mode"] == "hard":
            for pixel in range(unfolded_patch_depth.shape[2]):
                selected_depth = unfolded_patch_depth[:,:,pixel]
                distance_to_hypo = abs(hypos[level]-selected_depth.unsqueeze(2))
                occupied_mask = distance_to_hypo<=(intervals[level]/2)
                gt_occ_grid[occupied_mask]+=1
            gt_occ_grid = gt_occ_grid/gt_occ_grid.sum(dim=2,keepdim=True)
            gt_occ_grid[torch.isnan(gt_occ_grid)] = 0
        elif self.cfg["loss"]["gt_prob_mode"] == "soft":
            for pixel in range(unfolded_patch_depth.shape[2]):
                selected_depth = unfolded_patch_depth[:,:,pixel]
                distance_to_hypo = abs(hypos[level]-selected_depth.unsqueeze(2))
                distance_to_hypo /= intervals[level]
                mask = distance_to_hypo>1
                weights = 1-distance_to_hypo
                weights[mask] = 0
                gt_occ_grid+=weights
            gt_occ_grid = gt_occ_grid/gt_occ_grid.sum(dim=2,keepdim=True)
            gt_occ_grid[torch.isnan(gt_occ_grid)] = 0

        covered_mask = gt_occ_grid.sum(dim=2,keepdim=True) > 0
        occ_hypos_count = (gt_occ_grid>0).sum(dim=2,keepdim=True).repeat(1,1,D,1,1)
        edge_weight = occ_hypos_count
        final_mask = mask.unsqueeze(2) * covered_mask


        if self.cfg["loss"]["func"]=="BCE":
            est = torch.masked_select(prob_grids[level],final_mask)
            gt = torch.masked_select(gt_occ_grid,final_mask)
            tmp_loss = F.binary_cross_entropy(est,gt, reduction="none")
            edge_weight = torch.masked_select(edge_weight,final_mask)
            # Apply edge weight
            tmp_loss = tmp_loss * edge_weight
            # class balance
            num_positive = (gt>0).sum()
            num_negative = (gt==0).sum()
            num_total = gt.shape[0]
            alpha_positive = num_negative/float(num_total)
            alpha_negative = num_positive/float(num_total)
            weight = alpha_positive*(gt>0) + alpha_negative*(gt==0)
            tmp_loss = weight*tmp_loss
            tmp_loss = tmp_loss.mean()
            tmp_loss = loss_level_weights[level]*tmp_loss
            loss.append(tmp_loss)
        elif self.cfg["loss"]["func"]=="KL":
            # KL loss
            est = torch.masked_select(prob_grids[level],final_mask)
            gt = torch.masked_select(gt_occ_grid,final_mask)
            tmp_loss = F.kl_div(est.log(),gt, reduction="none", log_target=False)
            edge_weight = torch.masked_select(edge_weight,final_mask)
            # Apply edge weight
            tmp_loss = tmp_loss * edge_weight
            tmp_loss = tmp_loss.mean()
            tmp_loss = loss_level_weights[level]*tmp_loss
            loss.append(tmp_loss)

    loss = torch.stack(loss).mean()
    return loss

def depth_acc(est_depth, gt_depth, th=1.0):
    assert(est_depth.shape == gt_depth.shape)
    abs_error = torch.abs(est_depth - gt_depth)
    valid_pixels = torch.where(gt_depth > 0, 1, 0)
    acc_pixels = torch.where(abs_error <= th, 1, 0)

    abs_error *= valid_pixels
    acc_pixels *= valid_pixels

    mae = abs_error.sum() / (valid_pixels.sum()+1e-5)
    acc = acc_pixels.sum() / (valid_pixels.sum()+1e-5)

    return mae, acc
