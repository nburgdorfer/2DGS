import os
import sys
from tqdm import tqdm
import numpy as np
from cvt.common import set_random_seed
from cvt.visualization.util import print_csv, to_normal

from src.pipeline import Pipeline
from src.config import load_config, load_scenes, get_argparser
from src.evaluation.eval_2d import eval_2d
from src.evaluation.eval_3d import dtu_point_eval
from src.tools.consensus_filtering import consensus_filter

from src.gs_comps import network_gui

parser = get_argparser()
ARGS = parser.parse_args()

#### Load Configuration ####
cfg = load_config(os.path.join(ARGS.config_path, f"{ARGS.dataset}.yaml"))
cfg["dataset"] = ARGS.dataset
set_random_seed(cfg["seed"])

#### Start GUI server ####
network_gui.init(cfg["visualization"]["ip"], cfg["visualization"]["port"])

#### Load Scenes ####
scene_dict = load_scenes(os.path.join(ARGS.config_path, "scenes", "inference.yaml"))

total_samples = len(scene_dict["scenes"])
avg_mae = np.zeros((total_samples))
avg_auc = np.zeros((total_samples))
avg_percentages = np.zeros((total_samples, 4))
avg_acc = np.zeros((total_samples))
avg_comp = np.zeros((total_samples))
avg_fscore = np.zeros((total_samples))
for i, scene in enumerate(scene_dict["scenes"]):
    print(f"\n----Running 2DGS on {cfg['dataset']} - {scene}----")
    pipeline = Pipeline(cfg=cfg, config_path=ARGS.config_path, scene=scene)
    #pipeline.run()
    #pipeline.render()

    ######## 2D EVALUATION ####
    print("\n---Evaluating depth maps---")
    mae, auc, percentages = eval_2d(pipeline.depth_path, pipeline.opacity_path, pipeline.dataset)
    avg_mae[i] = mae
    avg_auc[i] = auc
    avg_percentages[i] = percentages

    #### 3D EVALUATION ####
    print("\n---Evaluating point cloud---")
    consensus_filter(cfg, pipeline.depth_path, pipeline.opacity_path, pipeline.image_path, pipeline.output_path, pipeline.dataset, scene)
    to_normal(os.path.join(pipeline.output_path, f"{scene}.ply"), os.path.join(pipeline.output_path, f"{scene}_normal.ply"))
    acc, comp, prec, rec, fscore = dtu_point_eval(cfg, scene, method=ARGS.model)
    avg_acc[i] = acc
    avg_comp[i] = comp
    avg_fscore[i] = fscore

print("\n---MAE list---")
print_csv(avg_mae)

print("\n---AUC list---")
print_csv(avg_auc)

print("\n---F-Score list---")
print_csv(avg_fscore)

avg_mae = avg_mae.mean()
avg_auc = avg_auc.mean()
avg_percentages = avg_percentages.mean(axis=0)
avg_acc = avg_acc.mean()
avg_comp = avg_comp.mean()
avg_fscore = avg_fscore.mean()
print(f"\n---Average---\nMAE: {avg_mae:6.3f}{pipeline.dataset.units}")
print(f"AUC: {avg_auc:6.3f}")
print(f"Percentages: {avg_percentages[0]:3.2f}%  |  {avg_percentages[1]:3.2f}%  |  {avg_percentages[2]:3.2f}%  |  {avg_percentages[3]:3.2f}%")
print(f"ACC: {avg_acc:6.3f}")
print(f"COMP: {avg_comp:6.3f}")
print(f"F-Score: {avg_fscore:6.3f}")
