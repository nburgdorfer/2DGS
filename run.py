import os
import sys
from tqdm import tqdm
from cvt.common import set_random_seed

from src.pipeline import Pipeline
from src.config import load_config, load_scenes, get_argparser

parser = get_argparser()
ARGS = parser.parse_args()

#### Load Configuration ####
cfg = load_config(os.path.join(ARGS.config_path, f"{ARGS.dataset}.yaml"))
cfg["dataset"] = ARGS.dataset
set_random_seed(cfg["seed"])

#### Start GUI server ####
network_gui.init(cfg["visualization"]["ip"], cfg["visualization"]["port"])

#### Load Scene Lists ####
scenes = load_scenes(os.path.join(ARGS.config_path, "scenes", "inference.yaml"))
print(scenes)

for network in scenes.keys():
    if scenes[network]["scenes"] == None:
        continue

    total_samples = len(scenes[network]["scenes"])
    for i, scene in enumerate(scenes[network]["scenes"]):
        print(f"\n----Running DMFNet on {network} - {scene}----")
        scene_dict = {network: {"scenes": [scene]}}
        pipeline = Pipeline(cfg=cfg, config_path=ARGS.config_path, log_path=ARGS.log_path, inference_scene=scene_dict)
        pipeline.inference()
