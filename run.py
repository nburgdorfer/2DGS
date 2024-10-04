import os
import sys
from tqdm import tqdm
from cvt.common import set_random_seed

from src.pipeline import Pipeline
from src.config import load_config, load_scenes, get_argparser
from src.gaussian_renderer import network_gui

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
for i, scene in enumerate(scene_dict["scenes"]):
    print(f"\n----Running 2DGS on {cfg['dataset']} - {scene}----")
    pipeline = Pipeline(cfg=cfg, config_path=ARGS.config_path, scene=scene)
    pipeline.run()
    pipeline.render()
