import argparse
import yaml
import os

def get_argparser():
    parser = argparse.ArgumentParser(description='2D Gaussian Splatting.')
    parser.add_argument('--config_path', type=str, help='Path to config file.', required=True)
    parser.add_argument('--dataset', type=str, help='Current dataset being used.', required=True, choices=["DTU", "TNT"])
    parser.add_argument('--model', type=str, help='Current model to be used.', required=True, choices=["2DGS"])
    return parser

def save_config(output_path, cfg):
    with open(os.path.join(output_path, "config.yaml"), 'w') as config_file:
        yaml.dump(cfg, config_file)

def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.full_load(f)
    return cfg

def load_scenes(scenes_file):
    with open(scenes_file, 'r') as f:
        scenes = yaml.full_load(f)
    return scenes
