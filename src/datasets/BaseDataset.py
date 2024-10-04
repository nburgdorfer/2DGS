
def build_dataset(cfg, scene):
    if cfg["dataset"] == 'TNT':
        from src.datasets.TNT import TNT as Dataset
    elif cfg["dataset"] == 'DTU':
        from src.datasets.DTU import DTU as Dataset
    else:
        raise Exception(f"Unknown Dataset {self.cfg['dataset']}")

    return Dataset(cfg, scene)

class BaseDataset():
    def __init__(self, cfg, scene):
        self.cfg = cfg
        self.data_path = self.cfg["data_path"]
        self.device = self.cfg["device"]
        self.scene = scene
        self.crop_h = self.cfg["camera"]["crop_h"]
        self.crop_w = self.cfg["camera"]["crop_w"]
        self.scale = self.cfg["inference"]["scale"]

    def get_cameras(self):
        raise NotImplementedError()

    def get_images(self):
        raise NotImplementedError()

    def get_depths(self):
        raise NotImplementedError()

    def get_points(self):
        raise NotImplementedError()
