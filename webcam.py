# Copyright (c) OpenMMLab. All rights reserved.

from mmcv import Config

from mmpose.apis.webcam import WebcamExecutor
from mmpose.apis.webcam.nodes import model_nodes


def set_device(cfg: Config, device: str):
    """Set model device in config.

    Args:
        cfg (Config): Webcam config
        device (str): device indicator like "cpu" or "cuda:0"
    """

    device = device.lower()
    assert device == 'cpu' or device.startswith('cuda:')

    for node_cfg in cfg.executor_cfg.nodes:
        if node_cfg.type in model_nodes.__all__:
            node_cfg.update(device=device)

    return cfg


def run_webcam(config, device):
    cfg = Config.fromfile(config)
    cfg = set_device(cfg, device)

    webcam_exe = WebcamExecutor(**cfg.executor_cfg)
    webcam_exe.run()
