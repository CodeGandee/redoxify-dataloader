import os
import copy
from typing import Dict, Union

from torch.utils.data import DataLoader
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmengine.config import Config, ConfigDict
from redoxify.plugin.mmdetection.datasets.RedoxMMDetDataset import RedoxMMDetDataset
from redoxify.plugin.mmdetection.datasets.utils import yolov5_collate

ConfigType = Union[Dict, Config, ConfigDict]

@RUNNERS.register_module()
class RedoxMMYoloRunner(Runner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

    @classmethod
    def from_cfg(cls, cfg: ConfigType) -> 'RedoxMMYoloRunner':
        redox_dataset_config = cfg.get('redox_dataset_config')
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        dataset = RedoxMMDetDataset.from_redox_cfg(redox_dataset_config, num_gpus=world_size, device_id=local_rank)
        train_loader = DataLoader(dataset, collate_fn=yolov5_collate)
        cfg = {k: v for k, v in cfg.items() if k not in ['redox_dataset_config', 'RedoxMMRunner', 'sys']}
        cfg = copy.deepcopy(cfg)
        runner = cls(
            model=cfg['model'],
            work_dir=cfg['work_dir'],
            train_dataloader=train_loader,
            val_dataloader=cfg.get('val_dataloader'),
            test_dataloader=cfg.get('test_dataloader'),
            train_cfg=cfg.get('train_cfg'),
            val_cfg=cfg.get('val_cfg'),
            test_cfg=cfg.get('test_cfg'),
            auto_scale_lr=cfg.get('auto_scale_lr'),
            optim_wrapper=cfg.get('optim_wrapper'),
            param_scheduler=cfg.get('param_scheduler'),
            val_evaluator=cfg.get('val_evaluator'),
            test_evaluator=cfg.get('test_evaluator'),
            default_hooks=cfg.get('default_hooks'),
            custom_hooks=cfg.get('custom_hooks'),
            data_preprocessor=cfg.get('data_preprocessor'),
            load_from=cfg.get('load_from'),
            resume=cfg.get('resume', False),
            launcher=cfg.get('launcher', 'none'),
            env_cfg=cfg.get('env_cfg', dict(dist_cfg=dict(backend='nccl'))),
            log_processor=cfg.get('log_processor'),
            log_level=cfg.get('log_level', 'INFO'),
            visualizer=cfg.get('visualizer'),
            default_scope=cfg.get('default_scope', 'mmengine'),
            randomness=cfg.get('randomness', dict(seed=None)),
            experiment_name=cfg.get('experiment_name'),
            cfg=cfg,
        )

        return runner