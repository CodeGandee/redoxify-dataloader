# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Optional, Union
from torch.utils.data import DataLoader
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

from redoxify.plugin.mmdetection.datasets.RedoxMMDetDataset import RedoxMMDetDataset
from redoxify.plugin.mmdetection.datasets.utils import pseudo_collate, yolov5_collate

DATA_BATCH = Optional[Union[dict, tuple, list]]

@HOOKS.register_module()
class RedoxLoaderSwitchHook(Hook):
    """Switch data pipeline at switch_epoch.

    Args:
        switch_epoch (int): switch pipeline at this epoch.
        switch_pipeline (list[dict]): the pipeline to switch to.
    """

    def __init__(self, switch_epoch, redox_config, mm_scope='mmyolo'):
        self.switch_epoch = switch_epoch
        self.redox_config = redox_config
        self.mm_scope = mm_scope
        self._has_switched = False

    def beforebefore_train_iterrun(self, runner) -> None:
        now_iter = runner.iter
        if now_iter >= 1000 and not self._has_switched:
            runner.logger.info('Switch pipeline now!')
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            world_size = int(os.environ.get('WORLD_SIZE', 1))
            dataset = RedoxMMDetDataset.from_redox_cfg(self.redox_config, num_gpus=world_size, device_id=local_rank)
            if self.mm_scope == 'mmyolo':
                train_loader = DataLoader(dataset, collate_fn=yolov5_collate)
            else:
                train_loader = DataLoader(dataset, collate_fn=pseudo_collate)
            runner.train_dataloader = train_loader
            self._has_switched = True
            
    def before_train_epoch(self, runner):
        """switch pipeline."""
        epoch = runner.epoch
        train_loader = runner.train_dataloader
        if epoch >= self.switch_epoch and not self._has_switched:
            runner.logger.info('Switch pipeline now!')
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            world_size = int(os.environ.get('WORLD_SIZE', 1))
            dataset = RedoxMMDetDataset.from_redox_cfg(self.redox_config, num_gpus=world_size, device_id=local_rank)
            if self.mm_scope == 'mmyolo':
                train_loader = DataLoader(dataset, collate_fn=yolov5_collate)
            else:
                train_loader = DataLoader(dataset, collate_fn=pseudo_collate)
            runner.train_dataloader = train_loader
            self._has_switched = True