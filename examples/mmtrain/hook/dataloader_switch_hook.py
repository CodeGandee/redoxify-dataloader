# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union
from mmengine.hooks import Hook

from mmengine.registry import HOOKS

DATA_BATCH = Optional[Union[dict, tuple, list]]

@HOOKS.register_module()
class DataloaderSwitchHook(Hook):
    """Switch data pipeline at switch_epoch.

    Args:
        switch_epoch (int): switch pipeline at this epoch.
        switch_pipeline (list[dict]): the pipeline to switch to.
    """

    def __init__(self, switch_iter, switch_dataloader):
        self.switch_iter = switch_iter
        self.switch_dataloader = switch_dataloader
        self._restart_dataloader = False
        self._has_switched = False

    def _before_iter(self,
                     runner,
                     batch_idx: int,
                     data_batch: DATA_BATCH = None,
                     mode: str = 'train') -> None:
        """switch pipeline."""
        if batch_idx >= self.switch_iter and not self._has_switched:
            runner.logger.info('Switch pipeline now!')
            # The dataset pipeline cannot be updated when persistent_workers
            # is True, so we need to force the dataloader's multi-process
            # restart. This is a very hacky approach.
            runner.train_dataloader = self.switch_dataloader
            if hasattr(runner.train_dataloader, 'persistent_workers'
                       ) and runner.train_dataloader.persistent_workers is True:
                runner.train_dataloader._DataLoader__initialized = False
                runner.train_dataloader._iterator = None
                self._restart_dataloader = True
            self._has_switched = True