from typing import Any, Mapping, Sequence
import torch

def pseudo_collate(batch: Sequence) -> Any:
    assert len(batch)==1, "Batch size of dataloader should be 1 if you are using RedoxDetDataset"
    data_batch = [data for data in batch[0]]
    return data_batch