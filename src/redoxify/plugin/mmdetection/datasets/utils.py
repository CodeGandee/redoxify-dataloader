from typing import Any, Mapping, Sequence

def pseudo_collate(batch: Sequence) -> Any:
    assert len(batch)==1, "Batch size of dataloader should be 1 if you are using RedoxDetDataset"
    keys = batch[0][0].keys()
    data_batch = {key: [data[key] for data in batch[0]] for key in keys}
    return data_batch
