import sys
from torch.utils.data import DataLoader
from mmengine.dataset import DefaultSampler
from tqdm import tqdm
from mmdet.datasets import CocoDataset
from mmengine.dataset import pseudo_collate
from mmcv.transforms import LoadImageFromFile
from mmdet.datasets.transforms import LoadAnnotations, Pad, RandomFlip, Resize, RandomCrop, YOLOXHSVRandomAug, PackDetInputs

data_root = '/mnt/data/coco2017/'

pipelines = [
    LoadImageFromFile(),
    LoadAnnotations(with_bbox=True),
    RandomCrop(crop_size=(0.5,0.5), crop_type='relative_range'),
    RandomFlip(prob=0.5),
    Resize(scale=(640, 640), keep_ratio=True),
    Pad(size=(640,640)),
    YOLOXHSVRandomAug(),
    PackDetInputs()
]

dataset = CocoDataset(data_root=data_root, 
                      ann_file='annotations/instances_train2017.json',
                      data_prefix=dict(img='train2017/'),
                      pipeline=pipelines)

dataloader = DataLoader(dataset=dataset, batch_size=4, num_workers=4,
                        sampler=DefaultSampler(dataset=dataset, shuffle=True), collate_fn=pseudo_collate)

pbar = tqdm(total=len(dataloader))
for idx, data in enumerate(dataloader):
    pbar.update(1)