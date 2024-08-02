import os, sys
import numpy as np
import cv2
import time
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append('/workspace/redoxify-dataloader/')
sys.path.append('/workspace/redoxify-dataloader/src')
from examples.mmtrain.redox_config import redox_dataset_config
from redoxify.datasets.RedoxBaseDataset import RedoxBaseDataset
from redoxify.plugin.mmdetection.datasets.RedoxMMDetDataset import RedoxMMDetDataset
from redoxify.plugin.mmdetection.datasets.utils import pseudo_collate

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def draw_bboxes(img, bboxes, labels):
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, str(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return img

def test_RedoxBaseDataset(dataset_config):
    dataset = RedoxMMDetDataset.from_redox_cfg(dataset_config, num_gpus=1, device_id=0)
    dataloader = DataLoader(dataset, collate_fn=pseudo_collate)
    start_time = time.time()
    pbar = tqdm(total=len(dataloader))
    for i, data in enumerate(dataloader):
        inputs = data['inputs']
        data_samples = data['data_samples']
        for j in range(len(inputs)):
            img = inputs[j].cpu().numpy().transpose(1, 2, 0)
            img = img.astype(np.uint8).copy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # img = (img * 255).astype(np.uint8)
            labels = data_samples[j].gt_instances.labels.cpu().numpy()
            bboxes = data_samples[j].gt_instances.bboxes.cpu().numpy()
            img = draw_bboxes(img, bboxes, labels)
            cv2.imwrite(f'temp/xx{i}_{j}.jpg', img)
        if i == 10:
            break
if __name__ == '__main__':
    test_RedoxBaseDataset(redox_dataset_config)