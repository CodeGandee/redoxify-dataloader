import sys, os
from tqdm import tqdm
sys.path.append('/workspace/redoxify-data-loader')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from redoxify.plugin.mmdetection.dataloader_builder import build_dataloader_from_cfg
from examples.mmtrain.redox_config import redox_dataset_config as redox_dataset_config
from mmengine.registry.default_scope import DefaultScope
DefaultScope.get_instance('task', scope_name='mmdet')
import cv2
import numpy as np

def draw_and_save_img(image, bboxes, class_labels, output_file):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for bbox, class_label in zip(bboxes, class_labels):
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, str(class_label), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite(output_file, image)


redox_dataset_config['pipeline_cfg']['batch_size']=4
redox_dataset_config['pipeline_cfg']['num_workers']=1
train_loader = build_dataloader_from_cfg(redox_dataset_config)

data_len = len(train_loader)
pbar = tqdm(total=data_len)
for idx, data in enumerate(train_loader):
    # print(idx, data)
    image = data['inputs'][0].permute(1,2,0).cpu().numpy()
    bboxes=data['data_samples'][0].gt_instances.bboxes.cpu().numpy()
    labels=data['data_samples'][0].gt_instances.labels.cpu().numpy()
    draw_and_save_img(image, bboxes, labels, f'./debug/{idx}.jpg')
    pbar.update(1)
    if idx>50:
        break
    