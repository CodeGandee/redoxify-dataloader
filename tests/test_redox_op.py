import os, sys
import cv2
import torch
import numpy as np
from numpy import random
import albumentations as A

sys.path.append("/workspace/redoxify-dataloader/src/")

from redoxify.functionals.cuda_clahe import _clahe, old_clahe


def cv_clahe(img, clip_limit=2.0, grid_size=(8, 8)):
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
    img_rgb = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
    return img_rgb

image_filename = "/workspace/redoxify-dataloader/tests/fuji.jpg"

img = cv2.imread(image_filename)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_rgb_tensor = torch.from_numpy(img_rgb)
img_tensor = torch.from_numpy(img)

fix_clip_limit = 4.0

a_pipe = A.Compose(
    [
        A.CLAHE(
            clip_limit=(fix_clip_limit, fix_clip_limit),
            tile_grid_size=(8, 8),
            always_apply=True,
            p=1.0,
        )
    ]
)

albu_aug_img = a_pipe(image=img)["image"]
cv2.imwrite(f"temp2/albu_clahe.png", albu_aug_img)

redox_aug_tensor = _clahe(
    img_rgb_tensor,
    clip_limit=torch.tensor([fix_clip_limit, fix_clip_limit]),
    tile_grid_size=torch.tensor([8, 8]),
    probability=torch.tensor(1.0),
)
redox_aug_tensor_old = old_clahe(
    img_rgb_tensor,
    clip_limit=torch.tensor([fix_clip_limit, fix_clip_limit]),
    tile_grid_size=torch.tensor([8, 8]),
    probability=torch.tensor(1.0),
)
redox_aug_img = redox_aug_tensor.cpu().numpy()
redox_aug_img = cv2.cvtColor(redox_aug_img, cv2.COLOR_RGB2BGR)

redox_aug_img_old = redox_aug_tensor_old.cpu().numpy()
redox_aug_img_old = cv2.cvtColor(redox_aug_img_old, cv2.COLOR_RGB2BGR)


difference = np.abs(
    albu_aug_img.astype(np.int32) - redox_aug_img.astype(np.int32)
).mean()
print(f"Mean difference between OpenCV CLAHE and Redox CLAHE: {difference.item()}")
# redox_aug_img = cv2.cvtColor(redox_aug_img, cv2.COLOR_RGB2BGR)
# cv2.imwrite(f"temp2/redox_clahe.png", redox_aug_img)
cv2.imwrite(f"temp2/redox_clahe.png", redox_aug_img)
cv2.imwrite(f"temp2/redox_clahe_old.png", redox_aug_img_old)

cv_clahe_img = cv_clahe(img, clip_limit=fix_clip_limit, grid_size=(8, 8))
cv2.imwrite(f"temp2/cv_clahe.png", cv_clahe_img)
