from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import cv2
import torch
import numpy as np
from numpy import random
from nvidia.dali.plugin.pytorch import DALIGenericIterator

from kornia.color.hsv import rgb_to_hsv, hsv_to_rgb


def kornia_hsv(img, hue_multiplier, saturation_multiplier, value_multiplier):
    img_rgb = img.permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_hsv = rgb_to_hsv(img_rgb)
    img_hsv[:, 0, :, :] = (img_hsv[:, 0, :, :] * hue_multiplier) % (2 * torch.pi)
    img_hsv[:, 1, :, :] = (img_hsv[:, 1, :, :] * saturation_multiplier).clamp(0, 1.0)
    img_hsv[:, 2, :, :] = (img_hsv[:, 2, :, :] * value_multiplier).clamp(0, 1.0)
    img_rgb = hsv_to_rgb(img_hsv)
    img_rgb = (img_rgb * 255.0).byte()
    return img_rgb.squeeze(0).permute(1, 2, 0)


def mm_hsv(image, hue, saturation, value):
    hsv_gains = np.array([hue, saturation, value], dtype=np.float32)
    hue, sat, val = cv2.split(
            cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    table_list = np.arange(0, 256, dtype=hsv_gains.dtype)
    lut_hue = ((table_list * hsv_gains[0]) % 180).astype(np.uint8)
    lut_sat = np.clip(table_list * hsv_gains[1], 0, 255).astype(np.uint8)
    lut_val = np.clip(table_list * hsv_gains[2], 0, 255).astype(np.uint8)
    
    im_hsv = cv2.merge(
        (cv2.LUT(hue, lut_hue), cv2.LUT(sat,
                                        lut_sat), cv2.LUT(val, lut_val)))
    return cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)

batch_size = 1
image_filename = "/workspace/redoxify-dataloader/tests/fuji.jpg"

img = cv2.imread(image_filename)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

@pipeline_def()
def hsv_pipeline(device, hue, saturation, value):
    jpeg_data = fn.readers.file(files=[image_filename])[0]
    images = fn.decoders.image(
        jpeg_data, device="cpu" if device == "cpu" else "mixed"
    )
    gains = fn.random.uniform(range=[-1, 1], shape=3) * [hue, saturation, value] + 1
    converted = fn.hsv(images, hue=gains[0], saturation=gains[1], value=gains[2])
    return images, converted, gains

# pipe_cpu = hsv_pipeline(
#     device="gpu",
#     hue=0.15,
#     saturation=0.0,
#     value=0.,
#     batch_size=batch_size,
#     num_threads=1,
#     device_id=0,
# )
# pipe_cpu.build()
# dali_iter = DALIGenericIterator(pipe_cpu, ["img1", "hsv", 'gains'])

hsv_mm = mm_hsv(img, 1.015, 1.4, 0.8)
cv2.imwrite(f'temp/hsv_mm_{0}.jpg', hsv_mm)
hsv_kor = kornia_hsv(torch.from_numpy(img_rgb).to(device="cuda"), torch.tensor(1.015), torch.tensor(1.4), torch.tensor(0.8))
hsv_kor = cv2.cvtColor(hsv_kor.cpu().numpy(), cv2.COLOR_RGB2BGR)
cv2.imwrite(f'temp/hsv_kor_{0}.jpg', cv2.cvtColor(hsv_kor.cpu().numpy(), cv2.COLOR_RGB2BGR))