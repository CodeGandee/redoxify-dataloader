from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import cv2
import numpy as np
from numpy import random
from nvidia.dali.plugin.pytorch import DALIGenericIterator

from kornia.color.hsv import rgb_to_hsv, hsv_to_rgb

def kornia_hsv(image, hue, saturation, value):
    image = image.permute(0, 3, 1, 2)
    image = image.float() / 255.0
    image = rgb_to_hsv(image)
    image[:, 0, :, :] = (image[:, 0, :, :] + hue) % 1.0
    image[:, 1, :, :] = image[:, 1, :, :] * saturation
    image[:, 2, :, :] = image[:, 2, :, :] * value
    image = hsv_to_rgb(image)
    image = (image * 255.0).byte()
    return image.permute(0, 2, 3, 1)

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

@pipeline_def()
def hsv_pipeline(device, hue, saturation, value):
    jpeg_data = fn.readers.file(files=[image_filename])[0]
    images = fn.decoders.image(
        jpeg_data, device="cpu" if device == "cpu" else "mixed"
    )
    gains = fn.random.uniform(range=[-1, 1], shape=3) * [hue, saturation, value] + 1
    converted = fn.hsv(images, hue=gains[0], saturation=gains[1], value=gains[2])
    return images, converted, gains

pipe_cpu = hsv_pipeline(
    device="gpu",
    hue=0.15,
    saturation=0.0,
    value=0.,
    batch_size=batch_size,
    num_threads=1,
    device_id=0,
)
pipe_cpu.build()
dali_iter = DALIGenericIterator(pipe_cpu, ["img1", "hsv", 'gains'])
for i, data in enumerate(dali_iter):
    img_hsv = data[0]['hsv']
    print(data[0]["gains"])
    hsv_dali = img_hsv.cpu().numpy()[0]
    hsv_dali = cv2.cvtColor(hsv_dali, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'temp/hsv_dali_{i}.jpg', hsv_dali)
    hsv_mm = mm_hsv(img, 0.0, 0.0, 0.0)
    cv2.imwrite(f'temp/hsv_mm_{2}.jpg', hsv_mm)
    break
