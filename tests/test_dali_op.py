from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import cv2
import numpy as np
from numpy import random
from nvidia.dali.plugin.pytorch import DALIGenericIterator

def hsv_augment(image, hue=0.0, sat=1.0, val=1.0):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue_channel, sat_channel, val_channel = cv2.split(hsv_image)
    hue_channel = (hue_channel.astype(np.float32) + hue * 180) % 180
    hue_channel = hue_channel.astype(np.uint8)
    sat_channel = np.clip(sat_channel.astype(np.float32) * sat, 0, 255).astype(np.uint8)
    val_channel = np.clip(val_channel.astype(np.float32) * val, 0, 255).astype(np.uint8)
    enhanced_hsv_image = cv2.merge([hue_channel, sat_channel, val_channel])
    enhanced_bgr_image = cv2.cvtColor(enhanced_hsv_image, cv2.COLOR_HSV2BGR)
    return enhanced_bgr_image

def mm_hsv(image, hue, saturation, value):
    hsv_gains = np.array([hue, saturation, value], dtype=np.float32)
    hue, sat, val = cv2.split(
            cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    table_list = np.arange(0, 256, dtype=hsv_gains.dtype)
    lut_hue = ((table_list + hsv_gains[0]) % 180).astype(np.uint8)
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
    converted = fn.hsv(images, hue=hue, saturation=saturation, value=value)
    return images, converted

pipe_cpu = hsv_pipeline(
    device="gpu",
    hue=0,
    saturation=1.9,
    value=1.0,
    batch_size=batch_size,
    num_threads=1,
    device_id=0,
)
pipe_cpu.build()
dali_iter = DALIGenericIterator(pipe_cpu, ["img1", "hsv"])
for i, data in enumerate(dali_iter):
    img_hsv = data[0]['hsv']
    hsv_dali = img_hsv.cpu().numpy()[0]
    hsv_dali = cv2.cvtColor(hsv_dali, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'temp/hsv_dali_{i}.jpg', hsv_dali)
    hsv_mm = mm_hsv(img, 0, 1.5, 1.0)
    cv2.imwrite(f'temp/hsv_mm_{i}.jpg', hsv_mm)
    break