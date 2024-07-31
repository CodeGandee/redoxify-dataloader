from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import numpy as np

def draw_and_save_img(image, bboxes, class_labels, qualities, output_file):
    import cv2
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for bbox, class_label, quality in zip(bboxes, class_labels, qualities):
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1*image.shape[1]), int(y1*image.shape[0]), int(x2*image.shape[1]), int(y2*image.shape[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, str(class_label), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, str(quality), (x2-20, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite(output_file, image)


pipe = Pipeline(batch_size=1, num_threads=1, device_id=0)
with pipe:
    inputs = fn.readers.tfrecord(
        path=["/workspace/redoxify_example/record/shards/detection_record-00000-of-00008"],
        index_path=["/workspace/redoxify_example/record/shards/detection_index-00000-of-00008"],
        features={
        "image": tfrec.FixedLenFeature((), tfrec.string, ""),
        "labels": tfrec.VarLenFeature(tfrec.int64, -1),
        "bboxes": tfrec.VarLenFeature(tfrec.float32, 0.0),
        "qualities": tfrec.VarLenFeature(tfrec.float32, 0.0),
    },
    )
    jpegs = inputs["image"]
    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
    resized = fn.resize(images, device="gpu", resize_shorter=1080.0)
    pipe.set_outputs(resized, inputs["bboxes"], inputs["labels"], inputs["qualities"])
pipe.build()

iterator = DALIGenericIterator([pipe], ['images', 'bboxes', 'labels', 'qualities'])

for idx, data in enumerate(iterator):
    image = data[0]['images'][0]
    bboxes = data[0]['bboxes'][0]
    class_labels = data[0]['labels'][0]
    qualities = data[0]['qualities'][0]
    print(image.shape)
    draw_and_save_img(image.cpu().numpy(), bboxes.cpu().numpy().reshape(-1,4), class_labels.cpu().numpy(), qualities.cpu().numpy(), f'./tmp/{idx}.jpg')
    if idx == 10:
        break