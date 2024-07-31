import os
import io
from PIL import Image
import hashlib
import logging
import glob
from subprocess import call
import numpy as np
import tensorflow.compat.v1 as tf
import contextlib2
from pycocotools.coco import COCO

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def open_sharded_output_tfrecords(exit_stack, base_path, num_shards):
  """Opens all TFRecord shards for writing and adds them to an exit stack.

  Args:
    exit_stack: A context2.ExitStack used to automatically closed the TFRecords
      opened in this function.
    base_path: The base path for all shards
    num_shards: The number of shards

  Returns:
    The list of opened TFRecords. Position k in the list corresponds to shard k.
  """
  tf_record_output_filenames = [
      '{}-{:05d}-of-{:05d}'.format(base_path, idx, num_shards)
      for idx in range(num_shards)
  ]

  tfrecords = [
      exit_stack.enter_context(tf.python_io.TFRecordWriter(file_name))
      for file_name in tf_record_output_filenames
  ]

  return tfrecords

def create_tf_example(encoded_jpg, bboxes, category_ids, quality_list):
    key = hashlib.sha256(encoded_jpg).hexdigest()
    if isinstance(bboxes, np.ndarray):
        bboxes = bboxes.flatten()
        bboxes = bboxes.tolist()
    if isinstance(category_ids, np.ndarray):
        category_ids = category_ids.tolist()
    if isinstance(quality_list, np.ndarray):
        quality_list = quality_list.tolist()
    feature_dict = {
            'image':
                    bytes_feature(encoded_jpg),
            'bboxes':
                    float_list_feature(bboxes),
            'labels':
                    int64_list_feature(category_ids),
            'qualities':
                    float_list_feature(quality_list),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return key, example

def draw_and_save_img(image, bboxes, class_labels, output_file):
    import cv2
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for bbox, class_label in zip(bboxes, class_labels):
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1*image.shape[1]), int(y1*image.shape[0]), int(x2*image.shape[1]), int(y2*image.shape[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, str(class_label), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite(output_file, image)

if __name__ == "__main__":
    image_path = "./data/"
    output_record_dir = './record/shards/'
    output_record_prefix = 'detection_record'
    output_index_prefix = 'detection_index'
    num_shards = 8
    image_path_list = glob.glob(os.path.join(image_path, '*.jpg'))
    os.makedirs(output_record_dir, exist_ok=True)
    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = open_sharded_output_tfrecords(
                tf_record_close_stack, os.path.join(output_record_dir, output_record_prefix), num_shards)
        for idx, image_path in enumerate(image_path_list):
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            encoded_jpg_io = io.BytesIO()
            # encode the image to jpeg format
            image.save(encoded_jpg_io, format='JPEG')
            encoded_jpg = encoded_jpg_io.getvalue()
            with open(image_path.replace('.jpg', '.txt'), 'r') as f:
                lines = f.readlines()
                bboxes = []
                class_labels = []
                qualities = []
                for line in lines[1:]:
                    line = line.strip().split()
                    bboxes.append([float(x) for x in line[:4]])
                    class_labels.append(int(line[4]))
                    qualities.append(float(line[5]))
            print(max(class_labels))
            key, tf_example = create_tf_example(encoded_jpg, np.array(bboxes), class_labels, qualities)
            shard_idx = idx % num_shards
            if tf_example:
                output_tfrecords[shard_idx].write(tf_example.SerializeToString())
    tfrecord_files = glob.glob(os.path.join(output_record_dir, f'{output_record_prefix}*'))
    tfrecord2idx_script = "./tfrecord2idx"

    for tfrecord_file in tfrecord_files:
        tfrecord_idx = tfrecord_file.replace(output_record_prefix, output_index_prefix)
        if not os.path.isfile(tfrecord_idx):
            call([tfrecord2idx_script, tfrecord_file, tfrecord_idx])