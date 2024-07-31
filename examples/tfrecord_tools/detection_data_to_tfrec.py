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
    
if __name__ == "__main__":
    image_path = "./data/"
    output_record_path = './record/example.record'
    output_record_idx_path = './record/example.idx'
    image_path_list = glob.glob(os.path.join(image_path, '*.jpg'))
    
    output_tfrecord = tf.python_io.TFRecordWriter(output_record_path)
    for image_path in image_path_list:
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
            for line in lines[1:]:
                line = line.strip().split()
                bboxes.append([float(x) for x in line[:4]])
                class_labels.append(int(line[4]))
                qualities.append(float(line[5]))
        key, tf_example = create_tf_example(encoded_jpg, np.array(bboxes), class_labels, qualities)
        if tf_example:
            output_tfrecord.write(tf_example.SerializeToString())
                
    tfrecord2idx_script = "./tfrecord2idx"
    if not os.path.isfile(output_record_idx_path):
        call([tfrecord2idx_script, output_record_path, output_record_idx_path])