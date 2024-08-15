import os
import io
import PIL
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
            # 'qualities':
            #         float_list_feature(quality_list),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return key, example
    
def coco_image_annotation_iterator(coco_annotation_file, images_dir, normalize_bbox=True):
    coco = COCO(coco_annotation_file)
    categories = coco.loadCats(coco.getCatIds())
    sorted_categories = sorted(categories, key=lambda x: x['id'])
    category_mapping = {cat['id']: idx for idx, cat in enumerate(sorted_categories)}
    img_ids = coco.getImgIds()
    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(images_dir, img_info['file_name'])
        img_height = img_info.get('height', None)
        img_width = img_info.get('width', None)
        with tf.gfile.GFile(img_path, 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = PIL.Image.open(encoded_jpg_io)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            logging.info('Converting %s to RGB', img_info['file_name'])
            encoded_jpg_io = io.BytesIO()
            image.save(encoded_jpg_io, format='JPEG')
            encoded_jpg = encoded_jpg_io.getvalue()
        if img_height is None or img_width is None:
            img_height, img_width = image.size
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            ann['category_id'] = category_mapping[ann['category_id']]
        
        bboxes = np.array([ann['bbox'] for ann in anns])
        if bboxes.size < 4:
            bboxes = np.zeros((1, 4), dtype=np.float32)
        if normalize_bbox:
            bboxes /= np.array([img_width, img_height, img_width, img_height])
            bboxes = np.clip(bboxes, 0, 1)
        class_labels = np.array([ann['category_id'] for ann in anns])
        if class_labels.size < 1:
            class_labels = np.zeros(1, dtype=np.int64)
        #convert xywh to xyxy
        bboxes[:, 2] += bboxes[:, 0]
        bboxes[:, 3] += bboxes[:, 1]
        #FIXME generate random quality values, replace with your actual quality values
        qualities = np.random.rand(class_labels.size).astype(np.float32)
        # draw_and_save_img(np.array(image), bboxes, class_labels, f'./tmp/{img_info["file_name"]}')
        yield encoded_jpg, bboxes, class_labels, qualities

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
    coco_annotation_file = '/mnt/data/coco2017/annotations/instances_train2017.json'
    images_dir = '/mnt/data/coco2017/train2017'
    output_dir = './record/coco_mini/'
    record_file_prefix = 'coco_train_record'
    index_file_prefix = 'coco_train_index'
    num_shards = 8
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, record_file_prefix)
    with contextlib2.ExitStack() as tf_record_close_stack, \
            tf.gfile.GFile(coco_annotation_file, 'r') as fid:
        output_tfrecords = open_sharded_output_tfrecords(
                tf_record_close_stack, output_path, num_shards)
        coco_iterator = coco_image_annotation_iterator(coco_annotation_file, images_dir)
        for idx, (image, bboxes, class_labels, qualities) in enumerate(coco_iterator):
            if (idx+1) % 1000 == 0:
                logging.info('On image %d', idx+1)
            if idx>800:
                break
            key, tf_example = create_tf_example(image, bboxes, class_labels, qualities)
            shard_idx = idx % num_shards
            if tf_example:
                output_tfrecords[shard_idx].write(tf_example.SerializeToString())
                
    tfrecord_files = glob.glob(f'{output_path}*')
    tfrecord2idx_script = "./examples/tfrecord_tools/tfrecord2idx"

    for tfrecord_file in tfrecord_files:
        tfrecord_idx = tfrecord_file.replace(record_file_prefix, index_file_prefix)
        if not os.path.isfile(tfrecord_idx):
            call([tfrecord2idx_script, tfrecord_file, tfrecord_idx])