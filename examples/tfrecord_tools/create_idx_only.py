
from subprocess import call
import glob
import os

output_path = '/workspace/github/redoxify-dataloader/record/coco_train/'
tfrecord_files = glob.glob(f'{output_path}*')
tfrecord2idx_script = "/workspace/github/redoxify-dataloader/examples/tfrecord_tools/tfrecord2idx"

for tfrecord_file in tfrecord_files:
    tfrecord_idx = tfrecord_file.replace('train_record', 'train_index')
    if not os.path.isfile(tfrecord_idx):
        call([tfrecord2idx_script, tfrecord_file, tfrecord_idx])
        