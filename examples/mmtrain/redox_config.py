import os
import glob
import nvidia.dali.tfrecord as tfrec
from nvidia.dali.types import DALIDataType
from redoxify.datablocks.ImagesBlock import ImageSpec
from redoxify.datablocks.VectorsBlock import VectorSpec
from redoxify.datablocks.BoxesBlock import BoxSpec
from redoxify.readers import TFRecordReader, TFReaderConfig, TFRecordFile
from redoxify.RedoxTypes import DataKey

from redoxify.transforms.RandomCropWithBoxes import (
    CropConfig, CropInputOutputMap, LabelCropSetting, ImageCropSetting
)
from redoxify.transforms.Resize import (
    ResizeConfig, ResizeInputOutputMap, BoxResizeSetting, ImageResizeSetting
)
from redoxify.transforms.Pad import (
    PadConfig, PadInputOutputMap, ImagePadSetting, BoxPadSetting
)

from redoxify.transforms.pytorch.Mosaic import (
    MosaicConfig, MosaicInputOutputMap, ImageMosaicSetting, BoxMosaicSetting, LabelMosaicSetting
)

from redoxify.transforms.pytorch.Mixup import (
    MixupConfig, MixupInputOutputMap, ImageMixupSetting, BoxMixupSetting, LabelMixupSetting
)
from redoxify.transforms.RandomSingleDirectionFlip import (
    RandomSingleDirectionFlipConfig, RandomSingleDirectionFlipInputOutputMap, ImageRandomSingleDirectionFlipSetting, BoxRandomSingleDirectionFlipSetting
)
from redoxify.transforms.RandomHSVAug import (
    RandomHSVConfig, ImageRandomHSVSetting, RandomHSVInputOutputMap
)
from redoxify.pipelines.PipelineBuilder import SingleOutputSpec
tf_feature_spec = {
        "image": tfrec.FixedLenFeature((), tfrec.string, ""),
        "labels": tfrec.VarLenFeature(tfrec.int64, -1),
        "bboxes": tfrec.VarLenFeature(tfrec.float32, 0.0),
        "qualities": tfrec.VarLenFeature(tfrec.float32, 0.0),
    }
datablock_spec = {
    "image": ImageSpec(encoding='jpg',channel=3),
    "labels": VectorSpec(dtype=DALIDataType.INT64),
    "bboxes": BoxSpec(),
    "qualities": VectorSpec(),
}


# record_files = ["/workspace/redoxify_example/record/example.record"]
# index_files = ["/workspace/redoxify_example/record/example.idx"]
record_files = sorted(glob.glob("/workspace/redoxify_example/record/coco_train/*record*"))
index_files = sorted(glob.glob("/workspace/redoxify_example/record/coco_train/*index*"))
tf_files = [TFRecordFile(record_file=rec_file, index_file=idx_file) for rec_file, idx_file in zip(record_files, index_files)]
reader_cfg = TFReaderConfig(tf_feature_spec=tf_feature_spec, datablock_spec=datablock_spec, random_shuffle=True)

# cropping config
# there are two types of labels(labels and qualities). two crop specs are needed
# bboxes together with labels are cropped at the first stage, and qualities are cropped at the second stage
# the refrerence boxes are the same for both stages, so output boxe key at the first stage should be different from the ref_box_key
image_crop_setting = ImageCropSetting(image_key=DataKey("image"), output_key=DataKey("image"))
labels_crop_setting = LabelCropSetting(ref_box_key=DataKey("bboxes",), crop_box_key=DataKey("bboxes"),
                                crop_label_key=DataKey("labels"),
                                output_box_key=DataKey("bboxes"), output_label_key=DataKey("labels"))
# qualities_crop_spec = LabelCropSpec(ref_box_key=DataKey("bboxes"), crop_box_key=DataKey("bboxes"),
#                                 crop_label_key=DataKey("qualities"),
#                                 output_box_key=DataKey("bboxes"), output_label_key=DataKey("qualities"))
crop_map = CropInputOutputMap(image_crop_settings=[image_crop_setting], label_crop_settings=[labels_crop_setting])
# crop config, crop args randomly generated w/h ratio between aspect_ratio_wh_min and aspect_ratio_wh_max, 
# and box size/original image size between box_length_min and box_length_max
crop_cfg = CropConfig(aspect_ratio_wh_min=0.5, aspect_ratio_wh_max=2.0, box_length_min=0.5, box_length_max=1.0)
crop_cfg._all_boxes_above_threshold = False

# resize config, specific image and boxes to be resized together
# if you do not want to change the aspect ratio, set keep_aspect_ratio to True
# not_larger mode: both width and height of the resized image will be less than or equal to the target size
# not_smaller mode: both width and height of the resized image will be greater than or equal to the target size
# for example, if the original image size is 1280x720, and the target size is 640x640, 
# it will be resized to 640x360 in not_larger mode, and 1138x640 in not_smaller mode
# the resize result is not certain if keep_aspect_ratio is True and original image size is variable
image_resize_setting = ImageResizeSetting(image_key=DataKey("image"), output_key=DataKey("image"))
box_resize_setting = BoxResizeSetting(box_key=DataKey("bboxes"), output_key=DataKey("bboxes"))
resize_map = ResizeInputOutputMap(image_resize_settings=[image_resize_setting], box_resize_settings=[box_resize_setting])
resize_cfg = ResizeConfig(target_height=640, target_width=640, keep_aspect_ratio=True, keep_aspect_ratio_mode='not_larger')

# if you want to keep the aspect ratio, and hope to get certain image size, you can use pad transform
# the pad transform will pad the image to the target size, and the padding values are determined by fill_values
# the boxes will be padded accordingly
# the anchor point of the original image in the padded image is fixed at the top-left corner
img_pad_setting = ImagePadSetting(image_key=DataKey("image"), output_key=DataKey("image"))
box_pad_setting = BoxPadSetting(box_key=DataKey("bboxes"), output_key=DataKey("bboxes"))
pad_map = PadInputOutputMap(image_pad_settings=[img_pad_setting], box_pad_settings=[box_pad_setting])
pad_cfg = PadConfig(target_height=640, target_width=640, fill_values=128.0)


img_mosaic_setting = ImageMosaicSetting(image_key=DataKey("image"), output_key=DataKey("image"))
box_mosaic_setting = BoxMosaicSetting(box_key=DataKey("bboxes"), output_key=DataKey("bboxes"))
label_mosaic_setting = LabelMosaicSetting(label_key=DataKey("labels"), output_key=DataKey("labels"))
mosaic_map = MosaicInputOutputMap(image_mosaic_settings=[img_mosaic_setting], box_mosaic_settings=[box_mosaic_setting], label_mosaic_settings=[label_mosaic_setting])
mosaic_cfg = MosaicConfig(mosaic_height=640, mosaic_width=640, fill_values=128.0)

img_mixup_setting = ImageMixupSetting(image_key=DataKey("image"), output_key=DataKey("image"))
box_mixup_setting = BoxMixupSetting(box_key=DataKey("bboxes"), output_key=DataKey("bboxes"))
label_mixup_setting = LabelMixupSetting(label_key=DataKey("labels"), output_key=DataKey("labels"))
mixup_map = MixupInputOutputMap(image_mixup_settings=[img_mixup_setting], box_mixup_settings=[box_mixup_setting], label_mixup_settings=[label_mixup_setting])
mixup_cfg = MixupConfig(mixup_lower_ratio=0.25, mixup_upper_ratio=0.75, fill_values=128.0)

# random flip config, flip the image and boxes horizontally(you can choose horizontal or vertical) with a probability of 0.5
img_flip_setting = ImageRandomSingleDirectionFlipSetting(image_key=DataKey("image"), output_key=DataKey("image"))
box_flip_setting = BoxRandomSingleDirectionFlipSetting(box_key=DataKey("bboxes"), output_key=DataKey("bboxes"))
flip_map = RandomSingleDirectionFlipInputOutputMap(image_flip_settings=[img_flip_setting], box_flip_settings=[box_flip_setting])
flip_cfg = RandomSingleDirectionFlipConfig(probability=0.5, flip_direction='horizontal')

# To change the hue, the saturation, and/or the value of the image, pass the corresponding coefficients. 
# Remember that the hue is an additive delta argument, while for saturation and value, the arguments are multiplicative.
# hsv args will randomly generated by:
#   new hue = hue+/-hue_delta
#   new saturation = saturation*(1+/-saturation_delta)
#   new value = value*(1+/-value_delta)
img_hsv_setting = ImageRandomHSVSetting(image_key=DataKey("image"), output_key=DataKey("image"))
hsv_map = RandomHSVInputOutputMap(image_hsv_settings=[img_hsv_setting])
hsv_cfg = RandomHSVConfig(hue_delta=15, saturation_delta=0.2, value_delta=0.2)

output_map = dict(
    images=SingleOutputSpec(input_key=DataKey("image"), pad_for_batch=True, split_batch_into_list=True),
    bboxes=SingleOutputSpec(input_key=DataKey("bboxes"), pad_for_batch=True, split_batch_into_list=True),
    classes=SingleOutputSpec(input_key=DataKey("labels"), pad_for_batch=True, split_batch_into_list=True),
)

# output_map = dict(
#     images=SingleOutputSpec(input_key=DataKey("mixup_image"), pad_for_batch=True, split_batch_into_list=True),
#     bboxes=SingleOutputSpec(input_key=DataKey("mixup_bboxes"), pad_for_batch=True, split_batch_into_list=True),
#     classes=SingleOutputSpec(input_key=DataKey("mixup_labels"), pad_for_batch=True, split_batch_into_list=True),
# )

mm_backend_args = None
albu_train_transforms = [
    # dict(
    #     type='ShiftScaleRotate',
    #     shift_limit=0.0625,
    #     scale_limit=0.0,
    #     rotate_limit=0,
    #     interpolation=1,
    #     p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    # dict(
    #     type='OneOf',
    #     transforms=[
    #         dict(
    #             type='RGBShift',
    #             r_shift_limit=10,
    #             g_shift_limit=10,
    #             b_shift_limit=10,
    #             p=1.0),
    #         dict(
    #             type='HueSaturationValue',
    #             hue_shift_limit=20,
    #             sat_shift_limit=30,
    #             val_shift_limit=20,
    #             p=1.0)
    #     ],
    #     p=0.1),
    # dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    # dict(type='ChannelShuffle', p=0.1),
    # dict(
    #     type='OneOf',
    #     transforms=[
    #         dict(type='Blur', blur_limit=3, p=1.0),
    #         dict(type='MedianBlur', blur_limit=3, p=1.0)
    #     ],
    #     p=0.1),
]

img_scale = (640, 640)
mm_pipeline = [
    dict(type='Resize', 
         scale=img_scale, 
         keep_ratio=True),
    dict(type='Pad', 
         size=img_scale, 
         pad_val=dict(img=(114,114,114,))
         ),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmyolo.Mosaic', 
         img_scale=img_scale, 
         center_ratio_range=(1.0,1.0),
         use_cached=True,
         pad_val=114.0),
    dict(
        type='mmyolo.YOLOXMixUp',
        img_scale=img_scale,
        use_cached=True,
        ratio_range=(0.9, 1.1),
        pad_val=114.0),

    # dict(type='mmdet.RandomFlip', prob=0.5),
    # dict(type='mmdet.FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    # dict(
    #     type='mmdet.Albu',
    #     transforms=albu_train_transforms,
    #     bbox_params=dict(
    #         type='BboxParams',
    #         format='pascal_voc',
    #         label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
    #     keymap={
    #         'img': 'image',
    #         'gt_bboxes': 'bboxes',
    #     },
    # ),
    # dict(type='mmdet.PackDetInputs')
]

redox_dataset_config = dict(
    pipeline_cfg=dict(
        batch_size=8,
        num_workers=8,
    ),
    reader=dict(
        type="TFRecordReader",
        tf_files=tf_files,
        config=reader_cfg,
    ),
    transform_sequence=[
        dict(
            type='RandomCropWithBoxes',
            config=crop_cfg,
            inout_map=crop_map
        ),
        # dict(
        #     type='Resize',
        #     config=resize_cfg,
        #     inout_map=resize_map
        # ),
        # dict(
        #     type='Pad',
        #     config=pad_cfg,
        #     inout_map=pad_map
        # ),
        # dict(
        #     type='RandomSingleDirectionFlip',
        #     config=flip_cfg,
        #     inout_map=flip_map
        # ),
        # dict(
        #     type='RandomHSVAug',
        #     config=hsv_cfg,
        #     inout_map=hsv_map
        # ),
        # dict(
        #     type='Mosaic',
        #     config=mosaic_cfg,
        #     inout_map=mosaic_map
        # ),
        # dict(
        #     type='Mixup',
        #     config=mixup_cfg,
        #     inout_map=mixup_map
        # ),
    ],
    output_map=output_map,
    normalized_bbox=True,
    mm_config=dict(
        image_key='images',
        bbox_key='bboxes',
        mm_key_mapping={"images": "img", "classes": "gt_bboxes_labels", "bboxes": "gt_bboxes"},
        mm_pipeline=mm_pipeline)
        # mm_pipeline=None)
)

