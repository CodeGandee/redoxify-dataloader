<a id="readme-top"></a>

<br />

<div align="center">

<h3 align="center">Redoxify</h3>

  <p align="center">
   NVIDIA DALI-based data loader integrated into your favorite DL frameworks (such as mmdetection).
  </p>
</div>

## About The Project

Redoxify leverages the NVIDIA DALI (Data Loading Library) framework to create a highly efficient data loader for deep learning tasks, specifically designed for use with PyTorch. The goal is to minimize the time spent on data loading and augmentation, allowing users to focus more on model training. With Redoxify, users can generate a complete DALI-based dataloader pipeline using only a configuration file, making it easy to use DALI without needing to understand its complexities.

### Features
 - **Configuration-Driven Pipelines**: Generate a complete data loading and augmentation pipeline using a simple configuration file.
 - **GPU and CPU Support**: Data loading and augmentation can be performed on both GPU and CPU.
 - **Efficient Data Loading**: Rapid data reading from files directly into GPU memory.
 - **Seamless Data Augmentation**: Various augmentation operations are applied directly on the GPU, significantly reducing preprocessing time.
 - **Integration with PyTorch**: Easy integration with PyTorch for seamless data processing and training.
 - **User-Friendly**: No need to write complex DALI code; simply provide a configuration file.

_For details, please refer to the [Documentation]_

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

### Installation

To install Redoxify, follow these steps:

1. Install the required dependencies:
    - pytorch <=2.3.0, >=1.9.0
    - for CUDA 11.0:
        ```sh
        pip install --extra-index-url https://pypi.nvidia.com --upgrade nvidia-dali-cuda110
        ```
    - for CUDA 12.0:
        ```sh
        pip install --extra-index-url https://pypi.nvidia.com --upgrade nvidia-dali-cuda120
        ```

2. Clone this repository:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

### Usage
Here's an very simple example of how to use Redoxify in your project:
1. **Create a Configuration File(a .py file)**
    ```python
    tf_feature_spec = {
            "image": tfrec.FixedLenFeature((), tfrec.string, ""),
            "labels": tfrec.VarLenFeature(tfrec.int64, -1),
            "bboxes": tfrec.VarLenFeature(tfrec.float32, 0.0),
        }
    datablock_spec = {
        "image": ImageSpec(encoding='jpg',channel=3),
        "labels": VectorSpec(dtype=DALIDataType.INT64),
        "bboxes": BoxSpec(),
    }
    record_files = sorted(glob.glob("/path/to/your/records/*record*"))
    index_files = sorted(glob.glob("/path/to/your/records/*index*"))
    tf_files = [TFRecordFile(record_file=rec_file, index_file=idx_file) for rec_file, idx_file in zip(record_files, index_files)]
    reader_cfg = TFReaderConfig(tf_feature_spec=tf_feature_spec, datablock_spec=datablock_spec, random_shuffle=True)

    image_crop_setting = ImageCropSetting(image_key=DataKey("image"), output_key=DataKey("image"))
    labels_crop_setting = LabelCropSetting(ref_box_key=DataKey("bboxes",), crop_box_key=DataKey("bboxes"),
                                    crop_label_key=DataKey("labels"),
                                    output_box_key=DataKey("bboxes"), output_label_key=DataKey("labels"))
    crop_map = CropInputOutputMap(image_crop_settings=[image_crop_setting], label_crop_settings=[labels_crop_setting])
    crop_cfg = CropConfig(aspect_ratio_wh_min=0.5, aspect_ratio_wh_max=2.0, box_length_min=0.5, box_length_max=1.0)
    crop_cfg._all_boxes_above_threshold = False

    image_resize_setting = ImageResizeSetting(image_key=DataKey("image"), output_key=DataKey("image"))
    box_resize_setting = BoxResizeSetting(box_key=DataKey("bboxes"), output_key=DataKey("bboxes"))
    resize_map = ResizeInputOutputMap(image_resize_settings=[image_resize_setting], box_resize_settings=[box_resize_setting])
    resize_cfg = ResizeConfig(target_height=640, target_width=640, keep_aspect_ratio=True, keep_aspect_ratio_mode='not_larger')

    output_map = dict(
        images=SingleOutputSpec(input_key=DataKey("image"), pad_for_batch=True, split_batch_into_list=True),
        bboxes=SingleOutputSpec(input_key=DataKey("bboxes"), pad_for_batch=True, split_batch_into_list=True),
        classes=SingleOutputSpec(input_key=DataKey("labels"), pad_for_batch=True, split_batch_into_list=True)
    )

    redox_loader_config = dict(
        pipeline_cfg=dict(
            batch_size=8,
            num_workers=8,
        ),
        reader=dict(
            tf_files=tf_files,
            reader_cfg=reader_cfg,
            do_not_split_tfrec=False,
        ),
        transform_sequence=[
            dict(
                type='RandomCropWithBoxes',
                config=crop_cfg,
                inout_map=crop_map
            ),
            dict(
                type='Resize',
                config=resize_cfg,
                inout_map=resize_map
            )
        ],
        output_map=output_map,
    )
    ```    

2. **Load the Configuration and Initialize the Data Loader**

## Customization

Redoxify allows you to define your data augmentation pipeline through a configuration file. You can add, remove, or modify the augmentation operations as needed. DALI provides a wide range of operations that can be used in your pipeline, such as:

 - Image Resizing
 - Normalization
 - Cropping
 - Color Adjustments
 - Random Augmentations

 Refer to the documentation for more details on the available operations and their usage.

## Contributing
We welcome contributions to enhance this project. Feel free to submit issues and pull requests on GitHub.

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.

## Acknowledgements
 - NVIDIA DALI for providing an excellent library for efficient data loading and augmentation.

## Contact
For any questions or inquiries, please contact