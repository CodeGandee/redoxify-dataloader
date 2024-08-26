from typing import Union, Dict, List, Tuple
from attrs import define, field
import attrs.validators as av
import copy

import nvidia.dali.fn as fn
from nvidia.dali import types as dali_types
from nvidia.dali.types import DALIDataType

from redoxify.datablocks.DataBlock import DataBlock, DALINode
from redoxify.functionals.cuda_affine import dali_affine
from redoxify.RedoxTypes import DataKey

from redoxify.datablocks.ImagesBlock import ImagesBlock, ImageSpec

from redoxify.datablocks.BoxesBlock import BoxesBlock, BoxSpec

from redoxify.datablocks.VectorsBlock import VectorsBlock, VectorSpec

from redoxify.transforms.BaseTransform import (
    BaseTransform,
    TransformParams,
    TransformInput,
    TransformOutput,
    TransformContext,
)


@define(kw_only=True, eq=False)
class ImageAffineSetting:
    image_key: DataKey = field()
    output_key: DataKey = field()


@define(kw_only=True, eq=False)
class BoxAffineSetting:
    box_key: DataKey = field()
    output_key: DataKey = field()


@define(kw_only=True, eq=False)
class LabelAffineSetting:
    label_key: DataKey = field()
    output_key: DataKey = field()


@define(kw_only=True, eq=False)
class AffineInputOutputMap:
    image_affine_settings: List[ImageAffineSetting] = field(factory=list)
    box_affine_settings: List[BoxAffineSetting] = field(factory=list)
    label_affine_settings: List[LabelAffineSetting] = field(factory=list)


@define(kw_only=True, eq=False)
class AffineConfig:
    max_rotate_degree: Union[float, DALINode] = field(
        default=10.0, validator=av.instance_of((float, DALINode))
    )
    max_translate_ratio: Union[float, DALINode] = field(
        default=0.1, validator=av.instance_of((float, DALINode))
    )
    scaling_ratio_range: Union[List[float], Tuple[float, float], DALINode] = field(
        default=(0.5, 1.5), validator=av.instance_of((List, Tuple, DALINode))
    )
    max_shear_degree: Union[float, DALINode] = field(
        default=2.0, validator=av.instance_of((float, DALINode))
    )
    border: Union[List[int], Tuple[int, int], DALINode] = field(
        default=(0, 0), validator=av.instance_of((List, Tuple, DALINode))
    )
    border_val: Union[List[int], Tuple[int, int, int], DALINode] = field(
        default=(114, 114, 114), validator=av.instance_of((List, Tuple, DALINode))
    )
    min_bbox_size: Union[float, DALINode] = field(
        default=2.0, validator=av.instance_of((float, DALINode))
    )
    min_area_ratio: Union[float, DALINode] = field(
        default=0.1, validator=av.instance_of((float, DALINode))
    )
    max_aspect_ratio: Union[float, DALINode] = field(
        default=20.0, validator=av.instance_of((float, DALINode))
    )
    probability: float = field(default=0.5, validator=av.instance_of(float))


@define(kw_only=True, eq=False)
class AffineParams(TransformParams):
    affine_indices: List[int] = field(factory=list)


class RandomAffine(BaseTransform):
    def __init__(
        self, random_affine_config: AffineConfig, inout_map: AffineInputOutputMap
    ):
        self.m_affine_config = random_affine_config
        self.m_inout_map = inout_map
        assert (
            len(self.m_inout_map.image_affine_settings) > 0
        ), "image_affine_settings must be provided"

    def do_transform(
        self, input_data: TransformInput, context: TransformContext = None
    ) -> TransformOutput:
        """
        perform random crop on the images and boxes
        """

        output = self._do_transform(input_data.data_blocks)

        affine_params = AffineParams(affine_indices=[0, 0, 0, 0])
        final_output = TransformOutput(data_blocks=output, params=affine_params)
        return final_output

    def _do_transform(
        self, input_data: Dict[str, DataBlock], *args, **kwargs
    ) -> Dict[str, DataBlock]:
        for img_setting in self.m_inout_map.image_affine_settings:
            assert (
                img_setting.image_key.main_key in input_data
            ), f"image key {img_setting.image_key} not found in data"
        for box_setting in self.m_inout_map.box_affine_settings:
            assert (
                box_setting.box_key.main_key in input_data
            ), f"box key {box_setting.box_key} not found in data"
        for label_setting in self.m_inout_map.label_affine_settings:
            assert (
                label_setting.label_key.main_key in input_data
            ), f"label key {label_setting.label_key} not found in data"
        assert len(self.m_inout_map.image_affine_settings) == len(
            self.m_inout_map.box_affine_settings
        ), "number of image and box settings must be the same"
        assert len(self.m_inout_map.image_affine_settings) == len(
            self.m_inout_map.label_affine_settings
        ), "number of image and label settings must be the same"

        image_key = self.m_inout_map.image_affine_settings[0].image_key
        image_datablock: ImagesBlock = input_data[image_key.main_key]

        config = self.m_affine_config
        output_data = copy.copy(input_data)

        for img_setting, boxes_setting, labels_setting in zip(
            self.m_inout_map.image_affine_settings,
            self.m_inout_map.box_affine_settings,
            self.m_inout_map.label_affine_settings,
        ):
            image_key = img_setting.image_key
            output_image_key = img_setting.output_key
            boxes_key = boxes_setting.box_key
            output_boxes_key = boxes_setting.output_key
            labels_key = labels_setting.label_key
            output_labels_key = labels_setting.output_key
            image_datablock = input_data[image_key.main_key]
            boxes_datablock = input_data[boxes_key.main_key]
            labels_datablock = input_data[labels_key.main_key]
            out_image_blk = ImagesBlock()
            out_boxes_blk = BoxesBlock()
            out_labels_blk = VectorsBlock()
            for sub_key in image_datablock.get_keys():
                image_data = image_datablock.get_data(sub_key)
                image_data_spec = image_datablock.get_spec(sub_key)
                boxes_data = boxes_datablock.get_data(sub_key)
                boxes_data_spec = boxes_datablock.get_spec(sub_key)
                labels_data = labels_datablock.get_data(sub_key)
                labels_data_spec = labels_datablock.get_spec(sub_key)
                scaling_ratio = fn.random.uniform(range=config.scaling_ratio_range)
                if config.max_rotate_degree == 0:
                    rotate_degree = dali_types.Constant([0.0])
                else:
                    rotate_degree = fn.random.uniform(
                        range=[-config.max_rotate_degree, config.max_rotate_degree]
                    )

                if config.max_translate_ratio == 0:
                    translate_ratio = dali_types.Constant([0.5, 0.5])
                else:
                    translate_ratio = fn.random.uniform(
                        range=[0.5-config.max_translate_ratio, 0.5+config.max_translate_ratio],
                        shape=2,
                    )

                if config.max_shear_degree == 0:
                    shear_degree_xy = dali_types.Constant([0.0, 0.0])
                else:
                    shear_degree_xy = fn.random.uniform(
                        range=[-config.max_shear_degree, config.max_shear_degree],
                        shape=2,
                    )
                affined_image, affined_boxes, affined_labels = dali_affine(
                    image_data.gpu(),
                    boxes_data.gpu(),
                    labels_data.gpu(),
                    rotate_degree=rotate_degree.gpu(),
                    translate_ratio=translate_ratio.gpu(),
                    scaling_ratio=scaling_ratio.gpu(),
                    shear_degree_xy=shear_degree_xy.gpu(),
                    border=config.border,
                    border_val=config.border_val,
                    min_bbox_size=config.min_bbox_size,
                    min_area_ratio=config.min_area_ratio,
                    max_aspect_ratio=config.max_aspect_ratio,
                )
                image_data_spec.height = fn.shapes(affined_image)[0]
                image_data_spec.width = fn.shapes(affined_image)[1]

                out_image_blk.add_data(sub_key, fn.copy(affined_image), image_data_spec)
                out_boxes_blk.add_data(sub_key, affined_boxes, boxes_data_spec)
                out_labels_blk.add_data(sub_key, affined_labels, labels_data_spec)
            output_data[output_image_key.main_key] = out_image_blk
            output_data[output_boxes_key.main_key] = out_boxes_blk
            output_data[output_labels_key.main_key] = out_labels_blk
        return output_data
