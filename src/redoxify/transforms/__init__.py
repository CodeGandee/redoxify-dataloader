from redoxify.transforms.BaseTransform import BaseTransform, TransformInput, TransformOutput, TransformContext, TransformParams
from redoxify.transforms.RandomCropWithBoxes import (
    RandomCropWithBoxes, CropConfig, CropInputOutputMap, LabelCropSetting, ImageCropSetting
)
from redoxify.transforms.Resize import (
    Resize, ResizeConfig, ResizeInputOutputMap, BoxResizeSetting, ImageResizeSetting
)
from redoxify.transforms.pytorch.Mosaic import (
    Mosaic, MosaicConfig, MosaicInputOutputMap, ImageMosaicSetting, BoxMosaicSetting, LabelMosaicSetting
)

from redoxify.transforms.pytorch.Mixup import (
    Mixup, MixupConfig, MixupInputOutputMap, ImageMixupSetting, BoxMixupSetting, LabelMixupSetting
)


from redoxify.transforms.Pad import (
    Pad, PadConfig, PadInputOutputMap, ImagePadSetting, BoxPadSetting
)
from redoxify.transforms.RandomSingleDirectionFlip import (
    RandomSingleDirectionFlip, RandomSingleDirectionFlipConfig, RandomSingleDirectionFlipInputOutputMap, ImageRandomSingleDirectionFlipSetting, BoxRandomSingleDirectionFlipSetting
)
from redoxify.transforms.RandomHSVAug import (
    RandomHSVAug, RandomHSVConfig, ImageRandomHSVSetting, RandomHSVInputOutputMap
)
from redoxify.transforms.DataBlockGrouping import (
    GroupingConfig, GroupingInputOutputMap, GroupingParams, GroupMemberSetting, DataBlockGrouping
)

__all__ = [
    'BaseTransform', 'TransformInput', 'TransformOutput', 'TransformContext', 'TransformParams',
    'RandomCropWithBoxes', 'CropConfig', 'CropInputOutputMap', 'LabelCropSetting', 'ImageCropSetting',
    'Resize', 'ResizeConfig', 'ResizeInputOutputMap', 'BoxResizeSetting', 'ImageResizeSetting',
    'Pad', 'PadConfig', 'PadInputOutputMap', 'ImagePadSetting', 'BoxPadSetting',
    'RandomSingleDirectionFlip', 'RandomSingleDirectionFlipConfig', 'RandomSingleDirectionFlipInputOutputMap', 'ImageRandomSingleDirectionFlipSetting', 'BoxRandomSingleDirectionFlipSetting',
    'RandomHSVAug', 'RandomHSVConfig', 'ImageRandomHSVSetting', 'RandomHSVInputOutputMap',
    'GroupingConfig', 'GroupingInputOutputMap', 'GroupingParams', 'GroupMemberSetting', 'DataBlockGrouping'
]

_TRANSFORM_CLASS_MAP = {
    'RandomCropWithBoxes': RandomCropWithBoxes,
    'Resize': Resize,
    'Mosaic': Mosaic,
    'Mixup': Mixup,
    'Pad': Pad,
    'RandomSingleDirectionFlip': RandomSingleDirectionFlip,
    'RandomHSVAug': RandomHSVAug,
    'DataBlockGrouping': DataBlockGrouping
}
