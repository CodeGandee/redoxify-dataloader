from redoxify.transforms.BaseTransform import (
    BaseTransform,
    TransformInput,
    TransformOutput,
    TransformContext,
    TransformParams,
)
from redoxify.transforms.RandomCropWithBoxes import (
    RandomCropWithBoxes,
    CropConfig,
    CropInputOutputMap,
    LabelCropSetting,
    ImageCropSetting,
)
from redoxify.transforms.Resize import (
    Resize,
    ResizeConfig,
    ResizeInputOutputMap,
    BoxResizeSetting,
    ImageResizeSetting,
)
from redoxify.transforms.pytorch.Mosaic import (
    Mosaic,
    MosaicConfig,
    MosaicInputOutputMap,
    ImageMosaicSetting,
    BoxMosaicSetting,
    LabelMosaicSetting,
)

from redoxify.transforms.pytorch.Mixup import (
    Mixup,
    MixupConfig,
    MixupInputOutputMap,
    ImageMixupSetting,
    BoxMixupSetting,
    LabelMixupSetting,
)

from redoxify.transforms.pytorch.Blur import (
    Blur,
    BlurConfig,
    BlurInputOutputMap,
    ImageBlurSetting,
)

from redoxify.transforms.pytorch.MedianBlur import (
    MedianBlur,
    MedianBlurConfig,
    MedianBlurInputOutputMap,
    ImageMedianBlurSetting,
)

from redoxify.transforms.Pad import (
    Pad,
    PadConfig,
    PadInputOutputMap,
    ImagePadSetting,
    BoxPadSetting,
)

from redoxify.transforms.pytorch.CLAHE import (
    Clahe,
    ClaheConfig,
    ClaheInputOutputMap,
    ImageClaheSetting,
)

from redoxify.transforms.RandomSingleDirectionFlip import (
    RandomSingleDirectionFlip,
    RandomSingleDirectionFlipConfig,
    RandomSingleDirectionFlipInputOutputMap,
    ImageRandomSingleDirectionFlipSetting,
    BoxRandomSingleDirectionFlipSetting,
)
from redoxify.transforms.pytorch.RandomHSVAug import (
    RandomHSVAug,
    RandomHSVConfig,
    ImageRandomHSVSetting,
    RandomHSVInputOutputMap,
)

from redoxify.transforms.pytorch.RandomAffine import (
    RandomAffine,
    AffineConfig,
    AffineInputOutputMap,
    ImageAffineSetting,
    BoxAffineSetting,
    LabelAffineSetting,
)

from redoxify.transforms.DataBlockGrouping import (
    GroupingConfig,
    GroupingInputOutputMap,
    GroupingParams,
    GroupMemberSetting,
    DataBlockGrouping,
)

__all__ = [
    "BaseTransform",
    "TransformInput",
    "TransformOutput",
    "TransformContext",
    "TransformParams",
    "RandomCropWithBoxes",
    "CropConfig",
    "CropInputOutputMap",
    "LabelCropSetting",
    "ImageCropSetting",
    "Resize",
    "ResizeConfig",
    "ResizeInputOutputMap",
    "BoxResizeSetting",
    "ImageResizeSetting",
    "Pad",
    "PadConfig",
    "PadInputOutputMap",
    "ImagePadSetting",
    "BoxPadSetting",
    "Clahe",
    "ClaheConfig",
    "ImageClaheSetting",
    "ClaheInputOutputMap",
    "RandomSingleDirectionFlip",
    "RandomSingleDirectionFlipConfig",
    "RandomSingleDirectionFlipInputOutputMap",
    "ImageRandomSingleDirectionFlipSetting",
    "BoxRandomSingleDirectionFlipSetting",
    "RandomHSVAug",
    "RandomHSVConfig",
    "ImageRandomHSVSetting",
    "RandomHSVInputOutputMap",
    "RandomAffine",
    "AffineConfig",
    "AffineInputOutputMap",
    "ImageAffineSetting",
    "BoxAffineSetting",
    "LabelAffineSetting",
    "GroupingConfig",
    "GroupingInputOutputMap",
    "GroupingParams",
    "GroupMemberSetting",
    "DataBlockGrouping",
]

_TRANSFORM_CLASS_MAP = {
    "RandomCropWithBoxes": RandomCropWithBoxes,
    "Resize": Resize,
    "Mosaic": Mosaic,
    "Mixup": Mixup,
    "Blur": Blur,
    "CLAHE": Clahe,
    "MedianBlur": MedianBlur,
    "Pad": Pad,
    "RandomSingleDirectionFlip": RandomSingleDirectionFlip,
    "RandomHSVAug": RandomHSVAug,
    "DataBlockGrouping": DataBlockGrouping,
    "RandomAffine": RandomAffine,
}
