from mmyolo.datasets.yolov5_coco import YOLOv5CocoDataset
from mmyolo.registry import DATASETS

@DATASETS.register_module()
class YOLOv5PersonDataset(YOLOv5CocoDataset):
    METAINFO = {
        "classes": (
            "body",
            "head",
            "face",
        ),
        # palette is a list of color tuples, which is used for visualization.
        "palette": [
            (220, 20, 60),
            (119, 11, 32),
            (0, 0, 142),
        ],
    }
