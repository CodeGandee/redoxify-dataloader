# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import cv2
from tqdm import tqdm
import os
import numpy as np
import os.path as osp

from mmdet.utils import setup_cache_size_limit_of_dynamo
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmyolo.registry import RUNNERS
from mmyolo.utils import is_metainfo_lower

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# os.environ['AWS_ACCESS_KEY_ID'] = 'minio'
# os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio123'
# os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://192.168.13.183:9000'


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument(
        "--config",
        help="train config file path",
        default="examples/mmtrain/configs/yolov8_s_ori.py",
    )
    parser.add_argument("--work-dir", help="the dir to save logs and models")
    parser.add_argument(
        "--amp",
        action="store_true",
        default=False,
        help="enable automatic-mixed-precision training",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        type=str,
        const="auto",
        help="If specify checkpoint path, resume from it, while if not "
        "specify, try to auto resume from the latest checkpoint "
        "in the work directory.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    # replace the ${key} with the value of cfg.key
    # cfg = replace_cfg_vals(cfg)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(args.config))[0]
        )

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == "AmpOptimWrapper":
            print_log(
                "AMP training is already enabled in your config.",
                logger="current",
                level=logging.WARNING,
            )
        else:
            assert optim_wrapper == "OptimWrapper", (
                "`--amp` is only supported when the optimizer wrapper type is "
                f"`OptimWrapper` but got {optim_wrapper}."
            )
            cfg.optim_wrapper.type = "AmpOptimWrapper"
            cfg.optim_wrapper.loss_scale = "dynamic"

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == "auto":
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # Determine whether the custom metainfo fields are all lowercase
    is_metainfo_lower(cfg)

    # build the runner from config
    if "runner_type" not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    dataloader = runner.train_dataloader
    pbar = tqdm(total=len(dataloader))
    for i, data in enumerate(dataloader):
        pbar.update()
        # visual_data_samples(data, f"temp/test_{i}")
        if i > 100:
            break


def visual_data_samples(data, save_path_prefix="temp/xx"):
    inputs = data["inputs"]
    lbs = data["data_samples"]["bboxes_labels"].cpu().numpy()
    for j in range(len(inputs)):
        img = inputs[j].cpu().numpy().transpose(1, 2, 0)
        img = img.astype(np.uint8).copy()
        label_bboxes = lbs[lbs[:, 0] == j]
        labels = label_bboxes[:, 1]
        bboxes = label_bboxes[:, 2:]
        for bidx, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                str(labels[bidx]),
                (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
        cv2.imwrite(f"{save_path_prefix}_{j}.jpg", img)


if __name__ == "__main__":
    main()
