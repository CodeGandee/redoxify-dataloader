# functions used in dali python_function
import torch
import random
import numpy as np

import pickle
import uuid

from nvidia.dali.plugin.pytorch.fn import torch_python_function
from nvidia.dali.fn import python_function
from nvidia.dali.pipeline import DataNode as DALINode
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import numpy as np


def _gen_uuid_gpu(torch_data: list[torch.Tensor]):
    uid = torch.tensor(
        np.frombuffer(uuid.uuid4().bytes, dtype=np.uint8), device=torch_data[0].device
    )
    return [uid] * len(torch_data)


def _gen_uuid_cpu(numpy_data: list[np.ndarray]):
    uid = np.frombuffer(uuid.uuid4().bytes, dtype=np.uint8)
    return [uid] * len(numpy_data)


def _print_batch_data_gpu(torch_data: list[torch.Tensor], msg: list[torch.Tensor]):
    uuid = _gen_uuid_gpu(torch_data)
    usr_msg = msg[0].cpu().numpy().tobytes().decode()
    for i, d in enumerate(torch_data):
        print(f"[gpu]  {usr_msg}  [{uuid[i]}] index: {i}, value: {d}")


def _print_batch_data_cpu(numpy_data: list[np.ndarray], msg: list[np.ndarray]):
    uuid = _gen_uuid_cpu(numpy_data)
    usr_msg = msg[0].tobytes().decode()
    for i, d in enumerate(numpy_data):
        print(f"[cpu]  {usr_msg}  [{uuid[i]}] index: {i}, value: {d}")


def print_batch_data_info_gpu(data: DALINode, msg: DALINode):
    func = torch_python_function(
        data.gpu(),
        msg.gpu(),
        function=_print_batch_data_gpu,
        batch_processing=True,
        num_outputs=0,
        device="gpu",
    )
    return func


def print_batch_data_info_cpu(data: DALINode, msg: DALINode):
    func = python_function(
        data,
        msg,
        function=_print_batch_data_cpu,
        batch_processing=True,
        num_outputs=0,
        device="cpu",
    )
    return func


def _print_data_gpu(torch_data: torch.Tensor, msg: torch.Tensor):
    uuidlist = _gen_uuid_gpu([torch_data])
    print_uid = uuid.UUID(bytes=uuidlist[0].cpu().numpy().tobytes())
    usr_msg = msg.cpu().numpy().tobytes().decode()
    print(f"[gpu]  {usr_msg}  [{print_uid}] value: {torch_data}")


def _print_data_cpu(numpy_data: np.ndarray, msg: np.ndarray):
    uuidlist = _gen_uuid_cpu([numpy_data])
    print_uid = uuid.UUID(bytes=uuidlist[0].tobytes())
    usr_msg = msg.tobytes().decode()
    print(f"[cpu]  {usr_msg}  [{print_uid}] value: {numpy_data}")


def print_data_info_gpu(data: DALINode, msg: DALINode):
    func = torch_python_function(
        data.gpu(),
        msg.gpu(),
        function=_print_data_gpu,
        batch_processing=False,
        num_outputs=0,
        device="gpu",
    )
    return func


def print_data_info_cpu(data: DALINode, msg: DALINode):
    func = python_function(
        data,
        msg,
        function=_print_data_cpu,
        batch_processing=False,
        num_outputs=0,
        device="cpu",
    )
    return func


if __name__ == "__main__":
    from nvidia.dali import pipeline_def
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    import cv2
    import numpy as np
    from numpy import random
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    image_filename = "/workspace/redoxify-dataloader/tests/fuji.jpg"
    @pipeline_def()
    def hsv_pipeline(device, hue, saturation, value):
        jpeg_data = fn.readers.file(files=[image_filename])[0]
        images = fn.decoders.image(
            jpeg_data, device="cpu" if device == "cpu" else "mixed"
        )
        converted = fn.hsv(images, hue=hue, saturation=saturation, value=value)
        str_to_tensor = lambda x: torch.tensor(
            np.frombuffer(x.encode(), dtype=np.uint8)
        )
        fake_value = types.Constant(
            np.zeros((0,), dtype=np.int64), dtype=types.DALIDataType.INT64
        )
        # msg = types.Constant(str_to_tensor("pring the shape of images"), device="gpu")
        msg = types.Constant(
            np.frombuffer("pring the shape of images".encode(), dtype=np.uint8),
            device="gpu",
        )

        print_data_info_gpu(fake_value, msg)
        return images, converted

    pipe_cpu = hsv_pipeline(
        device="gpu",
        hue=0,
        saturation=1.9,
        value=1.0,
        batch_size=1,
        num_threads=1,
        device_id=0,
    )
    pipe_cpu.build()
    dali_iter = DALIGenericIterator(pipe_cpu, ["img1", "hsv"])
    for i, data in enumerate(dali_iter):
        img_hsv = data[0]["hsv"]
        break
