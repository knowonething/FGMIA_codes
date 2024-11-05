import os.path
from dataclasses import dataclass

import PIL.Image
import onnxruntime
import torch
from tqdm import tqdm

from src.share.mydatasets.dataset_transfroms import get_transfroms


@dataclass
class PDMConfig001:
    data_name = "celeba"
    data_rootPath = "~/dataset/celeba_prepared/"
    data_fileName = "all.txt"
    data_splitTrain = 0.8
    data_size = 112
    data_channels = 3
    data_crop_size = 108
    data_crop_offset = 0
    data_h_flip_p = 0.0
    data_crop = True

    onnx_path = "~/models/FaceModel/IR_R100_Glint360K.onnx"
    onnx_output_path = "~/dataset/celeba_prepared/data_IR_R100_Glint360K"
    target_output_shape = (512,)


if __name__ == "__main__":
    config = PDMConfig001()
    print(config)

    transform = get_transfroms(config)
    targetModel = onnxruntime.InferenceSession(os.path.expanduser(config.onnx_path))
    data_test_name = targetModel.get_inputs()[0].name

    with open(os.path.join(os.path.expanduser(config.data_rootPath),config.data_fileName), "r") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            one_line = line.strip().split(" ")
            img_path = one_line[0]
            class_id = int(one_line[1])

            data_path = os.path.join(os.path.expanduser(config.data_rootPath), "data", img_path)
            img = PIL.Image.open(data_path)
            img_trans = transform(img)
            target_path = os.path.join(os.path.expanduser(config.onnx_output_path), img_path.replace(".png", ".pth"))
            if not os.path.exists(os.path.dirname(target_path)):
                os.makedirs(os.path.dirname(target_path))

            data_raw = {data_test_name: img_trans.cpu().unsqueeze(0).numpy()}
            data_output = targetModel.run(None, data_raw)[0]
            target_output = torch.reshape(torch.from_numpy(data_output), config.target_output_shape)
            torch.save(target_output, target_path)