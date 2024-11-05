import PIL
import torch
from PIL import Image
from diffusers import ModelMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config
from torchvision import transforms

from src.utils.models.utils.InceptionResnet import IResNet, IBasicBlock
from src.utils.models.utils.facenet_pytorch import InceptionResnetV1
from src.utils.models.utils.mtcnn_pytorch import MTCNN_model


class MTCNN(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self,
                 **kwargs):
        super().__init__()
        self.model = MTCNN_model()

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    # mtcnn = MTCNN()
    # mtcnn.model.pnet.load_state_dict(
    #     torch.load(
    #         "/home/miao/Documents/expers/my_expers/e1/github/facenet-pytorch/data/pnet.pt",
    #         map_location="cpu"),
    #     strict=False,
    # )
    # mtcnn.model.rnet.load_state_dict(
    #     torch.load(
    #         "/home/miao/Documents/expers/my_expers/e1/github/facenet-pytorch/data/rnet.pt",
    #         map_location="cpu"),
    #     strict=False,
    # )
    # mtcnn.model.onet.load_state_dict(
    #     torch.load(
    #         "/home/miao/Documents/expers/my_expers/e1/github/facenet-pytorch/data/onet.pt",
    #         map_location="cpu"),
    #     strict=False,
    # )
    # mtcnn.save_pretrained("/home/miao/Documents/expers/my_expers/e1/exper02/outputs/targets/mtcnn_bin")

    mtcnn = MTCNN.from_pretrained("/home/miao/Documents/expers/my_expers/e1/exper02/outputs/targets/mtcnn_bin")
    img = Image.open("/home/miao/Documents/expers/my_expers/e1/github/facenet-pytorch/data/test_images/angelina_jolie/1.jpg")
    boxes = mtcnn(img)
    print(boxes.shape)

    postProcess = transforms.Compose([
        transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),  # 反归一化
        transforms.Lambda(lambda x: x.clamp(0, 1)),
        transforms.ToPILImage(),
        transforms.Resize((112,112)),
    ])
    imgaaa = postProcess(boxes)
    imgaaa.show()
