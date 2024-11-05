import torch
from diffusers import ModelMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config

from src.utils.models.utils.InceptionResnet import IResNet, IBasicBlock
from src.utils.models.utils.facenet_pytorch import InceptionResnetV1


class Facenet(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(self,
                 **kwargs):
        super().__init__()

        self.model = InceptionResnetV1()

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    facenet = Facenet()
    facenet.model.load_state_dict(
        torch.load(
            "/home/miao/Documents/expers/my_expers/e1/exper02/outputs/targets/facenet/20180402-114759-vggface2.pt",
            map_location="cpu"),
        strict=False,
    )
    facenet.save_pretrained("/home/miao/Documents/expers/my_expers/e1/exper02/outputs/targets/facenet_bin")
    