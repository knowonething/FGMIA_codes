import torch
from diffusers import ModelMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config

from src.utils.models.utils.InceptionResnet import IResNet, IBasicBlock
from src.utils.models.utils.sphereface import sphere20a


class Sphereface(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(self,
                 **kwargs):
        super().__init__()
        self.model = sphere20a(feature=True)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    target = Sphereface()
    target.model.load_state_dict(
        torch.load("/home/miao/Documents/expers/my_expers/e1/exper02/outputs/targets/sphereface/sphere20a_20171020.pth",
                   map_location="cpu"),
        strict=False,
    )
    target.save_pretrained("/home/miao/Documents/expers/my_expers/e1/exper02/outputs/targets/sphereface_bin")