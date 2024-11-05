from diffusers import ModelMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config

from src.utils.models.utils.InceptionResnet import IResNet, IBasicBlock


class InceptionResNetV2(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(self,
                 layers,
                 dropout=0,
                 num_features=512,
                 width_per_group=64,
                 **kwargs):
        super().__init__()
        self.layers = layers
        self.dropout = dropout
        self.num_features = num_features
        self.width_per_group = width_per_group

        self.model = IResNet(IBasicBlock,
                             layers,
                             dropout=dropout,
                             num_features=num_features,
                             width_per_group=width_per_group,
                             **kwargs)
    def forward(self, x):
        return self.model(x)



if __name__ == "__main__":
    ir = InceptionResNetV2.from_pretrained("~/models/arcface/glint360k_cosface_r18_fp16_0.1_bin")
