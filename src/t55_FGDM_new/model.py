import os.path

from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler, PNDMScheduler, DDPMScheduler, \
    LMSDiscreteScheduler

from src.utils.models.arcface import InceptionResNetV2
from src.utils.models.facenet import Facenet
from src.utils.models.sphereface import Sphereface


def get_ldm_model(config):
    if config.model_name == "UNet2DConditionModel":
        return UNet2DConditionModel(
            sample_size=config.data_size,
            in_channels=config.data_channels,
            out_channels=config.data_channels,
            down_block_types=config.model_ldm_downBlockTypes,
            up_block_types=config.model_ldm_upBlockTypes,
            block_out_channels=config.model_ldm_blockOutChannels,
            layers_per_block=config.model_ldm_layersPerBlock,
            cross_attention_dim=config.model_ldm_crossAttentionDim,
        )
    else:
        raise NotImplementedError(f"model_name: {config.model_name} is not implemented")


def get_target_model(config):
    if config.target_name == "arcface":
        arcface = InceptionResNetV2.from_pretrained(config.target_path)
        return arcface
    elif config.target_name == "cosface":
        arcface = InceptionResNetV2.from_pretrained(config.target_path)
        return arcface
    elif config.target_name == "facenet":
        facenet = Facenet.from_pretrained(config.target_path)
        return facenet
    elif config.target_name == "sphereface":
        sphereface = Sphereface.from_pretrained(config.target_path)
        return sphereface
    else:
        raise NotImplementedError(f"target_name: {config.target_name} is not implemented")


def get_vae_model(config):
    if config.model_vae_modelName == "AutoencoderKL":
        return AutoencoderKL.from_pretrained(os.path.expanduser(config.model_vae_path))
    else:
        raise NotImplementedError(f"model_name: {config.vae_modelName} is not implemented")


def get_pipeline_scheduler(config):
    if config.model_pipeline_scheduler == "DDIMScheduler":
        return DDIMScheduler(num_train_timesteps=config.model_ldm_timesteps)
    elif config.model_pipeline_scheduler == "PNDMScheduler":
        return PNDMScheduler(num_train_timesteps=config.model_ldm_timesteps)
    elif config.model_pipeline_scheduler == "LMSDiscreteScheduler":
        return LMSDiscreteScheduler(num_train_timesteps=config.model_ldm_timesteps)
    elif config.model_pipeline_scheduler == "DDPMScheduler":
        return DDPMScheduler(num_train_timesteps=config.model_ldm_timesteps)
    else:
        raise NotImplementedError(f"model_name: {config.model_pipeline_scheduler} is not implemented")
