from typing import Union, List, Optional, Tuple

import torch
from diffusers import DiffusionPipeline, UNet2DConditionModel, UNet2DModel, PNDMScheduler, \
    LMSDiscreteScheduler, DDIMScheduler, ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor
from torchvision import transforms


class PDMTargetToImagePipeline(DiffusionPipeline):
    def __init__(
            self,
            unet: Union[UNet2DModel, UNet2DConditionModel],
            scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
    ):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
            self,
            batch_size: int = 1,
            num_inference_steps: Optional[int] = 50,
            target_hidden_stats: torch.Tensor = None,
            guidance_scale: Optional[float] = 1.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            **kwargs,
    ) -> Union[Tuple, ImagePipelineOutput]:

        latent_noise = randn_tensor(
            (
            batch_size, self.unet.config.in_channels, self.unet.config.sample_size[0], self.unet.config.sample_size[1]),
            generator=generator,
            device=self.unet.device,
        )

        self.scheduler.set_timesteps(num_inference_steps)

        latents_denoise = latent_noise
        for t in self.progress_bar(self.scheduler.timesteps):
            noise_pred = self.unet(latents_denoise, t, encoder_hidden_states=target_hidden_stats).sample
            latents_denoise = self.scheduler.step(noise_pred, t, latents_denoise).prev_sample

        image = latents_denoise
        if self.unet.config.in_channels == 3:
            postProcess = transforms.Compose([
                transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),  # 反归一化
                transforms.Lambda(lambda x: x.clamp(0, 255)),
                transforms.ToPILImage(),
            ])
        elif self.unet.config.in_channels == 1:
            postProcess = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Normalize(mean=[-1], std=[2]),
                transforms.Lambda(lambda x: x.clamp(0, 255)),
                transforms.ToPILImage()
            ])
        else:
            raise ValueError("unet.config.in_channels should be 1 or 3")

        images = [postProcess(img) for img in image]

        if not return_dict:
            return (images,)

        return ImagePipelineOutput(images=images)


if __name__ == "__main__":
    model = PDMTargetToImagePipeline.from_pretrained(
        "/home/miao/Documents/expers/my_expers/e1/exper02/outputs/05_PDM/PDMConfig601")
    print("Hello World")
