# FGMIA_codes

This project presents a Feature-Guided Model Inversion Attacks (FGMIA) method to reconstruct facial images of specific individuals from face recognition models. Our project proposes a novel solution that leverages the latent feature representations of the target model to guide the facial image generation, significantly improving the reconstruction quality. The code of this project is implemented in **PyTorch**, covering the full pipeline of training a conditional diffusion model and executing the model inversion attacks. Additionally, we have released the trained FGMIA model on the HuggingFace platform, which can be accessed [here](https://huggingface.co/MMCT/FGMIA_Models). We hope this open-source project can provide a valuable reference for related research.

## Requirements
environment.yml
```sh
conda env create -f environment.yml
```

## Attack
```sh
$ CUDA_VISIBLE_DEVICES="0" accelerate launch --config_file ./onegpu.conf src/t55_FGDM_new/attack.py --config 006  
```
