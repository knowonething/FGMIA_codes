# FGMIA_codes

This repository contains the source code for the paper: pass

code of FGMIA

datasets: ./dataset

training outputs: ./output

source code: ./src

target models and trained models: https://huggingface.co/MMCT/FGMIA_Models

## Requirements
environment.yml
```sh
conda env create -f environment.yml
```

## Attack
```sh
$ CUDA_VISIBLE_DEVICES="0" accelerate launch --config_file ./onegpu.conf src/t55_FGDM_new/attack.py --config 003  
```