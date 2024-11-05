import argparse
import inspect
import os

import PIL.Image
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from loguru import logger as log
from torchmetrics.image import FrechetInceptionDistance
from torchvision import transforms
from tqdm import tqdm

from src.share.mydatasets.dataset_transfroms import CenterCropWithRec
from src.share.utils.inits.init_dirs import initDirs
from src.share.utils.inits.init_logs import initLogs
from src.share.utils.inits.init_seeds import initSeeds
from src.share.utils.logConfig import logConfig
from src.share.utils.metrics.KNNDistance import KNNCosDistance
from src.share.utils.metrics.lpips import LearnedPerceptualImagePatchSimilarityMin, \
    LearnedPerceptualImagePatchSimilarityAvg
from src.t55_FGDM_new import AttackConfig
from src.t55_FGDM_new.model import get_target_model
from src.t55_FGDM_new.pipeline_PDM import PDMTargetToImagePipeline


def main(config_name):
    print(config_name)
    confClass = None
    members = inspect.getmembers(AttackConfig, inspect.isclass)
    for name, member in members:
        if name == config_name:
            confClass = member
    if confClass is None:
        log.error("Error of conf")
        exit()
    config = confClass()

    initDirs(config)
    initLogs(config)
    initSeeds(config)
    sample_dir = os.path.join(config.init_outputDir, "samples")
    logs_dir = os.path.join(config.init_outputDir, "logs")
    tensorboard_dir = os.path.join(config.init_outputDir, "tensorboard")
    log.info("Config: \n{}", logConfig(confClass))

    if config.model_DMIA_name == "PDM":
        pipeline = PDMTargetToImagePipeline.from_pretrained(config.model_DMIA)
    else:
        raise ValueError("Unknown target model name")
    model = pipeline.unet
    noise_scheduler = pipeline.scheduler

    targetmodel = get_target_model(config)
    feature_path = os.path.join(config.target_path, "features.pth")
    features = torch.load(feature_path, map_location="cpu")

    accelerate = Accelerator()

    pipeline = pipeline.to(accelerate.device)

    targetmodel = targetmodel.to(accelerate.device)
    features = features.to(accelerate.device)

    log.info("target_model_features:{}", features.shape)

    # generate and attack

    if config.MIA_gen:
        log.warning("************************Begin MIA_GEN*********************")
        for target_label, target_features in tqdm(enumerate(features)):
            output_dir = os.path.join(sample_dir, f"{target_label}")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with torch.no_grad():
                target_features = torch.reshape(target_features, (1, -1, config.model_DMIA_crossAttentionDim))
                target_hidden_stats = target_features.to(accelerate.device).repeat(config.MIA_batch, 1, 1)

                images = pipeline(
                    batch_size=config.MIA_batch,
                    num_inference_steps=config.MIA_steps,
                    target_hidden_stats=target_hidden_stats,
                    guidance_scale=1.0,
                    generator=None,
                    output_type="pil",
                ).images

                for i, image in enumerate(images):
                    image.save(os.path.join(output_dir, f"{i}.png"))
        log.info("************************End MIA_GEN*********************")
    # end generate and attack
    pre_process_crop = transforms.Compose([
        transforms.ToTensor(),
        CenterCropWithRec(config.data_crop_size, config.data_crop_offset),
        transforms.ToPILImage(),
        transforms.Resize(config.data_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    pre_process = transforms.Compose([
        transforms.Resize(config.data_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    postProcess = transforms.Compose([
        transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),
        transforms.Lambda(lambda x: x.clamp(0, 255)),
        transforms.ToPILImage(),
        transforms.Resize(config.data_crop_size)
    ])

    if config.MIA_make_raw_imgs:
        log.info("Begin Make raw images")

        with open(os.path.expanduser(config.MIA_ACC_src), "r") as f, torch.no_grad():
            output_raw_dir = os.path.join(sample_dir, "raw")
            if not os.path.exists(output_raw_dir):
                os.makedirs(output_raw_dir)
            lines = f.readlines()
            base = os.path.join(os.path.expanduser(config.MIA_ACC_src_basedir), "data")
            for line in tqdm(lines):
                src_path, src_label = line.strip().split(" ")
                src_path = os.path.join(base, src_path)
                src_label = int(src_label)
                src_img = pre_process_crop(PIL.Image.open(src_path)).to(accelerate.device)
                img_output = postProcess(src_img.cpu())
                img_output.save(os.path.join(output_raw_dir, f"{src_label}.png"))
        log.info("End Make raw images")

    if config.MIA_cal_acc:
        log.info("Begin cal acc")
        cos_accept_sum = 0
        cos_accept_one = 0
        topk = (1, 3, 5)
        topk_res = torch.zeros((1, len(topk))).to(accelerate.device)

        with open(os.path.expanduser(config.MIA_ACC_src), "r") as f, torch.no_grad():
            lines = f.readlines()
            for line in tqdm(lines):
                src_path, src_label = line.strip().split(" ")
                src_label = int(src_label)
                dst_path = os.path.join(sample_dir, f'{src_label}')
                dst_imgs_path = os.listdir(dst_path)
                dst_imgs_imgs = [PIL.Image.open(os.path.join(dst_path, i)) for i in dst_imgs_path]
                dst_imgs = torch.stack([pre_process(i) for i in dst_imgs_imgs]).to(accelerate.device)
                dst_features = F.normalize(targetmodel(dst_imgs))
                logits = torch.einsum("ik,jk->ij", dst_features, features)
                dis, pred = torch.topk(logits, k=max(topk), dim=-1, largest=True, sorted=True)
                pred = pred.t()
                dis = dis.t()
                dis_correct = dis > config.target_accept

                target_labels = torch.Tensor([src_label]).to(accelerate.device).expand_as(pred)
                correct = pred.eq(target_labels)
                correct = correct & dis_correct
                res = []
                for k in topk:
                    correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                    res.append(correct_k / len(dst_imgs))
                topk_res += torch.stack(res).t()
            topk_res /= len(lines)
        log.info("topk_res:{}", topk_res)

    if config.MIA_cal_knn:
        log.info("Begin cal knn")
        knnCosDistance = KNNCosDistance()

        with open(os.path.expanduser(config.MIA_ACC_src), "r") as f, torch.no_grad():
            lines = f.readlines()
            base = os.path.join(os.path.expanduser(config.MIA_ACC_src_basedir), "data")
            for line in tqdm(lines):
                src_path, src_label = line.strip().split(" ")
                src_path = os.path.join(base, src_path)
                src_label = int(src_label)
                dst_path = os.path.join(sample_dir, f'{src_label}')
                dst_imgs_path = os.listdir(dst_path)
                dst_imgs_imgs = [PIL.Image.open(os.path.join(dst_path, i)) for i in dst_imgs_path]
                dst_imgs = torch.stack([pre_process(i) for i in dst_imgs_imgs]).to(accelerate.device)
                src_img = pre_process_crop(PIL.Image.open(src_path)).to(accelerate.device)

                src_feature = F.normalize(targetmodel(src_img.unsqueeze(dim=0)))
                dst_features = F.normalize(targetmodel(dst_imgs))
                d_cos = torch.einsum("ik,jk->ij", src_feature, dst_features).view((-1,))
                knnCosDistance.update(d_cos)
            knn = knnCosDistance.compute()
            log.info("KNN CosDistance:{}", knn)

    if config.MIA_cal_FID:
        FID = FrechetInceptionDistance(feature=2048, normalize=True).to(accelerate.device)

        with open(os.path.expanduser(config.MIA_ACC_src), "r") as f, torch.no_grad():
            lines = f.readlines()
            base = os.path.join(os.path.expanduser(config.MIA_ACC_src_basedir), "data")
            for line in tqdm(lines):
                src_path, src_label = line.strip().split(" ")
                src_path = os.path.join(base, src_path)
                src_label = int(src_label)
                dst_path = os.path.join(sample_dir, f'{src_label}')
                dst_imgs_path = os.listdir(dst_path)
                if hasattr(config, "MIA_cal_batch"):
                    dst_imgs_path = dst_imgs_path[:config.MIA_cal_batch]
                dst_imgs_imgs = [PIL.Image.open(os.path.join(dst_path, i)) for i in dst_imgs_path]
                dst_imgs = torch.stack([pre_process(i) for i in dst_imgs_imgs]).to(accelerate.device)
                src_imgs_path = os.listdir(os.path.dirname(src_path))
                src_imgs_imgs = [PIL.Image.open(os.path.join(os.path.dirname(src_path), i)) for i in src_imgs_path]
                src_imgs = torch.stack([pre_process_crop(i) for i in src_imgs_imgs]).to(accelerate.device)

                FID.update(src_imgs, real=True)
                FID.update(dst_imgs, real=False)
            fid = FID.compute()
            log.info("FID:{}", fid)

    if config.MIA_cal_lpips:
        log.info("Begin cal lpips min")
        lpips = LearnedPerceptualImagePatchSimilarityMin(net_type='vgg').to(accelerate.device)

        with open(os.path.expanduser(config.MIA_ACC_src), "r") as f, torch.no_grad():
            lines = f.readlines()
            base = os.path.join(os.path.expanduser(config.MIA_ACC_src_basedir), "data")
            for line in tqdm(lines):
                src_path, src_label = line.strip().split(" ")
                src_path = os.path.join(base, src_path)
                src_label = int(src_label)
                dst_path = os.path.join(sample_dir, f'{src_label}')
                dst_imgs_path = os.listdir(dst_path)
                dst_imgs_imgs = [PIL.Image.open(os.path.join(dst_path, i)) for i in dst_imgs_path]
                dst_imgs = torch.stack([pre_process(i) for i in dst_imgs_imgs]).to(accelerate.device)
                src_img = pre_process_crop(PIL.Image.open(src_path)).unsqueeze(dim=0).to(accelerate.device)
                lpips.update(src_img, dst_imgs)
            lpipsss = lpips.compute()
            log.info("LPIPS:{}", lpipsss)

    if config.MIA_cal_lpips:
        log.info("Begin cal lpips Avg")
        lpips = LearnedPerceptualImagePatchSimilarityAvg(net_type='vgg').to(accelerate.device)

        with open(os.path.expanduser(config.MIA_ACC_src), "r") as f, torch.no_grad():
            lines = f.readlines()
            base = os.path.join(os.path.expanduser(config.MIA_ACC_src_basedir), "data")
            for line in tqdm(lines):
                src_path, src_label = line.strip().split(" ")
                src_path = os.path.join(base, src_path)
                src_label = int(src_label)
                dst_path = os.path.join(sample_dir, f'{src_label}')
                dst_imgs_path = os.listdir(dst_path)
                dst_imgs_imgs = [PIL.Image.open(os.path.join(dst_path, i)) for i in dst_imgs_path]
                dst_imgs = torch.stack([pre_process(i) for i in dst_imgs_imgs]).to(accelerate.device)
                src_img = pre_process_crop(PIL.Image.open(src_path)).unsqueeze(dim=0).to(accelerate.device)
                lpips.update(src_img, dst_imgs)
            lpipsss = lpips.compute()
            log.info("LPIPS-AVG:{}", lpipsss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='conf')
    args = parser.parse_args()

    main("AttackConfig" + args.config)
