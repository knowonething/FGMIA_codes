from dataclasses import dataclass

@dataclass
class AttackConfig003:
    init_outputDir: str = "outputs/FGMIA_Models/FGMIAs/Sphereface-FGMIA"

    target_name = "sphereface"
    target_path = "outputs/targets/sphereface_bin"

    model_DMIA_name = "PDM"
    model_DMIA = "outputs/FGMIA_Models/FGDMs/Sphereface-FG"
    model_DMIA_crossAttentionDim = 512
    MIA_steps = 50
    MIA_batch = 32
    MIA_cal_batch = 16
    model_target_num = 1000

    data_size = (112, 96)
    data_channels = 3
    data_crop_size = [130, 130]
    data_crop_offset = 0

    debug = False
    debug_time = 1

    target_accept = 0.20

    MIA_gen = True
    MIA_make_raw_imgs = False
    MIA_cal_acc = True
    MIA_cal_knn = True
    MIA_cal_lpips = True
    MIA_cal_FID = True

    MIA_ACC_src = "dataset/celeba_prepared/mini1000_2_register.txt"
    MIA_ACC_src_basedir = "dataset/celeba_prepared"

@dataclass
class AttackConfig004:
    init_outputDir: str = "outputs/FGMIA_Models/FGMIAs/Facenet-FGMIA"

    target_name = "facenet"
    target_path = "outputs/targets/facenet_bin"

    model_DMIA_name = "PDM"
    model_DMIA = "outputs/FGMIA_Models/FGDMs/Facenet-FG"
    model_DMIA_crossAttentionDim = 512
    MIA_steps = 50
    MIA_batch = 32
    model_target_num = 1000

    data_size = (112, 112)
    data_channels = 3
    data_crop_size = [112, 112]
    data_crop_offset = 0

    debug = False
    debug_time = 1

    target_accept = 0.20

    MIA_gen = True
    MIA_make_raw_imgs = True
    MIA_cal_acc = True
    MIA_cal_knn = True
    MIA_cal_lpips = True
    MIA_cal_FID = True

    MIA_ACC_src = "dataset/celeba_prepared/mini1000_2_register.txt"
    MIA_ACC_src_basedir = "dataset/celeba_prepared"


@dataclass
class AttackConfig005:
    init_outputDir: str = "outputs/FGMIA_Models/FGMIAs/Arcface-FGMIA"

    target_name = "cosface"
    target_path = "outputs/targets/arccosfaces/cosface18"

    model_DMIA_name = "PDM"
    model_DMIA = "outputs/FGMIA_Models/FGDMs/Cosface18-FG"
    model_DMIA_crossAttentionDim = 512
    MIA_steps = 50
    MIA_batch = 32
    model_target_num = 1000

    data_size = (112, 112)
    data_channels = 3
    data_crop_size = [112, 112]
    data_crop_offset = 0

    debug = False
    debug_time = 1

    target_accept = 0.20

    MIA_gen = True
    MIA_make_raw_imgs = False
    MIA_cal_acc = True
    MIA_cal_knn = True
    MIA_cal_lpips = True
    MIA_cal_FID = True

    MIA_ACC_src = "dataset/celeba_prepared/mini1000_2_register.txt"
    MIA_ACC_src_basedir = "dataset/celeba_prepared"


@dataclass
class AttackConfig006:
    init_outputDir: str = "outputs/FGMIA_Models/FGMIAs/Cosface-FGMIA"

    target_name = "arcface"
    target_path = "outputs/targets/arccosfaces/arcface18"

    model_DMIA_name = "PDM"
    model_DMIA = "outputs/FGMIA_Models/FGDMs/Arcface18-FG"
    model_DMIA_crossAttentionDim = 512
    MIA_steps = 50
    MIA_batch = 32
    model_target_num = 1000

    data_size = (112, 112)
    data_channels = 3
    data_crop_size = [112, 112]
    data_crop_offset = 0

    debug = False
    debug_time = 1

    target_accept = 0.20

    MIA_gen = True
    MIA_make_raw_imgs = False
    MIA_cal_acc = True
    MIA_cal_knn = True
    MIA_cal_lpips = True
    MIA_cal_FID = True

    MIA_ACC_src = "dataset/celeba_prepared/mini1000_2_register.txt"
    MIA_ACC_src_basedir = "dataset/celeba_prepared"

# ---------------------------------------- end of facenet/sphereface/cosface/arcface ----------------------------------------
