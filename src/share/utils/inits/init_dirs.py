import os


def initDirs(config):
    if not os.path.exists(config.init_outputDir):
        os.makedirs(config.init_outputDir)
    sample_dir = os.path.join(config.init_outputDir, "samples")
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    logs_dir = os.path.join(config.init_outputDir, "logs")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    tensorboard_dir = os.path.join(config.init_outputDir, "tensorboard")
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
