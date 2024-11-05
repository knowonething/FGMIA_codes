import os.path
import random

from loguru import logger as log
from tqdm import tqdm

if __name__ == "__main__":
    src = os.path.join(os.path.expanduser("~/"), "dataset/VGG-Face2")

    src_test = os.path.join(src, "data/test/")
    src_train = os.path.join(src, "data/train/")

    target_dir = os.path.join(os.path.expanduser("~/"), "dataset/vggface2")
    target_data_dir = os.path.join(target_dir, "data/")

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        os.makedirs(os.path.join(target_dir, "data"))

    # # 合并目录 1
    # for filename in tqdm(os.listdir(src_test)):
    #     src_file = os.path.join(src_test, filename)
    #     dst_file = os.path.join(target_data_dir, filename)
    #     if os.path.isfile(src_file):
    #         shutil.copy(src_file, dst_file)
    #     else:
    #         shutil.copytree(src_file, dst_file)
    #
    # # 合并目录 2
    # for filename in tqdm(os.listdir(src_train)):
    #     src_file = os.path.join(src_train, filename)
    #     dst_file = os.path.join(target_data_dir, filename)
    #     if os.path.isfile(src_file):
    #         shutil.copy(src_file, dst_file)
    #     else:
    #         shutil.copytree(src_file, dst_file)

    # 将所有图片存入内存中
    all_images = []
    for dirname in os.listdir(target_data_dir):
        if os.path.isfile(dirname):
            log.error("error in " + dirname)
            continue
        all_images.append({dirname: os.listdir(os.path.join(target_data_dir, dirname))})

    with open(os.path.join(target_dir, "all.txt"), "a") as f:
        for i, one_dir in enumerate(all_images):
            dirname = list(one_dir.keys())[0]
            for filename in list(one_dir.values())[0]:
                s = "%s %s\n" % (os.path.join(dirname, filename), i)
                f.write(s)

    # 生成不重复的随机整数集合
    random_ints = list(range(0, 9131))
    random.shuffle(random_ints)
    random_ints = random_ints[:8000]

    # 将随机数最为索引，将文件与标签写入文件
    with open(os.path.join(target_dir, "mini1000_1.txt"), "a") as f:
        for i in tqdm(range(0, 1000)):
            dirname = list(all_images[random_ints[i]].keys())[0]
            for filename in list(all_images[random_ints[i]].values())[0]:
                s = "%s %s\n" % (os.path.join(dirname, filename), i)
                f.write(s)

    with open(os.path.join(target_dir, "mini1000_2.txt"), "a") as f:
        for i in tqdm(range(1000, 2000)):
            dirname = list(all_images[random_ints[i]].keys())[0]
            for filename in list(all_images[random_ints[i]].values())[0]:
                s = "%s %s\n" % (os.path.join(dirname, filename), i - 1000)
                f.write(s)

    with open(os.path.join(target_dir, "mini2000.txt"), "a") as f:
        for i in tqdm(range(2000, 4000)):
            dirname = list(all_images[random_ints[i]].keys())[0]
            for filename in list(all_images[random_ints[i]].values())[0]:
                s = "%s %s\n" % (os.path.join(dirname, filename), i - 2000)
                f.write(s)

    with open(os.path.join(target_dir, "mini4000.txt"), "a") as f:
        for i in tqdm(range(4000, 8000)):
            dirname = list(all_images[random_ints[i]].keys())[0]
            for filename in list(all_images[random_ints[i]].values())[0]:
                s = "%s %s\n" % (os.path.join(dirname, filename), i - 4000)
                f.write(s)
