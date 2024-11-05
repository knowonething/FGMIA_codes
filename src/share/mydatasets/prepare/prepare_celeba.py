import os.path
import shutil

from loguru import logger as log
from tqdm import tqdm

if __name__ == "__main__":
    log.info("Hello World")

    src_root = os.path.expanduser("~/dataset/CelebA/")
    dst_root = os.path.expanduser("~/dataset/celeba_prepared/")

    # Create dst_root
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)
    if not os.path.exists(os.path.join(dst_root, "data")):
        os.makedirs(os.path.join(dst_root, "data"))

    image_lists = []
    for i in range(10178):
        image_lists.append({f"{i}": []})
        if not os.path.exists(os.path.join(dst_root, "data", f"{i}")):
            os.makedirs(os.path.join(dst_root, "data", f"{i}"))

    # 读取数据集列表
    with open(os.path.join(src_root, "Anno/identity_CelebA.txt"), "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]

    for line in lines:
        image_name, image_id = line.split()
        image_id = int(image_id)
        image_name = image_name.replace(".jpg", "")
        image_lists[image_id][f"{image_id}"].append(image_name)

    # 按照image_id的数量排序
    sorted_image_lists = sorted(image_lists, key=lambda x: len(x.copy().popitem()[1]), reverse=True)

    # 复制图片
    for image_list in tqdm(image_lists):
        image_id = list(image_list.keys())[0]
        image_name_list = list(image_list.values())[0]
        for image_name in image_name_list:
            src_image_path = os.path.join(src_root, "Img/img_align_celeba_png", f"{image_name}.png")
            dst_image_path = os.path.join(dst_root, "data", image_id, f"{image_name}.png")
            # log.info("Copy {} to {}", src_image_path, dst_image_path)
            # shutil.copy(src_image_path, dst_image_path)
    log.info("Done Copy")

    # 将image_lists写入文件all.txt
    log.info("Write all images to all.txt")
    with open(os.path.join(dst_root, "all.txt"), "w") as f:
        for image_list in tqdm(image_lists):
            image_id = list(image_list.keys())[0]
            image_name_list = list(image_list.values())[0]
            for image_name in image_name_list:
                line = f"{image_id}/{image_name}.png {image_id}\n"
                f.write(line)
    log.info("Logging to all.txt ends")

    # 将前面2000个,奇数mini1000_1.txt， 偶数mini1000_2.txt
    log.info("Write first 2000 images to mini1000_1.txt and mini1000_2.txt")
    with (open(os.path.join(dst_root, "mini1000_1.txt"), "w") as f1,
          open(os.path.join(dst_root, "mini1000_2.txt"), "w") as f2):
        for i in tqdm(range(2000)):
            image_id = list(sorted_image_lists[i].keys())[0]
            image_name_list = list(sorted_image_lists[i].values())[0]
            for image_name in image_name_list:
                line = f"{image_id}/{image_name}.png {i // 2}\n"
                if i % 2 == 0:
                    f1.write(line)
                else:
                    f2.write(line)
    log.info("Logging to mini1000_1.txt and mini1000_2.txt ends")

    # 将剩下的写入train.txt
    log.info("Write the rest images to rest.txt")
    with open(os.path.join(dst_root, "rest.txt"), "w") as f:
        for i in tqdm(range(2000, len(sorted_image_lists))):
            image_id = list(sorted_image_lists[i].keys())[0]
            image_name_list = list(sorted_image_lists[i].values())[0]
            for image_name in image_name_list:
                line = f"{image_id}/{image_name}.png {i - 2000}\n"
                f.write(line)
    log.info("Logging to rest.txt ends")
