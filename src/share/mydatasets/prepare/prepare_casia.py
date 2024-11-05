import os.path
import random

from loguru import logger as log

if __name__ == "__main__":
    base = os.path.join(os.path.expanduser("~/"), "dataset/casia_webface/")

    all_images_class = []
    for dirname in os.listdir(os.path.join(base, "CASIA-WebFace")):
        if os.path.isfile(dirname):
            log.error("error in " + dirname)
            continue
        all_images_class.append(dirname)

    #     打乱顺序
    random.shuffle(all_images_class)

    #     写入前1000个
    with open(os.path.join(base, "mini1000_1.txt"), "a") as f:
        for i in range(0, 1000):
            label = all_images_class[i]
            for filename in os.listdir(os.path.join(base, "CASIA-WebFace", label)):
                s = "%s %s\n" % (os.path.join("CASIA-WebFace", label, filename), i)
                f.write(s)

    with open(os.path.join(base, "mini1000_2.txt"), "a") as f:
        for i in range(1000, 2000):
            label = all_images_class[i]
            for filename in os.listdir(os.path.join(base, "CASIA-WebFace", label)):
                s = "%s %s\n" % (os.path.join("CASIA-WebFace", label, filename), i - 1000)
                f.write(s)
    with open(os.path.join(base, "mini2000.txt"), "a") as f:
        for i in range(2000, 4000):
            label = all_images_class[i]
            for filename in os.listdir(os.path.join(base, "CASIA-WebFace", label)):
                s = "%s %s\n" % (os.path.join("CASIA-WebFace", label, filename), i - 2000)
                f.write(s)
    with open(os.path.join(base, "mini4000.txt"), "a") as f:
        for i in range(4000, 8000):
            label = all_images_class[i]
            for filename in os.listdir(os.path.join(base, "CASIA-WebFace", label)):
                s = "%s %s\n" % (os.path.join("CASIA-WebFace", label, filename), i - 4000)
                f.write(s)

    with open(os.path.join(base, "all.txt"), "a") as f:
        for i in range(len(all_images_class)):
            label = all_images_class[i]
            for filename in os.listdir(os.path.join(base, "CASIA-WebFace", label)):
                s = "%s %s\n" % (os.path.join("CASIA-WebFace", label, filename), i)
                f.write(s)
