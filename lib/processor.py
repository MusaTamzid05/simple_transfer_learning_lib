from lib.image_augmentar import ImageAugmentar
import os

class Processor:
    def __init__(self, src_dir):
        self.src_dir = src_dir

    def run(self, save_dir_path,  num = 10):
        image_names = os.listdir(self.src_dir)

        for image_name in image_names:
            image_path = os.path.join(self.src_dir, image_name)
            img_aug = ImageAugmentar(image_path = image_path)
            img_aug.generate(save_dir_path = save_dir_path, num = num)







