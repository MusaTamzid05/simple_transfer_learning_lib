from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from PIL import Image

class ImageAugmentar:
    def __init__(self, image_path):
        self.image = load_img(image_path)

    def generate(self, save_dir_path, num = 10):
        if os.path.isdir(save_dir_path) == False:
            os.mkdir(save_dir_path)

        datagen = ImageDataGenerator(rotation_range = 90, brightness_range = [0.2, 1.0])
        samples = np.expand_dims(self.image, 0)


        itr = datagen.flow(samples, batch_size = 1)

        for _ in range(num):
            batch = itr.next()
            image = batch[0].astype(np.uint8)
            image = Image.fromarray(image)

            self.save(image = image, save_dir_path = save_dir_path)




    def save(self, image,  save_dir_path):
        i = len(os.listdir(save_dir_path))
        save_path = os.path.join(save_dir_path, f"{i}.jpg")
        image.save(save_path)
        print(f"{save_path} saved")






