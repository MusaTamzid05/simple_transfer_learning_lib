import os
import cv2
import numpy as np
from sklearn.utils import shuffle


class DataPreperer:
    def __init__(self, dir_path, image_size):
        self._init_train_path(dir_path = dir_path)
        self.image_size = image_size

    def _init_train_path(self, dir_path):
        names = os.listdir(dir_path)
        self.train_dir_paths = []

        for name in names:
            self.train_dir_paths.append(os.path.join(dir_path, name))

    @staticmethod
    def process_image(image, image_size):

        if type(image) == str:
            image = cv2.imread(image)


        image = cv2.resize(image, (image_size, image_size))
        return image / 255


    def _load_data(self):
        X = []
        y = []

        labels = {}

        for index, label_path in enumerate(self.train_dir_paths):
            label = label_path.split(os.path.sep)[-1]
            labels[label] = index

            for path_name in  os.listdir(label_path):
                current_path = os.path.join(label_path, path_name)
                X.append(DataPreperer.process_image(current_path, self.image_size))
                y.append(index)

        self.labels = labels
        X = np.array(X)
        y = np.array(y)

        X, y = shuffle(X, y)

        return X, y

    def run(self):
        X, y = self._load_data()
        return X, y





