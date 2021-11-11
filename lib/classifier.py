from tensorflow import keras
import os
import cv2
import numpy as np

class Classifier:
    def __init__(self, model_path = None, input_shape = (224, 224, 3)):
        if model_path is None:
            self._init_model(input_shape)

        self.model = keras.models.load_model(os.path.join(model_path, "model.h5"))


    def _init_model(self, input_shape):
        base_model = keras.applications.VGG16(weights = "imagenet", input_shape = input_shape, include_top = False)
        base_model.trainable = False

        inputs = keras.Input(shape = input_shape)
        x = base_model(inputs, training = False)

        x = keras.layers.Conv2D(64, (3,3), activation = "relu")(x)
        x = keras.layers.MaxPooling2D(pool_size = (3, 3))(x)
        x = keras.layers.Dropout(0.6)(x)
        x = keras.layers.Flatten()(x)

        outputs = keras.layers.Dense(1, activation = "sigmoid")(x)
        model = keras.Model(inputs, outputs)

        model.compile(
                loss = "binary_crossentropy",
                optimizer = "adam",
                metrics = ["accuracy"]
                )

        self.model = model
        model.summary()


    def fit(self, X, y,  epochs = 50, batch_size = 10, save_dir = "train_model"):
        self.model.fit(X, y, epochs = epochs, batch_size = batch_size)
        self.model.save(os.path.join(save_dir, "model.h5"))


    def predict(self, image):
        image = np.expand_dims(image, 0)
        prediction = self.model.predict(image)[0][0]

        print(round(prediction))





