from tensorflow import keras

class Classifier:
    def __init__(self, model_path = None, input_shape = (224, 224, 3)):
        if model_path is None:
            self._init_model(input_shape)


    def _init_model(self, input_shape):
        base_model = keras.applications.MobileNet(weights = "imagenet", input_shape = input_shape, include_top = False)
        base_model.trainable = False

        inputs = keras.Input(shape = input_shape)
        x = base_model(inputs, training = False)
        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(1, activation = "sigmoid")(x)
        model = keras.Model(inputs, outputs)

        model.compile(
                loss = "binary_crossentropy",
                optimizer = "rmsprop",
                metrics = ["accuracy"]
                )

        self.model = model
        model.summary()




