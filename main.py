from lib.processor import Processor
from lib.classifier import Classifier
from lib.data_preparer import DataPreperer

def main():
    processor = Processor(src_dir = "data/close")
    processor.run(save_dir_path = "close", num = 100)


def train():
    input_shape = (224, 224, 3)
    cls = Classifier(input_shape = input_shape)
    data_preparer = DataPreperer(dir_path = "./result", image_size = input_shape[0])
    X, y = data_preparer.run()

    print(X.shape)
    print(y.shape)

if __name__ == "__main__":
    train()
