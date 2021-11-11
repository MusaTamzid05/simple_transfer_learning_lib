from lib.processor import Processor
from lib.classifier import Classifier
from lib.data_preparer import DataPreperer
from lib.utils import limit_gpu
import cv2

def main():
    processor = Processor(src_dir = "data/close")
    processor.run(save_dir_path = "close", num = 100)



def run_video():
    input_shape = (224, 224, 3)
    limit_gpu()
    cls = Classifier(input_shape = input_shape, model_path = "train_model")
    cap = cv2.VideoCapture(0)
    running = True

    while running:
        ret, frame = cap.read()

        if ret == False:
            running = False
            continue

        processed_frame = DataPreperer.process_image(image = frame , image_size = input_shape[0])
        print(cls.predict(processed_frame))

        cv2.imshow("Test", frame)


        if cv2.waitKey(1) & 0xFF == ord("q"):
            running = False
            continue


def train():
    input_shape = (224, 224, 3)
    limit_gpu()
    data_preparer = DataPreperer(dir_path = "./result", image_size = input_shape[0])
    X, y = data_preparer.run()

    print(X.shape)
    print(y.shape)

    cls = Classifier(input_shape = input_shape)
    cls.fit(X, y, epochs = 8)

if __name__ == "__main__":
    run_video()
