from lib.processor import Processor
from lib.classifier import Classifier

def main():
    processor = Processor(src_dir = "data/close")
    processor.run(save_dir_path = "close", num = 100)


def train():
    cls = Classifier()

if __name__ == "__main__":
    train()
