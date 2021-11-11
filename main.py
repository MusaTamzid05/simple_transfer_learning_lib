from lib.processor import Processor

def main():
    processor = Processor(src_dir = "data/close")
    processor.run(save_dir_path = "close", num = 100)



if __name__ == "__main__":
    main()
