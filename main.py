from lib.image_augmentar import ImageAugmentar


def main():
    image_aug = ImageAugmentar(image_path = "avatar.jpg")
    image_aug.generate("test1")



if __name__ == "__main__":
    main()
