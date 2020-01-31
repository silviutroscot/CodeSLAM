import glob
import os
from PIL import Image
from read_protobuf import DATA_ROOT_PATH

TRAINING_SET_PATH = 'data/train_0/train/0'
print(TRAINING_SET_PATH)

def create_intensity_images_from_rgb_images_folder(path):
    print("create intensity is called")
    for filename in glob.iglob(path + '**/0/photo/*', recursive=True):
        # don't duplicate intensity images
        if not filename.endswith("_intensity.jpg"):
            img = Image.open(filename).convert('L')
            image_name_without_extension = filename.split('.')[0]
            intensity_image_name = image_name_without_extension + '_intensity.jpg'
            print(intensity_image_name)
            img.save(intensity_image_name)

def resize_intensity_images(path, new_width, new_height):
    for filename in glob.iglob(path + '**/0/photo/*_intensity.jpg', recursive=True):
        print(filename)
        img = Image.open(filename)
        resized_image = img.resize((new_width, new_height))
        image_name_without_extension = filename.split('.')[0]
        resized_intensity_image_name = image_name_without_extension + '_resized.jpg'
        resized_image.save(resized_intensity_image_name, "JPEG", optimize=True)
        # remove the original intensity image
        os.remove(filename)

def remove_unresized_intensity_images(path):
    for filename in glob.iglob(path + '**/0/photo/*_intensity.jpg', recursive=True):
        os.remove(filename)

if __name__ == '__main__':
    create_intensity_images_from_rgb_images_folder(TRAINING_SET_PATH)
    resize_intensity_images(TRAINING_SET_PATH, 256, 192)
