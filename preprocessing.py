import glob
import os
from PIL import Image
from read_protobuf import DATA_ROOT_PATH

TRAINING_SET_PATH = os.path.join(DATA_ROOT_PATH,'train_0/train/0')
print(TRAINING_SET_PATH)

def create_intensity_images_from_rgb_images_folder(path):
    for filename in glob.iglob(path + '**/*/photo/*', recursive=True):
        img = Image.open(filename).convert('L')
        image_name_without_extension = filename.split('.')[0]
        intensity_image_name = image_name_without_extension+'_intensity.jpg'
        print(intensity_image_name)
        img.save(intensity_image_name)


if __name__ == '__main__':
    create_intensity_images_from_rgb_images_folder(TRAINING_SET_PATH)