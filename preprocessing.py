import glob
import multiprocessing as mp
import os
from PIL import Image
from read_protobuf import DATA_ROOT_PATH

TRAINING_SET_PATH = 'data/train_0/train/0'
DATASET_SUBFOLDERS = range(0, 18)
IMAGE_NEW_WIDTH = 256
IMAGE_NEW_HEIGHT = 192

def create_intensity_images_from_rgb_images_folder(path, subfolder):
    for filename in glob.iglob(path + '**/' + str(subfolder) + '/*/photo/*', recursive=True):
        # don't duplicate intensity images
        if not filename.endswith("_intensity.jpg"):
            img = Image.open(filename).convert('L')
            image_name_without_extension = filename.split('.')[0]
            intensity_image_name = image_name_without_extension + '_intensity.jpg'
            print(intensity_image_name)
            img.save(intensity_image_name)

def resize_intensity_images(path, new_width, new_height, subfolder):
    for filename in glob.iglob(path + '**/*/photo/*_intensity.jpg', recursive=True):
        print(filename)
        img = Image.open(filename)
        resized_image = img.resize((new_width, new_height))
        image_name_without_extension = filename.split('.')[0]
        resized_intensity_image_name = image_name_without_extension + '_resized.jpg'
        resized_image.save(resized_intensity_image_name, "JPEG", optimize=True)
        # remove the original intensity image
        os.remove(filename)

def remove_intensity_images(path):
    for filename in glob.iglob(path + '**/*/photo/*_intensity.jpg', recursive=True):
        os.remove(filename)
    for filename in glob.iglob(path + '**/*/photo/*_resized.jpg', recursive=True):
        os.remove(filename)

if __name__ == '__main__':
    # preprocess each of the 17 subfolders of the training in parallel
    pool = mp.Pool(mp.cpu_count())
    #[pool.apply(create_intensity_images_from_rgb_images_folder, args=(TRAINING_SET_PATH, subfolder)) for subfolder in DATASET_SUBFOLDERS]
    [pool.apply(resize_intensity_images, args=(TRAINING_SET_PATH, IMAGE_NEW_WIDTH, IMAGE_NEW_HEIGHT, subfolder)) for subfolder in DATASET_SUBFOLDERS]
    pool.close()
    