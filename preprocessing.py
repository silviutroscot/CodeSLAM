import glob
import multiprocessing as mp
import numpy as np
import os
from PIL import Image
import statistics
import sys


TRAINING_SET_PATH = 'data/train/'
DATASET_SUBFOLDERS = range(0, 15)
IMAGE_NEW_WIDTH = 256
IMAGE_NEW_HEIGHT = 192

def create_intensity_images_from_rgb_images_folder(path, subfolder):
    print("create intensity is called")
    for filename in glob.iglob(path + str(subfolder) + '/*/photo/*', recursive=True):
        # don't duplicate intensity images
        if not filename.endswith("_intensity.jpg"):
            img = Image.open(filename).convert('L')
            image_name_without_extension = filename.split('.')[0]
            intensity_image_name = image_name_without_extension + '_intensity.jpg'
            print(intensity_image_name)
            img.save(intensity_image_name)

def resize_intensity_images(path, new_width, new_height, subfolder):
    for filename in glob.iglob(path + str(subfolder) + '/*/photo/*_intensity.jpg', recursive=True):
        print(filename)
        img = Image.open(filename)
        resized_image = img.resize((new_width, new_height))
        image_name_without_extension = filename.split('.')[0]
        resized_intensity_image_name = image_name_without_extension + '_resized.jpg'
        resized_image.save(resized_intensity_image_name, "JPEG", optimize=True)
        # remove the original intensity image
        os.remove(filename)

def scale_depth(image, average):
    for i in [0, len(image)-1]:
        image[i] = average / (average + image[i])

def normalize_depth_values(path, subfolder):
    for filename in glob.iglob(path + str(subfolder) + '/*/depth/*[0-9].png', recursive=True):
        print(filename)
        img = Image.open(filename)
        size = img.size
        image_values = img.histogram()
        average_depth = statistics.mean(image_values)
        if average_depth is 0:
            average_depth = 0.000001
        scale_depth(image_values, average_depth)
        image_array = np.array(image_values, dtype=np.float32)
        image = Image.new("L", size)
        image.putdata(image_array)
        normalized_image_name = filename.split('.')[0] + '_normalized.png'
        image.save(normalized_image_name, "PNG", optimize=True)

def remove_normalized_depth_images(path):
     for filename in glob.iglob(path + '**/*/depth/*_normalized.png', recursive=True):
        os.remove(filename)

def remove_intensity_images(path):
    for filename in glob.iglob(path + '**/*/photo/*_intensity.jpg', recursive=True):
        os.remove(filename)
    for filename in glob.iglob(path + '**/*/photo/*_resized.jpg', recursive=True):
        os.remove(filename)

if __name__ == '__main__':
    pool = mp.Pool(mp.cpu_count())
    # if no args are passed, don't alter images
    if len(sys.argv) is 1:
        print("You should specify resizing (r) or converting to intensity (i).")
    elif len(sys.argv) is 2:
        if str(sys.argv[1]) is "i":
            [pool.apply(create_intensity_images_from_rgb_images_folder, args=(TRAINING_SET_PATH, subfolder)) for subfolder in DATASET_SUBFOLDERS]
        elif str(sys.argv[1]) is "r":
            [pool.apply(resize_intensity_images, args=(TRAINING_SET_PATH, IMAGE_NEW_WIDTH, IMAGE_NEW_HEIGHT, subfolder)) for subfolder in DATASET_SUBFOLDERS]
        elif str(sys.argv[1]) is "n":
            [pool.apply(normalize_depth_values, args=(TRAINING_SET_PATH, subfolder)) for subfolder in DATASET_SUBFOLDERS]
        else:
            print("Invalid argument: use 'i' for convert images to intensity images, 'r' to resize the intensity images, 'n' for depth normalization or any combination of them")
    elif len(sys.argv) is 3:
        if (str(sys.argv[1]) is "i" and str(sys.argv[2]) is "r") or (str(sys.argv[2]) is "i" and str(sys.argv[1]) is "r"):
            [pool.apply(create_intensity_images_from_rgb_images_folder, args=(TRAINING_SET_PATH, subfolder)) for subfolder in DATASET_SUBFOLDERS]
            [pool.apply(resize_intensity_images, args=(TRAINING_SET_PATH, IMAGE_NEW_WIDTH, IMAGE_NEW_HEIGHT, subfolder)) for subfolder in DATASET_SUBFOLDERS]
        elif (str(sys.argv[1]) is "i" and str(sys.argv[2]) is "n") or (str(sys.argv[1]) is "n" and str(sys.argv[2]) is "i"):
            [pool.apply(create_intensity_images_from_rgb_images_folder, args=(TRAINING_SET_PATH, subfolder)) for subfolder in DATASET_SUBFOLDERS]
            [pool.apply(normalize_depth_values, args=(TRAINING_SET_PATH, subfolder)) for subfolder in DATASET_SUBFOLDERS]
        elif (str(sys.argv[1]) is "r" and str(sys.argv[2]) is "n") or (str(sys.argv[1]) is "n" and str(sys.argv[2]) is "r"):
            [pool.apply(resize_intensity_images, args=(TRAINING_SET_PATH, IMAGE_NEW_WIDTH, IMAGE_NEW_HEIGHT, subfolder)) for subfolder in DATASET_SUBFOLDERS]
            [pool.apply(normalize_depth_values, args=(TRAINING_SET_PATH, subfolder)) for subfolder in DATASET_SUBFOLDERS]
        else:
            print("Invalid arguments: use 'i' for convert images to intensity images, 'r' to resize the intensity images, 'n' for depth normalization or any combination of them")
    elif len(sys.argv) is 4:
        if ((str(sys.argv[1]) is "i" and str(sys.argv[2]) is "r" and str(sys.argv[3]) is "n") or
            (str(sys.argv[1]) is "i" and str(sys.argv[2]) is "n" and str(sys.argv[3]) is "r") or 
            (str(sys.argv[1]) is "r" and str(sys.argv[2]) is "i" and str(sys.argv[3]) is "n") or 
            (str(sys.argv[1]) is "r" and str(sys.argv[2]) is "n" and str(sys.argv[3]) is "i") or 
            (str(sys.argv[1]) is "n" and str(sys.argv[2]) is "i" and str(sys.argv[3]) is "r") or 
            (str(sys.argv[1]) is "n" and str(sys.argv[2]) is "r" and str(sys.argv[3]) is "i")):

            [pool.apply(create_intensity_images_from_rgb_images_folder, args=(TRAINING_SET_PATH, subfolder)) for subfolder in DATASET_SUBFOLDERS]
            [pool.apply(resize_intensity_images, args=(TRAINING_SET_PATH, IMAGE_NEW_WIDTH, IMAGE_NEW_HEIGHT, subfolder)) for subfolder in DATASET_SUBFOLDERS]
            [pool.apply(normalize_depth_values, args=(TRAINING_SET_PATH, subfolder)) for subfolder in DATASET_SUBFOLDERS]
    else:
        print("Too many arguments: use 'i' for convert images to intensity images, 'r' to resize the intensity images, 'n' for depth normalization or any combination of them")
    
    pool.close()
    
