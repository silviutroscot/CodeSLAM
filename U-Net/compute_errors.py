import glob
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from Unet import ResnetUnetHybrid
import image_utils

subfolders = range(0, 1000)
test_dir = '../data/val'

def show_test_files():
    # build test files list
    test_paths = [os.path.join(os.path.join(test_dir, str(f))) for f in subfolders]
    test_img_paths = []
    for img_path in test_paths:
        current_image_path = os.path.join(img_path, 'photo')
        for filename in glob.iglob(current_image_path + '/*', recursive=True):
            test_img_paths.append(filename)
   
    test_img_paths.sort()
    # build labels list
    test_label_paths = []
    for img_path in test_paths:
        current_image_path = os.path.join(img_path, 'depth')
        for filename in glob.iglob(current_image_path + '/*', recursive=True):
            test_label_paths.append(filename)

    test_label_paths.sort()

    return test_img_paths, test_label_paths

if __name__ == '__main__':
    
