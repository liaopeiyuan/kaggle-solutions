import cv2
import numpy as np
import random
import torch
def random_cropping(image, target_shape=(128, 128), p=0.5):
    zeros = np.zeros(target_shape)
    target_w, target_h = target_shape
    width, height = image.shape
    if random.random() < p:
        start_x = random.randint(0, target_w - width)
        start_y = random.randint(0, target_h - height)
        zeros[start_x:start_x+width, start_y:start_y+height] = image
    else:
        start_x = (target_w - width)//2
        start_y = (target_h - height)//2
        zeros[start_x:start_x+width, start_y:start_y+height] = image
    return zeros
def TTA_cropps(image, target_shape=(128, 128, 3)):
    width, height, d = image.shape
    target_w, target_h, d = target_shape
    start_x = (target_w - width) // 2
    start_y = (target_h - height) // 2
    starts = [[start_x, start_y], [0, 0], [ 2 * start_x, 0],
              [0, 2 * start_y], [2 * start_x, 2 * start_y]]
    images = []
    for start_index in starts:
        image_ = image.copy()
        x, y = start_index

        zeros = np.zeros(target_shape)
        zeros[x:x + width, y: y+height, :] = image_
        image_ = zeros.copy()
        image_ = (torch.from_numpy(image_).div(255)).float()
        image_ = image_.permute(2, 0, 1)
        images.append(image_)

        zeros = np.fliplr(zeros)
        image_ = zeros.copy()
        image_ = (torch.from_numpy(image_).div(255)).float()
        image_ = image_.permute(2, 0, 1)
        images.append(image_)

    return images



def random_cropping3d(image, target_shape=(8, 128, 128), p=0.5):
    zeros = np.zeros(target_shape)
    target_l,target_w, target_h = target_shape
    length, width, height = image.shape
    if random.random() < p:
        start_x = random.randint(0, target_w - width)
        start_y = random.randint(0, target_h - height)
    else:
        start_x = (target_w - width) // 2
        start_y = (target_h - height) // 2
    zeros[:target_l,start_x:start_x+width, start_y:start_y+height] = image
    return zeros
def random_erase(image, p=0.5):
    if random.random() < p:
        pass
    return image
def random_flip(image, p=0.5):
    if random.random() < p:
        if len(image.shape) == 2:
            image = np.flip(image, 1)
        elif len(image.shape) == 3:
            image = np.transpose(image, (1, 2, 0))
            image = np.flip(image, 1)
            image = np.transpose(image, (2, 0, 1))
    return image

def random_gaussian_noise(gray, sigma=0.5, p=0.5):
    if random.random() < p:
        gray = gray.astype(np.float32)/255
        H,W  = gray.shape

        noise = np.random.normal(0,sigma,(H,W))
        noisy = gray + noise

        gray = (np.clip(noisy,0,1)*255).astype(np.uint8)
    return gray



