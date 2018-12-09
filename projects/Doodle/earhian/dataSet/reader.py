from torch.utils.data import Dataset
import random
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import math
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
BASE_SIZE = 256
def get_image(raw_strokes, size=256, lw=6, time_color=True):
    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    for t, stroke in enumerate(raw_strokes):
        for i in range(len(stroke[0]) - 1):
            color = 255 - min(t, 10) * 13 if time_color else 255
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
    if size != BASE_SIZE:
        return cv2.resize(img, (size, size))
    else:
        return img
def get_image_colorful(raw_strokes, size=256, lw=6, time_color=True, time_length=3):
    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    length = len(raw_strokes)
    factor = 1
    if length > time_length:
        factor = (length // time_length) + 1
    images = []
    for t, stroke in enumerate(raw_strokes):
        for i in range(len(stroke[0]) - 1):
            color = 255 - min(t, 10) * 13 if time_color else 255
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
        if t % factor == 0:
            image = img.copy()
            if size != BASE_SIZE:
                image = cv2.resize(image, (size, size))
            images.append(image)
            img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    image = img.copy()
    if size != BASE_SIZE:
        image = cv2.resize(image, (size, size))
    for index in range(len(images), time_length):
        images.append(image)
    return np.array(images)
def get_image_list(raw_strokes, size=256, lw=6, time_color=True, time_length=3):
    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    length = len(raw_strokes)
    factor = 1
    if length > time_length:
        factor = (length // time_length) + 1
    images = []
    for t, stroke in enumerate(raw_strokes):
        for i in range(len(stroke[0]) - 1):
            color = 255 - min(t, 10) * 13 if time_color else 255
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
        if t % factor == 0:
            image = img.copy()
            if size != BASE_SIZE:
                image = cv2.resize(image, (size, size))
            images.append(image)
            img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    image = img.copy()
    if size != BASE_SIZE:
        image = cv2.resize(image, (size, size))
    for index in range(len(images), time_length):
        images.append(image)
    return np.array(images)


def drawing_to_image_with_color_v2(drawing, H, W):
    point = []
    time = []
    for t, (x, y) in enumerate(drawing):
        point.append(np.array((x, y), np.float32).T)
        time.append(np.full(len(x), t))

    point = np.concatenate(point).astype(np.float32)
    time = np.concatenate(time).astype(np.int32)

    # --------
    image = np.full((H, W, 3), 0, np.uint8)
    x_max = point[:, 0].max()
    x_min = point[:, 0].min()
    y_max = point[:, 1].max()
    y_min = point[:, 1].min()
    w = x_max - x_min
    h = y_max - y_min
    # print(w,h)

    s = max(w, h)
    norm_point = (point - [x_min, y_min]) / s
    norm_point = (norm_point - [w / s * 0.5, h / s * 0.5]) * max(W, H) * 0.85
    norm_point = np.floor(norm_point + [W / 2, H / 2]).astype(np.int32)

    # --------
    T = time.max() + 1
    P = len(point) - T

    colors = plt.cm.jet(np.arange(0, P + 1) / (P))
    colors = (colors[:, :3] * 255).astype(np.uint8)
    p_num = 0
    for t in range(T):
        p = norm_point[time == t]
        x, y = p.T
        image[y, x] = 255
        N = len(p)
        for i in range(N - 1):
            color = colors[p_num]
            color = [int(color[2]), int(color[1]), int(color[0])]
            x0, y0 = p[i]
            x1, y1 = p[i + 1]
            cv2.line(image, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)
            p_num += 1

    return image
def drawing_to_image_with_color(drawing, H=112, W=112, time_length=8):
    point = []
    time = []
    for t, (x, y) in enumerate(drawing):
        point.append(np.array((x, y), np.float32).T)
        time.append(np.full(len(x), t))

    point = np.concatenate(point).astype(np.float32)
    time = np.concatenate(time).astype(np.int32)

    # --------
    image = np.full((H, W), 0, np.uint8)
    x_max = point[:, 0].max()
    x_min = point[:, 0].min()
    y_max = point[:, 1].max()
    y_min = point[:, 1].min()
    w = x_max - x_min
    h = y_max - y_min
    # print(w,h)

    s = max(w, h)
    norm_point = (point - [x_min, y_min]) / s
    norm_point = (norm_point - [w / s * 0.5, h / s * 0.5]) * max(W, H) * 0.85
    norm_point = np.floor(norm_point + [W / 2, H / 2]).astype(np.int32)

    # --------
    T = time.max() + 1
    P = len(point) - T
    factor = 1
    if T > time_length:
        factor = (T//time_length) + 1
    p_num = 0
    images = []
    for t in range(T):
        p = norm_point[time == t]
        x, y = p.T
        image[y, x] = 255
        N = len(p)
        for i in range(N - 1):
            x0, y0 = p[i]
            x1, y1 = p[i + 1]
            cv2.line(image, (x0, y0), (x1, y1), 255, 1, cv2.LINE_AA)
            p_num += 1
        if (t+1) % factor == 0:
            images.append(image)
    for index in range(len(images), time_length):
        images.append(image)

    return np.array(images)

class QDDRDataset(Dataset):

    def __init__(self, csv_id=None, num_preclass=110, time_length=3,mode='train', size=112,transform=None, isGetList=False):
        super(QDDRDataset, self).__init__()
        if mode in ['train', 'valid']:
            if csv_id is None:
                self.csv = pd.read_csv('./input/valid_{}k_100.csv'.format(num_preclass))
            else:
                self.csv = pd.read_csv('./input/shuffle_csv_200_{}k/train_k{}.csv.gz'.format(num_preclass, csv_id))
        else:
            self.csv = pd.read_csv('./input/test_simplified.csv')

        self.dict_label = self.load_labels()
        self.transform = transform
        if mode == 'valid':
            # self.sample_valid()
            pass
        self.isGetList = isGetList
        self.size = size
        self.mode = mode
        self.time_length = time_length


    def __len__(self):
        return len(self.csv)
    def sample_valid(self, num_valid=100):
        self.csv = self.csv.sample(340 * num_valid)
        self.csv = self.csv.reset_index()

    def load_labels(self):
        label = pd.read_csv('./input/label.csv')
        labelName = label['name'].tolist()
        labelId = label['id'].tolist()
        dict_label = {}
        for (name, id) in zip(labelName, labelId):
            dict_label[name] = int(id)
        return dict_label
    def __getitem__(self, index):
        #Index(['countrycode', 'drawing', 'recognized', 'timestamp', 'word', 'y', 'cv'], dtype='object')
        if self.mode in ['train', 'valid']:
            drawing = eval(self.csv['drawing'][index])
            word = self.csv['word'][index]
            label = self.dict_label[word]
            recognized = self.csv['recognized'][index]
            recognized = int(recognized)
            if self.transform is not None:
                input = self.transform(drawing, label, recognized)
            return input
        elif self.mode in ['test']:
            drawing = eval(self.csv['drawing'][index])
            name = self.csv['key_id'][index]
            if self.transform is not None:
                image = self.transform(drawing)
            return image, str(name)

        

if __name__ == '__main__':
    csv = pd.read_csv('../input/test_simplified.csv')
    for index in range(100):
        drawing = eval(csv['drawing'][index])
        images = drawing_to_image_with_color(drawing)
        print(images.shape)

