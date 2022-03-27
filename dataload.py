import os
import glob
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch

SEED = 233
CROP_SIZE = 96
RESIZE_SIZE = 224
BATCH_SIZE = 64

TRAIN_DIR = "./train/"
TEST_DTR = "./test/"

TRAIN_CSV = pd.read_csv("train_labels.csv")
TEST_CSV = pd.read_csv("sample_submission.csv")
WSI_CSV = pd.read_csv("patch_id_wsi.csv")


def seed_everything(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


seed_everything()


# -------------------- 函数定义 --------------------#


def ImageLoader(path):
    """
    图像loader
    将图像读取为PIL.Image RGB模式
    """
    return Image.open(path).convert('RGB')


def ImageLoader_cv2(path):
    """
    图像loader
    将图像读取为PIL.Image RGB模式
    """
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def read_image_list(csv_data, data_dir):
    """
    读取img_list
    返回图像的相对位置的list
    """
    image_list = [os.path.join(data_dir, '{}.tif'.format(e_i)) for e_i in csv_data['id'].values]
    return image_list


def read_label_list(csv_data):
    """
    读取label_list
    返回图像对应的label
    """
    return csv_data['label'].values.reshape(-1, 1)


def get_mean_std():
    """
    获取图像avr std
    对图像进行标准化之后传入网络，计算22W张图片的mean和std
    """
    if not os.path.exists("train_mean_std.npy"):
        print("Start computing statistics of 220025 images")
        dark_th = 10 / 255
        bright_th = 245 / 255
        too_dark_idx = []
        too_bright_idx = []

        x_tot = np.zeros(3)
        x2_tot = np.zeros(3)
        counted_ones = 0

        for f_path in read_image_list(TRAIN_CSV, TRAIN_DIR):
            # 读取归一化之后的图像
            imagearray = np.array(ImageLoader(f_path)).reshape(-1, 3) / 255
            if imagearray.max() < dark_th:  # 图像是否过暗
                too_dark_idx.append(f_path)
                continue
            if imagearray.min() > bright_th:  # 图像是否过亮
                too_bright_idx.append(f_path)
                continue

            x_tot += imagearray.mean(axis=0)
            x2_tot += (imagearray ** 2).mean(axis=0)
            counted_ones += 1

        channel_avr = x_tot / counted_ones
        channel_std = np.sqrt(x2_tot / counted_ones - channel_avr ** 2)
        np.save("train_mean_std.npy", np.append(channel_avr, channel_std))

        print("Computing finished: {} images\n".format(counted_ones), "-" * 20)
        return channel_avr, channel_std

    else:
        print("-" * 30, "\nReading existed file")
        avr_std = np.load("train_mean_std.npy")
        channel_avr = avr_std[:3]
        channel_std = avr_std[3:]
        return channel_avr, channel_std


def get_x_trans_origin():
    """
    获取图像的transform.Compose
    """
    # mean=[0.485, 0.456, 0.406]
    # std=[0.229, 0.224, 0.225]

    all_mean, all_std = get_mean_std()

    x_trans = transforms.Compose([
        transforms.CenterCrop(CROP_SIZE),
        transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
        transforms.RandomChoice([
            transforms.RandomRotation((0, 0)),
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomVerticalFlip(p=1),
            transforms.RandomRotation((90, 90)),
            transforms.RandomRotation((180, 180)),
            transforms.RandomRotation((270, 270)),
            transforms.Compose([
                transforms.RandomHorizontalFlip(p=1),
                transforms.RandomRotation((90, 90)),
            ]),
            transforms.Compose([
                transforms.RandomHorizontalFlip(p=1),
                transforms.RandomRotation((270, 270)),
            ])
        ]),
        transforms.ToTensor(),
        transforms.Normalize(mean=all_mean, std=all_std)
    ])
    return x_trans


def get_x_trans_change():
    """
    获取图像的transform.Compose
    """
    # mean=[0.485, 0.456, 0.406]
    # std=[0.229, 0.224, 0.225]

    all_mean, all_std = get_mean_std()

    x_trans = transforms.Compose([
        transforms.CenterCrop(CROP_SIZE),
        transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
        transforms.RandomChoice([
            transforms.ColorJitter(brightness=0.5),
            transforms.ColorJitter(contrast=0.5),
            transforms.ColorJitter(saturation=0.5),
            transforms.ColorJitter(hue=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        ]),
        transforms.RandomChoice([
            transforms.RandomRotation((0, 0)),
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomVerticalFlip(p=1),
            transforms.RandomRotation((90, 90)),
            transforms.RandomRotation((180, 180)),
            transforms.RandomRotation((270, 270)),
            transforms.Compose([
                transforms.RandomHorizontalFlip(p=1),
                transforms.RandomRotation((90, 90)),
            ]),
            transforms.Compose([
                transforms.RandomHorizontalFlip(p=1),
                transforms.RandomRotation((270, 270)),
            ])
        ]),
        transforms.ToTensor(),
        transforms.Normalize(mean=all_mean, std=all_std)
    ])
    return x_trans


# -------------------- Dataset创建 --------------------#

class MyDataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None, loader=ImageLoader):
        self.image_data = read_image_list(csv_file, data_dir)
        self.label_data = read_label_list(csv_file)

        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        x = self.image_data[index]
        label = self.label_data[index]

        img = self.loader(x)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.image_data)


# -------------------- No-WSI Dataset构造 --------------------#

TRANS = get_x_trans_change()


# TRANS = get_x_trans_origin()


def train_val_data():
    train_part, val_part = train_test_split(TRAIN_CSV, stratify=TRAIN_CSV.label,
                                            test_size=0.1, random_state=SEED)

    train_data = MyDataset(csv_file=train_part, data_dir=TRAIN_DIR,
                           transform=TRANS, loader=ImageLoader)

    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE,
                                   shuffle=True, num_workers=4)

    valid_data = MyDataset(csv_file=val_part, data_dir=TRAIN_DIR,
                           transform=TRANS, loader=ImageLoader)
    valid_data_loader = DataLoader(valid_data, batch_size=BATCH_SIZE,
                                   shuffle=True, num_workers=4)

    train_val_loader = {"train": train_data_loader,
                        "val": valid_data_loader}

    print("-" * 30, "\nTrain data size: {}\nValid data size: {}\n".format(
        len(train_part), len(val_part)))

    print("Train data batch: {}\nValid data batch: {}\n".format(
        len(train_data_loader), len(valid_data_loader)))

    return train_val_loader


def test_data():
    test_datapart = MyDataset(csv_file=TEST_CSV, data_dir=TEST_DTR,
                              transform=TRANS, loader=ImageLoader)

    test_data_loader = DataLoader(test_datapart,
                                  batch_size=BATCH_SIZE, num_workers=4)

    print("-" * 30, "Test data size: {}\nTest data batch: {}\n".format(
        len(test_datapart), len(test_data_loader)))

    return test_data_loader


# -------------------- WSI Dataset构造 --------------------#

def wsi_split():
    not_in_wsi = TRAIN_CSV.set_index('id').drop(WSI_CSV.id)

    # 用于训练以及验证的WSI-label
    val_wsi = WSI_CSV.groupby(by='wsi')['id'].count().sample(frac=0.15,
                                                             random_state=SEED).index
    trn_wsi = [i[0] for i in WSI_CSV.groupby(by='wsi')['id'] if i[0] not in val_wsi]

    train_image = []
    valid_image = []

    for each_wsi, wsi_data in WSI_CSV.groupby(by='wsi'):
        if each_wsi in trn_wsi:
            train_image.append(wsi_data)
        else:
            valid_image.append(wsi_data)

    wsi_train = pd.concat(train_image)
    wsi_valid = pd.concat(valid_image)

    wsi_train = pd.merge(wsi_train, TRAIN_CSV, how="inner",
                         left_on="id", right_on="id").drop(["wsi"], axis=1)

    wsi_valid = pd.merge(wsi_valid, TRAIN_CSV, how="inner",
                         left_on="id", right_on="id").drop(["wsi"], axis=1)

    wsi_valid = pd.concat([not_in_wsi.reset_index(), wsi_valid])

    return wsi_train, wsi_valid


def train_val_data_wsi():
    train_part, val_part = wsi_split()

    train_data = MyDataset(csv_file=train_part, data_dir=TRAIN_DIR,
                           transform=TRANS, loader=ImageLoader)

    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE,
                                   shuffle=True, num_workers=4)

    valid_data = MyDataset(csv_file=val_part, data_dir=TRAIN_DIR,
                           transform=TRANS, loader=ImageLoader)
    valid_data_loader = DataLoader(valid_data, batch_size=BATCH_SIZE,
                                   shuffle=True, num_workers=4)

    train_val_loader = {"train": train_data_loader,
                        "val": valid_data_loader}

    print("-" * 30, "\nTrain data size: {}\nValid data size: {}\n".format(
        len(train_part), len(val_part)))

    print("Train data batch: {}\nValid data batch: {}\n".format(
        len(train_data_loader), len(valid_data_loader)))

    return train_val_loader
