import os
import glob
import copy
import cv2
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import models, transforms, utils
from torch.utils.data import Dataset, DataLoader, random_split
from pretrainedmodels.models import se_resnext50_32x4d, se_resnext101_32x4d
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

SEED = 233
CROP_SIZE = 96
RESIZE_SIZE = 224
BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
MAX_EPOCH = 15

TRAIN_DIR = "./train/"
TEST_DTR = "./test/"

TRAIN_CSV = pd.read_csv("train_labels.csv")
TEST_CSV = pd.read_csv("sample_submission.csv")
WSI_CSV = pd.read_csv("patch_id_wsi.csv")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def seed_everything(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


seed_everything()


# -------------------- Define Function --------------------#


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


def tta_test():
    # "./train/00001b2b5609af42ab0ab276dd4cd41c3e7745b5.tif"
    # Image.open().convert('RGB')

    all_mean, all_std = get_mean_std()

    change_01 = transforms.Compose([
        transforms.CenterCrop(CROP_SIZE),
        transforms.Resize((RESIZE_SIZE, RESIZE_SIZE))
    ])

    change_02 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=all_mean, std=all_std)
    ])

    change_list = [
        transforms.ColorJitter(brightness=0.5),
        transforms.ColorJitter(contrast=0.5),
        transforms.ColorJitter(saturation=0.5),
        transforms.ColorJitter(hue=0.5),

        transforms.RandomHorizontalFlip(p=1),
        transforms.RandomVerticalFlip(p=1),

        transforms.RandomRotation((0, 0)),
        transforms.RandomRotation((90, 90)),
        transforms.RandomRotation((180, 180)),
        transforms.RandomRotation((270, 270)),
    ]

    x_all = transforms.Lambda(
        lambda image:
        torch.stack([change_02(each(change_01(image))) for each in change_list])
    )

    return x_all


def wsi_split():
    not_in_wsi = TRAIN_CSV.set_index('id').drop(WSI_CSV.id)

    # 用于训练以及验证的WSI-label
    val_wsi = WSI_CSV.groupby(by='wsi')['id'].count().sample(frac=0.10,
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


def wsi_kfold_split():
    not_in_wsi = TRAIN_CSV.set_index('id').drop(WSI_CSV.id)

    grouped_l = list(WSI_CSV.groupby(by='wsi'))

    grouped_normal = grouped_l[:128]  # 正常的影像wsi
    grouped_tumor = grouped_l[128:]  # 带有tumor的wsi

    random.shuffle(grouped_normal)
    random.shuffle(grouped_tumor)

    k_f_data = {}

    kf_5 = KFold(n_splits=5, shuffle=True, random_state=SEED)
    not_in_wsi_kf = list(kf_5.split(not_in_wsi))

    for k in range(5):
        v_normal = grouped_normal[int(k / 5 * len(grouped_normal)):
                                  int((k + 1) / 5 * len(grouped_normal))]
        v_tumor = grouped_tumor[int(k / 5 * len(grouped_tumor)):
                                int((k + 1) / 5 * len(grouped_tumor))]  # validation

        t_normal = [_ for _ in grouped_normal if _ not in v_normal]
        t_tumor = [_ for _ in grouped_tumor if _ not in v_tumor]  # train

        temp_v = [_v[1] for _v in v_normal] + [_v[1] for _v in v_tumor]
        temp_t = [_t[1] for _t in t_normal] + [_t[1] for _t in t_tumor]

        random.shuffle(temp_t)
        random.shuffle(temp_v)

        wsi_k_t = pd.concat(temp_t)  # wsi train
        wsi_k_v = pd.concat(temp_v)  # wsi valid

        img_k_train = pd.merge(wsi_k_t, TRAIN_CSV, how="inner",
                               left_on="id", right_on="id").drop(["wsi"], axis=1)

        img_k_valid = pd.merge(wsi_k_v, TRAIN_CSV, how="inner",
                               left_on="id", right_on="id").drop(["wsi"], axis=1)

        # 加上不在WSI中的CV
        img_k_train = pd.concat([not_in_wsi.iloc[not_in_wsi_kf[k][0]].reset_index(),
                                 img_k_train])
        img_k_valid = pd.concat([not_in_wsi.iloc[not_in_wsi_kf[k][1]].reset_index(),
                                 img_k_valid])

        k_f_data[k] = {"train": img_k_train,
                       "val": img_k_valid}

    return k_f_data


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


# -------------------- Dataset构造 --------------------#
TRANS = get_x_trans_change()
TRANS_test = tta_test()

KF_Dataset = {}
KF_length = {}
k_wsi_csv = wsi_kfold_split()

for cnt in range(5):
    print(cnt + 1)

    train_part = k_wsi_csv[cnt]["train"]
    val_part = k_wsi_csv[cnt]["val"]

    train_data = MyDataset(csv_file=train_part, data_dir=TRAIN_DIR,
                           transform=TRANS, loader=ImageLoader)
    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE,
                                   shuffle=True, num_workers=4)

    valid_data = MyDataset(csv_file=val_part, data_dir=TRAIN_DIR,
                           transform=TRANS, loader=ImageLoader)
    valid_data_loader = DataLoader(valid_data, batch_size=BATCH_SIZE,
                                   shuffle=True, num_workers=4)

    KF_Dataset[cnt] = {"train": train_data_loader,
                       "val": valid_data_loader}
    KF_length[cnt] = {"train": len(train_data),
                      "val": len(valid_data)}

    print("-" * 30, "\nTrain data size: {}\nValid data size: {}\n".format(
        len(train_part), len(val_part)))
    print("Train data batch: {}\nValid data batch: {}\n".format(
        len(train_data_loader), len(valid_data_loader)))

test_data = MyDataset(
    csv_file=TEST_CSV,
    data_dir=TEST_DTR,
    transform=TRANS_test,
    loader=ImageLoader
)

test_data_loader = DataLoader(
    test_data,
    batch_size=TEST_BATCH_SIZE,
    num_workers=4
)


# -------------------- Build Network --------------------#


class SE_ResNext_50(nn.Module):

    def __init__(self, ):
        super(SE_ResNext_50, self).__init__()

        model = se_resnext50_32x4d()
        self.model_layer = nn.Sequential(*list(model.children())[:-1])
        self.linear_layer = nn.Linear(2048, 1)

    def forward(self, x):
        x = self.model_layer(x)  # [-1, 2048, 1, 1]

        batch = x.shape[0]
        conc = x.view(batch, -1)
        out = self.linear_layer(conc)

        return out


class DenseNet169_plus(nn.Module):

    def __init__(self, pretrained=True):
        super(DenseNet169_plus, self).__init__()

        # self.densenet_layer = nn.Sequential(*list(model.children())[:-1])
        self.densenet = models.densenet169(pretrained=pretrained)

        self.Linear_layer = nn.Linear(1000 + 2, 16)
        self.bn = nn.BatchNorm1d(16)
        self.dropout = nn.Dropout(0.2)
        self.elu = nn.ELU()
        self.out = nn.Linear(16, 1)

    def forward(self, x):
        x = self.densenet(x)  # [-1, 1000]

        batch = x.shape[0]  # [-1, 1000]
        max_pool, _ = torch.max(x, 1, keepdim=True)  # [-1, 1]
        avg_pool = torch.mean(x, 1, keepdim=True)  # [-1, 1]

        x = x.view(batch, -1)  # [-1, 1000]
        conc = torch.cat((x, max_pool, avg_pool), 1)  # [-1, 1002]

        conc = self.Linear_layer(conc)
        conc = self.elu(conc)
        conc = self.bn(conc)
        conc = self.dropout(conc)

        out = self.out(conc)

        return out


class SE_DenseNet169_plus(nn.Module):

    def __init__(self):
        super(SE_DenseNet169_plus, self).__init__()

        densenet = models.densenet169(pretrained=True)
        features = densenet.features

        self.dense_block_1 = nn.Sequential(*list(features.children())[:5])  # [-1, 256, 56, 56]
        self.se_block_1 = nn.Sequential(nn.AdaptiveMaxPool2d(1),
                                        nn.Conv2d(256, 16, kernel_size=1,
                                                  stride=1, bias=False),
                                        nn.ReLU(),
                                        nn.Conv2d(16, 256, kernel_size=1,
                                                  stride=1, bias=False),
                                        nn.Sigmoid(),
                                        )
        self.Transition_1 = nn.Sequential(*list(features.children())[5:6])

        self.dense_block_2 = nn.Sequential(*list(features.children())[6:7])  # [-1, 512, 28, 28]
        self.se_block_2 = nn.Sequential(nn.AdaptiveMaxPool2d(1),
                                        nn.Conv2d(512, 32, kernel_size=1,
                                                  stride=1, bias=False),
                                        nn.ReLU(),
                                        nn.Conv2d(32, 512, kernel_size=1,
                                                  stride=1, bias=False),
                                        nn.Sigmoid(),
                                        )
        self.Transition_2 = nn.Sequential(*list(features.children())[7:8])

        self.dense_block_3 = nn.Sequential(*list(features.children())[8:9])  # [-1, 1280, 14, 14]
        self.se_block_3 = nn.Sequential(nn.AdaptiveMaxPool2d(1),
                                        nn.Conv2d(1280, 80, kernel_size=1,
                                                  stride=1, bias=False),
                                        nn.ReLU(),
                                        nn.Conv2d(80, 1280, kernel_size=1,
                                                  stride=1, bias=False),
                                        nn.Sigmoid(),
                                        )
        self.Transition_3 = nn.Sequential(*list(features.children())[9:10])

        self.dense_block_4 = nn.Sequential(*list(features.children())[10:11])  # [-1, 1664, 7, 7]
        self.se_block_4 = nn.Sequential(nn.AdaptiveMaxPool2d(1),
                                        nn.Conv2d(1664, 104, kernel_size=1,
                                                  stride=1, bias=False),
                                        nn.ReLU(),
                                        nn.Conv2d(104, 1664, kernel_size=1,
                                                  stride=1, bias=False),
                                        nn.Sigmoid(),
                                        )
        self.BN_4 = nn.Sequential(*list(features.children())[11:])
        self.relu_avg = nn.Sequential(nn.ReLU(),
                                      nn.AdaptiveAvgPool2d(1))
        self.classifier = densenet.classifier

        self.Linear_layer = nn.Linear(1000 + 2, 16)
        self.bn = nn.BatchNorm1d(16)
        self.dropout = nn.Dropout(0.2)
        self.elu = nn.ELU()
        self.out = nn.Linear(16, 1)

    def forward(self, x):
        x1 = self.dense_block_1(x)  # [-1, 256, 56, 56]
        x1_se = self.se_block_1(x1)  # [-1, 256, 1, 1]
        x1 = x1 * x1_se.expand_as(x1)
        x1 = self.Transition_1(x1)

        x2 = self.dense_block_2(x1)  # [-1, 512, 28, 28]
        x2_se = self.se_block_2(x2)  # [-1, 512, 1, 1]
        x2 = x2 * x2_se.expand_as(x2)
        x2 = self.Transition_2(x2)

        x3 = self.dense_block_3(x2)  # [-1, 1280, 14, 14]
        x3_se = self.se_block_3(x3)  # [-1, 1280, 1, 1]
        x3 = x3 * x3_se.expand_as(x3)
        x3 = self.Transition_3(x3)

        x4 = self.dense_block_4(x3)  # [-1, 1664, 7, 7]
        x4_se = self.se_block_4(x4)  # [-1, 1664, 1, 1]
        x4 = x4 * x4_se.expand_as(x4)
        x4 = self.BN_4(x4)
        x4 = self.relu_avg(x4)
        x4 = torch.flatten(x4, 1)  # [-1, 1664]
        x_out = self.classifier(x4)  # [-1, 1000]

        batch = x_out.shape[0]  # [-1, 1000]
        max_pool, _ = torch.max(x_out, 1, keepdim=True)  # [-1, 1]
        avg_pool = torch.mean(x_out, 1, keepdim=True)  # [-1, 1]

        x_out = x_out.view(batch, -1)  # [-1, 1000]
        conc = torch.cat((x_out, max_pool, avg_pool), 1)  # [-1, 1002]

        conc = self.Linear_layer(conc)
        conc = self.elu(conc)
        conc = self.bn(conc)
        conc = self.dropout(conc)

        out = self.out(conc)

        return out


# -------------------- Tensorboard --------------------#


def train(max_epoch=MAX_EPOCH):
    for idx in range(5):
        print("CV {}/5 starts!!!".format(idx + 1))
        data_loader = KF_Dataset[idx]

        net = SE_DenseNet169_plus()
        net.to(DEVICE)

        optimizer = optim.Adam(net.parameters(), lr=1e-4)  # 定义优化器和目标函数
        criterion = nn.BCEWithLogitsLoss()  # 自带sigmoid
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)

        best_model_wts = copy.deepcopy(net.state_dict())
        best_acc = 0.0
        best_epoch = 0

        for epoch in range(max_epoch):
            print('Epoch {}/{}\n'.format(epoch, max_epoch - 1), '-' * 30)

            for phase in ["train", "val"]:

                y_true = []
                y_pred = []

                if phase == "train":
                    net.train()
                else:
                    net.eval()

                running_loss = 0.0
                running_corrects = 0

                # Iterate data
                for iteration, (x, y) in tqdm(enumerate(data_loader[phase])):

                    x = x.to(DEVICE)  # FloatType
                    y = y.to(DEVICE).float()  # LongType → FloatType
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = net(x)  # outputs shape & y shape [64, 1]
                        loss = criterion(outputs, y)

                        proba = nn.Sigmoid()(outputs)
                        preds = torch.round(proba)  # 将输出进行Sigmoid

                        y_true.append(y.data.detach().cpu().numpy())  # 截断反向梯度流，转为numpy
                        y_pred.append(proba.detach().cpu().numpy())

                        if phase == 'train':
                            # backward + optimizer
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item()  # get loss number
                    running_corrects += torch.sum(preds == y.data)

                y_true = np.vstack(y_true).reshape(-1)
                y_pred = np.vstack(y_pred).reshape(-1)

                if phase == "train":
                    epoch_auc = roc_auc_score(y_true, y_pred)
                    epoch_loss = running_loss / KF_length[idx]["train"]
                    epoch_acc = running_corrects.double() / KF_length[idx]["train"]

                    # 注意摆放位置
                    scheduler.step(epoch_acc)
                else:
                    epoch_auc = roc_auc_score(y_true, y_pred)
                    epoch_loss = running_loss / KF_length[idx]["val"]
                    epoch_acc = running_corrects.double() / KF_length[idx]["val"]

                print('{} Loss: {:.4f} Acc: {:.4f}, Auc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc, epoch_auc))

                # 保存模型，当验证集合准确率最高的时候保存
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(net.state_dict())
                if phase == 'val':
                    print("Now epoch {} is the best epoch".format(best_epoch))

        net.load_state_dict(best_model_wts)
        torch.save(net.state_dict(), "{}_cv{}_best_epoch{}.pth".format(net._get_name(), idx, best_epoch))
        print("-" * 30, "\nFinish Training cv{}, {} Epoch\nThe Best is epoch {}\n".format(idx, MAX_EPOCH, best_epoch))


def predict():
    for idx in range(5):
        net = SE_DenseNet169_plus()
        net.to("cpu")

        preds = []
        weight_path = glob.glob("SE_DenseNet169_plus_cv{}*pth".format(idx))
        if len(weight_path) != 0:
            print("-" * 30, "\nLoading weight")
            print("Test data size: {}\nTest data batch: {}\n".format(
                len(test_data), len(test_data_loader)))
            net.load_state_dict(torch.load(weight_path[0]))
        else:
            raise FileExistsError("Not exist weight file!")

        # batch_size, n_crops, c, h, w = data.size()
        # data = data.view(-1, c, h, w)
        # output = model(data)
        # output = output.view(batch_size, n_crops, -1).mean(1)

        net.to(DEVICE)
        net.eval()
        print("Start testing cv-{}".format(idx))

        with torch.no_grad():
            # 测试阶段不占用显存
            for batch_i, (x_test, target) in tqdm(enumerate(test_data_loader)):
                test_batch_size, n_crops, c, h, w = x_test.size()
                x_test = x_test.view(-1, c, h, w)
                x_test = x_test.to(DEVICE)
                out = net(x_test)

                batch_pred = nn.Sigmoid()(out)  # 将输出进行Sigmoid
                batch_pred = batch_pred.view(test_batch_size, n_crops, -1).mean(1)  # 将tta的进行整合

                batch_pred = list(batch_pred.detach().cpu().numpy())

                preds.append(batch_pred)

        test_pred = pd.DataFrame({"imgs": test_data.image_data,
                                  "preds": np.vstack(preds).reshape(-1)})
        test_pred["imgs"] = test_pred["imgs"].apply(lambda x: x.split("/")[-1][:-4])

        sub = pd.merge(TEST_CSV, test_pred, left_on="id", right_on="imgs")
        sub = sub[['id', 'preds']]
        sub.columns = ['id', 'label']

        if os.path.exists("./output"):
            sub.to_csv("./output/{}_{}_cv{}_TTA_proba.csv".format(time.strftime("%m%d"), net._get_name(), idx))
        else:
            os.mkdir("./output")
            sub.to_csv("./output/{}_{}_cv{}_TTA_proba.csv".format(time.strftime("%m%d"), net._get_name(), idx))
        print("File Saved!")


def calculate():
    # 计算各种指标
    y_all_true = np.array([])
    y_all_pred = np.array([])
    y_all_proba = np.array([])

    for idx in range(5):

        data_loader = KF_Dataset[idx]

        net = SE_ResNext_50()
        net.to("cpu")

        weight_path = glob.glob("SE_ResNext_50_cv{}*pth".format(idx))
        if len(weight_path) != 0:
            print("-" * 30, "\nLoading weight")
            net.load_state_dict(torch.load(weight_path[0]))
        else:
            raise FileExistsError("Not exist weight file!")

        net.to(DEVICE)
        net.eval()
        print("Start calculating cv-{}".format(idx))

        # for phase in ["val", "train"]:
        for phase in ["val"]:
            y_true = []
            y_pred = []
            y_proba = []

            with torch.no_grad():

                for iteration, (x, y) in tqdm(enumerate(data_loader[phase])):
                    x = x.to(DEVICE)
                    y = y.to(DEVICE).float()  # LongType → FloatType
                    outputs = net(x)

                    proba = nn.Sigmoid()(outputs)  # 将输出进行Sigmoid
                    preds = torch.round(proba)

                    y_true.append(y.data.detach().cpu().numpy())  # 截断反向梯度流，转为numpy
                    y_pred.append(preds.detach().cpu().numpy())  # [0,1]取值
                    y_proba.append(proba.detach().cpu().numpy())  # 概率取值

            y_true = np.vstack(y_true).reshape(-1)
            y_pred = np.vstack(y_pred).reshape(-1)
            y_proba = np.vstack(y_proba).reshape(-1)

            epoch_acc = accuracy_score(y_true, y_pred)
            epoch_auc = roc_auc_score(y_true, y_proba)
            epoch_precision = precision_score(y_true, y_pred)
            epoch_recall = recall_score(y_true, y_pred)
            epoch_f1 = f1_score(y_true, y_pred)

            print('\n{} ACC:{:.4f}, Auc: {:.4f}'
                  ' precision: {:.4f}, recall: {:.4f}, f1-score: {:.4f}\n'.format(
                phase, epoch_acc, epoch_auc, epoch_precision, epoch_recall, epoch_f1))

            if phase == "val":
                y_all_true = np.hstack((y_all_true, y_true))
                y_all_pred = np.hstack((y_all_pred, y_pred))
                y_all_proba = np.hstack((y_all_proba, y_proba))

    all_acc = accuracy_score(y_all_true, y_all_pred)
    all_auc = roc_auc_score(y_all_true, y_all_proba)
    all_precision = precision_score(y_all_true, y_all_pred)
    all_recall = recall_score(y_all_true, y_all_pred)
    all_f1 = f1_score(y_all_true, y_all_pred)

    print('\nALL ACC:{:.4f}, Auc: {:.4f}'
          ' precision: {:.4f}, recall: {:.4f}, f1-score: {:.4f}\n'.format(
        all_acc, all_auc, all_precision, all_recall, all_f1))


# -------------------- Model --------------------#

# train()
# torch.cuda.empty_cache()
# predict()

calculate()

# tta_file = glob.glob("output/0410*")
# df_tta = pd.read_csv("sample_submission.csv")
# df_tta["label"] = 0
# for each_csv in tta_file:
#     temp = pd.read_csv(each_csv, index_col=0)
#     df_tta["label"] += temp["label"]
#
# df_tta['label'] /= len(tta_file)
# df_tta.to_csv("output/TTA_0410_SE_DenseNet169_WSIall_CV5.csv", index=False)
