import os
import glob
import copy
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models, transforms, utils
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from pretrainedmodels.models import se_resnext50_32x4d
from torchsummary import summary

SEED = 233
CROP_SIZE = 96
RESIZE_SIZE = 224
BATCH_SIZE = 64
MAX_EPOCH = 12

TRAIN_DIR = "./train/"
TEST_DTR = "./test/"

TRAIN_CSV = pd.read_csv("train_labels.csv")
TEST_CSV = pd.read_csv("sample_submission.csv")
WSI_CSV = pd.read_csv("patch_id_wsi.csv")

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def seed_everything(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
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


# -------------------- Dataset构造 --------------------#


train_part, val_part = train_test_split(TRAIN_CSV, stratify=TRAIN_CSV.label,
                                        test_size=0.1, random_state=SEED)

TRANS = get_x_trans_change()
# TRANS = get_x_trans_origin()

train_data = MyDataset(
    csv_file=train_part,
    data_dir=TRAIN_DIR,
    transform=TRANS,
    loader=ImageLoader
)
train_data_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4
)

valid_data = MyDataset(
    csv_file=val_part,
    data_dir=TRAIN_DIR,
    transform=TRANS,
    loader=ImageLoader
)
valid_data_loader = DataLoader(
    valid_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4
)

DATA_LOADER = {"train": train_data_loader,
               "val": valid_data_loader}

test_data = MyDataset(
    csv_file=TEST_CSV,
    data_dir=TEST_DTR,
    transform=TRANS,
    loader=ImageLoader
)

test_data_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    num_workers=4
)

print("-" * 30, "\nTrain data size: {}\nValid data size: {}\n".format(len(train_part), len(val_part)))
print("Train data batch: {}\nValid data batch: {}\n".format(len(train_data_loader), len(valid_data_loader)))


# -------------------- Build Network --------------------#

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


class DenseNet169_org(nn.Module):

    def __init__(self, pretrained=True):
        super(DenseNet169_org, self).__init__()
        model = models.densenet169(pretrained=pretrained)
        self.densenet_layer = nn.Sequential(*list(model.children())[:-1])

        self.Avg_Pooling = nn.AdaptiveAvgPool2d(1)

        self.Linear_layer = nn.Linear(1664 + 2, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.densenet_layer(x)  # [-1, 1664, 7, 7]

        batch = x.shape[0]  # [-1, 1664, 7, 7]
        max_pool, _ = torch.max(x, 1, keepdim=True)  # [-1, 1, 7, 7]
        avg_pool = torch.mean(x, 1, keepdim=True)  # [-1, 1, 7, 7]

        conc = torch.cat((x, max_pool, avg_pool), 1)  # [-1, 1666, 7, 7]
        conc = self.Avg_Pooling(conc)  # [-1, 1666, 1, 1]
        conc = conc.view(batch, -1)
        out = self.Linear_layer(conc)

        return out


class SE_ResNext(nn.Module):

    def __init__(self, ):
        super(SE_ResNext, self).__init__()

        model = se_resnext50_32x4d()
        self.model_layer = nn.Sequential(*list(model.children())[:-1])
        self.linear_layer = nn.Linear(2048, 1)

    def forward(self, x):
        x = self.model_layer(x)  # [-1, 2048, 1, 1]

        batch = x.shape[0]
        conc = x.view(batch, -1)
        out = self.linear_layer(conc)

        return out


# -------------------- Tensorboard --------------------#

writer = SummaryWriter('runs/experient')


def train(max_epoch=MAX_EPOCH):
    net = SE_ResNext()
    net.to(DEVICE)

    weight_path = glob.glob("SE_ResNext*pth")
    if len(weight_path) != 0:
        net.load_state_dict(torch.load(weight_path[0]))
        print("-" * 30, "\nLOAD local weight...")

    optimizer = optim.Adam(net.parameters(), lr=1e-4)  # 定义优化器和目标函数
    criterion = nn.BCEWithLogitsLoss()  # 自带sigmoid

    # # step4: 统计指标：平滑处理之后的损失，还有混淆矩阵
    # loss_meter = meter.AverageValueMeter()
    # confusion_matrix = meter.ConfusionMeter(2)

    dummy_input = torch.rand(BATCH_SIZE, 3, 224, 224).to(DEVICE)
    writer.add_graph(net, dummy_input)

    best_model_wts = copy.deepcopy(net.state_dict())
    best_acc = 0.0
    best_epoch = 0
    last_acc = 0

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
            writer_loss = 0.0
            writer_corrects = 0

            # Iterate data
            for iteration, (x, y) in tqdm(enumerate(DATA_LOADER[phase])):

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
                writer_loss += loss.item()

                # Tensor board
                if phase == "train" and iteration % 1000 == 999:
                    writer.add_scalar("training batch loss",
                                      writer_loss / 1000,
                                      epoch * len(train_data_loader) + iteration)
                    writer.add_scalar("training batch acc",
                                      writer_corrects / 1000 * BATCH_SIZE,
                                      epoch * len(train_data_loader) + iteration)
                    writer_loss = 0.0
                    writer_corrects = 0

            y_true = np.vstack(y_true).reshape(-1)
            y_pred = np.vstack(y_pred).reshape(-1)

            if phase == "train":
                epoch_auc = roc_auc_score(y_true, y_pred)
                epoch_loss = running_loss / len(train_data)
                epoch_acc = running_corrects.double() / len(train_data)
            else:
                epoch_auc = roc_auc_score(y_true, y_pred)
                epoch_loss = running_loss / len(valid_data)
                epoch_acc = running_corrects.double() / len(valid_data)

            print('{} Loss: {:.4f} Acc: {:.4f}, Auc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_auc))

            # 保存模型，当验证集合准确率最高的时候保存
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                best_model_wts = copy.deepcopy(net.state_dict())
                print("Now epoch {} is the best epoch".format(best_epoch))

            # # 保存模型，当验证集合准确率上升的时候就保存
            # if phase == 'val':
            #     if epoch_acc > last_acc:
            #         last_acc = epoch_acc
            #         best_epoch = epoch
            #         best_model_wts = copy.deepcopy(net.state_dict())
            #     else:
            #         last_acc = epoch_acc

    writer.close()
    net.load_state_dict(best_model_wts)
    torch.save(net.state_dict(), "SE_ResNext_no_best_epoch{}.pth".format(best_epoch))
    print("-" * 30, "Finish Training {} Epoch\nThe Best is epoch {}".format(MAX_EPOCH, best_epoch))

    return net


def predict():
    net = SE_ResNext()
    net.to("cpu")

    preds = []
    weight_path = glob.glob("SE_ResNext*pth")
    if len(weight_path) != 0:
        print("-" * 30, "\nLoading weight")
        print("Test data size: {}\nTest data batch: {}\n".format(
            len(test_data), len(test_data_loader)))
        net.load_state_dict(torch.load(weight_path[1]))
    else:
        raise FileExistsError("Not exist weight file!")

    net.to(DEVICE)
    net.eval()
    print("Start testing")

    with torch.no_grad():
        # 测试阶段不占用显存
        for batch_i, (x_test, target) in tqdm(enumerate(test_data_loader)):
            x_test = x_test.to(DEVICE)
            out = net(x_test)
            # batch_pred = torch.round(nn.Sigmoid()(out))  # 将输出进行Sigmoid
            batch_pred = nn.Sigmoid()(out)  # 将输出进行Sigmoid

            batch_pred = list(batch_pred.detach().cpu().numpy())

            preds.append(batch_pred)

    test_pred = pd.DataFrame({"imgs": test_data.image_data,
                              "preds": np.vstack(preds).reshape(-1)})
    test_pred["imgs"] = test_pred["imgs"].apply(lambda x: x.split("/")[-1][:-4])

    sub = pd.merge(TEST_CSV, test_pred, left_on="id", right_on="imgs")
    sub = sub[['id', 'preds']]
    sub.columns = ['id', 'label']

    if os.path.exists("./output"):
        sub.to_csv("./output/{}_SE_ResNext_no_proba.csv".format(time.strftime("%m%d")))
    else:
        os.mkdir("./output")
        sub.to_csv("./output/{}_SE_ResNext_no_proba.csv".format(time.strftime("%m%d")))
    print("File Saved!")


train()
torch.cuda.empty_cache()
predict()
