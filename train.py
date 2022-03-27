import torch
from focalloss import FocalLoss
from SEDenseNet import DenseNet169_plus, create_DenseNet169
from dataload import train_val_data, train_val_data_wsi, test_data

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# train_val_loader = train_val_data()
train_val_loader = train_val_data_wsi()

test_loader = test_data()
