import os
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torchvision.utils import make_grid, save_image
import torch.nn as nn
import matplotlib.pyplot as plt

from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp

# from SEDenseNet import DenseNet169_plus

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'


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


# ------------- Load image ------------- #
img_dir = 'train'
# img_name = 'c18f2d887b7ae4f6742ee445113fa1aef383ed77.tif'  # label为1 fig1
# img_name = 'a24ce148f6ffa7ef8eefb4efb12ebffe8dd700da.tif'  # label为1 fig2
# img_name = '7f6ccae485af121e0b6ee733022e226ee6b0c65f.tif'  # label为1 fig3
img_name = '401ed2905877a6bb7c411408187cb36324c8a1ab.tif'  # label为1 fig4
# img_name = 'ddaee21647f182ecc9b30401745428904ee162c8.tif'
# img_name = 'd21e66df5c8ad45acd148c8ddf221c4961a23101.tif'
img_path = os.path.join(img_dir, img_name)

pil_img = Image.open(img_path)

# ------------- preprocess image ------------- #

torch_img = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])(pil_img).to(device)

avr_std = np.load("train_mean_std.npy")
channel_avr = avr_std[:3]
channel_std = avr_std[3:]

normed_torch_img = transforms.Normalize(channel_avr, channel_std)(torch_img)[None]

dense_plus = DenseNet169_plus()
dense_plus.load_state_dict(torch.load("0329_DenseNet169_plus_cv0_best_epoch13.pth",
                                      map_location=device))

# config = dict(model_type='densenet', arch=dense_plus, layer_name='densenet_features_norm5')
# config = dict(model_type='densenet', arch=dense_plus, layer_name='densenet_features_transition3_norm')
config = dict(model_type='densenet', arch=dense_plus, layer_name='densenet_features_denseblock4_denselayer1_conv1')

config['arch'].to(device).eval()

cams = [[cls.from_config(**config) for cls in (GradCAM, GradCAMpp)]]

images = []
for gradcam, gradcam_pp in cams:
    mask, _ = gradcam(normed_torch_img)
    heatmap, result = visualize_cam(mask, torch_img)

    mask_pp, _ = gradcam_pp(normed_torch_img)
    heatmap_pp, result_pp = visualize_cam(mask_pp, torch_img)

    images.extend([torch_img.cpu(), heatmap, result, heatmap_pp, result_pp])

grid_image = make_grid(images, nrow=5)

transforms.ToPILImage()(grid_image).show()
save_image(grid_image, "figure/fig_4_dense4_conv1.png")

# npgrid = grid_image.cpu().numpy()
#
# plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')
#
# ax = plt.gca()
# ax.xaxis.set_visible(False)
# ax.yaxis.set_visible(False)
#
# plt.title("origin                GradCAM                   GradCAM++     ")

# plt.savefig(export_img, bbox_inches='tight', pad_inches=0.1)
# plt.clf()
