from torchvision.transforms import InterpolationMode
from argparse import Namespace
import argparse

import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from torch.autograd import Variable

# from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import time
import os
# import random
# import copy
# import cv2
# import glob
# import math
# import sys

# from model_repurpose import FewShotCNN
# from labeller import Labeller
from utils_repurpose import tensor2image, imshow, horizontal_concat, imsave
from perceptual_model import VGG16_for_Perceptual
from stylegan2 import Generator


################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("--generator_domain1_dir", type=str,
                    help='path to the domain1 generator model')
parser.add_argument("--generator_domain2_dir", type=str,
                    help='path to the domain2 generator model')
parser.add_argument("--save_path_root", type=str,
                    help='path to save the results')
parser.add_argument("--source_image", type=str, help="Source image file")
parser.add_argument("--source_mask", type=str)
parser.add_argument("--target_image", type=str, help="Target image file")
parser.add_argument("--target_mask", type=str)
parser.add_argument("--image_size", type=int, default=256, help='image size')
parser.add_argument("--n_samples", type=int, default=1,
                    help='number of training samples')
parser.add_argument("--imshow_size", type=int,
                    default=3, help='image show size')
parser.add_argument("--latent_dim", type=int,
                    default=512, help='latent dimension')
parser.add_argument("--truncation", type=float, default=0.7, help='truncation')
parser.add_argument("--n_test", type=int, default=1,
                    help='number of test images')
parser.add_argument("--id_dir", type=str, default=1, help='id to save files')
parser.add_argument("--sample_z_path", type=str,
                    help='path to load the sample_z')
parser.add_argument("--save_iterations", type=bool,
                    help='whether to save iterations')
parser.add_argument("--mask_guided_iterations", type=int,
                    default=1002, help='number of the iterations')
parser.add_argument("--lr", type=float, default=0.01, help='learning rate')
parser.add_argument("--n_mean_latent", type=int,
                    default=10000, help='n_mean_latent')

################################################################################

args = parser.parse_args()

generator_path = args.generator_domain1_dir
target_model_path = args.generator_domain2_dir
save_path_root = args.save_path_root
image_size = args.image_size
n_samples = args.n_samples
imshow_size = args.imshow_size
latent_dim = args.latent_dim
truncation = args.truncation
n_test = args.n_test
id_dir = args.id_dir
sample_z_path = args.sample_z_path
save_iterations = args.save_iterations
mask_guided_iterations = args.mask_guided_iterations
lr = args.lr
n_mean_latent = args.n_mean_latent

###############################################
save_path = save_path_root+str(id_dir)+"/"

if sample_z_path:
    save_path = save_path_root+sample_z_path.split('/')[-1].split('.')[0]+"/"

if not os.path.exists(save_path):
    os.makedirs(save_path)
#############################################

if save_iterations:
    iterations_path = save_path+'iterations/'

    if not os.path.exists(iterations_path):
        os.makedirs(iterations_path)
#############################################


device = 'cuda:0'

generator = Generator(image_size, latent_dim, 8)
generator_ckpt = torch.load(generator_path, map_location='cpu')
generator.load_state_dict(generator_ckpt["g_ema"], strict=False)
generator.eval().to(device)
print(f'[StyleGAN2 generator loaded] {generator_path}\n')

classes = ['background', 'semantic_part']

# 1.0 This whole part generates mainly calculate dimension of the latent space
with torch.no_grad():
    trunc_mean = generator.mean_latent(4096).detach().clone()
    latent = generator.get_latent(torch.randn(
        n_samples, latent_dim, device=device))
    imgs_gen, features = generator([latent],
                                   truncation=truncation,
                                   truncation_latent=trunc_mean,
                                   input_is_latent=True,
                                   randomize_noise=True)
    torch.cuda.empty_cache()


@torch.no_grad()
def concat_features(features):
    h = max([f.shape[-2] for f in features])
    w = max([f.shape[-1] for f in features])
    return torch.cat([torch.nn.functional.interpolate(f, (h, w), mode='nearest') for f in features], dim=1)


data = dict(
    features=concat_features(features).cpu(),
)
print(' data[features].shape[1] is: {}'.format(
    data['features'].shape[1]))  # 5376


print('Start Testing')
device = 'cuda'
generator.eval().to(device)
# 1.9 No need for segmentation model anymore
# print('loading the model')
# net = FewShotCNN(data['features'].shape[1], len(classes), size='S')
# net.load_state_dict(torch.load(
#     save_segmentation_model_path+segmentation_part+'.pt'))
# net.eval().to('cpu')
# print('model is loaded')


###############################################################################
# Helper codes ################################################################
###############################################################################


class Convert2Uint8(torch.nn.Module):
    '''
    Resize input when the target dim is not divisible by the input dim
    '''

    def __init__(self):
        super().__init__()

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        img = torch.round(torch.mul(img, 255))
        return img


class ToOneHot(torch.nn.Module):
    '''
    Convert input to one-hot encoding
    '''

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, img):
        """
        Args:
            img (Tensor): Image to be scaled of shape (1, h, w).

        Returns:
            Tensor: Rescaled image.
        """
        img = img.long()[0]
        # img = torch.nn.functional.one_hot(img, num_classes=self.num_classes)
        img = img.permute(2, 0, 1)
        return img


class MapVal(torch.nn.Module):
    '''
    Map a list of value to another
    '''

    def __init__(self, src_vals, dst_vals):
        super().__init__()
        assert len(src_vals) == len(
            dst_vals), "src_vals and dst_vals must of equal length"
        self.src_vals = src_vals
        self.dst_vals = dst_vals

    def forward(self, img):
        for s, d in zip(self.src_vals, self.dst_vals):
            img[img == s] = d
        return img


def get_transforms(opts):
    transforms_dict = {
        'transform_gt_train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * opts.output_nc, [0.5] * opts.output_nc)]),
        'transform_source': transforms.Compose([
            transforms.Resize(
                (256, 256), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor(),
            Convert2Uint8(),
            MapVal(opts.src_vals, opts.dst_vals),
            # ToOneHot(opts.label_nc)
        ])
    }
    return transforms_dict


num_gap = 12 + 1  # self-defined
gap = 255 // num_gap
ILM = 1 * gap  # present in 1, 2, 3, 4
RNFL_o = 2 * gap  # NFL/FCL in DME, present in 2
IPL_INL = 3 * gap
INL_OPL = 4 * gap
OPL_o = 5 * gap  # OPL/ONL in DME
ISM_ISE = 6 * gap
IS_OS = 7 * gap
OS_RPE = 8 * gap
# not sure whether they are the same
RPE = 9 * gap
# RPEDC = 10 * gap
# RPE = 11 * gap

BM = 10 * gap

AROI_LABELS = [ILM, IPL_INL, RPE, BM]  # [19, 57, 171, 190]
FLUID_LABELS = [80, 160, 240]
OP_LABELS = [ILM, RPE]
INSTRUMENT_LABELS = [100, 200]
op_map = [1, 3]
fluid_map = [0, 0, 0]
instrument_map = [5, 6]

transform_dict = get_transforms(Namespace(
    output_nc=1, label_nc=2, src_vals=[2, 3, 4]+AROI_LABELS+INSTRUMENT_LABELS+FLUID_LABELS,
    dst_vals=[5, 2, 6]+[1, 2, 3, 4]+instrument_map+[0, 0, 0]))
bscan_transform = transform_dict['transform_gt_train']
label_transform = transform_dict['transform_source']


def load_bscan(bscan_path):
    """Load and add a batch dimension
    """
    image = Image.open(bscan_path).convert('RGB')
    image = bscan_transform(image)
    return image


def load_label(label_path):
    """Load and add a batch dimension
    """
    image = Image.open(label_path).convert('L')
    image = label_transform(image)
    return image


###############################################################################
# Helper codes end ############################################################
###############################################################################

with torch.no_grad():
    # 1.1 Generate or load a random latent vector
    # sample_z = torch.randn(n_test, latent_dim, device=device)

    # if sample_z_path:
    #     sample_z = torch.load(sample_z_path)

    # torch.save(sample_z, save_path+"sample_z.pt")
    # 1.2 Create or reconstruct images from the latent vector
    # imgs_gen, features = generator([sample_z.to(device)],
    #                                truncation=truncation,
    #                                truncation_latent=trunc_mean.to(device),
    #                                input_is_latent=False,
    #                                randomize_noise=False)
    # torch.cuda.empty_cache()

    # 2.1 Load the source image
    # Here we assume n_test = 1
    imgs_gen = load_bscan(args.source_image)  # source is iOCT

    # 1.3 Create masks from the features
    # source_features_tens = [torch.tensor(f, device='cpu') for f in features]
    # out = net(concat_features(source_features_tens))
    # predictions = out.data.max(1)[1].cpu().numpy()

    # masks = np.zeros((n_test, image_size, image_size, len(classes)))

    # 2.2 Load the source mask
    masks = load_label(args.source_mask)  # assume to be [1, h, w]

    # for i in range(n_test):
    #     for c in range(len(classes)):
    #         masks[i, :, :, c] = (predictions[i, :, :] == c)

    # 1.4 Load target generator
    targ_generator = Generator(image_size, latent_dim, 8).to(device)
    targ_generator = nn.parallel.DataParallel(targ_generator)

    targ_generator_ckpt = torch.load(target_model_path)
    targ_generator.load_state_dict(targ_generator_ckpt["g_ema"], strict=False)
    targ_generator.eval().to(device)
    print(
        f'[StyleGAN2 generator for target style loaded] {target_model_path}\n')

    ########################################################

    targ_truncation = float(1)
    targ_mean_latent = None
    # 1.5 Generate a target image with the same latent
    # targ_imgs_gen, targ_features = targ_generator([sample_z.to(device)],
    #                                               truncation=targ_truncation,
    #                                               truncation_latent=targ_mean_latent,
    #                                               input_is_latent=False,
    #                                               randomize_noise=False)

    # torch.cuda.empty_cache()

    # 2.3 Load the target image
    targ_imgs_gen = load_bscan(args.target_image)  # target is OCT

    # targ_features_tens = [torch.tensor(t, device='cpu') for t in targ_features]
    # target_out = net(concat_features(targ_features_tens))
    # targ_predictions = target_out.data.max(1)[1].cpu().numpy()

    # targ_masks = np.zeros((n_test, image_size, image_size, len(classes)))

    # for i in range(n_test):
    #     for c in range(len(classes)):
    #         targ_masks[i, :, :, c] = (targ_predictions[i, :, :] == c)

    # 2.4 Load the target mask
    targ_masks = load_label(args.target_mask)

    #####################################################
    # 1.6 Save created images
    # img_name = save_path+"org_source.png"
    # img_tens = (imgs_gen.clamp_(-1., 1.).detach().squeeze().permute(1,
    #             2, 0).cpu().numpy())*0.5 + 0.5
    # pil_img = Image.fromarray((img_tens*255).astype(np.uint8))
    # pil_img.save(img_name)

    # img_name = save_path+"org_target.png"
    # img_tens = (targ_imgs_gen.clamp_(-1.,
    #             1.).detach().squeeze().permute(1, 2, 0).cpu().numpy())*0.5 + 0.5
    # pil_img = Image.fromarray((img_tens*255).astype(np.uint8))
    # pil_img.save(img_name)

#################################################################################################


def caluclate_loss(synth_img, img, perceptual_net, mask, MSE_Loss, image_resolution):

    img_p = torch.nn.Upsample(scale_factor=(
        256/image_resolution), mode='bilinear')(img)
    real_0, real_1, real_2, real_3 = perceptual_net(img_p)
    synth_p = torch.nn.Upsample(scale_factor=(
        256/image_resolution), mode='bilinear')(synth_img)
    synth_0, synth_1, synth_2, synth_3 = perceptual_net(synth_p)

    perceptual_loss = 0
    mask = torch.nn.Upsample(scale_factor=(
        256/image_resolution), mode='bilinear')(mask)
    perceptual_loss += MSE_Loss(synth_0*mask.expand(1,
                                64, 256, 256), real_0*mask.expand(1, 64, 256, 256))
    perceptual_loss += MSE_Loss(synth_1*mask.expand(1,
                                64, 256, 256), real_1*mask.expand(1, 64, 256, 256))
    mask = torch.nn.Upsample(scale_factor=(64/256), mode='bilinear')(mask)
    perceptual_loss += MSE_Loss(synth_2*mask.expand(1,
                                256, 64, 64), real_2*mask.expand(1, 256, 64, 64))
    mask = torch.nn.Upsample(scale_factor=(32/64), mode='bilinear')(mask)
    perceptual_loss += MSE_Loss(synth_3*mask.expand(1,
                                512, 32, 32), real_3*mask.expand(1, 512, 32, 32))

    return perceptual_loss

#################################################################################################


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)
#################################################################################################


torch.cuda.empty_cache()
image_resolution = image_size
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#################################################################################################

# 1.7 Transform the source and target images and masks
# transform = transforms.Compose([transforms.ToTensor()])
# TODO: Check the dimension, set to 1 or 3?
img_source = imgs_gen[:, :, :].unsqueeze(0).to(
    device)  # (1,3,image_size,image_size) (1,3,256,256)
img_target = targ_imgs_gen[:, :, :].unsqueeze(0).to(
    device)  # (1,3,image_size,image_size) (1,3,256,256)
# # (1,3,image_size,image_size) (1,3,256,256)
# mask = transforms.ToTensor()(masks[0, :, :, -1]).unsqueeze(0).to(device)
# # (1,3,image_size,image_size) (1,3,256,256)
# targ_mask = transforms.ToTensor()(
#     targ_masks[0, :, :, -1]).unsqueeze(0).to(device)

# 2.5 Expand masks


def horizontal_expand(label, feature, to_expand=20):
    if isinstance(label, torch.Tensor):
        label_copy = label.clone()
    else:
        label_copy = label.copy()
    x, y = np.where(label_copy == feature)
    xacc, yacc = x, y
    for i in range(to_expand):
        xacc = np.concatenate([xacc, x])
        yacc = np.concatenate([yacc, y+i])
    label_copy[xacc, yacc] = feature
    return label_copy


def expand_label(label, instrument_label=2, shadow_label=4, expansion_instrument=30,
                 expansion_shadow=60):
    """The input size is expected to be [h, w]
    """
    # For label 2 4 (instrument & its mirroring), we horizontally expand
    # a couple of pixels rightward
    label = horizontal_expand(label, instrument_label,
                              to_expand=expansion_instrument)
    # shadows are generally broader
    label = horizontal_expand(label, shadow_label, to_expand=expansion_shadow)
    return label


def get_shadow(label, instrument_label=2, shadow_label=4, top_layer_label=1, img_width=256, img_height=256):
    shadow_x = np.array([], dtype=np.int64)
    shadow_y = np.array([], dtype=np.int64)
    # Requirements for the shadow label:
    # 1. Horizontally after the starting of the instrument/mirroring & before the
    #    ending of the instrument/mirroring
    # 2. Vertically below the lower bound of instrument/mirroring
    x, y = np.where(np.logical_or(label==instrument_label, label==shadow_label)) # (1024, 512)
    if len(x) == 0:
        return shadow_x, shadow_y
    left_bound = np.min(y)
    right_bound = np.max(y)
    accumulated_min_lowerbound = 0
    for i in range(left_bound, right_bound):
        instrument_above = np.where(np.logical_or(label[:, i] == instrument_label, label[:, i] == shadow_label))[0]
        if len(instrument_above) == 0:
            if accumulated_min_lowerbound == 0:
                continue
            else:
                # set to current recorded lowest shadow
                instrument_lowerbound = accumulated_min_lowerbound
        else:
            # print("instrument_above", instrument_above, len(instrument_above))
            instrument_lowerbound = np.max(instrument_above)
            if accumulated_min_lowerbound == 0:
                # initialize
                accumulated_min_lowerbound = instrument_lowerbound
            else:
                accumulated_min_lowerbound = max(accumulated_min_lowerbound, instrument_lowerbound)
        x_vertical = np.arange(instrument_lowerbound, img_height) # upperbound to bottom
        y_vertical = np.full_like(x_vertical, i)
        shadow_x = np.concatenate([shadow_x, x_vertical])
        shadow_y = np.concatenate([shadow_y, y_vertical])
    return shadow_x, shadow_y


# 2.5.1 expand instruments and shadows in source
mask = expand_label(masks[0, :, :], instrument_label=5, shadow_label=6,
                    expansion_instrument=15, expansion_shadow=15)  # (256, 256)
# 2.5.2 Select classes of interest (instrument, its mirroring and the shadow below)
classes_of_interest = [5, 6]
mask_copy = np.zeros_like(mask)
for c in classes_of_interest:
    mask_copy[mask == c] = 1
# 2.5.3 Get the shadow and set to intrested
shadow_x, shadow_y = get_shadow(
    mask, instrument_label=5, shadow_label=6, top_layer_label=1)
mask_copy[shadow_x, shadow_y] = 1
mask = torch.as_tensor(mask_copy)

# (1,1,image_resolution,image_resolution)
# mask_0 = mask[:, 0, :, :].unsqueeze(0)
# 2.6 set mask 0
mask_0 = mask.unsqueeze(0)
mask_1 = mask_0.clone()
mask_1 = 1-(mask_1)  # (1,1,image_resolution,image_resolution)

# 2.7 Set target mask to the same as mask
# targ_mask_0 = targ_mask[:, 0, :, :].unsqueeze(
#     0)  # (1,1,image_resolution,image_resolution)
# targ_mask_1 = targ_mask_0.clone()
# targ_mask_1 = 1-(targ_mask_1)  # (1,1,image_resolution,image_resolution)
targ_mask_0 = mask_0.clone()
targ_mask_1 = mask_1.clone()

#################################################################################################
mask_0 = mask_0.to(device)
mask_1 = mask_1.to(device)
targ_mask_0 = targ_mask_0.to(device)
targ_mask_1 = targ_mask_1.to(device)
cross_over_source = (img_source*mask_1)+(img_target*targ_mask_0)
cross_over_source_image = tensor2image(cross_over_source.to('cpu'))  # [256, 256, n_channels(1)]

cross_over_target = (img_source*mask_0)+(img_target*targ_mask_1)
cross_over_target_image = tensor2image(cross_over_target.to('cpu'))

img_name = save_path+"naive_crossover_source.png"
img_tens = (cross_over_source.clamp_(-1.,
            1.).detach().squeeze(0).permute(1, 2, 0).cpu().numpy())*0.5 + 0.5
if img_tens.shape[-1] == 3:
    pil_format = 'RGB'
else:
    pil_format = 'L'
    img_tens = img_tens[:, :, 0]
pil_img = Image.fromarray((img_tens*255).astype(np.uint8), mode=pil_format)
pil_img.save(img_name)

img_name = save_path+"naive_crossover_target.png"
img_tens = (cross_over_target.clamp_(-1.,
            1.).detach().squeeze(0).permute(1, 2, 0).cpu().numpy())*0.5 + 0.5
if img_tens.shape[-1] == 3:
    pil_format = 'RGB'
else:
    pil_format = 'L'
    img_tens = img_tens[:, :, 0]
pil_img = Image.fromarray((img_tens*255).astype(np.uint8), mode=pil_format)
pil_img.save(img_name)

img_name = save_path+"mask_source_0.png"
img_tens = mask_0[0, :, :].cpu().numpy()
pil_img = Image.fromarray((img_tens*255).astype(np.uint8))
pil_img.save(img_name)

img_name = save_path+"mask_source_1.png"
img_tens = mask_1[0, :, :].cpu().numpy()
pil_img = Image.fromarray((img_tens*255).astype(np.uint8))
pil_img.save(img_name)

img_name = save_path+"mask_target_0.png"
img_tens = targ_mask_0[0, :, :].cpu().numpy()
pil_img = Image.fromarray((img_tens*255).astype(np.uint8))
pil_img.save(img_name)

img_name = save_path+"mask_target_1.png"
img_tens = targ_mask_1[0, :, :].cpu().numpy()
pil_img = Image.fromarray((img_tens*255).astype(np.uint8))
pil_img.save(img_name)

print("naive_crossover_source_target images are saved")

###############################################################################

g_ema = generator
with torch.no_grad():
    noise_sample = torch.randn(n_mean_latent, 512, device=device)
    latent_out = g_ema.style(noise_sample)
    # 1.7 Create a new latent as optimization starting point
    latent_mean = latent_out.mean(0)
    latent_std = ((latent_out - latent_mean).pow(2).sum() /
                  n_mean_latent) ** 0.5

print("latent works")
# 1.8 Create random noise
noises_single = g_ema.make_noise()
noises = []
for noise in noises_single:
    noises.append(noise.repeat(img_source.shape[0], 1, 1, 1).normal_())

latent_in = latent_mean.detach().clone().unsqueeze(
    0).repeat(img_source.shape[0], 1)
latent_in_1 = latent_in

latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)

# 1.9 Both latents and noise are optimized? (Only latent_in is in optimizer)
latent_in.requires_grad = True

for noise in noises:
    noise.requires_grad = True

perceptual_net = VGG16_for_Perceptual(n_layers=[2, 4, 14, 21]).to(
    device)  # conv1_1,conv1_2,conv2_2,conv3_3

MSE_Loss = nn.MSELoss(reduction="mean")
optimizer = optim.Adam([latent_in], lr=lr)

print("Start embeding mask on target and source images")
loss_list = []
latent_path = []

# TODO: manage size elsewhere
mask_1 = mask_1.unsqueeze(0)
mask_0 = mask_0.unsqueeze(0)

for i in range(mask_guided_iterations):
    t = i / mask_guided_iterations
    optimizer.param_groups[0]["lr"] = lr

    synth_img, _ = g_ema([latent_in], input_is_latent=True, noise=noises)

    batch, channel, height, width = synth_img.shape

    if height > image_size:
        factor = height // image_size

        synth_img = synth_img.reshape(
            batch, channel, height // factor, factor, width // factor, factor
        )
        synth_img = synth_img.mean([3, 5])

    loss_wl1 = caluclate_loss(synth_img, img_source,
                              perceptual_net, mask_1, MSE_Loss, image_size)
    loss_wl0 = caluclate_loss(synth_img, img_target,
                              perceptual_net, mask_0, MSE_Loss, image_size)
    mse_w0 = F.mse_loss(synth_img*mask_1.expand(1, 3, image_size, image_size),
                        img_source*mask_1.expand(1, 3, image_size, image_size))
    mse_w1 = F.mse_loss(synth_img*mask_0.expand(1, 3, image_size, image_size),
                        img_target*mask_0.expand(1, 3, image_size, image_size))
    mse_crossover = 3*(F.mse_loss(synth_img.float(),
                       cross_over_source.float()))
    p_loss = 2*((loss_wl0)+loss_wl1)
    mse_loss = (mse_w0)+mse_w1
    loss = (p_loss)+(mse_loss)+(mse_crossover)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    noise_normalize_(noises)

    lr_schedule = optimizer.param_groups[0]['lr']

    if (i + 1) % 100 == 0:
        latent_path.append(latent_in.detach().clone())

        loss_np = loss.detach().cpu().numpy()
        loss_0 = loss_wl0.detach().cpu().numpy()
        loss_1 = loss_wl1.detach().cpu().numpy()
        mse_0 = mse_w0.detach().cpu().numpy()
        mse_1 = mse_w1.detach().cpu().numpy()
        mse_loss = mse_loss.detach().cpu().numpy()

        print("iter{}: loss -- {},  loss0 --{},  loss1 --{}, mse0--{}, mse1--{}, mseTot--{}, lr--{}".format(i,
              loss_np, loss_0, loss_1, mse_0, mse_1, mse_loss, lr_schedule))

        if save_iterations:
            img_name = iterations_path+"{}_D1.png".format(str(i).zfill(6))
            img_gen, _ = g_ema([latent_path[-1]],
                               input_is_latent=True, noise=noises)
            img_tens = (
                img_gen.clamp_(-1., 1.).detach().squeeze().permute(1, 2, 0).cpu().numpy())*0.5 + 0.5
            pil_img = Image.fromarray((img_tens*255).astype(np.uint8))
            pil_img.save(img_name)

    if i == (mask_guided_iterations-1):
        img_name = save_path+"{}_D1.png".format(str(i).zfill(6))
        img_gen, _ = g_ema([latent_path[-1]],
                           input_is_latent=True, noise=noises)
        img_tens = (img_gen.clamp_(-1., 1.).detach().squeeze().permute(1,
                    2, 0).cpu().numpy())*0.5 + 0.5
        pil_img = Image.fromarray((img_tens*255).astype(np.uint8))
        pil_img.save(img_name)

##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
