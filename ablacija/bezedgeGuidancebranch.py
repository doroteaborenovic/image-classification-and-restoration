# ==============================================================================
# ABLACIJA 9: BEZ EDGE GUIDANCE GRANE (w/o Edge Guidance Branch)
# LOSS: Kompletan originalni RestauracijaLoss (Charb + SSIM + Sobel + FFT + CustomPerceptual + Color + Freq)
# Protokol: 25 Epoha (Sepia Dataset) + 5 Epoha Fine-Tuning (Trening Skup) + Eval
# ==============================================================================

import os
import sys
import zipfile
import random
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 1. LPIPS I TABULATE (za evaluaciju)
try:
    import lpips
except ImportError:
    os.system("pip install -q lpips tabulate")
    import lpips
from tabulate import tabulate

# 2. GOOGLE DRIVE I PUTANJE
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
except Exception:
    pass

DRIVE_PROJECT_DIR = '/content/drive/MyDrive/Projekat_Model'
os.makedirs(DRIVE_PROJECT_DIR, exist_ok=True)
DIR_ABLACIJA_CKPT = os.path.join(DRIVE_PROJECT_DIR, 'ablacija_checkpoints_customloss')
os.makedirs(DIR_ABLACIJA_CKPT, exist_ok=True)

LOCAL_SEPIA_DIR = '/content/dataset_sepia'
ZIP_SEPIA_PATH = os.path.join(DRIVE_PROJECT_DIR, 'dataset_sepia_1.zip')

# Raspakivanje Sepia dataseta
if not os.path.exists(LOCAL_SEPIA_DIR) or len(os.listdir(LOCAL_SEPIA_DIR)) == 0:
    if os.path.exists(ZIP_SEPIA_PATH):
        print(f"[INFO] Raspakujem {ZIP_SEPIA_PATH} u {LOCAL_SEPIA_DIR}...")
        with zipfile.ZipFile(ZIP_SEPIA_PATH, 'r') as zip_ref:
            zip_ref.extractall(LOCAL_SEPIA_DIR)
        print("[INFO] Sepia dataset uspešno raspakovan.")
    else:
        raise FileNotFoundError(f"[GREŠKA] Fajl {ZIP_SEPIA_PATH} nije pronađen na Drive-u!")

# Pronalaženje glavnih foldera
def pronadji_glavne_foldere(tip="TRENING"):
    moguce = [
        f"/content/drive/MyDrive/Projekat_Model/dataset/{tip}",
        f"/content/drive/MyDrive/Projekat_Model/dataset_njihov/{tip}_NJIHOV" if tip == "TRENING" else f"/content/drive/MyDrive/Projekat_Model/dataset_njihov/VALIDACIJA_NJIHOVA",
        f"./dataset/{tip}",
        f"/content/{tip}"
    ]
    for b in moguce:
        if os.path.exists(b):
            c = os.path.join(b, "clean")
            d = os.path.join(b, "degraded")
            if os.path.exists(c) and os.path.exists(d) and len(os.listdir(d)) > 0:
                return c, d
    raise FileNotFoundError(f"[GREŠKA] Nije pronađen glavni folder za {tip}")

DIR_TRAIN_CLEAN, DIR_TRAIN_DEGRADED = pronadji_glavne_foldere("TRENING")
DIR_VAL_CLEAN, DIR_VAL_DEGRADED = pronadji_glavne_foldere("VALIDACIJA")

# Pronalaženje Sepia foldera (/train/0 i /train/1)
def pronadji_sepia_foldere(base_dir, fallback_clean_dir):
    for root, dirs, _ in os.walk(base_dir):
        if 'clean' in dirs and 'degraded' in dirs:
            c, d = os.path.join(root, 'clean'), os.path.join(root, 'degraded')
            if len(os.listdir(d)) > 0:
                return c, d
    train_dir = os.path.join(base_dir, 'train')
    if os.path.exists(train_dir):
        subdirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        if '0' in subdirs and '1' in subdirs:
            return os.path.join(train_dir, '1'), os.path.join(train_dir, '0')
        if '0' in subdirs:
            return fallback_clean_dir, os.path.join(train_dir, '0')
    d_0 = os.path.join(base_dir, 'train', '0')
    if os.path.exists(d_0) and len(os.listdir(d_0)) > 0:
        return fallback_clean_dir, d_0

    raise FileNotFoundError(f"[GREŠKA] Nije moguće mapirati sepia strukturu u {base_dir}")

DIR_SEPIA_CLEAN, DIR_SEPIA_DEGRADED = pronadji_sepia_foldere(LOCAL_SEPIA_DIR, DIR_TRAIN_CLEAN)
print(f"[INFO] Sepia Clean: {DIR_SEPIA_CLEAN}")
print(f"[INFO] Sepia Degraded: {DIR_SEPIA_DEGRADED}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eval_lpips_fn = lpips.LPIPS(net='alex', verbose=False).to(device).eval()

BATCH_SIZE = 4
LR_PRETRAIN = 2e-4
LR_FINETUNE = 5e-5
IMG_SIZE = 256
EPOCHS_SEPIA = 25
EPOCHS_FT = 5

# ==============================================================================
# DATASET
# ==============================================================================
class PairedDataset(Dataset):
    def __init__(self, clean_dir, degraded_dir, img_size=256, train=False):
        self.clean_dir = clean_dir
        self.degraded_dir = degraded_dir
        d_files = set(os.listdir(degraded_dir))
        c_files = set(os.listdir(clean_dir))
        self.files = sorted([f for f in (d_files & c_files) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if len(self.files) == 0:
            self.files = sorted([f for f in os.listdir(degraded_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.img_size = img_size
        self.train = train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        c_p = os.path.join(self.clean_dir, fname)
        d_p = os.path.join(self.degraded_dir, fname)

        if not os.path.exists(c_p):
            c_p = os.path.join(self.clean_dir, os.listdir(self.clean_dir)[idx % len(os.listdir(self.clean_dir))])

        c_img = cv2.resize(cv2.cvtColor(cv2.imread(c_p), cv2.COLOR_BGR2RGB), (self.img_size, self.img_size))
        d_img = cv2.resize(cv2.cvtColor(cv2.imread(d_p), cv2.COLOR_BGR2RGB), (self.img_size, self.img_size))

        c_t = torch.from_numpy(c_img).permute(2, 0, 1).float() / 255.0
        d_t = torch.from_numpy(d_img).permute(2, 0, 1).float() / 255.0

        if self.train:
            if random.random() > 0.5:
                c_t, d_t = torch.flip(c_t, dims=[2]), torch.flip(d_t, dims=[2])
            if random.random() > 0.5:
                c_t, d_t = torch.flip(c_t, dims=[1]), torch.flip(d_t, dims=[1])

        return d_t, c_t, fname

# ==============================================================================
# VAŠI ORIGINALNI GUBICI (RESTAURACIJA LOSS)
# ==============================================================================
def gaussian(window_size: int, sigma: float) -> Tensor:
    gauss = torch.tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11):
        super().__init__()
        self.window_size = window_size
        self.channel = 3
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        self.register_buffer('window', _2D_window.expand(self.channel, 1, window_size, window_size).contiguous())

    def forward(self, img1: Tensor, img2: Tensor) -> Tensor:
        _, channel, _, _ = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            _1D_window = gaussian(self.window_size, 1.5).unsqueeze(1)
            _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
            window = _2D_window.expand(channel, 1, self.window_size, self.window_size).to(img1.device)

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)
        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-6)
        return 1.0 - ssim_map.mean()

class SobelLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)
        ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).unsqueeze(0).unsqueeze(0)
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        p_gray = torch.mean(pred, dim=1, keepdim=True)
        t_gray = torch.mean(target, dim=1, keepdim=True)
        gx_p = F.conv2d(p_gray, self.kx, padding=1)
        gy_p = F.conv2d(p_gray, self.ky, padding=1)
        gx_t = F.conv2d(t_gray, self.kx, padding=1)
        gy_t = F.conv2d(t_gray, self.ky, padding=1)
        return F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)

class ColorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pred_blur = F.avg_pool2d(pred, kernel_size=5, stride=1, padding=2)
        target_blur = F.avg_pool2d(target, kernel_size=5, stride=1, padding=2)
        return F.l1_loss(pred_blur, target_blur)

class CustomPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.scales = [1, 2, 4]
        dx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)
        dy = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).unsqueeze(0).unsqueeze(0)
        lap = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).unsqueeze(0).unsqueeze(0)
        self.register_buffer('dx', dx.repeat(3, 1, 1, 1))
        self.register_buffer('dy', dy.repeat(3, 1, 1, 1))
        self.register_buffer('lap', lap.repeat(3, 1, 1, 1))

    def extract_features(self, x: Tensor) -> list[Tensor]:
        feats = []
        for s in self.scales:
            x_scaled = F.interpolate(x, scale_factor=1.0 / s, mode='bilinear', align_corners=False) if s > 1 else x
            fx = F.conv2d(x_scaled, self.dx, padding=1, groups=3)
            fy = F.conv2d(x_scaled, self.dy, padding=1, groups=3)
            flap = F.conv2d(x_scaled, self.lap, padding=1, groups=3)
            feats.extend([fx, fy, flap])
        return feats

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pred_feats = self.extract_features(pred)
        target_feats = self.extract_features(target)
        loss = 0.0
        for pf, tf in zip(pred_feats, target_feats):
            loss += F.l1_loss(pf, tf)
        return loss / len(pred_feats)

class SoftHardExampleMiningLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        error_map = torch.abs(pred - target).mean(dim=1, keepdim=True).detach()
        return 1.0 + torch.sigmoid((error_map - error_map.mean()) / (error_map.std() + 1e-6))

class FrequencyConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pred_low = F.avg_pool2d(pred, kernel_size=5, stride=1, padding=2)
        tgt_low = F.avg_pool2d(target, kernel_size=5, stride=1, padding=2)
        pred_high = pred - pred_low
        tgt_high = target - tgt_low
        return 0.4 * F.l1_loss(pred_low, tgt_low) + 0.6 * F.l1_loss(pred_high, tgt_high)

class RestauracijaLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.sobel = SobelLoss()
        self.ssim = SSIMLoss()
        self.perceptual = CustomPerceptualLoss()
        self.soft_hem = SoftHardExampleMiningLoss()
        self.freq_consistency = FrequencyConsistencyLoss()
        self.color_loss_fn = ColorLoss()

    def single_scale_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        hem_weight = self.soft_hem(pred, target)
        char_diff = torch.sqrt((pred - target) ** 2 + self.eps)
        char_loss = torch.mean(char_diff * hem_weight)

        ssim_loss = self.ssim(pred, target)
        edge_loss = self.sobel(pred, target)
        percep_loss = self.perceptual(pred, target)
        freq_loss = self.freq_consistency(pred, target)
        color_loss = self.color_loss_fn(pred, target)

        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')
        fft_loss = F.l1_loss(torch.real(pred_fft), torch.real(target_fft)) + \
                   F.l1_loss(torch.imag(pred_fft), torch.imag(target_fft))

        return (
            0.40 * char_loss +
            0.10 * ssim_loss +
            0.10 * edge_loss +
            0.15 * fft_loss +
            0.10 * percep_loss +
            0.10 * color_loss +
            0.05 * freq_loss
        )

    def multi_scale_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        loss = self.single_scale_loss(pred, target)
        for scale in [0.5, 0.25]:
            p = F.interpolate(pred, scale_factor=scale, mode='bilinear', align_corners=False)
            t = F.interpolate(target, scale_factor=scale, mode='bilinear', align_corners=False)
            loss += scale * torch.mean(torch.sqrt((p - t) ** 2 + self.eps))
        return loss

    def forward(self, pred_outputs: Tensor | dict[str, Tensor], target: Tensor) -> Tensor:
        if isinstance(pred_outputs, dict):
            total_loss = self.multi_scale_loss(pred_outputs['out'], target)
            if 'aux3' in pred_outputs:
                total_loss += 0.3 * torch.mean(torch.sqrt((pred_outputs['aux3'] - target) ** 2 + self.eps))
            if 'aux2' in pred_outputs:
                total_loss += 0.15 * torch.mean(torch.sqrt((pred_outputs['aux2'] - target) ** 2 + self.eps))
            return total_loss
        return self.multi_scale_loss(pred_outputs, target)


# ==============================================================================
# ARHITEKTURA MODELA (BEZ EDGE GUIDANCE GRANE)
# ==============================================================================
class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1, dilation: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(x))

class RecursiveDenseRestorationBlock(nn.Module):
    def __init__(self, channels: int, num_recursions: int = 3):
        super().__init__()
        self.num_recursions = num_recursions
        self.conv = DepthwiseSeparableConv2d(channels, channels, 3, padding=1)
        self.gn = nn.GroupNorm(4, channels)
        self.fusion = nn.Conv2d(channels * num_recursions, channels, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        outputs = []
        out = x
        for _ in range(self.num_recursions):
            out = F.relu(self.gn(self.conv(out)) + x)
            outputs.append(out)
        return self.fusion(torch.cat(outputs, dim=1))

class SpectralDecompositionRestorationBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.low_conv = nn.Sequential(
            DepthwiseSeparableConv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(4, channels),
            nn.ReLU(inplace=False)
        )
        self.high_conv = nn.Sequential(
            DepthwiseSeparableConv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(4, channels),
            nn.ReLU(inplace=False)
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, 2, 1),
            nn.Softmax(dim=1)
        )
        self.fuse = nn.Conv2d(channels * 2, channels, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        low = F.interpolate(F.avg_pool2d(x, kernel_size=2), size=x.shape[2:], mode='bilinear', align_corners=False)
        high = x - low
        low_feat = self.low_conv(low)
        high_feat = self.high_conv(high)
        w = self.gate(torch.cat([low_feat, high_feat], dim=1))
        fused = w[:, 0:1] * low_feat + w[:, 1:2] * high_feat
        return self.fuse(torch.cat([fused, x], dim=1))

class SpatialEncoderRestorationBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(4, out_ch),
            nn.ReLU(inplace=False)
        )
        self.dense_micro = RecursiveDenseRestorationBlock(out_ch, num_recursions=3)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.conv(x)
        x = self.dense_micro(x)
        return self.pool(x), x

class AsymmetricCrossBridgeRestoration(nn.Module):
    def __init__(self, spatial_ch: int, spectral_ch: int, out_ch: int):
        super().__init__()
        self.spatial_to_spectral = nn.Sequential(
            nn.Conv2d(spatial_ch, spectral_ch, 1, bias=False),
            nn.GroupNorm(4, spectral_ch),
            nn.ReLU(inplace=False)
        )
        self.spectral_to_spatial = nn.Sequential(
            nn.Conv2d(spectral_ch, spatial_ch, 1, bias=False),
            nn.GroupNorm(4, spatial_ch),
            nn.ReLU(inplace=False)
        )
        self.fuse = nn.Conv2d(spatial_ch + spectral_ch, out_ch, 1, bias=False)

    def forward(self, spatial_feat: Tensor, spectral_feat: Tensor) -> Tensor:
        s_enh = spectral_feat + self.spatial_to_spectral(F.adaptive_avg_pool2d(spatial_feat, spatial_feat.shape[2:]))
        sp_enh = spatial_feat + self.spectral_to_spatial(F.interpolate(spectral_feat, size=spatial_feat.shape[2:], mode='bilinear', align_corners=False))
        min_h = min(spatial_feat.shape[2], spectral_feat.shape[2])
        min_w = min(spatial_feat.shape[3], spectral_feat.shape[3])
        return self.fuse(torch.cat([F.adaptive_avg_pool2d(sp_enh, (min_h, min_w)), F.adaptive_avg_pool2d(s_enh, (min_h, min_w))], dim=1))

class GatedFusionRestorationBlock(nn.Module):
    def __init__(self, spatial_ch: int, spectral_ch: int, out_ch: int):
        super().__init__()
        self.spatial_proj = nn.Conv2d(spatial_ch, out_ch, 1, bias=False)
        self.spectral_proj = nn.Conv2d(spectral_ch, out_ch, 1, bias=False)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_ch * 2, out_ch // 4, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(out_ch // 4, out_ch * 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, spatial: Tensor, spectral: Tensor) -> Tensor:
        s = self.spatial_proj(spatial)
        sp = self.spectral_proj(F.interpolate(spectral, size=spatial.shape[2:], mode='bilinear', align_corners=False))
        gates = self.gate(torch.cat([s, sp], dim=1)).view(s.shape[0], -1, 1, 1)
        out_ch = s.shape[1]
        return gates[:, :out_ch] * s + gates[:, out_ch:] * sp

class DamageAttentionRestorationModule(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1, bias=False),
            nn.GroupNorm(4, in_channels // 4),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )
        self.refine = nn.Sequential(
            DepthwiseSeparableConv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(4, in_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        attn = self.attention(x)
        return self.refine(x * attn) + x, attn

class DecoderRestorationBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, in_ch // 2, kernel_size=3, padding=1, bias=False)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch // 2 + skip_ch + 1, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(4, out_ch),
            nn.ReLU(inplace=False)
        )
        self.dense_micro = RecursiveDenseRestorationBlock(out_ch, num_recursions=2)
        self.spectral = SpectralDecompositionRestorationBlock(out_ch)

    def forward(self, x: Tensor, skip: Tensor, damage_map: Tensor) -> Tensor:
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        dm = F.interpolate(damage_map, size=skip.shape[2:], mode='bilinear', align_corners=False)
        feat = self.dense_micro(self.conv(torch.cat([x, skip, dm], dim=1)))
        return self.spectral(feat)

class DilatedContextBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        mid = channels // 4
        self.c1 = nn.Conv2d(channels, mid, 3, padding=1, dilation=1, bias=False)
        self.c2 = nn.Conv2d(channels, mid, 3, padding=2, dilation=2, bias=False)
        self.c3 = nn.Conv2d(channels, mid, 3, padding=4, dilation=4, bias=False)
        self.c4 = nn.Conv2d(channels, mid, 3, padding=8, dilation=8, bias=False)
        self.fusion = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.GroupNorm(4, channels)

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.bn(self.fusion(torch.cat([self.c1(x), self.c2(x), self.c3(x), self.c4(x)], dim=1))) + x)

class GatedSkipConnection(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, skip: Tensor) -> Tensor:
        return skip * self.gate(skip)

class ContrastColorRecovery(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 3):
        super().__init__()
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 3, padding=1, bias=False),
            nn.GroupNorm(4, in_ch // 2),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_ch // 2, out_ch, 3, padding=1)
        )
        self.global_adjust = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, in_ch // 4, 1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_ch // 4, out_ch * 2, 1),
        )

    def forward(self, x: Tensor, input_img: Tensor) -> Tensor:
        loc = self.local_conv(x)
        gain, bias = torch.chunk(self.global_adjust(x), 2, dim=1)
        gain = torch.sigmoid(gain).view(x.shape[0], -1, 1, 1) * 2.0
        bias = torch.tanh(bias).view(x.shape[0], -1, 1, 1) * 0.5
        return torch.clamp(input_img + loc * gain + bias, 0.0, 1.0)

# GLAVNI MODEL ZA ABLACIJU 9: BEZ EDGE GUIDANCE GRANE
class Restauracija_NoEdgeBranch(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_ch: int = 32):
        super().__init__()
        # NEMA self.edge_branch NITI self.edge_fusion
        self.spatial_block1 = SpatialEncoderRestorationBlock(in_channels, base_ch)
        self.spatial_block2 = SpatialEncoderRestorationBlock(base_ch, base_ch * 2)
        self.spatial_block3 = SpatialEncoderRestorationBlock(base_ch * 2, base_ch * 4)
        self.spatial_block4 = SpatialEncoderRestorationBlock(base_ch * 4, base_ch * 8)
        
        self.spectral_init = nn.Sequential(nn.Conv2d(in_channels, base_ch, 3, padding=1, bias=False), nn.GroupNorm(4, base_ch), nn.ReLU(inplace=False))
        self.spectral_block1 = SpectralDecompositionRestorationBlock(base_ch)
        self.spectral_pool1 = nn.MaxPool2d(2)
        self.spec_proj1 = nn.Sequential(nn.Conv2d(base_ch, base_ch * 2, 1, bias=False), nn.GroupNorm(4, base_ch * 2), nn.ReLU(inplace=False))
        self.spectral_block2 = SpectralDecompositionRestorationBlock(base_ch * 2)
        self.spectral_pool2 = nn.MaxPool2d(2)
        self.spec_proj2 = nn.Sequential(nn.Conv2d(base_ch * 2, base_ch * 4, 1, bias=False), nn.GroupNorm(4, base_ch * 4), nn.ReLU(inplace=False))
        self.spectral_block3 = SpectralDecompositionRestorationBlock(base_ch * 4)
        self.spectral_pool3 = nn.MaxPool2d(2)
        self.spec_proj3 = nn.Sequential(nn.Conv2d(base_ch * 4, base_ch * 8, 1, bias=False), nn.GroupNorm(4, base_ch * 8), nn.ReLU(inplace=False))
        self.spectral_block4 = SpectralDecompositionRestorationBlock(base_ch * 8)
        
        self.cross1 = AsymmetricCrossBridgeRestoration(base_ch, base_ch, base_ch)
        self.cross2 = AsymmetricCrossBridgeRestoration(base_ch * 2, base_ch * 2, base_ch * 2)
        self.cross3 = AsymmetricCrossBridgeRestoration(base_ch * 4, base_ch * 4, base_ch * 4)
        self.cross4 = AsymmetricCrossBridgeRestoration(base_ch * 8, base_ch * 8, base_ch * 8)
        
        self.gated_fusion = GatedFusionRestorationBlock(base_ch * 8, base_ch * 8, base_ch * 8)
        self.damage_attention = DamageAttentionRestorationModule(base_ch * 8)
        self.bottleneck_refine = nn.Sequential(
            nn.Conv2d(base_ch * 8, base_ch * 8, 1, bias=False),
            nn.GroupNorm(4, base_ch * 8),
            nn.ReLU(inplace=False),
            DilatedContextBlock(base_ch * 8),
            RecursiveDenseRestorationBlock(base_ch * 8, num_recursions=2)
        )
        
        self.decoder4 = DecoderRestorationBlock(base_ch * 8, base_ch * 8, base_ch * 4)
        self.decoder3 = DecoderRestorationBlock(base_ch * 4, base_ch * 4, base_ch * 2)
        self.decoder2 = DecoderRestorationBlock(base_ch * 2, base_ch * 2, base_ch)
        self.decoder1 = DecoderRestorationBlock(base_ch, base_ch, base_ch)

        self.skip_gate1 = GatedSkipConnection(base_ch)
        self.skip_gate2 = GatedSkipConnection(base_ch * 2)
        self.skip_gate3 = GatedSkipConnection(base_ch * 4)
        self.skip_gate4 = GatedSkipConnection(base_ch * 8)

        self.skip_refine1 = nn.Sequential(RecursiveDenseRestorationBlock(base_ch, 2), SpectralDecompositionRestorationBlock(base_ch))
        self.skip_refine2 = nn.Sequential(RecursiveDenseRestorationBlock(base_ch * 2, 2), SpectralDecompositionRestorationBlock(base_ch * 2))
        self.skip_refine3 = nn.Sequential(RecursiveDenseRestorationBlock(base_ch * 4, 2), SpectralDecompositionRestorationBlock(base_ch * 4))
        self.skip_refine4 = nn.Sequential(RecursiveDenseRestorationBlock(base_ch * 8, 2), SpectralDecompositionRestorationBlock(base_ch * 8))

        self.final_refinement = nn.Sequential(RecursiveDenseRestorationBlock(base_ch, 2), SpectralDecompositionRestorationBlock(base_ch), RecursiveDenseRestorationBlock(base_ch, 2))
        self.contrast_color_recovery = ContrastColorRecovery(base_ch, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        input_img = x

        sp1 = self.spectral_block1(self.spectral_init(x))
        sp2 = self.spectral_block2(self.spec_proj1(self.spectral_pool1(sp1)))
        sp3 = self.spectral_block3(self.spec_proj2(self.spectral_pool2(sp2)))
        sp4 = self.spectral_block4(self.spec_proj3(self.spectral_pool3(sp3)))

        s1, s1_skip = self.spatial_block1(x)
        s2, s2_skip = self.spatial_block2(s1)
        s3, s3_skip = self.spatial_block3(s2)
        s4, s4_skip = self.spatial_block4(s3)

        c1, c2, c3, c4 = self.cross1(s1_skip, sp1), self.cross2(s2_skip, sp2), self.cross3(s3_skip, sp3), self.cross4(s4_skip, sp4)
        s4_enriched = s4 + F.adaptive_avg_pool2d(c4, s4.shape[2:])

        fused = self.gated_fusion(s4_enriched, sp4)
        attended, damage_map = self.damage_attention(fused)
        bottleneck_out = self.bottleneck_refine(attended)

        c4_r = F.interpolate(c4, size=s4_skip.shape[2:], mode='bilinear', align_corners=False)
        c3_r = F.interpolate(c3, size=s3_skip.shape[2:], mode='bilinear', align_corners=False)
        c2_r = F.interpolate(c2, size=s2_skip.shape[2:], mode='bilinear', align_corners=False)
        c1_r = F.interpolate(c1, size=s1_skip.shape[2:], mode='bilinear', align_corners=False)

        sk4 = self.skip_refine4(self.skip_gate4(s4_skip) + c4_r)
        sk3 = self.skip_refine3(self.skip_gate3(s3_skip) + c3_r)
        sk2 = self.skip_refine2(self.skip_gate2(s2_skip) + c2_r)
        sk1 = self.skip_refine1(self.skip_gate1(s1_skip) + c1_r)

        d4 = self.decoder4(bottleneck_out, sk4, damage_map)
        d3 = self.decoder3(d4, sk3, damage_map)
        d2 = self.decoder2(d3, sk2, damage_map)
        d1 = self.decoder1(d2, sk1, damage_map)

        if d1.shape[2:] != input_img.shape[2:]:
            d1 = F.interpolate(d1, size=input_img.shape[2:], mode='bilinear', align_corners=False)

        # BEZ EDGE BRANCH: Izlaz final_refinement-a direktno ide u CCR
        refined = self.final_refinement(d1)
        return self.contrast_color_recovery(refined, input_img)


# ==============================================================================
# TRENING I EVALUACIJA (SA RESTAURACIJA LOSS-OM)
# ==============================================================================
model_no_edge = Restauracija_NoEdgeBranch(base_ch=32).to(device)

sepia_ds = PairedDataset(DIR_SEPIA_CLEAN, DIR_SEPIA_DEGRADED, img_size=IMG_SIZE, train=True)
sepia_loader = DataLoader(sepia_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

train_ds = PairedDataset(DIR_TRAIN_CLEAN, DIR_TRAIN_DEGRADED, img_size=IMG_SIZE, train=True)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

# KORISTI SE VAŠ ORIGINALNI RESTAURACIJA LOSS
criterion = RestauracijaLoss().to(device)
scaler = torch.amp.GradScaler('cuda')

# FAZA 1: 25 EPOHA NA SEPIA DATASETU (NOVO IME CHECKPOINT-A)
CKPT_SEPIA = os.path.join(DIR_ABLACIJA_CKPT, "ablation9_no_edgebranch_sepia25ep_customloss.pth")
if os.path.exists(CKPT_SEPIA):
    print(f"✓ [Keš] Učitavam postojeći checkpoint sa 25 epoha Sepia: {CKPT_SEPIA}")
    model_no_edge.load_state_dict(torch.load(CKPT_SEPIA, map_location=device))
else:
    print(f"\n=======================================================")
    print(f" FAZA 1: Trening bez Edge grane ({EPOCHS_SEPIA} epoha na Sepia uz RestauracijaLoss)")
    print(f"=======================================================")
    opt_pretrain = torch.optim.AdamW(model_no_edge.parameters(), lr=LR_PRETRAIN, weight_decay=1e-4)
    for ep in range(EPOCHS_SEPIA):
        model_no_edge.train()
        ep_loss = 0.0
        for d_t, c_t, _ in sepia_loader:
            d_t, c_t = d_t.to(device), c_t.to(device)
            opt_pretrain.zero_grad()
            with torch.amp.autocast('cuda'):
                pred = model_no_edge(d_t)
                loss = criterion(pred, c_t)
            scaler.scale(loss).backward()
            scaler.step(opt_pretrain)
            scaler.update()
            ep_loss += loss.item()
        print(f" [Sepia Epoha {ep+1:02d}/{EPOCHS_SEPIA}] Loss: {ep_loss/len(sepia_loader):.4f}")
    torch.save(model_no_edge.state_dict(), CKPT_SEPIA)
    print(f"✓ Sačuvan bazni Sepia checkpoint: {CKPT_SEPIA}")

# FAZA 2: 5 EPOHA FINE-TUNINGA NA TRENING SKUPU
CKPT_FINAL = os.path.join(DIR_ABLACIJA_CKPT, "ablation9_no_edgebranch_final_customloss.pth")
if os.path.exists(CKPT_FINAL):
    print(f"✓ [Keš] Učitavam finalni dotrenirani model: {CKPT_FINAL}")
    model_no_edge.load_state_dict(torch.load(CKPT_FINAL, map_location=device))
else:
    print(f"\n=======================================================")
    print(f" FAZA 2: Fine-Tuning ({EPOCHS_FT} epoha na ciljnom datasetu uz RestauracijaLoss)")
    print(f"=======================================================")
    opt_ft = torch.optim.AdamW(model_no_edge.parameters(), lr=LR_FINETUNE, weight_decay=1e-4)
    for ep in range(EPOCHS_FT):
        model_no_edge.train()
        ep_loss = 0.0
        for d_t, c_t, _ in train_loader:
            d_t, c_t = d_t.to(device), c_t.to(device)
            opt_ft.zero_grad()
            with torch.amp.autocast('cuda'):
                pred = model_no_edge(d_t)
                loss = criterion(pred, c_t)
            scaler.scale(loss).backward()
            scaler.step(opt_ft)
            scaler.update()
            ep_loss += loss.item()
        print(f" [Fine-Tune Epoha {ep+1:02d}/{EPOCHS_FT}] Loss: {ep_loss/len(train_loader):.4f}")
    torch.save(model_no_edge.state_dict(), CKPT_FINAL)
    print(f"✓ Sačuvan finalni model: {CKPT_FINAL}")

# FAZA 3: EVALUACIJA NA VALIDACIJI
print(f"\n[INFO] Evaluacija modela (w/o Edge Guidance Branch) na validaciji...")
model_no_edge.eval()
val_files = sorted([f for f in os.listdir(DIR_VAL_DEGRADED) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

psnr_list, ssim_list, lpips_list = [], [], []

with torch.no_grad():
    for fname in val_files:
        c_p = os.path.join(DIR_VAL_CLEAN, fname)
        d_p = os.path.join(DIR_VAL_DEGRADED, fname)
        if not (os.path.exists(c_p) and os.path.exists(d_p)):
            continue

        c_img = cv2.resize(cv2.cvtColor(cv2.imread(c_p), cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
        d_img = cv2.resize(cv2.cvtColor(cv2.imread(d_p), cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0

        d_t = torch.from_numpy(d_img).permute(2, 0, 1).unsqueeze(0).to(device)
        out_t = torch.clamp(model_no_edge(d_t), 0.0, 1.0)
        out_np = (out_t.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8).astype(np.float32) / 255.0

        out_eval_t = torch.from_numpy(out_np).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
        c_eval_t = torch.from_numpy(c_img).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0

        psnr_v = psnr_metric(c_img, out_np, data_range=1.0)
        ssim_v = ssim_metric(c_img, out_np, channel_axis=2, data_range=1.0)
        lpips_v = eval_lpips_fn(out_eval_t, c_eval_t).item()

        psnr_list.append(psnr_v)
        ssim_list.append(ssim_v)
        lpips_list.append(lpips_v)

p_m, p_sd = np.mean(psnr_list), np.std(psnr_list)
s_m, s_sd = np.mean(ssim_list), np.std(ssim_list)
l_m, l_sd = np.mean(lpips_list), np.std(lpips_list)

rezultati = [[
    "9. w/o Edge Guidance Branch",
    f"{p_m:.2f} ± {p_sd:.2f}",
    f"{s_m:.4f} ± {s_sd:.4f}",
    f"{l_m:.4f} ± {l_sd:.4f}"
]]

print("\n" + "="*80)
print("REZULTAT EVALUACIJE: ABLACIJA 9 (w/o Edge Guidance Branch)")
print("="*80)
print(tabulate(rezultati, headers=["Konfiguracija", "PSNR [↑]", "SSIM [↑]", "LPIPS [↓]"], tablefmt="fancy_grid"))

csv_out = os.path.join(DRIVE_PROJECT_DIR, "rezultat_ablacija_9_no_edgebranch_customloss.csv")
pd.DataFrame(rezultati, columns=["Konfiguracija", "PSNR", "SSIM", "LPIPS"]).to_csv(csv_out, index=False)
print(f"✓ Rezultat sačuvan u: {csv_out}\n")
