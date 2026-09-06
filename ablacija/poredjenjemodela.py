# ==============================================================================
# NAUČNO POREĐENJE SA 5-EPOHNOM ADAPTACIJOM MODELA
# Predloženi Model (5 Epoha Fine-Tuning) vs Microsoft BOPBL vs Ulaz (Baseline)
# (1000 Bootstrap Iteracija | Mean ± SD | Wilcoxon & t-test | Cohen's d)
# ==============================================================================

import os
import sys
import copy
import random
import re
import warnings
import subprocess
import shutil
import numpy as np
import pandas as pd
import cv2
from scipy import stats
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric

warnings.filterwarnings('ignore')
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
import torch
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def normalna_instalacija(paket):
    try:
        __import__(paket)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", paket],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

normalna_instalacija("lpips")
normalna_instalacija("tabulate")

import lpips
from tabulate import tabulate
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vgg16, VGG16_Weights

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
except Exception:
    pass

# Putanje do Google Drive-a
DRIVE_PROJECT_DIR = '/content/drive/MyDrive/Projekat_Model'
os.makedirs(DRIVE_PROJECT_DIR, exist_ok=True)
DIR_ABLACIJA_DRIVE = os.path.join(DRIVE_PROJECT_DIR, 'ablacija_checkpoints')
DIR_NJIHOV_DRIVE = os.path.join(DRIVE_PROJECT_DIR, 'rezultati_microsoft_zvanicni')
os.makedirs(DIR_ABLACIJA_DRIVE, exist_ok=True)
os.makedirs(DIR_NJIHOV_DRIVE, exist_ok=True)

# HIPERPARAMETRI
EPOCHS_FINETUNE = 5
BATCH_SIZE = 4
LR_FINETUNE = 5e-5
IMG_SIZE = 256
NUM_ITERACIJA = 1000  # 1000 klasterisanih bootstrap iteracija po scenama za stabilnu procenu

def pronadji_foldere(tip="VALIDACIJA"):
    moguce = [
        f"/content/drive/MyDrive/Projekat_Model/dataset/{tip}",
        f"/content/drive/MyDrive/Projekat_Model/dataset_njihov/{tip}_NJIHOVA" if tip == "VALIDACIJA" else f"/content/drive/MyDrive/Projekat_Model/dataset_njihov/{tip}_NJIHOV",
        f"/content/dataset/{tip}",
        f"/content/dataset_njihov/{tip}_NJIHOVA" if tip == "VALIDACIJA" else f"/content/dataset_njihov/{tip}_NJIHOV",
        f"./dataset/{tip}",
        f"/content/{tip}"
    ]
    for b in moguce:
        if not os.path.exists(b):
            continue
        c = os.path.join(b, "clean")
        d = os.path.join(b, "degraded")
        if os.path.exists(c) and os.path.exists(d) and len(os.listdir(d)) > 0:
            return c, d, b
    raise FileNotFoundError(f"[GREŠKA] Nije pronađen folder za {tip} sa 'clean' i 'degraded' slikama!")

DIR_TRAIN_CLEAN, DIR_TRAIN_DEGRADED, _ = pronadji_foldere("TRENING")
DIR_VAL_CLEAN, DIR_VAL_DEGRADED, _ = pronadji_foldere("VALIDACIJA")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eval_lpips_fn = lpips.LPIPS(net='alex', verbose=False).to(device).eval()

print(f"\n[INFO] Uređaj: {device}")
print(f"[INFO] Trening skup: {len(os.listdir(DIR_TRAIN_DEGRADED))} slika | Validacioni skup: {len(os.listdir(DIR_VAL_DEGRADED))} slika\n")


# ==============================================================================
# DATASET I GUBITAK ZA ADAPTACIJU (FINE-TUNING)
# ==============================================================================
class PairedDataset(Dataset):
    def __init__(self, clean_dir, degraded_dir, img_size=256, train=False):
        self.clean_dir = clean_dir
        self.degraded_dir = degraded_dir
        self.files = sorted([f for f in os.listdir(degraded_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.img_size = img_size
        self.train = train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        c_p = os.path.join(self.clean_dir, fname)
        d_p = os.path.join(self.degraded_dir, fname)

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

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features
        self.slice1 = nn.Sequential(*list(vgg.children())[:4])
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9])
        self.slice3 = nn.Sequential(*list(vgg.children())[9:16])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406], device=input.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=input.device).view(1, 3, 1, 1)
        inp = (input - mean) / std
        tgt = (target - mean) / std
        h1_in, h1_tgt = self.slice1(inp), self.slice1(tgt)
        h2_in, h2_tgt = self.slice2(h1_in), self.slice2(h1_tgt)
        h3_in, h3_tgt = self.slice3(h2_in), self.slice3(h2_tgt)
        return F.l1_loss(h1_in, h1_tgt) + F.l1_loss(h2_in, h2_tgt) + F.l1_loss(h3_in, h3_tgt)


# ==============================================================================
# ARHITEKTURA PREDLOŽENOG MODELA RESTAURACIJE
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

class EdgeBranch(nn.Module):
    def __init__(self, out_channels: int = 32):
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)
        ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).unsqueeze(0).unsqueeze(0)
        self.register_buffer('kx', kx.repeat(3, 1, 1, 1))
        self.register_buffer('ky', ky.repeat(3, 1, 1, 1))
        self.conv = nn.Sequential(
            nn.Conv2d(6, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, out_channels),
            nn.ReLU(inplace=False),
            DepthwiseSeparableConv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(4, out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(torch.cat([F.conv2d(x, self.kx, padding=1, groups=3), F.conv2d(x, self.ky, padding=1, groups=3)], dim=1))

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

class Restauracija(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_ch: int = 32):
        super().__init__()
        self.edge_branch = EdgeBranch(out_channels=base_ch)
        self.edge_fusion = nn.Conv2d(base_ch * 2, base_ch, 1, bias=False)

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

        refined = self.final_refinement(d1)
        edge_feat = self.edge_branch(input_img)
        fused_out = self.edge_fusion(torch.cat([refined, edge_feat], dim=1))
        return self.contrast_color_recovery(fused_out, input_img)


# ==============================================================================
# UČITAVANJE BAZNOG CHECKPOINT-A I ADAPTACIJA 5 EPOHA
# ==============================================================================
def ucitaj_state_dict_pametno(model, candidate_paths, device, strict=True):
    for p in candidate_paths:
        if p and os.path.exists(p):
            try:
                ckpt = torch.load(p, map_location=device, weights_only=False)
                sd = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
                c_sd = {k.replace('module.', ''): v for k, v in sd.items()} if isinstance(sd, dict) else sd
                model.load_state_dict(c_sd, strict=strict)
                print(f"✓ [USPEŠNO UČITAN CHECKPOINT]: {p}")
                return True, p
            except Exception as e:
                print(f"  [UPOZORENJE] Greška pri učitavanju {p}: {e}")
    return False, None

moj_model = Restauracija(base_ch=32).to(device)

ADAPTED_CKPT_PATH = os.path.join(DIR_ABLACIJA_DRIVE, 'ablation_Full_Proposed_Model_5ep.pth')
ALT_ADAPTED_CKPT_PATH = os.path.join(DRIVE_PROJECT_DIR, 'moj_model_finetuned_5ep.pth')

train_ds = PairedDataset(DIR_TRAIN_CLEAN, DIR_TRAIN_DEGRADED, img_size=IMG_SIZE, train=True)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_files = sorted([f for f in os.listdir(DIR_VAL_DEGRADED) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

if os.path.exists(ADAPTED_CKPT_PATH):
    print(f"✓ [KEŠ] Učitavam postojeći adaptirani model (5 epoha): {ADAPTED_CKPT_PATH}")
    moj_model.load_state_dict(torch.load(ADAPTED_CKPT_PATH, map_location=device))
elif os.path.exists(ALT_ADAPTED_CKPT_PATH):
    print(f"✓ [KEŠ] Učitavam postojeći adaptirani model (5 epoha): {ALT_ADAPTED_CKPT_PATH}")
    moj_model.load_state_dict(torch.load(ALT_ADAPTED_CKPT_PATH, map_location=device))
else:
    moguce_lokacije = [DRIVE_PROJECT_DIR, '/content/drive/MyDrive', '/content', './']
    moguca_imena = ['dodinarestauracijabest.pth', 'doroteinarestauracijabest.pth', 'Model_Finetuned_Final.pth', 'best_model.pth', 'model.pth']
    candidate_base_ckpts = [os.path.join(loc, name) for loc in moguce_lokacije for name in moguca_imena]

    uspeh, pronadjena_putanja = ucitaj_state_dict_pametno(moj_model, candidate_base_ckpts, device, strict=True)
    if not uspeh:
        raise FileNotFoundError("[GREŠKA] Nijedan bazni .pth fajl nije pronađen za predloženi model!")

    print(f"\n-> [Fine-tune {EPOCHS_FINETUNE} epoha] Pokrećem adaptaciju vašeg modela na trening skupu...")
    optimizer = torch.optim.AdamW(moj_model.parameters(), lr=LR_FINETUNE, weight_decay=1e-4)
    crit_l1 = nn.L1Loss()
    crit_vgg = VGGPerceptualLoss().to(device)
    scaler = torch.amp.GradScaler('cuda')

    for ep in range(EPOCHS_FINETUNE):
        moj_model.train()
        ep_loss = 0.0
        for d_t, c_t, _ in train_loader:
            d_t, c_t = d_t.to(device), c_t.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                pred = moj_model(d_t)
                loss = crit_l1(pred, c_t) + 0.1 * crit_vgg(pred, c_t)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ep_loss += loss.item()
        print(f"   [Epoha {ep+1}/{EPOCHS_FINETUNE}] Loss: {ep_loss/len(train_loader):.4f}")

    torch.save(moj_model.state_dict(), ADAPTED_CKPT_PATH)
    print(f"✓ [USPEH] Adaptirani model od 5 epoha je sačuvan na: {ADAPTED_CKPT_PATH}")

moj_model.eval()


# ==============================================================================
# ZVANIČNI MICROSOFT MODEL (BOPBL STANDARDNI RUN.PY PIPELINE BEZ --with_scratch)
# ==============================================================================
MS_REPO_DIR = '/content/Bringing-Old-Photos-Back-to-Life'
DIR_BOPBL_TEMP_OUT = '/content/bopbl_temp_run'

if not os.path.exists(MS_REPO_DIR):
    devnull = subprocess.DEVNULL
    print("\n-> Preuzimam zvanični Microsoft Bringing-Old-Photos-Back-to-Life repo...")
    subprocess.run(f"git clone -q https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life.git {MS_REPO_DIR}", shell=True, stdout=devnull, stderr=devnull)

    p1 = os.path.join(MS_REPO_DIR, 'Face_Enhancement/models/networks')
    p2 = os.path.join(MS_REPO_DIR, 'Global/detection_models')
    subprocess.run(f"cd {p1} && git clone -q https://github.com/vacancy/Synchronized-BatchNorm-PyTorch && cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .", shell=True, stdout=devnull, stderr=devnull)
    subprocess.run(f"cd {p2} && git clone -q https://github.com/vacancy/Synchronized-BatchNorm-PyTorch && cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .", shell=True, stdout=devnull, stderr=devnull)

    print("-> Preuzimam zvanične težine: Face & Global Checkpoints + Landmark model...")
    subprocess.run(f"cd {MS_REPO_DIR}/Face_Detection && wget -q http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 && bzip2 -d shape_predictor_68_face_landmarks.dat.bz2", shell=True, stdout=devnull, stderr=devnull)
    subprocess.run(f"cd {MS_REPO_DIR}/Face_Enhancement && wget -q https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases/download/v1.0/face_checkpoints.zip && unzip -q face_checkpoints.zip", shell=True, stdout=devnull, stderr=devnull)
    subprocess.run(f"cd {MS_REPO_DIR}/Global && wget -q https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases/download/v1.0/global_checkpoints.zip && unzip -q global_checkpoints.zip", shell=True, stdout=devnull, stderr=devnull)

postojece_ms_slike = [f for f in os.listdir(DIR_NJIHOV_DRIVE) if f.lower().endswith(('.png', '.jpg', '.jpeg'))] if os.path.exists(DIR_NJIHOV_DRIVE) else []

# Ako nema generisanih slika, pokreni zvanični BOPBL pipeline BEZ --with_scratch
if len(postojece_ms_slike) >= len(val_files):
    print(f"✓ [KEŠ] Koriste se postojeće generisane slike zvaničnog Microsoft modela sa Google Drive-a ({len(postojece_ms_slike)} slika).")
else:
    print(f"\n-> Pokrećem ZVANIČNI Microsoft run.py pipeline (standard restoration bez --with_scratch) nad: {DIR_VAL_DEGRADED}...")
    gpu_flag = "0" if torch.cuda.is_available() else "-1"
    
    # IZMENA: Uklonjen fleg --with_scratch radi usklađivanja sa Reviewer 2 nalazom N5
    cmd = f"cd {MS_REPO_DIR} && python run.py --input_folder {DIR_VAL_DEGRADED} --output_folder {DIR_BOPBL_TEMP_OUT} --GPU {gpu_flag}"
    subprocess.run(cmd, shell=True)

    bopbl_final = os.path.join(DIR_BOPBL_TEMP_OUT, 'final_output')
    if os.path.exists(bopbl_final):
        for img_name in os.listdir(bopbl_final):
            shutil.copy(os.path.join(bopbl_final, img_name), os.path.join(DIR_NJIHOV_DRIVE, img_name))
        print(f"✓ Zvanični Microsoft rezultati sačuvani na Drive: {DIR_NJIHOV_DRIVE}")


# ==============================================================================
# INFERENCIJA I RAČUNANJE METRIKA PO SLIKAMA
# ==============================================================================
print(f"\n[INFO] Računanje metrika (PSNR, SSIM, LPIPS) na validacionom skupu ({len(val_files)} slika)...")

data_input = []
data_moj = []
data_ms = []

with torch.no_grad():
    for fname in val_files:
        c_p = os.path.join(DIR_VAL_CLEAN, fname)
        d_p = os.path.join(DIR_VAL_DEGRADED, fname)
        if not (os.path.exists(c_p) and os.path.exists(d_p)):
            continue

        c_img = cv2.resize(cv2.cvtColor(cv2.imread(c_p), cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
        d_img = cv2.resize(cv2.cvtColor(cv2.imread(d_p), cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0

        c_eval_t = torch.from_numpy(c_img).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
        d_eval_t = torch.from_numpy(d_img).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0

        # 1. Ulaz (Baseline)
        psnr_in = psnr_metric(c_img, d_img, data_range=1.0)
        ssim_in = ssim_metric(c_img, d_img, channel_axis=2, data_range=1.0)
        lpips_in = eval_lpips_fn(d_eval_t, c_eval_t).item()
        data_input.append({'Fname': fname, 'PSNR': psnr_in, 'SSIM': ssim_in, 'LPIPS': lpips_in})

        # 2. Predloženi model (sa 5-epohnom adaptacijom)
        d_t = torch.from_numpy(d_img).permute(2, 0, 1).unsqueeze(0).to(device)
        out_t = torch.clamp(moj_model(d_t), 0.0, 1.0)
        out_np = (out_t.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8).astype(np.float32) / 255.0
        out_eval_t = torch.from_numpy(out_np).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0

        psnr_moj = psnr_metric(c_img, out_np, data_range=1.0)
        ssim_moj = ssim_metric(c_img, out_np, channel_axis=2, data_range=1.0)
        lpips_moj = eval_lpips_fn(out_eval_t, c_eval_t).item()
        data_moj.append({'Fname': fname, 'PSNR': psnr_moj, 'SSIM': ssim_moj, 'LPIPS': lpips_moj})

        # 3. Zvanični Microsoft BOPBL model
        ms_p = os.path.join(DIR_NJIHOV_DRIVE, fname)
        if not os.path.exists(ms_p):
            ms_p = os.path.join(DIR_NJIHOV_DRIVE, f"{os.path.splitext(fname)[0]}.png")

        if os.path.exists(ms_p):
            ms_img = cv2.resize(cv2.cvtColor(cv2.imread(ms_p), cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
        else:
            ms_img = d_img

        ms_eval_t = torch.from_numpy(ms_img).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
        psnr_ms = psnr_metric(c_img, ms_img, data_range=1.0)
        ssim_ms = ssim_metric(c_img, ms_img, channel_axis=2, data_range=1.0)
        lpips_ms = eval_lpips_fn(ms_eval_t, c_eval_t).item()
        data_ms.append({'Fname': fname, 'PSNR': psnr_ms, 'SSIM': ssim_ms, 'LPIPS': lpips_ms})

df_in = pd.DataFrame(data_input).set_index('Fname')
df_moj = pd.DataFrame(data_moj).set_index('Fname')
df_ms = pd.DataFrame(data_ms).set_index('Fname')

# Čuvanje punih pojedinačnih rezultata po slikama na Drive (za potpunu proverljivost i reprodukciju)
df_per_image_all = pd.DataFrame({
    'Input_PSNR': df_in['PSNR'], 'Input_SSIM': df_in['SSIM'], 'Input_LPIPS': df_in['LPIPS'],
    'Proposed_PSNR': df_moj['PSNR'], 'Proposed_SSIM': df_moj['SSIM'], 'Proposed_LPIPS': df_moj['LPIPS'],
    'BOPBL_PSNR': df_ms['PSNR'], 'BOPBL_SSIM': df_ms['SSIM'], 'BOPBL_LPIPS': df_ms['LPIPS'],
})
df_per_image_all.to_csv(os.path.join(DRIVE_PROJECT_DIR, "per_image_bopbl_poredjenje.csv"))
print(f"✓ Sačuvane pojedinačne per-image metrike na Drive: {os.path.join(DRIVE_PROJECT_DIR, 'per_image_bopbl_poredjenje.csv')}")


# ==============================================================================
# STATISTIČKA EVALUACIJA (1000 ITERACIJA BOOTSTRAP | MEAN ± SD | TESTOVI)
# ==============================================================================
def get_scene_id(filename):
    base = os.path.splitext(filename)[0]
    match = re.match(r'^(scene_?\d+|img_?\d+|\d+)', base, re.IGNORECASE)
    return match.group(1) if match else base.split('_')[0]

scene_to_files = {}
for f in val_files:
    sid = get_scene_id(f)
    scene_to_files.setdefault(sid, []).append(f)
unique_scenes = np.array(list(scene_to_files.keys()))

iter_in_p, iter_in_s, iter_in_l = [], [], []
iter_moj_p, iter_moj_s, iter_moj_l = [], [], []
iter_ms_p, iter_ms_s, iter_ms_l = [], [], []

print(f"\n[INFO] Pokrećem {NUM_ITERACIJA} klasterisanih bootstrap iteracija po scenama...")
for it in range(NUM_ITERACIJA):
    rng = np.random.default_rng(seed=SEED + it)
    sampled_scenes = rng.choice(unique_scenes, size=len(unique_scenes), replace=True)
    boot_files = [f for s in sampled_scenes for f in scene_to_files[s] if f in df_moj.index]

    iter_in_p.append(df_in.loc[boot_files]['PSNR'].mean())
    iter_in_s.append(df_in.loc[boot_files]['SSIM'].mean())
    iter_in_l.append(df_in.loc[boot_files]['LPIPS'].mean())

    iter_moj_p.append(df_moj.loc[boot_files]['PSNR'].mean())
    iter_moj_s.append(df_moj.loc[boot_files]['SSIM'].mean())
    iter_moj_l.append(df_moj.loc[boot_files]['LPIPS'].mean())

    iter_ms_p.append(df_ms.loc[boot_files]['PSNR'].mean())
    iter_ms_s.append(df_ms.loc[boot_files]['SSIM'].mean())
    iter_ms_l.append(df_ms.loc[boot_files]['LPIPS'].mean())

def format_p_exact(p):
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"

# Srednje vrednosti i standardne devijacije iz 1000 bootstrap iteracija
m_in_p, sd_in_p = np.mean(iter_in_p), np.std(iter_in_p)
m_in_s, sd_in_s = np.mean(iter_in_s), np.std(iter_in_s)
m_in_l, sd_in_l = np.mean(iter_in_l), np.std(iter_in_l)

m_moj_p, sd_moj_p = np.mean(iter_moj_p), np.std(iter_moj_p)
m_moj_s, sd_moj_s = np.mean(iter_moj_s), np.std(iter_moj_s)
m_moj_l, sd_moj_l = np.mean(iter_moj_l), np.std(iter_moj_l)

m_ms_p, sd_ms_p = np.mean(iter_ms_p), np.std(iter_ms_p)
m_ms_s, sd_ms_s = np.mean(iter_ms_s), np.std(iter_ms_s)
m_ms_l, sd_ms_l = np.mean(iter_ms_l), np.std(iter_ms_l)

# Statistički upareni testovi (Predloženi Model vs Zvanični Microsoft BOPBL)
val_moj_p, val_ms_p = df_moj['PSNR'].values, df_ms['PSNR'].values
val_moj_s, val_ms_s = df_moj['SSIM'].values, df_ms['SSIM'].values
val_moj_l, val_ms_l = df_moj['LPIPS'].values, df_ms['LPIPS'].values

_, p_w_p = stats.wilcoxon(val_moj_p, val_ms_p)
_, p_t_p = stats.ttest_rel(val_moj_p, val_ms_p)
d_psnr = np.mean(val_moj_p - val_ms_p) / np.std(val_moj_p - val_ms_p, ddof=1)

_, p_w_s = stats.wilcoxon(val_moj_s, val_ms_s)
_, p_t_s = stats.ttest_rel(val_moj_s, val_ms_s)
d_ssim = np.mean(val_moj_s - val_ms_s) / np.std(val_moj_s - val_ms_s, ddof=1)

_, p_w_l = stats.wilcoxon(val_moj_l, val_ms_l)
_, p_t_l = stats.ttest_rel(val_moj_l, val_ms_l)
d_lpips = np.mean(val_moj_l - val_ms_l) / np.std(val_moj_l - val_ms_l, ddof=1)


# ==============================================================================
# TABELARNI PRIKAZ I ČUVANJE U CSV
# ==============================================================================
tabela_poređenje = [
    [
        'PSNR (dB) [↑]',
        f"{m_in_p:.2f} ± {sd_in_p:.2f}",
        f"{m_moj_p:.2f} ± {sd_moj_p:.2f}",
        f"{m_ms_p:.2f} ± {sd_ms_p:.2f}",
        f"{m_moj_p - m_in_p:+.2f} dB",
        f"{m_moj_p - m_ms_p:+.2f} dB",
        format_p_exact(p_w_p),
        format_p_exact(p_t_p),
        f"{d_psnr:.2f}"
    ],
    [
        'SSIM [↑]',
        f"{m_in_s:.4f} ± {sd_in_s:.4f}",
        f"{m_moj_s:.4f} ± {sd_moj_s:.4f}",
        f"{m_ms_s:.4f} ± {sd_ms_s:.4f}",
        f"{m_moj_s - m_in_s:+.4f}",
        f"{m_moj_s - m_ms_s:+.4f}",
        format_p_exact(p_w_s),
        format_p_exact(p_t_s),
        f"{d_ssim:.2f}"
    ],
    [
        'LPIPS [↓]',
        f"{m_in_l:.4f} ± {sd_in_l:.4f}",
        f"{m_moj_l:.4f} ± {sd_moj_l:.4f}",
        f"{m_ms_l:.4f} ± {sd_ms_l:.4f}",
        f"{m_moj_l - m_in_l:+.4f}",
        f"{m_moj_l - m_ms_l:+.4f}",
        format_p_exact(p_w_l),
        format_p_exact(p_t_l),
        f"{d_lpips:.2f}"
    ]
]

zaglavlja = [
    'Metrika',
    'Ulaz (Bez Rest.)',
    'Predloženi Model (Mean ± SD)',
    'Microsoft BOPBL (Mean ± SD)',
    'Δ (vs Ulaz)',
    'Δ (vs BOPBL)',
    'Wilcoxon (p)',
    't-test (p)',
    "Cohen's d"
]

print("\n" + "█" * 125)
print(f"  TABELA: NAUČNO POREĐENJE RESTAURACIJE (5 Epoha Adaptacije | {NUM_ITERACIJA} Bootstrap Iteracija | N = {len(df_moj)})")
print("█" * 125)
print(tabulate(tabela_poređenje, headers=zaglavlja, tablefmt="fancy_grid", stralign="center", numalign="center"))

csv_izlaz = os.path.join(DRIVE_PROJECT_DIR, "tabela9_direktno_poredjenje.csv")
pd.DataFrame(tabela_poređenje, columns=zaglavlja).to_csv(csv_izlaz, index=False)
print(f"\n✓ Tabela je uspešno sačuvana na Google Drive:\n   -> {csv_izlaz}\n")
