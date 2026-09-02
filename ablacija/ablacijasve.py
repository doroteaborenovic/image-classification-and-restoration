# ==============================================================================
# NAUČNO METODOLOŠKA EVALUACIJA I ABLACIJA SA AUTOMATSKIM UČITAVANJEM TEŽINA
# Bootstrap: 1000 iteracija | 95% CI | Ulazni Baseline | Holm-Bonferroni
# ==============================================================================

import os
import sys
import copy
import random
import re
import warnings
import subprocess
import numpy as np
import pandas as pd
import cv2
from scipy import stats
from skimage.metrics import structural_similarity as ssim_metric

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

# ==============================================================================
# PUTANJE I PODEŠAVANJA
# ==============================================================================
DRIVE_PROJECT_DIR = '/content/drive/MyDrive/Projekat_Model'
DIR_ABLACIJA_DRIVE = os.path.join(DRIVE_PROJECT_DIR, 'ablacija_checkpoints')
os.makedirs(DIR_ABLACIJA_DRIVE, exist_ok=True)

EPOCHS_FINETUNE = 5
BATCH_SIZE = 4
LR_FINETUNE = 5e-5
IMG_SIZE = 256
NUM_ITERACIJA = 1000

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
print(f"[INFO] Trening skup: {len(os.listdir(DIR_TRAIN_DEGRADED))} | Validacioni skup: {len(os.listdir(DIR_VAL_DEGRADED))} slika\n")


# ==============================================================================
# DATASET I GUBITAK
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
# ARHITEKTURA MODELA
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
        x = self.dense_micro(self.conv(x))
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

    def forward(self, x: Tensor, skip: Tensor, damage_map: Tensor, use_spectral: bool = True) -> Tensor:
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        dm = F.interpolate(damage_map, size=skip.shape[2:], mode='bilinear', align_corners=False)
        feat = self.dense_micro(self.conv(torch.cat([x, skip, dm], dim=1)))
        return self.spectral(feat) if use_spectral else feat

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

    def forward(
        self,
        x: Tensor,
        use_spatial: bool = True,
        use_spectral: bool = True,
        use_cross_bridge: bool = True,
        use_gated_fusion: bool = True,
        use_damage_attention: bool = True,
        use_dilated_context: bool = True,
        use_gated_skips: bool = True,
        use_skip_refine: bool = True,
        use_edge_branch: bool = True,
        use_ccr: bool = True
    ) -> Tensor:
        input_img = x

        if use_spectral:
            sp1 = self.spectral_block1(self.spectral_init(x))
            sp2 = self.spectral_block2(self.spec_proj1(self.spectral_pool1(sp1)))
            sp3 = self.spectral_block3(self.spec_proj2(self.spectral_pool2(sp2)))
            sp4 = self.spectral_block4(self.spec_proj3(self.spectral_pool3(sp3)))
        else:
            sp1 = sp2 = sp3 = sp4 = None

        if use_spatial:
            s1, s1_skip = self.spatial_block1(x)
            s2, s2_skip = self.spatial_block2(s1)
            s3, s3_skip = self.spatial_block3(s2)
            s4, s4_skip = self.spatial_block4(s3)
        else:
            s1_skip, s2_skip, s3_skip, s4_skip = sp1, sp2, sp3, sp4
            s4 = sp4

        if not use_spectral:
            sp1, sp2, sp3, sp4 = s1_skip, s2_skip, s3_skip, s4_skip

        if use_cross_bridge and use_spatial and use_spectral:
            c1, c2, c3, c4 = self.cross1(s1_skip, sp1), self.cross2(s2_skip, sp2), self.cross3(s3_skip, sp3), self.cross4(s4_skip, sp4)
            s4_enriched = s4 + F.adaptive_avg_pool2d(c4, s4.shape[2:])
        else:
            c1, c2, c3, c4 = torch.zeros_like(s1_skip), torch.zeros_like(s2_skip), torch.zeros_like(s3_skip), torch.zeros_like(s4_skip)
            s4_enriched = s4

        if use_gated_fusion:
            fused = self.gated_fusion(s4_enriched, sp4)
        else:
            fused = (s4_enriched + F.interpolate(sp4, size=s4_enriched.shape[2:], mode='bilinear', align_corners=False)) * 0.5

        if use_damage_attention:
            attended, damage_map = self.damage_attention(fused)
        else:
            attended = fused
            damage_map = torch.zeros(fused.shape[0], 1, fused.shape[2], fused.shape[3], device=fused.device)

        if use_dilated_context:
            bottleneck_out = self.bottleneck_refine(attended)
        else:
            b = self.bottleneck_refine[2](self.bottleneck_refine[1](self.bottleneck_refine[0](attended)))
            bottleneck_out = self.bottleneck_refine[4](b)

        c4_r = F.interpolate(c4, size=s4_skip.shape[2:], mode='bilinear', align_corners=False) if (use_cross_bridge and use_spatial and use_spectral) else 0
        c3_r = F.interpolate(c3, size=s3_skip.shape[2:], mode='bilinear', align_corners=False) if (use_cross_bridge and use_spatial and use_spectral) else 0
        c2_r = F.interpolate(c2, size=s2_skip.shape[2:], mode='bilinear', align_corners=False) if (use_cross_bridge and use_spatial and use_spectral) else 0
        c1_r = F.interpolate(c1, size=s1_skip.shape[2:], mode='bilinear', align_corners=False) if (use_cross_bridge and use_spatial and use_spectral) else 0

        sk4 = (self.skip_gate4(s4_skip) if use_gated_skips else s4_skip) + c4_r
        sk3 = (self.skip_gate3(s3_skip) if use_gated_skips else s3_skip) + c3_r
        sk2 = (self.skip_gate2(s2_skip) if use_gated_skips else s2_skip) + c2_r
        sk1 = (self.skip_gate1(s1_skip) if use_gated_skips else s1_skip) + c1_r

        sk4_final = self.skip_refine4(sk4) if use_skip_refine else sk4
        sk3_final = self.skip_refine3(sk3) if use_skip_refine else sk3
        sk2_final = self.skip_refine2(sk2) if use_skip_refine else sk2
        sk1_final = self.skip_refine1(sk1) if use_skip_refine else sk1

        d4 = self.decoder4(bottleneck_out, sk4_final, damage_map, use_spectral=use_spectral)
        d3 = self.decoder3(d4, sk3_final, damage_map, use_spectral=use_spectral)
        d2 = self.decoder2(d3, sk2_final, damage_map, use_spectral=use_spectral)
        d1 = self.decoder1(d2, sk1_final, damage_map, use_spectral=use_spectral)

        if d1.shape[2:] != input_img.shape[2:]:
            d1 = F.interpolate(d1, size=input_img.shape[2:], mode='bilinear', align_corners=False)

        refined = self.final_refinement(d1)

        if use_edge_branch:
            edge_feat = self.edge_branch(input_img)
            fused_out = self.edge_fusion(torch.cat([refined, edge_feat], dim=1))
        else:
            fused_out = refined

        if use_ccr:
            return self.contrast_color_recovery(fused_out, input_img)
        else:
            loc = self.contrast_color_recovery.local_conv(fused_out)
            return torch.clamp(input_img + loc, 0.0, 1.0)


# ==============================================================================
# PAMETNO UČITAVANJE CHECKPOINT-A (ROBUSNO PRETRAŽIVANJE)
# ==============================================================================
def ucitaj_state_dict_pametno(model, candidate_paths, device, strict=True):
    for p in candidate_paths:
        if p and os.path.exists(p):
            try:
                ckpt = torch.load(p, map_location=device, weights_only=False)
                if isinstance(ckpt, dict):
                    if 'model_state_dict' in ckpt:
                        sd = ckpt['model_state_dict']
                    elif 'state_dict' in ckpt:
                        sd = ckpt['state_dict']
                    elif 'model' in ckpt:
                        sd = ckpt['model']
                    else:
                        sd = ckpt
                else:
                    sd = ckpt

                # Uklanjanje eventualnog 'module.' prefiksa iz DataParallel obuke
                c_sd = {k.replace('module.', ''): v for k, v in sd.items()}
                model.load_state_dict(c_sd, strict=strict)
                print(f"✓ [USPEŠNO UČITAN CHECKPOINT] {p}")
                return True, p
            except Exception as e:
                print(f"  [UPOZORENJE] Nije uspelo učitavanje {p}: {e}")
    return False, None

# Lista svih mogućih lokacija i ranijih imena modela
moguce_lokacije = [
    DRIVE_PROJECT_DIR,
    '/content/drive/MyDrive',
    '/content',
    './'
]

moguca_imena = [
    'Model_Finetuned_Final.pth',
    'best_model.pth',
    'model_restoration_heavy.pth',
    'model_restoration_moderate.pth',
    'dodinamrezajej.pth',
    'model.pth',
    'checkpoint.pth',
    'restoration_model.pth'
]

candidate_base_ckpts = [os.path.join(loc, name) for loc in moguce_lokacije for name in moguca_imena]

moj_model = Restauracija(base_ch=32).to(device)
uspeh, pronadjena_putanja = ucitaj_state_dict_pametno(moj_model, candidate_base_ckpts, device, strict=True)

if not uspeh:
    raise FileNotFoundError("[GREŠKA] Nijedan bazni model za restauraciju nije pronađen na navedenim putanjama!")

val_files = sorted([f for f in os.listdir(DIR_VAL_DEGRADED) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
train_ds = PairedDataset(DIR_TRAIN_CLEAN, DIR_TRAIN_DEGRADED, img_size=IMG_SIZE, train=True)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)


# ==============================================================================
# STATISTIČKE I KLASTERSKE FUNKCIJE
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

def ci95(vals):
    vals = np.array(vals)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return np.mean(vals), np.std(vals), lo, hi

def format_p_val(p):
    if p < 0.001:
        return "< 0.001 ***"
    elif p < 0.01:
        return f"{p:.4f} **"
    elif p < 0.05:
        return f"{p:.4f} *"
    else:
        return f"{p:.4f} (ns)"

def holm_bonferroni(p_vals):
    m = len(p_vals)
    indexed = sorted(enumerate(p_vals), key=lambda x: x[1])
    corrected = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj_p = min(1.0, p * (m - rank))
        running_max = max(running_max, adj_p)
        corrected[orig_idx] = running_max
    return corrected


# ==============================================================================
# 0. POLAZNI KVALITET ULAZA (BASELINE)
# ==============================================================================
print(f"\n[INFO] Računanje polaznih vrednosti za neobrađeni degradirani ulaz...")
cached_input_baseline = []

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

        mse_in = np.mean((c_img - d_img) ** 2)
        psnr_in = 10.0 * np.log10(1.0 / mse_in) if mse_in > 0 else 100.0
        ssim_in = ssim_metric(c_img, d_img, channel_axis=2, data_range=1.0)
        lpips_in = eval_lpips_fn(d_eval_t, c_eval_t).item()

        cached_input_baseline.append({'Fname': fname, 'PSNR': psnr_in, 'SSIM': ssim_in, 'LPIPS': lpips_in})

df_input_baseline = pd.DataFrame(cached_input_baseline).set_index('Fname')


# ==============================================================================
# 1. METODOLOŠKA ADAPTACIJA ILI UČITAVANJE POSTOJEĆIH CHECKPOINTA
# ==============================================================================
ablation_configs = [
    ("Full Proposed Model",                   dict()),
    ("1. w/o Spatial Encoder Stream",         dict(use_spatial=False)),
    ("2. w/o Spectral Encoder Stream",        dict(use_spectral=False)),
    ("3. w/o Asymmetric Cross-Bridge",        dict(use_cross_bridge=False)),
    ("4. w/o Gated Bottleneck Fusion",        dict(use_gated_fusion=False)),
    ("5. w/o Damage Attention Module",        dict(use_damage_attention=False)),
    ("6. w/o Bottleneck Dilated Context",     dict(use_dilated_context=False)),
    ("7. w/o Gated Skip Connections",         dict(use_gated_skips=False)),
    ("8. w/o Skip Refinement Blocks",         dict(use_skip_refine=False)),
    ("9. w/o Edge Guidance Branch",           dict(use_edge_branch=False)),
    ("10. w/o Contrast Color Recovery (CCR)", dict(use_ccr=False)),
]

def adaptiraj_ili_ucitaj_model(base_model, cfg_name, cfg_kwargs, epochs=EPOCHS_FINETUNE):
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', cfg_name)
    save_path = os.path.join(DIR_ABLACIJA_DRIVE, f"ablation_{sanitized}_5ep.pth")

    model_variant = copy.deepcopy(base_model).to(device)

    # 1. Provera da li već postoji sačuvana adaptirana verzija na Drive-u
    if os.path.exists(save_path):
        print(f"   ✓ [Keš] Učitavam postojeći checkpoint sa Drive-a: {cfg_name}")
        model_variant.load_state_dict(torch.load(save_path, map_location=device), strict=True)
        return model_variant

    # 2. Ako ne postoji, pokreće se 5 epoha adaptacije (metodološki neophodno)
    print(f"   -> [Fine-tune {epochs} ep.] Adaptacija za: {cfg_name}...")
    optimizer = torch.optim.AdamW(model_variant.parameters(), lr=LR_FINETUNE, weight_decay=1e-4)
    crit_l1 = nn.L1Loss()
    crit_vgg = VGGPerceptualLoss().to(device)
    scaler = torch.amp.GradScaler('cuda')

    for ep in range(epochs):
        model_variant.train()
        for d_t, c_t, _ in train_loader:
            d_t, c_t = d_t.to(device), c_t.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                pred = model_variant(d_t, **cfg_kwargs)
                loss = crit_l1(pred, c_t) + 0.1 * crit_vgg(pred, c_t)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    torch.save(model_variant.state_dict(), save_path)
    print(f"   ✓ Sačuvan novi checkpoint: {save_path}")
    return model_variant

cached_results = {}
print(f"\n[INFO] Učitavanje i evaluacija svih varijanti modela...")

for naziv, cfg in ablation_configs:
    active_model = adaptiraj_ili_ucitaj_model(moj_model, naziv, cfg)
    active_model.eval()

    res_list = []
    with torch.no_grad():
        for fname in val_files:
            c_p = os.path.join(DIR_VAL_CLEAN, fname)
            d_p = os.path.join(DIR_VAL_DEGRADED, fname)
            if not (os.path.exists(c_p) and os.path.exists(d_p)):
                continue

            c_img = cv2.resize(cv2.cvtColor(cv2.imread(c_p), cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
            d_img = cv2.resize(cv2.cvtColor(cv2.imread(d_p), cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0

            d_t = torch.from_numpy(d_img).permute(2, 0, 1).unsqueeze(0).to(device)
            out_t = torch.clamp(active_model(d_t, **cfg), 0.0, 1.0)
            out_np = (out_t.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8).astype(np.float32) / 255.0

            out_eval_t = torch.from_numpy(out_np).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
            c_eval_t = torch.from_numpy(c_img).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0

            mse = np.mean((c_img - out_np) ** 2)
            psnr_val = 10.0 * np.log10(1.0 / mse) if mse > 0 else 100.0
            ssim_val = ssim_metric(c_img, out_np, channel_axis=2, data_range=1.0)
            lpips_val = eval_lpips_fn(out_eval_t, c_eval_t).item()

            res_list.append({'Fname': fname, 'PSNR': psnr_val, 'SSIM': ssim_val, 'LPIPS': lpips_val})

    cached_results[naziv] = pd.DataFrame(res_list).set_index('Fname')


# ==============================================================================
# 2. MICROSOFT MODEL RESTAURACIJE (BOPBL)
# ==============================================================================
DIR_NJIHOV_ROOT = '/content/rezultati_njihov_model'
DIR_NJIHOV_OUTPUT = os.path.join(DIR_NJIHOV_ROOT, 'final_output')

if not os.path.exists(DIR_NJIHOV_OUTPUT) or len(os.listdir(DIR_NJIHOV_OUTPUT)) == 0:
    for alt in [os.path.join(DIR_NJIHOV_ROOT, 'stage_3_restore_output'), os.path.join(DIR_NJIHOV_ROOT, 'restored_image')]:
        if os.path.exists(alt) and len(os.listdir(alt)) > 0:
            DIR_NJIHOV_OUTPUT = alt
            break

cached_njihov_results = []
if os.path.exists(DIR_NJIHOV_OUTPUT):
    with torch.no_grad():
        for fname in val_files:
            c_p = os.path.join(DIR_VAL_CLEAN, fname)
            nj_p = os.path.join(DIR_NJIHOV_OUTPUT, fname)
            if not os.path.exists(nj_p):
                nj_p = os.path.join(DIR_NJIHOV_OUTPUT, f"{os.path.splitext(fname)[0]}.png")

            if not (os.path.exists(c_p) and os.path.exists(nj_p)):
                continue

            c_img = cv2.resize(cv2.cvtColor(cv2.imread(c_p), cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
            nj_img = cv2.resize(cv2.cvtColor(cv2.imread(nj_p), cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0

            c_eval_t = torch.from_numpy(c_img).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
            nj_eval_t = torch.from_numpy(nj_img).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0

            mse_nj = np.mean((c_img - nj_img) ** 2)
            psnr_nj = 10.0 * np.log10(1.0 / mse_nj) if mse_nj > 0 else 100.0
            ssim_nj = ssim_metric(c_img, nj_img, channel_axis=2, data_range=1.0)
            lpips_nj = eval_lpips_fn(nj_eval_t, c_eval_t).item()

            cached_njihov_results.append({'Fname': fname, 'PSNR': psnr_nj, 'SSIM': ssim_nj, 'LPIPS': lpips_nj})

df_njihov = pd.DataFrame(cached_njihov_results).set_index('Fname') if cached_njihov_results else pd.DataFrame(columns=['PSNR', 'SSIM', 'LPIPS'])


# ==============================================================================
# 3. KLASTERISANI BOOTSTRAP PO SCENAMA (NUM_ITERACIJA = 1000)
# ==============================================================================
raw_boot_data = {cfg_name: {'PSNR': [], 'SSIM': [], 'LPIPS': []} for cfg_name, _ in ablation_configs}
runs_in_psnr, runs_in_ssim, runs_in_lpips = [], [], []
runs_nj_psnr, runs_nj_ssim, runs_nj_lpips = [], [], []

for iter_idx in range(NUM_ITERACIJA):
    rng = np.random.default_rng(seed=SEED + iter_idx)
    sampled_scenes = rng.choice(unique_scenes, size=len(unique_scenes), replace=True)
    boot_files = [f for s in sampled_scenes for f in scene_to_files[s] if f in df_input_baseline.index]

    df_in_sub = df_input_baseline.loc[boot_files]
    runs_in_psnr.append(df_in_sub['PSNR'].mean())
    runs_in_ssim.append(df_in_sub['SSIM'].mean())
    runs_in_lpips.append(df_in_sub['LPIPS'].mean())

    for naziv, _ in ablation_configs:
        df_sub = cached_results[naziv].loc[boot_files]
        raw_boot_data[naziv]['PSNR'].append(df_sub['PSNR'].mean())
        raw_boot_data[naziv]['SSIM'].append(df_sub['SSIM'].mean())
        raw_boot_data[naziv]['LPIPS'].append(df_sub['LPIPS'].mean())

    if not df_njihov.empty:
        df_nj_sub = df_njihov.loc[[f for f in boot_files if f in df_njihov.index]]
        runs_nj_psnr.append(df_nj_sub['PSNR'].mean())
        runs_nj_ssim.append(df_nj_sub['SSIM'].mean())
        runs_nj_lpips.append(df_nj_sub['LPIPS'].mean())


# ==============================================================================
# 4. TABELA 9: DIREKTNO POREĐENJE I BAZNA LINIJA ULAZA
# ==============================================================================
in_p_m, _, in_p_lo, in_p_hi = ci95(runs_in_psnr)
in_s_m, _, in_s_lo, in_s_hi = ci95(runs_in_ssim)
in_l_m, _, in_l_lo, in_l_hi = ci95(runs_in_lpips)

moj_full_p = cached_results["Full Proposed Model"]['PSNR'].values
moj_full_s = cached_results["Full Proposed Model"]['SSIM'].values
moj_full_l = cached_results["Full Proposed Model"]['LPIPS'].values

moj_p_m, _, moj_p_lo, moj_p_hi = ci95(raw_boot_data["Full Proposed Model"]['PSNR'])
moj_s_m, _, moj_s_lo, moj_s_hi = ci95(raw_boot_data["Full Proposed Model"]['SSIM'])
moj_l_m, _, moj_l_lo, moj_l_hi = ci95(raw_boot_data["Full Proposed Model"]['LPIPS'])

tabela_direktna = []
headers_direktna = ['Metrika', 'Ulaz (Bez Rest.)', 'Predloženi Model (95% CI)', 'Microsoft BOPBL', 'Δ (vs Ulaz)', 'Δ (vs BOPBL)', 'Wilcoxon', 't-test', "Cohen's d"]

if not df_njihov.empty:
    nj_full_p = df_njihov['PSNR'].values
    nj_full_s = df_njihov['SSIM'].values
    nj_full_l = df_njihov['LPIPS'].values

    nj_p_m, _, nj_p_lo, nj_p_hi = ci95(runs_nj_psnr)
    nj_s_m, _, nj_s_lo, nj_s_hi = ci95(runs_nj_ssim)
    nj_l_m, _, nj_l_lo, nj_l_hi = ci95(runs_nj_lpips)

    _, p_w_p = stats.wilcoxon(moj_full_p, nj_full_p)
    _, p_t_p = stats.ttest_rel(moj_full_p, nj_full_p)
    d_psnr = np.mean(moj_full_p - nj_full_p) / np.std(moj_full_p - nj_full_p, ddof=1)

    _, p_w_s = stats.wilcoxon(moj_full_s, nj_full_s)
    _, p_t_s = stats.ttest_rel(moj_full_s, nj_full_s)
    d_ssim = np.mean(moj_full_s - nj_full_s) / np.std(moj_full_s - nj_full_s, ddof=1)

    _, p_w_l = stats.wilcoxon(moj_full_l, nj_full_l)
    _, p_t_l = stats.ttest_rel(moj_full_l, nj_full_l)
    d_lpips = np.mean(moj_full_l - nj_full_l) / np.std(moj_full_l - nj_full_l, ddof=1)

    tabela_direktna = [
        ['PSNR (dB) [↑]', f"{in_p_m:.2f}", f"{moj_p_m:.2f} [{moj_p_lo:.2f}, {moj_p_hi:.2f}]", f"{nj_p_m:.2f} [{nj_p_lo:.2f}, {nj_p_hi:.2f}]", f"{moj_p_m - in_p_m:+.2f} dB", f"{moj_p_m - nj_p_m:+.2f} dB", format_p_val(p_w_p), format_p_val(p_t_p), f"{d_psnr:.2f}"],
        ['SSIM [↑]', f"{in_s_m:.4f}", f"{moj_s_m:.4f} [{moj_s_lo:.4f}, {moj_s_hi:.4f}]", f"{nj_s_m:.4f} [{nj_s_lo:.4f}, {nj_s_hi:.4f}]", f"{moj_s_m - in_s_m:+.4f}", f"{moj_s_m - nj_s_m:+.4f}", format_p_val(p_w_s), format_p_val(p_t_s), f"{d_ssim:.2f}"],
        ['LPIPS [↓]', f"{in_l_m:.4f}", f"{moj_l_m:.4f} [{moj_l_lo:.4f}, {moj_l_hi:.4f}]", f"{nj_l_m:.4f} [{nj_l_lo:.4f}, {nj_l_hi:.4f}]", f"{moj_l_m - in_l_m:+.4f}", f"{moj_l_m - nj_l_m:+.4f}", format_p_val(p_w_l), format_p_val(p_t_l), f"{d_lpips:.2f}"]
    ]
    pd.DataFrame(tabela_direktna, columns=headers_direktna).to_csv("tabela9.csv", index=False)


# ==============================================================================
# 5. TABELE ABLACIJE (ČIST FORMAT)
# ==============================================================================
summary_abl = []
raw_p_wilcoxon = []
raw_p_ttest = []
diffs_psnr = []
cohens_d_list = []
comp_names = []

for naziv, _ in ablation_configs:
    p_m, p_sd, p_lo, p_hi = ci95(raw_boot_data[naziv]['PSNR'])
    s_m, s_sd, s_lo, s_hi = ci95(raw_boot_data[naziv]['SSIM'])
    l_m, l_sd, l_lo, l_hi = ci95(raw_boot_data[naziv]['LPIPS'])

    summary_abl.append([
        naziv,
        f"{p_m:.2f} ± {p_sd:.2f} [{p_lo:.2f}, {p_hi:.2f}]",
        f"{s_m:.4f} ± {s_sd:.4f} [{s_lo:.4f}, {s_hi:.4f}]",
        f"{l_m:.4f} ± {l_sd:.4f} [{l_lo:.4f}, {l_hi:.4f}]"
    ])

    if naziv != "Full Proposed Model":
        var_p = cached_results[naziv]['PSNR'].values
        diff_p = var_p - moj_full_p
        _, p_w = stats.wilcoxon(moj_full_p, var_p)
        _, p_t = stats.ttest_rel(moj_full_p, var_p)
        d_val = np.mean(diff_p) / np.std(diff_p, ddof=1) if np.std(diff_p, ddof=1) > 0 else 0.0

        comp_names.append(naziv)
        diffs_psnr.append(np.mean(diff_p))
        raw_p_wilcoxon.append(p_w)
        raw_p_ttest.append(p_t)
        cohens_d_list.append(d_val)

adj_p_wilcoxon = holm_bonferroni(raw_p_wilcoxon)
adj_p_ttest = holm_bonferroni(raw_p_ttest)

headers_abl_mean = ['Konfiguracija Modela (5 Epoha Adaptacije)', 'PSNR (Mean ± SD, 95% CI) [↑]', 'SSIM (Mean ± SD, 95% CI) [↑]', 'LPIPS (Mean ± SD, 95% CI) [↓]']
pd.DataFrame(summary_abl, columns=headers_abl_mean).to_csv("tabela10.csv", index=False)

stat_abl_table = []
for name, d_psnr_val, p_w, p_w_adj, p_t_adj, d_val in zip(comp_names, diffs_psnr, raw_p_wilcoxon, adj_p_wilcoxon, adj_p_ttest, cohens_d_list):
    stat_abl_table.append([
        name,
        f"{d_psnr_val:+.2f} dB",
        format_p_val(p_w),
        format_p_val(p_w_adj),
        format_p_val(p_t_adj),
        f"{d_val:.2f}"
    ])

headers_abl_stat = ['Uklonjena Komponenta', 'Δ PSNR', 'Wilcoxon (Sirovo p)', 'Wilcoxon (Holm-Bonf.)', 't-test (Holm-Bonf.)', "Cohen's d"]


# ==============================================================================
# 6. ANALIZA PO KATEGORIJAMA (TEST CENTRALNE HIPOTEZE)
# ==============================================================================
def izdvoji_kategoriju(fname):
    fname_low = fname.lower()
    mapa = [
        (['pukot', 'crack'], 'Pukotine'),
        (['bud', 'mold', 'fung'], 'Buđ i biološko propadanje'),
        (['vlag', 'water', 'moist', 'anisotropic', 'mrlj', 'spot'], 'Vlaga i vodene mrlje'),
        (['boja', 'peel', 'paint', 'ljusten'], 'Ljuštenje boje'),
        (['zutilo', 'yellow', 'age', 'oxid', 'star', 'chemical', 'hemij'], 'Hemijsko starenje'),
        (['zamuc', 'blur', 'fft', 'lowpass', 'ostrin'], 'Gubitak oštrine'),
        (['pras', 'dust', 'ogreb', 'scratch'], 'Prašina i ogrebotine'),
        (['komb', 'comb'], 'Kombinovano oštećenje')
    ]
    for kljucevi, naziv in mapa:
        if any(k in fname_low for k in kljucevi):
            return naziv
    return "Ostale degradacije"

df_full = cached_results["Full Proposed Model"].copy()
df_no_spatial = cached_results["1. w/o Spatial Encoder Stream"].copy()
df_no_spectral = cached_results["2. w/o Spectral Encoder Stream"].copy()

df_full['Kategorija'] = [izdvoji_kategoriju(f) for f in df_full.index]
df_no_spatial['Kategorija'] = [izdvoji_kategoriju(f) for f in df_no_spatial.index]
df_no_spectral['Kategorija'] = [izdvoji_kategoriju(f) for f in df_no_spectral.index]

tabela_kategorije = []
for kat, grp in df_full.groupby('Kategorija'):
    idx = grp.index
    p_full = grp['PSNR'].mean()
    p_no_spat = df_no_spatial.loc[idx]['PSNR'].mean()
    p_no_spec = df_no_spectral.loc[idx]['PSNR'].mean()
    tabela_kategorije.append([
        kat, 
        len(idx), 
        f"{p_full:.2f} dB", 
        f"{p_no_spat:.2f} dB", 
        f"{p_no_spec:.2f} dB", 
        f"{p_no_spat - p_full:+.2f} dB", 
        f"{p_no_spec - p_full:+.2f} dB"
    ])

headers_kategorije = ['Kategorija Oštećenja', 'N Uzoraka', 'Full Model', 'w/o Spatial', 'w/o Spectral', 'Δ bez Spatial', 'Δ bez Spectral']


# ==============================================================================
# KONAČAN PRIKAZ ČISTIH TABELA U TERMINALU
# ==============================================================================
if not df_njihov.empty:
    print("\n" + "█" * 125)
    print(f"  1. DIREKTNO POREĐENJE I BAZNA LINIJA ULAZA [Klasterisani Bootstrap {NUM_ITERACIJA}× | 95% CI]")
    print("█" * 125)
    print(tabulate(tabela_direktna, headers=headers_direktna, tablefmt="fancy_grid", stralign="center", numalign="center"))

print("\n" + "█" * 125)
print(f"  2. ABLACIONA STUDIJA SA 5-EPOHNOM ADAPTACIJOM (N = {len(val_files)} slika, {len(unique_scenes)} scena)")
print("█" * 125)
print(tabulate(summary_abl, headers=headers_abl_mean, tablefmt="fancy_grid", stralign="center", numalign="center"))

print("\n" + "█" * 125)
print("  3. STATISTIČKA ZNAČAJNOST ABLACIJE SA HOLM-BONFERRONI KOREKCIJOM")
print("█" * 125)
print(tabulate(stat_abl_table, headers=headers_abl_stat, tablefmt="fancy_grid", stralign="center", numalign="center"))

print("\n" + "█" * 125)
print("  4. TEST CENTRALNE HIPOTEZE: PROSTORNI VS SPEKTRALNI TOK PO KATEGORIJAMA")
print("█" * 125)
print(tabulate(tabela_kategorije, headers=headers_kategorije, tablefmt="fancy_grid", stralign="center", numalign="center"))

print("\n✓ CSV fajlovi sačuvani za rad: 'tabela9.csv' i 'tabela10.csv'")
