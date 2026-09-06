# ==============================================================================
# FINALNA EVALUACIJA: PUNI MODEL + SVIH 10 ABLACIJA + STATISTIČKI TESTOVI
# (100% POKLAPANJE CHECKPOINT-A | 1000 BOOTSTRAP ITERACIJA | HOLM-BONFERRONI)
# ==============================================================================

import os
import sys
import random
import subprocess
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
from scipy import stats

try:
    import lpips
    from tabulate import tabulate
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lpips", "tabulate"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import lpips
    from tabulate import tabulate

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

NUM_BOOTSTRAP = 1000

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
except Exception:
    pass

DRIVE_PROJECT_DIR = '/content/drive/MyDrive/Projekat_Model'
DIR_ABLACIJA_CKPT = os.path.join(DRIVE_PROJECT_DIR, 'ablacija_checkpoints_customloss')
DIR_CKPT_PRAVI = os.path.join(DRIVE_PROJECT_DIR, 'ablacija_checkpoints_pravi')
DIR_CKPT_ALT = os.path.join(DRIVE_PROJECT_DIR, 'ablacija_checkpoints')

def pronadji_glavne_foldere(tip="VALIDACIJA"):
    moguce = [
        f"/content/drive/MyDrive/Projekat_Model/dataset/{tip}",
        f"/content/drive/MyDrive/Projekat_Model/dataset_njihov/{tip}_NJIHOV" if tip == "TRENING" else f"/content/drive/MyDrive/Projekat_Model/dataset_njihov/VALIDACIJA_NJIHOVA",
        f"./dataset/{tip}",
        f"/content/{tip}",
        f"/content/drive/MyDrive/Projekat_Model/{tip.lower()}"
    ]
    for b in moguce:
        if os.path.exists(b):
            c = os.path.join(b, "clean")
            d = os.path.join(b, "degraded")
            if os.path.exists(c) and os.path.exists(d) and len(os.listdir(d)) > 0:
                return c, d
            if os.path.exists(os.path.join(b, '0')) and os.path.exists(os.path.join(b, '1')):
                return os.path.join(b, '0'), os.path.join(b, '1')
    return None, None

DIR_VAL_CLEAN, DIR_VAL_DEGRADED = pronadji_glavne_foldere("VALIDACIJA")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 256
eval_lpips_fn = lpips.LPIPS(net='alex', verbose=False).to(device).eval()

# ==============================================================================
# DATASET
# ==============================================================================
class RestorationDataset(Dataset):
    def __init__(self, clean_dir, degraded_dir, img_size=256):
        self.img_size = img_size
        self.pairs = []
        d_files = sorted([f for f in os.listdir(degraded_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        for df in d_files:
            dp = os.path.join(degraded_dir, df)
            cp = os.path.join(clean_dir, df)
            if not os.path.exists(cp):
                parts = df.split('_')
                base_name = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else os.path.splitext(df)[0]
                clean_name = f"{base_name}_flip.jpg" if "_flip_" in df else f"{base_name}_clean.jpg"
                cp = os.path.join(clean_dir, clean_name)
            if os.path.exists(cp):
                self.pairs.append((dp, cp))

        print(f"[Dataset] Učitano {len(self.pairs)} validacionih parova slika.")
        if len(self.pairs) == 0:
            raise RuntimeError("[GREŠKA] Nisu pronađeni parovi slika u validacionom folderu!")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        d_p, c_p = self.pairs[idx]
        d_img = cv2.resize(cv2.cvtColor(cv2.imread(d_p), cv2.COLOR_BGR2RGB), (self.img_size, self.img_size))
        c_img = cv2.resize(cv2.cvtColor(cv2.imread(c_p), cv2.COLOR_BGR2RGB), (self.img_size, self.img_size))
        d_t = torch.from_numpy(d_img).permute(2, 0, 1).float() / 255.0
        c_t = torch.from_numpy(c_img).permute(2, 0, 1).float() / 255.0
        return d_t, c_t, os.path.basename(d_p)

# ==============================================================================
# GRADIVNI MODULI
# ==============================================================================
class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1, dilation: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(x))

class RecursiveDenseRestorationBlock(nn.Module):
    def __init__(self, channels: int, num_recursions: int = 2):
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
        self.low_conv = nn.Sequential(DepthwiseSeparableConv2d(channels, channels, 3, padding=1), nn.GroupNorm(4, channels), nn.ReLU(inplace=False))
        self.high_conv = nn.Sequential(DepthwiseSeparableConv2d(channels, channels, 3, padding=1), nn.GroupNorm(4, channels), nn.ReLU(inplace=False))
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels * 2, 2, 1), nn.Softmax(dim=1))
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
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.GroupNorm(4, out_ch), nn.ReLU(inplace=False))
        self.dense_micro = RecursiveDenseRestorationBlock(out_ch, num_recursions=3)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.conv(x)
        x = self.dense_micro(x)
        return self.pool(x), x

class AsymmetricCrossBridgeRestoration(nn.Module):
    def __init__(self, spatial_ch: int, spectral_ch: int, out_ch: int):
        super().__init__()
        self.spatial_to_spectral = nn.Sequential(nn.Conv2d(spatial_ch, spectral_ch, 1, bias=False), nn.GroupNorm(4, spectral_ch), nn.ReLU(inplace=False))
        self.spectral_to_spatial = nn.Sequential(nn.Conv2d(spectral_ch, spatial_ch, 1, bias=False), nn.GroupNorm(4, spatial_ch), nn.ReLU(inplace=False))
        self.fuse = nn.Conv2d(spatial_ch + spectral_ch, out_ch, 1, bias=False)
    def forward(self, spatial_feat: Tensor, spectral_feat: Tensor) -> Tensor:
        s_enh = spectral_feat + self.spatial_to_spectral(spatial_feat)
        sp_enh = spatial_feat + self.spectral_to_spatial(F.interpolate(spectral_feat, size=spatial_feat.shape[2:], mode='bilinear', align_corners=False))
        min_h, min_w = min(spatial_feat.shape[2], spectral_feat.shape[2]), min(spatial_feat.shape[3], spectral_feat.shape[3])
        return self.fuse(torch.cat([F.adaptive_avg_pool2d(sp_enh, (min_h, min_w)), F.adaptive_avg_pool2d(s_enh, (min_h, min_w))], dim=1))

class GatedFusionRestorationBlock(nn.Module):
    def __init__(self, spatial_ch: int, spectral_ch: int, out_ch: int):
        super().__init__()
        self.spatial_proj = nn.Conv2d(spatial_ch, out_ch, 1, bias=False)
        self.spectral_proj = nn.Conv2d(spectral_ch, out_ch, 1, bias=False)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(out_ch * 2, out_ch // 4, bias=False), nn.ReLU(inplace=False),
            nn.Linear(out_ch // 4, out_ch * 2, bias=False), nn.Sigmoid()
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
            nn.GroupNorm(4, in_channels // 4), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels // 4, 1, 1), nn.Sigmoid()
        )
        self.refine = nn.Sequential(DepthwiseSeparableConv2d(in_channels, in_channels, 3, padding=1), nn.GroupNorm(4, in_channels), nn.ReLU(inplace=False))
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        attn = self.attention(x)
        return self.refine(x * attn) + x, attn

class DecoderRestorationBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(in_ch, in_ch // 2, kernel_size=3, padding=1, bias=False))
        self.conv = nn.Sequential(nn.Conv2d(in_ch // 2 + skip_ch + 1, out_ch, 3, padding=1, bias=False), nn.GroupNorm(4, out_ch), nn.ReLU(inplace=False))
        self.dense_micro = RecursiveDenseRestorationBlock(out_ch, num_recursions=2)
        self.spectral = SpectralDecompositionRestorationBlock(out_ch)
    def forward(self, x: Tensor, skip: Tensor, damage_map: Tensor) -> Tensor:
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        dm = F.interpolate(damage_map, size=skip.shape[2:], mode='bilinear', align_corners=False)
        feat = self.dense_micro(self.conv(torch.cat([x, skip, dm], dim=1)))
        return self.spectral(feat)

class DecoderBlockNoSpectral(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(in_ch, in_ch // 2, 3, padding=1, bias=False))
        self.conv = nn.Sequential(nn.Conv2d(in_ch // 2 + skip_ch + 1, out_ch, 3, padding=1, bias=False), nn.GroupNorm(4, out_ch), nn.ReLU(inplace=False))
        self.dense_micro = RecursiveDenseRestorationBlock(out_ch, num_recursions=2)
    def forward(self, x: Tensor, skip: Tensor, damage_map: Tensor) -> Tensor:
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        dm = F.interpolate(damage_map, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.dense_micro(self.conv(torch.cat([x, skip, dm], dim=1)))

class DecoderRestorationBlock_NoDamageMap(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(in_ch, in_ch // 2, kernel_size=3, padding=1, bias=False))
        self.conv = nn.Sequential(nn.Conv2d(in_ch // 2 + skip_ch, out_ch, 3, padding=1, bias=False), nn.GroupNorm(4, out_ch), nn.ReLU(inplace=False))
        self.dense_micro = RecursiveDenseRestorationBlock(out_ch, num_recursions=2)
        self.spectral = SpectralDecompositionRestorationBlock(out_ch)
    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        feat = self.dense_micro(self.conv(torch.cat([x, skip], dim=1)))
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
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1, bias=False), nn.Sigmoid())
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
            nn.Conv2d(6, out_channels, 3, padding=1, bias=False), nn.GroupNorm(4, out_channels), nn.ReLU(inplace=False),
            DepthwiseSeparableConv2d(out_channels, out_channels, 3, padding=1), nn.GroupNorm(4, out_channels), nn.ReLU(inplace=False)
        )
    def forward(self, x: Tensor) -> Tensor:
        return self.conv(torch.cat([F.conv2d(x, self.kx, padding=1, groups=3), F.conv2d(x, self.ky, padding=1, groups=3)], dim=1))

class ContrastColorRecovery(nn.Module):
    def __init__(self, in_ch: int = 32, out_ch: int = 3):
        super().__init__()
        self.local_conv = nn.Sequential(nn.Conv2d(in_ch, in_ch // 2, 3, padding=1, bias=False), nn.GroupNorm(4, in_ch // 2), nn.ReLU(inplace=False), nn.Conv2d(in_ch // 2, out_ch, 3, padding=1))
        self.global_adjust = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_ch, in_ch // 4, 1, bias=False), nn.ReLU(inplace=False), nn.Conv2d(in_ch // 4, out_ch * 2, 1))
    def forward(self, x: Tensor, input_img: Tensor) -> Tensor:
        loc = self.local_conv(x)
        gain, bias = torch.chunk(self.global_adjust(x), 2, dim=1)
        gain = torch.sigmoid(gain).view(x.shape[0], -1, 1, 1) * 2.0
        bias = torch.tanh(bias).view(x.shape[0], -1, 1, 1) * 0.5
        return torch.clamp(input_img + loc * gain + bias, 0.0, 1.0)

# ==============================================================================
# DEFINICIJE ARHITEKTURA MODELA
# ==============================================================================

# PUNI MODEL
class FullRestauracija(nn.Module):
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
            nn.Conv2d(base_ch * 8, base_ch * 8, 1, bias=False), nn.GroupNorm(4, base_ch * 8), nn.ReLU(inplace=False),
            DilatedContextBlock(base_ch * 8), RecursiveDenseRestorationBlock(base_ch * 8, num_recursions=2)
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

# 1. BEZ SPATIAL STREAM-A
class Ablation1_NoSpatial(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 3, base_ch: int = 32):
        super().__init__()
        self.edge_branch = EdgeBranch(out_channels=base_ch)
        self.edge_fusion = nn.Conv2d(base_ch * 2, base_ch, 1, bias=False)
        self.spectral_init = nn.Sequential(nn.Conv2d(in_ch, base_ch, 3, padding=1, bias=False), nn.GroupNorm(4, base_ch), nn.ReLU(inplace=False))
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
        self.damage_attention = DamageAttentionRestorationModule(base_ch * 8)
        self.bottleneck_refine = nn.Sequential(
            nn.Conv2d(base_ch * 8, base_ch * 8, 1, bias=False), nn.GroupNorm(4, base_ch * 8), nn.ReLU(inplace=False),
            DilatedContextBlock(base_ch * 8), RecursiveDenseRestorationBlock(base_ch * 8, num_recursions=2)
        )
        self.decoder4 = DecoderRestorationBlock(base_ch * 8, base_ch * 8, base_ch * 4)
        self.decoder3 = DecoderRestorationBlock(base_ch * 4, base_ch * 4, base_ch * 2)
        self.decoder2 = DecoderRestorationBlock(base_ch * 2, base_ch * 2, base_ch)
        self.decoder1 = DecoderRestorationBlock(base_ch, base_ch, base_ch)
        self.final_refinement = nn.Sequential(RecursiveDenseRestorationBlock(base_ch, 2), SpectralDecompositionRestorationBlock(base_ch), RecursiveDenseRestorationBlock(base_ch, 2))
        self.contrast_color_recovery = ContrastColorRecovery(base_ch, out_ch)
    def forward(self, x: Tensor) -> Tensor:
        input_img = x
        sp1 = self.spectral_block1(self.spectral_init(x))
        sp2 = self.spectral_block2(self.spec_proj1(self.spectral_pool1(sp1)))
        sp3 = self.spectral_block3(self.spec_proj2(self.spectral_pool2(sp2)))
        sp4 = self.spectral_block4(self.spec_proj3(self.spectral_pool3(sp3)))
        attended, damage_map = self.damage_attention(sp4)
        b_out = self.bottleneck_refine(attended)
        d4 = self.decoder4(b_out, sp4, damage_map)
        d3 = self.decoder3(d4, sp3, damage_map)
        d2 = self.decoder2(d3, sp2, damage_map)
        d1 = self.decoder1(d2, sp1, damage_map)
        if d1.shape[2:] != input_img.shape[2:]:
            d1 = F.interpolate(d1, size=input_img.shape[2:], mode='bilinear', align_corners=False)
        refined = self.final_refinement(d1)
        edge_feat = self.edge_branch(input_img)
        return self.contrast_color_recovery(self.edge_fusion(torch.cat([refined, edge_feat], dim=1)), input_img)

# 2. BEZ SPECTRAL STREAM-A (100% POKLAPANJE)
class Ablation2_NoSpectral(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 3, base_ch: int = 32):
        super().__init__()
        self.edge_branch = EdgeBranch(out_channels=base_ch)
        self.edge_fusion = nn.Conv2d(base_ch * 2, base_ch, 1, bias=False)
        self.spatial_block1 = SpatialEncoderRestorationBlock(in_ch, base_ch)
        self.spatial_block2 = SpatialEncoderRestorationBlock(base_ch, base_ch * 2)
        self.spatial_block3 = SpatialEncoderRestorationBlock(base_ch * 2, base_ch * 4)
        self.spatial_block4 = SpatialEncoderRestorationBlock(base_ch * 4, base_ch * 8)
        self.damage_attention = DamageAttentionRestorationModule(base_ch * 8)
        self.bottleneck_refine = nn.Sequential(
            nn.Conv2d(base_ch * 8, base_ch * 8, 1, bias=False), nn.GroupNorm(4, base_ch * 8), nn.ReLU(inplace=False),
            DilatedContextBlock(base_ch * 8), RecursiveDenseRestorationBlock(base_ch * 8, num_recursions=2)
        )
        self.decoder4 = DecoderBlockNoSpectral(base_ch * 8, base_ch * 8, base_ch * 4)
        self.decoder3 = DecoderBlockNoSpectral(base_ch * 4, base_ch * 4, base_ch * 2)
        self.decoder2 = DecoderBlockNoSpectral(base_ch * 2, base_ch * 2, base_ch)
        self.decoder1 = DecoderBlockNoSpectral(base_ch, base_ch, base_ch)
        self.skip_gate1 = GatedSkipConnection(base_ch)
        self.skip_gate2 = GatedSkipConnection(base_ch * 2)
        self.skip_gate3 = GatedSkipConnection(base_ch * 4)
        self.skip_gate4 = GatedSkipConnection(base_ch * 8)
        self.skip_refine1 = nn.Sequential(RecursiveDenseRestorationBlock(base_ch, 2))
        self.skip_refine2 = nn.Sequential(RecursiveDenseRestorationBlock(base_ch * 2, 2))
        self.skip_refine3 = nn.Sequential(RecursiveDenseRestorationBlock(base_ch * 4, 2))
        self.skip_refine4 = nn.Sequential(RecursiveDenseRestorationBlock(base_ch * 8, 2))
        self.final_refinement = nn.Sequential(RecursiveDenseRestorationBlock(base_ch, 2), RecursiveDenseRestorationBlock(base_ch, 2))
        self.contrast_color_recovery = ContrastColorRecovery(base_ch, out_ch)

    def forward(self, x: Tensor) -> Tensor:
        input_img = x
        s1, s1_sk = self.spatial_block1(x)
        s2, s2_sk = self.spatial_block2(s1)
        s3, s3_sk = self.spatial_block3(s2)
        s4, s4_sk = self.spatial_block4(s3)
        attended, damage_map = self.damage_attention(s4)
        b_out = self.bottleneck_refine(attended)
        sk4 = self.skip_refine4(self.skip_gate4(s4_sk))
        sk3 = self.skip_refine3(self.skip_gate3(s3_sk))
        sk2 = self.skip_refine2(self.skip_gate2(s2_sk))
        sk1 = self.skip_refine1(self.skip_gate1(s1_sk))
        d4 = self.decoder4(b_out, sk4, damage_map)
        d3 = self.decoder3(d4, sk3, damage_map)
        d2 = self.decoder2(d3, sk2, damage_map)
        d1 = self.decoder1(d2, sk1, damage_map)
        if d1.shape[2:] != input_img.shape[2:]:
            d1 = F.interpolate(d1, size=input_img.shape[2:], mode='bilinear', align_corners=False)
        refined = self.final_refinement(d1)
        edge_feat = self.edge_branch(input_img)
        fused_out = self.edge_fusion(torch.cat([refined, edge_feat], dim=1))
        return self.contrast_color_recovery(fused_out, input_img)

# 3. BEZ ASYMMETRIC CROSS-BRIDGE
class Ablation3_NoCrossBridge(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 3, base_ch: int = 32):
        super().__init__()
        self.edge_branch = EdgeBranch(out_channels=base_ch)
        self.edge_fusion = nn.Conv2d(base_ch * 2, base_ch, 1, bias=False)
        self.spatial_block1 = SpatialEncoderRestorationBlock(in_ch, base_ch)
        self.spatial_block2 = SpatialEncoderRestorationBlock(base_ch, base_ch * 2)
        self.spatial_block3 = SpatialEncoderRestorationBlock(base_ch * 2, base_ch * 4)
        self.spatial_block4 = SpatialEncoderRestorationBlock(base_ch * 4, base_ch * 8)
        self.spectral_init = nn.Sequential(nn.Conv2d(in_ch, base_ch, 3, padding=1, bias=False), nn.GroupNorm(4, base_ch), nn.ReLU(inplace=False))
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
        self.gated_fusion = GatedFusionRestorationBlock(base_ch * 8, base_ch * 8, base_ch * 8)
        self.damage_attention = DamageAttentionRestorationModule(base_ch * 8)
        self.bottleneck_refine = nn.Sequential(
            nn.Conv2d(base_ch * 8, base_ch * 8, 1, bias=False), nn.GroupNorm(4, base_ch * 8), nn.ReLU(inplace=False),
            DilatedContextBlock(base_ch * 8), RecursiveDenseRestorationBlock(base_ch * 8, num_recursions=2)
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
        self.contrast_color_recovery = ContrastColorRecovery(base_ch, out_ch)
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
        fused = self.gated_fusion(s4, sp4)
        attended, damage_map = self.damage_attention(fused)
        b_out = self.bottleneck_refine(attended)
        sk4 = self.skip_refine4(self.skip_gate4(s4_skip))
        sk3 = self.skip_refine3(self.skip_gate3(s3_skip))
        sk2 = self.skip_refine2(self.skip_gate2(s2_skip))
        sk1 = self.skip_refine1(self.skip_gate1(s1_skip))
        d4 = self.decoder4(b_out, sk4, damage_map)
        d3 = self.decoder3(d4, sk3, damage_map)
        d2 = self.decoder2(d3, sk2, damage_map)
        d1 = self.decoder1(d2, sk1, damage_map)
        if d1.shape[2:] != input_img.shape[2:]:
            d1 = F.interpolate(d1, size=input_img.shape[2:], mode='bilinear', align_corners=False)
        refined = self.final_refinement(d1)
        edge_feat = self.edge_branch(input_img)
        return self.contrast_color_recovery(self.edge_fusion(torch.cat([refined, edge_feat], dim=1)), input_img)

# 4. BEZ GATED BOTTLENECK FUSION
class Ablation4_NoGatedFusion(nn.Module):
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
        self.simple_bottleneck_fuse = nn.Conv2d(base_ch * 16, base_ch * 8, 1, bias=False)
        self.damage_attention = DamageAttentionRestorationModule(base_ch * 8)
        self.bottleneck_refine = nn.Sequential(
            nn.Conv2d(base_ch * 8, base_ch * 8, 1, bias=False), nn.GroupNorm(4, base_ch * 8), nn.ReLU(inplace=False),
            DilatedContextBlock(base_ch * 8), RecursiveDenseRestorationBlock(base_ch * 8, num_recursions=2)
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
        sp4_resized = F.interpolate(sp4, size=s4_enriched.shape[2:], mode='bilinear', align_corners=False)
        fused = self.simple_bottleneck_fuse(torch.cat([s4_enriched, sp4_resized], dim=1))
        attended, damage_map = self.damage_attention(fused)
        b_out = self.bottleneck_refine(attended)
        c4_r = F.interpolate(c4, size=s4_skip.shape[2:], mode='bilinear', align_corners=False)
        c3_r = F.interpolate(c3, size=s3_skip.shape[2:], mode='bilinear', align_corners=False)
        c2_r = F.interpolate(c2, size=s2_skip.shape[2:], mode='bilinear', align_corners=False)
        c1_r = F.interpolate(c1, size=s1_skip.shape[2:], mode='bilinear', align_corners=False)
        sk4 = self.skip_refine4(self.skip_gate4(s4_skip) + c4_r)
        sk3 = self.skip_refine3(self.skip_gate3(s3_skip) + c3_r)
        sk2 = self.skip_refine2(self.skip_gate2(s2_skip) + c2_r)
        sk1 = self.skip_refine1(self.skip_gate1(s1_skip) + c1_r)
        d4 = self.decoder4(b_out, sk4, damage_map)
        d3 = self.decoder3(d4, sk3, damage_map)
        d2 = self.decoder2(d3, sk2, damage_map)
        d1 = self.decoder1(d2, sk1, damage_map)
        if d1.shape[2:] != input_img.shape[2:]:
            d1 = F.interpolate(d1, size=input_img.shape[2:], mode='bilinear', align_corners=False)
        refined = self.final_refinement(d1)
        edge_feat = self.edge_branch(input_img)
        return self.contrast_color_recovery(self.edge_fusion(torch.cat([refined, edge_feat], dim=1)), input_img)

# 5. BEZ DAMAGE ATTENTION
class Ablation5_NoDamageAttention(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 3, base_ch: int = 32):
        super().__init__()
        self.edge_branch = EdgeBranch(out_channels=base_ch)
        self.edge_fusion = nn.Conv2d(base_ch * 2, base_ch, 1, bias=False)
        self.spatial_block1 = SpatialEncoderRestorationBlock(in_ch, base_ch)
        self.spatial_block2 = SpatialEncoderRestorationBlock(base_ch, base_ch * 2)
        self.spatial_block3 = SpatialEncoderRestorationBlock(base_ch * 2, base_ch * 4)
        self.spatial_block4 = SpatialEncoderRestorationBlock(base_ch * 4, base_ch * 8)
        self.spectral_init = nn.Sequential(nn.Conv2d(in_ch, base_ch, 3, padding=1, bias=False), nn.GroupNorm(4, base_ch), nn.ReLU(inplace=False))
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
        self.bottleneck_refine = nn.Sequential(
            nn.Conv2d(base_ch * 8, base_ch * 8, 1, bias=False), nn.GroupNorm(4, base_ch * 8), nn.ReLU(inplace=False),
            DilatedContextBlock(base_ch * 8), RecursiveDenseRestorationBlock(base_ch * 8, num_recursions=2)
        )
        self.decoder4 = DecoderRestorationBlock_NoDamageMap(base_ch * 8, base_ch * 8, base_ch * 4)
        self.decoder3 = DecoderRestorationBlock_NoDamageMap(base_ch * 4, base_ch * 4, base_ch * 2)
        self.decoder2 = DecoderRestorationBlock_NoDamageMap(base_ch * 2, base_ch * 2, base_ch)
        self.decoder1 = DecoderRestorationBlock_NoDamageMap(base_ch, base_ch, base_ch)
        self.skip_gate1 = GatedSkipConnection(base_ch)
        self.skip_gate2 = GatedSkipConnection(base_ch * 2)
        self.skip_gate3 = GatedSkipConnection(base_ch * 4)
        self.skip_gate4 = GatedSkipConnection(base_ch * 8)
        self.skip_refine1 = nn.Sequential(RecursiveDenseRestorationBlock(base_ch, 2), SpectralDecompositionRestorationBlock(base_ch))
        self.skip_refine2 = nn.Sequential(RecursiveDenseRestorationBlock(base_ch * 2, 2), SpectralDecompositionRestorationBlock(base_ch * 2))
        self.skip_refine3 = nn.Sequential(RecursiveDenseRestorationBlock(base_ch * 4, 2), SpectralDecompositionRestorationBlock(base_ch * 4))
        self.skip_refine4 = nn.Sequential(RecursiveDenseRestorationBlock(base_ch * 8, 2), SpectralDecompositionRestorationBlock(base_ch * 8))
        self.final_refinement = nn.Sequential(RecursiveDenseRestorationBlock(base_ch, 2), SpectralDecompositionRestorationBlock(base_ch), RecursiveDenseRestorationBlock(base_ch, 2))
        self.contrast_color_recovery = ContrastColorRecovery(base_ch, out_ch)
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
        fused = self.gated_fusion(s4 + F.adaptive_avg_pool2d(c4, s4.shape[2:]), sp4)
        b_out = self.bottleneck_refine(fused)
        c4_r = F.interpolate(c4, size=s4_skip.shape[2:], mode='bilinear', align_corners=False)
        c3_r = F.interpolate(c3, size=s3_skip.shape[2:], mode='bilinear', align_corners=False)
        c2_r = F.interpolate(c2, size=s2_skip.shape[2:], mode='bilinear', align_corners=False)
        c1_r = F.interpolate(c1, size=s1_skip.shape[2:], mode='bilinear', align_corners=False)
        sk4 = self.skip_refine4(self.skip_gate4(s4_skip) + c4_r)
        sk3 = self.skip_refine3(self.skip_gate3(s3_skip) + c3_r)
        sk2 = self.skip_refine2(self.skip_gate2(s2_skip) + c2_r)
        sk1 = self.skip_refine1(self.skip_gate1(s1_skip) + c1_r)
        d4 = self.decoder4(b_out, sk4)
        d3 = self.decoder3(d4, sk3)
        d2 = self.decoder2(d3, sk2)
        d1 = self.decoder1(d2, sk1)
        if d1.shape[2:] != input_img.shape[2:]:
            d1 = F.interpolate(d1, size=input_img.shape[2:], mode='bilinear', align_corners=False)
        refined = self.final_refinement(d1)
        edge_feat = self.edge_branch(input_img)
        return self.contrast_color_recovery(self.edge_fusion(torch.cat([refined, edge_feat], dim=1)), input_img)

# 6. BEZ BOTTLENECK DILATED CONTEXT
class Ablation6_NoDilatedContext(nn.Module):
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
            nn.Conv2d(base_ch * 8, base_ch * 8, 1, bias=False), nn.GroupNorm(4, base_ch * 8), nn.ReLU(inplace=False),
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
        b_out = self.bottleneck_refine(attended)
        c4_r = F.interpolate(c4, size=s4_skip.shape[2:], mode='bilinear', align_corners=False)
        c3_r = F.interpolate(c3, size=s3_skip.shape[2:], mode='bilinear', align_corners=False)
        c2_r = F.interpolate(c2, size=s2_skip.shape[2:], mode='bilinear', align_corners=False)
        c1_r = F.interpolate(c1, size=s1_skip.shape[2:], mode='bilinear', align_corners=False)
        sk4 = self.skip_refine4(self.skip_gate4(s4_skip) + c4_r)
        sk3 = self.skip_refine3(self.skip_gate3(s3_skip) + c3_r)
        sk2 = self.skip_refine2(self.skip_gate2(s2_skip) + c2_r)
        sk1 = self.skip_refine1(self.skip_gate1(s1_skip) + c1_r)
        d4 = self.decoder4(b_out, sk4, damage_map)
        d3 = self.decoder3(d4, sk3, damage_map)
        d2 = self.decoder2(d3, sk2, damage_map)
        d1 = self.decoder1(d2, sk1, damage_map)
        if d1.shape[2:] != input_img.shape[2:]:
            d1 = F.interpolate(d1, size=input_img.shape[2:], mode='bilinear', align_corners=False)
        refined = self.final_refinement(d1)
        edge_feat = self.edge_branch(input_img)
        return self.contrast_color_recovery(self.edge_fusion(torch.cat([refined, edge_feat], dim=1)), input_img)

# 7. BEZ GATED SKIP CONNECTIONS
class Ablation7_NoGatedSkips(nn.Module):
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
            nn.Conv2d(base_ch * 8, base_ch * 8, 1, bias=False), nn.GroupNorm(4, base_ch * 8), nn.ReLU(inplace=False),
            DilatedContextBlock(base_ch * 8), RecursiveDenseRestorationBlock(base_ch * 8, num_recursions=2)
        )
        self.decoder4 = DecoderRestorationBlock(base_ch * 8, base_ch * 8, base_ch * 4)
        self.decoder3 = DecoderRestorationBlock(base_ch * 4, base_ch * 4, base_ch * 2)
        self.decoder2 = DecoderRestorationBlock(base_ch * 2, base_ch * 2, base_ch)
        self.decoder1 = DecoderRestorationBlock(base_ch, base_ch, base_ch)
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
        b_out = self.bottleneck_refine(attended)
        c4_r = F.interpolate(c4, size=s4_skip.shape[2:], mode='bilinear', align_corners=False)
        c3_r = F.interpolate(c3, size=s3_skip.shape[2:], mode='bilinear', align_corners=False)
        c2_r = F.interpolate(c2, size=s2_skip.shape[2:], mode='bilinear', align_corners=False)
        c1_r = F.interpolate(c1, size=s1_skip.shape[2:], mode='bilinear', align_corners=False)
        sk4 = self.skip_refine4(s4_skip + c4_r)
        sk3 = self.skip_refine3(s3_skip + c3_r)
        sk2 = self.skip_refine2(s2_skip + c2_r)
        sk1 = self.skip_refine1(s1_skip + c1_r)
        d4 = self.decoder4(b_out, sk4, damage_map)
        d3 = self.decoder3(d4, sk3, damage_map)
        d2 = self.decoder2(d3, sk2, damage_map)
        d1 = self.decoder1(d2, sk1, damage_map)
        if d1.shape[2:] != input_img.shape[2:]:
            d1 = F.interpolate(d1, size=input_img.shape[2:], mode='bilinear', align_corners=False)
        refined = self.final_refinement(d1)
        edge_feat = self.edge_branch(input_img)
        return self.contrast_color_recovery(self.edge_fusion(torch.cat([refined, edge_feat], dim=1)), input_img)

# 8. BEZ SKIP REFINEMENT BLOKOVA
class Ablation8_NoSkipRefine(nn.Module):
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
            nn.Conv2d(base_ch * 8, base_ch * 8, 1, bias=False), nn.GroupNorm(4, base_ch * 8), nn.ReLU(inplace=False),
            DilatedContextBlock(base_ch * 8), RecursiveDenseRestorationBlock(base_ch * 8, num_recursions=2)
        )
        self.decoder4 = DecoderRestorationBlock(base_ch * 8, base_ch * 8, base_ch * 4)
        self.decoder3 = DecoderRestorationBlock(base_ch * 4, base_ch * 4, base_ch * 2)
        self.decoder2 = DecoderRestorationBlock(base_ch * 2, base_ch * 2, base_ch)
        self.decoder1 = DecoderRestorationBlock(base_ch, base_ch, base_ch)
        self.skip_gate1 = GatedSkipConnection(base_ch)
        self.skip_gate2 = GatedSkipConnection(base_ch * 2)
        self.skip_gate3 = GatedSkipConnection(base_ch * 4)
        self.skip_gate4 = GatedSkipConnection(base_ch * 8)
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
        b_out = self.bottleneck_refine(attended)
        c4_r = F.interpolate(c4, size=s4_skip.shape[2:], mode='bilinear', align_corners=False)
        c3_r = F.interpolate(c3, size=s3_skip.shape[2:], mode='bilinear', align_corners=False)
        c2_r = F.interpolate(c2, size=s2_skip.shape[2:], mode='bilinear', align_corners=False)
        c1_r = F.interpolate(c1, size=s1_skip.shape[2:], mode='bilinear', align_corners=False)
        sk4 = self.skip_gate4(s4_skip) + c4_r
        sk3 = self.skip_gate3(s3_skip) + c3_r
        sk2 = self.skip_gate2(s2_skip) + c2_r
        sk1 = self.skip_gate1(s1_skip) + c1_r
        d4 = self.decoder4(b_out, sk4, damage_map)
        d3 = self.decoder3(d4, sk3, damage_map)
        d2 = self.decoder2(d3, sk2, damage_map)
        d1 = self.decoder1(d2, sk1, damage_map)
        if d1.shape[2:] != input_img.shape[2:]:
            d1 = F.interpolate(d1, size=input_img.shape[2:], mode='bilinear', align_corners=False)
        refined = self.final_refinement(d1)
        edge_feat = self.edge_branch(input_img)
        return self.contrast_color_recovery(self.edge_fusion(torch.cat([refined, edge_feat], dim=1)), input_img)

# 9. BEZ EDGE GUIDANCE BRANCHE
class Ablation9_NoEdgeBranch(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 3, base_ch: int = 32):
        super().__init__()
        self.spatial_block1 = SpatialEncoderRestorationBlock(in_ch, base_ch)
        self.spatial_block2 = SpatialEncoderRestorationBlock(base_ch, base_ch * 2)
        self.spatial_block3 = SpatialEncoderRestorationBlock(base_ch * 2, base_ch * 4)
        self.spatial_block4 = SpatialEncoderRestorationBlock(base_ch * 4, base_ch * 8)
        self.spectral_init = nn.Sequential(nn.Conv2d(in_ch, base_ch, 3, padding=1, bias=False), nn.GroupNorm(4, base_ch), nn.ReLU(inplace=False))
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
            nn.Conv2d(base_ch * 8, base_ch * 8, 1, bias=False), nn.GroupNorm(4, base_ch * 8), nn.ReLU(inplace=False),
            DilatedContextBlock(base_ch * 8), RecursiveDenseRestorationBlock(base_ch * 8, num_recursions=2)
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
        self.contrast_color_recovery = ContrastColorRecovery(base_ch, out_ch)
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
        fused = self.gated_fusion(s4 + F.adaptive_avg_pool2d(c4, s4.shape[2:]), sp4)
        attended, damage_map = self.damage_attention(fused)
        b_out = self.bottleneck_refine(attended)
        c4_r = F.interpolate(c4, size=s4_skip.shape[2:], mode='bilinear', align_corners=False)
        c3_r = F.interpolate(c3, size=s3_skip.shape[2:], mode='bilinear', align_corners=False)
        c2_r = F.interpolate(c2, size=s2_skip.shape[2:], mode='bilinear', align_corners=False)
        c1_r = F.interpolate(c1, size=s1_skip.shape[2:], mode='bilinear', align_corners=False)
        sk4 = self.skip_refine4(self.skip_gate4(s4_skip) + c4_r)
        sk3 = self.skip_refine3(self.skip_gate3(s3_skip) + c3_r)
        sk2 = self.skip_refine2(self.skip_gate2(s2_skip) + c2_r)
        sk1 = self.skip_refine1(self.skip_gate1(s1_skip) + c1_r)
        d4 = self.decoder4(b_out, sk4, damage_map)
        d3 = self.decoder3(d4, sk3, damage_map)
        d2 = self.decoder2(d3, sk2, damage_map)
        d1 = self.decoder1(d2, sk1, damage_map)
        if d1.shape[2:] != input_img.shape[2:]:
            d1 = F.interpolate(d1, size=input_img.shape[2:], mode='bilinear', align_corners=False)
        refined = self.final_refinement(d1)
        return self.contrast_color_recovery(refined, input_img)

# 10. BEZ CONTRAST COLOR RECOVERY (CCR) - TAČNO SA direct_conv (100% POKLAPANJE)
class Ablation10_NoCCR(nn.Module):
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
            nn.Conv2d(base_ch * 8, base_ch * 8, 1, bias=False), nn.GroupNorm(4, base_ch * 8), nn.ReLU(inplace=False),
            DilatedContextBlock(base_ch * 8), RecursiveDenseRestorationBlock(base_ch * 8, num_recursions=2)
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
        self.direct_conv = nn.Sequential(
            nn.Conv2d(base_ch, base_ch // 2, 3, padding=1, bias=False),
            nn.GroupNorm(4, base_ch // 2),
            nn.ReLU(inplace=False),
            nn.Conv2d(base_ch // 2, out_channels, 3, padding=1)
        )

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
        return torch.clamp(input_img + self.direct_conv(fused_out), 0.0, 1.0)

# ==============================================================================
# SIGURAN LOADER I EVALUATOR
# ==============================================================================
def build_and_load_model(model_class, ckpt_path):
    m = model_class().to(device)
    raw_state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(raw_state, dict):
        if 'state_dict' in raw_state:
            raw_state = raw_state['state_dict']
        elif 'model' in raw_state:
            raw_state = raw_state['model']
        elif 'model_state_dict' in raw_state:
            raw_state = raw_state['model_state_dict']

    cleaned_state = {}
    for k, v in raw_state.items():
        key = k[7:] if k.startswith('module.') else k
        cleaned_state[key] = v

    model_state = m.state_dict()
    matched_state = {}
    for k, v in cleaned_state.items():
        if k in model_state and model_state[k].shape == v.shape:
            matched_state[k] = v

    m.load_state_dict(matched_state, strict=False)
    m.eval()
    print(f"  [CKPT] Učitano {len(matched_state)}/{len(model_state)} tenzora iz: {os.path.basename(ckpt_path)}")
    return m

def evaluate_single_model(model, val_loader):
    model.eval()
    psnr_per_img, ssim_per_img, lpips_per_img, fnames = [], [], [], []

    with torch.no_grad():
        for d_t, c_t, fn in val_loader:
            d_t, c_t = d_t.to(device), c_t.to(device)
            out_t = torch.clamp(model(d_t), 0.0, 1.0)

            c_np = c_t.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            out_np = out_t.squeeze(0).cpu().numpy().transpose(1, 2, 0)

            psnr_v = psnr_metric(c_np, out_np, data_range=1.0)
            ssim_v = ssim_metric(c_np, out_np, channel_axis=2, data_range=1.0)

            out_eval = out_t * 2.0 - 1.0
            c_eval = c_t * 2.0 - 1.0
            lpips_v = eval_lpips_fn(out_eval, c_eval).item()

            psnr_per_img.append(psnr_v)
            ssim_per_img.append(ssim_v)
            lpips_per_img.append(lpips_v)
            fnames.append(fn[0])

    return np.array(psnr_per_img), np.array(ssim_per_img), np.array(lpips_per_img), fnames

# ==============================================================================
# STATISTIČKE FUNKCIJE
# ==============================================================================
def bootstrap_metrics(psnr_arr, ssim_arr, lpips_arr, num_bootstraps=1000, seed=42):
    n = len(psnr_arr)
    rng = np.random.default_rng(seed=seed)
    boot_p, boot_s, boot_l = [], [], []
    for _ in range(num_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        boot_p.append(np.mean(psnr_arr[idx]))
        boot_s.append(np.mean(ssim_arr[idx]))
        boot_l.append(np.mean(lpips_arr[idx]))
    return (
        np.mean(boot_p), np.std(boot_p),
        np.mean(boot_s), np.std(boot_s),
        np.mean(boot_l), np.std(boot_l)
    )

def format_p_exact(p):
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"

def holm_bonferroni(p_vals):
    m = len(p_vals)
    sorted_idx = np.argsort(p_vals)
    sorted_p = np.array(p_vals)[sorted_idx]
    adj = np.zeros(m)
    c_max = 0.0
    for i in range(m):
        v = min((m - i) * sorted_p[i], 1.0)
        c_max = max(c_max, v)
        adj[i] = c_max
    out = np.zeros(m)
    out[sorted_idx] = np.minimum(adj, 1.0)
    return out

# ==============================================================================
# GLAVNI TOK EVALUACIJE
# ==============================================================================
val_ds = RestorationDataset(clean_dir=DIR_VAL_CLEAN, degraded_dir=DIR_VAL_DEGRADED, img_size=IMG_SIZE)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

# 1. EVALUACIJA PUNOG MODELA
print("\n" + "="*85)
print("1. EVALUACIJA: PUNI PREDLOŽENI MODEL (BASELINE)")
print("="*85)

CKPT_FULL = None
for fld in [DIR_ABLACIJA_CKPT, DIR_CKPT_PRAVI, DIR_CKPT_ALT, DRIVE_PROJECT_DIR]:
    for fn in ["full_model_final_customloss.pth", "full_model_sepia25ep_customloss.pth", "model_final.pth", "full_model_final.pth"]:
        fp = os.path.join(fld, fn)
        if os.path.exists(fp):
            CKPT_FULL = fp
            break
    if CKPT_FULL:
        break

if not CKPT_FULL:
    print("[UPOZORENJE] Puni model checkpoint nije pronađen. Koristim default FullRestauracija.")
    full_model = FullRestauracija().to(device)
else:
    full_model = build_and_load_model(FullRestauracija, CKPT_FULL)

full_psnr, full_ssim, full_lpips, fnames_all = evaluate_single_model(full_model, val_loader)
fp_m, fp_sd, fs_m, fs_sd, fl_m, fl_sd = bootstrap_metrics(full_psnr, full_ssim, full_lpips, num_bootstraps=NUM_BOOTSTRAP)

print("\n" + "-"*55)
print(f"✓ PUNI MODEL REZULTATI ({NUM_BOOTSTRAP} Bootstrap iteracija):")
print(f"  PSNR  : {fp_m:.2f} ± {fp_sd:.2f} dB")
print(f"  SSIM  : {fs_m:.4f} ± {fs_sd:.4f}")
print(f"  LPIPS : {fl_m:.4f} ± {fl_sd:.4f}")
print("-" * 55 + "\n")

per_image_data = {
    'Filename': fnames_all,
    'Full_PSNR': full_psnr,
    'Full_SSIM': full_ssim,
    'Full_LPIPS': full_lpips
}

# 2. DEFINICIJA SVIH 10 ABLACIONIH MODELA
ABLACIJE_LISTA = [
    ("1. w/o Spatial Encoder Stream", ["ablation1_no_spatial_final_fixed.pth", "ablation1_no_spatial_sepia25ep_fixed.pth", "ablation1_no_spatial_final.pth"], Ablation1_NoSpatial),
    ("2. w/o Spectral Encoder Stream", ["ablation2_no_spectral_final_stable.pth", "ablation2_no_spectral_sepia25ep_stable.pth", "ablation2_no_spectral_final.pth"], Ablation2_NoSpectral),
    ("3. w/o Asymmetric Cross-Bridge", ["ablation3_no_crossbridge_final_fixed.pth", "ablation3_no_crossbridge_sepia25ep_fixed.pth", "ablation3_no_crossbridge_final.pth"], Ablation3_NoCrossBridge),
    ("4. w/o Gated Bottleneck Fusion", ["ablation4_no_gatedfusion_final_customloss.pth", "ablation4_no_gatedfusion_sepia25ep_customloss.pth", "ablation4_no_gatedfusion_final.pth"], Ablation4_NoGatedFusion),
    ("5. w/o Damage Attention Module", ["ablation5_no_damageattention_final_customloss.pth", "ablation5_no_damageattention_sepia25ep_customloss.pth", "ablation5_no_damageattention_final.pth"], Ablation5_NoDamageAttention),
    ("6. w/o Bottleneck Dilated Context", ["ablation6_no_dilatedcontext_final_customloss.pth", "ablation6_no_dilatedcontext_sepia25ep_customloss.pth", "ablation6_no_dilatedcontext_final.pth"], Ablation6_NoDilatedContext),
    ("7. w/o Gated Skip Connections", ["ablation7_no_gatedskips_final_customloss.pth", "ablation7_no_gatedskips_sepia25ep_customloss.pth", "ablation7_no_gatedskips_final.pth"], Ablation7_NoGatedSkips),
    ("8. w/o Skip Refinement Blocks", ["ablation8_no_skiprefine_final_customloss.pth", "ablation8_no_skiprefine_sepia25ep_customloss.pth", "ablation8_no_skiprefine_final.pth"], Ablation8_NoSkipRefine),
    ("9. w/o Edge Guidance Branch", ["ablation9_no_edgebranch_final_v2.pth", "ablation9_no_edgebranch_sepia25ep_v2.pth", "ablation9_no_edgebranch_final.pth"], Ablation9_NoEdgeBranch),
    ("10. w/o Contrast Color Recovery", ["ablation_no_ccr_final.pth", "ablation_no_ccr_sepia25ep.pth", "ablation10_no_ccr_final_customloss.pth"], Ablation10_NoCCR)
]

sve_metrike = [[
    "Full Proposed Model",
    f"{fp_m:.2f} ± {fp_sd:.2f}",
    f"{fs_m:.4f} ± {fs_sd:.4f}",
    f"{fl_m:.4f} ± {fl_sd:.4f}"
]]

stat_podaci = []

for naziv, ckpt_names, model_cls in ABLACIJE_LISTA:
    nadjen_fajl = None
    for folder in [DIR_ABLACIJA_CKPT, DIR_CKPT_PRAVI, DIR_CKPT_ALT, DRIVE_PROJECT_DIR]:
        for fn in ckpt_names:
            fp = os.path.join(folder, fn)
            if os.path.exists(fp):
                nadjen_fajl = fp
                break
        if nadjen_fajl:
            break

    if not nadjen_fajl:
        print(f"[PRESKOČENO] Fajl za '{naziv}' nije pronađen.")
        continue

    print(f"\n[EVALUACIJA] {naziv}...")
    abl_model = build_and_load_model(model_cls, nadjen_fajl)
    abl_p, abl_s, abl_l, _ = evaluate_single_model(abl_model, val_loader)

    short_k = f"Abl_{naziv.split('.')[0]}"
    per_image_data[f"{short_k}_PSNR"] = abl_p
    per_image_data[f"{short_k}_SSIM"] = abl_s
    per_image_data[f"{short_k}_LPIPS"] = abl_l

    # 1000 Bootstrap procena
    p_m, p_sd, s_m, s_sd, l_m, l_sd = bootstrap_metrics(abl_p, abl_s, abl_l, num_bootstraps=NUM_BOOTSTRAP)

    print(f"  --> PSNR : {p_m:.2f} dB (± {p_sd:.2f})")
    print(f"  --> SSIM : {s_m:.4f} (± {s_sd:.4f})")
    print(f"  --> LPIPS: {l_m:.4f} (± {l_sd:.4f})")

    sve_metrike.append([naziv, f"{p_m:.2f} ± {p_sd:.2f}", f"{s_m:.4f} ± {s_sd:.4f}", f"{l_m:.4f} ± {l_sd:.4f}"])

    # Statistički upareni testovi slika-po-slika
    diff = abl_p - full_psnr
    delta = p_m - fp_m

    try:
        _, w_p = stats.wilcoxon(abl_p, full_psnr, zero_method='pratt')
    except Exception:
        w_p = 1.0

    try:
        _, t_p = stats.ttest_rel(abl_p, full_psnr)
    except Exception:
        t_p = 1.0

    sd_d = np.std(diff, ddof=1)
    d_eff = np.mean(diff) / (sd_d + 1e-8)

    stat_podaci.append({
        'name': naziv,
        'delta': delta,
        'w_raw_p': w_p,
        't_raw_p': t_p,
        'd': d_eff
    })

# Čuvanje punog per-image CSV fajla
csv_per_img = os.path.join(DRIVE_PROJECT_DIR, "per_image_ablation_all_variants.csv")
pd.DataFrame(per_image_data).to_csv(csv_per_img, index=False)
print(f"\n✓ Sačuvane kompletne pojedinačne metrike po slikama na Drive:\n   -> {csv_per_img}")

# ==============================================================================
# ISPIS TABELA I ČUVANJE U CSV
# ==============================================================================

# TABELA 1: SVE METRIKE (PSNR, SSIM, LPIPS)
print("\n" + "="*85)
print(f"  TABELA 1: EVALUACIJA SVIH METRIKA ({NUM_BOOTSTRAP} BOOTSTRAP ITERACIJA)")
print("="*85)
print(tabulate(sve_metrike, headers=["Model / Ablacija", "PSNR [↑]", "SSIM [↑]", "LPIPS [↓]"], tablefmt="fancy_grid", stralign="center", numalign="center"))

csv_metrike = os.path.join(DRIVE_PROJECT_DIR, "rezultati_evaluacije_svih_modela.csv")
pd.DataFrame(sve_metrike[1:], columns=["Model / Ablacija", "PSNR", "SSIM", "LPIPS"]).to_csv(csv_metrike, index=False)
print(f"✓ Tabela metrika sačuvana na: {csv_metrike}")

# TABELA 2: STATISTIČKA ZNAČAJNOST (HOLM-BONFERRONI)
if len(stat_podaci) > 0:
    w_raw_list = [x['w_raw_p'] for x in stat_podaci]
    t_raw_list = [x['t_raw_p'] for x in stat_podaci]
    w_holm = holm_bonferroni(w_raw_list)
    t_holm = holm_bonferroni(t_raw_list)

    tabela_stat = []
    for i, item in enumerate(stat_podaci):
        tabela_stat.append([
            item['name'],
            f"{item['delta']:+.2f} dB",
            format_p_exact(item['w_raw_p']),
            format_p_exact(w_holm[i]),
            format_p_exact(t_holm[i]),
            f"{item['d']:.2f}"
        ])

    zaglavlja = ["Uklonjena Komponenta", "Δ PSNR", "Wilcoxon (Sirovo p)", "Wilcoxon (Holm-Bonf.)", "t-test (Holm-Bonf.)", "Cohen's d"]
    print("\n" + "="*95)
    print("  TABELA 2: STATISTIČKA ZNAČAJNOST ABLACIONE STUDIJE (HOLM-BONFERRONI KOREKCIJA)")
    print("="*95)
    print(tabulate(tabela_stat, headers=zaglavlja, tablefmt="fancy_grid", stralign="center", numalign="center"))

    csv_stat = os.path.join(DRIVE_PROJECT_DIR, "tabela_statisticka_znacajnost_ablacije.csv")
    pd.DataFrame(tabela_stat, columns=zaglavlja).to_csv(csv_stat, index=False)
    print(f"✓ Statistička tabela sačuvana na: {csv_stat}\n")
