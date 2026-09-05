#!/usr/bin/env python3
# ==============================================================================
# STATISTIČKA ANALIZA (5 ITERACIJA + BROJ PARAMETARA + ZNAČAJNOST PREMA REFERENCI)
# Referentne vrednosti sa slike: PSNR = 29.70 dB | SSIM = 0.8693 | LPIPS = 0.2428
# Protokol: 25 epoha Sepia + 5 epoha Fine-Tuning
# ==============================================================================

import os
import random
import argparse
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
import lpips
from scipy import stats

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

# REFERENTNE VREDNOSTI SA TVOJE SLIKE
REF_PSNR = 29.70
REF_SSIM = 0.8693
REF_LPIPS = 0.2428

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
            parts = df.split('_')
            base_name = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else os.path.splitext(df)[0]
            clean_name = f"{base_name}_flip.jpg" if "_flip_" in df else f"{base_name}_clean.jpg"
            cp = os.path.join(clean_dir, clean_name)
            if not os.path.exists(cp):
                cp = os.path.join(clean_dir, df)
            if os.path.exists(cp):
                self.pairs.append((dp, cp, df))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        d_p, c_p, fname = self.pairs[idx]
        d_img = cv2.resize(cv2.cvtColor(cv2.imread(d_p), cv2.COLOR_BGR2RGB), (self.img_size, self.img_size))
        c_img = cv2.resize(cv2.cvtColor(cv2.imread(c_p), cv2.COLOR_BGR2RGB), (self.img_size, self.img_size))
        d_t = torch.from_numpy(d_img).permute(2, 0, 1).float() / 255.0
        c_t = torch.from_numpy(c_img).permute(2, 0, 1).float() / 255.0
        return d_t, c_t, fname

# ==============================================================================
# STRUKTURA SLOJEVA (IDENTIČNA ČEKPOINTIMA)
# ==============================================================================
class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class SpatialEncoderRestorationBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(4, out_ch),
            nn.ReLU(inplace=False),
            DepthwiseSeparableConv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.ReLU(inplace=False)
        )
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        feat = self.conv(x)
        return self.pool(feat), feat

class SpectralDecompositionRestorationBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.low_conv = nn.Sequential(DepthwiseSeparableConv2d(channels, channels, 3, padding=1), nn.GroupNorm(4, channels), nn.ReLU(inplace=False))
        self.high_conv = nn.Sequential(DepthwiseSeparableConv2d(channels, channels, 3, padding=1), nn.GroupNorm(4, channels), nn.ReLU(inplace=False))
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels * 2, 2, 1), nn.Softmax(dim=1))
        self.fuse = nn.Conv2d(channels * 2, channels, 1, bias=False)
    def forward(self, x):
        low = F.interpolate(F.avg_pool2d(x, kernel_size=2), size=x.shape[2:], mode='bilinear', align_corners=False)
        high = x - low
        l_f = self.low_conv(low)
        h_f = self.high_conv(high)
        w = self.gate(torch.cat([l_f, h_f], dim=1))
        return self.fuse(torch.cat([w[:, 0:1] * l_f + w[:, 1:2] * h_f, x], dim=1))

class AsymmetricCrossBridgeRestoration(nn.Module):
    def __init__(self, spatial_ch, spectral_ch, out_ch):
        super().__init__()
        self.spatial_to_spectral = nn.Sequential(nn.Conv2d(spatial_ch, spectral_ch, 1, bias=False), nn.GroupNorm(4, spectral_ch), nn.ReLU(inplace=False))
        self.spectral_to_spatial = nn.Sequential(nn.Conv2d(spectral_ch, spatial_ch, 1, bias=False), nn.GroupNorm(4, spatial_ch), nn.ReLU(inplace=False))
        self.fuse = nn.Conv2d(spatial_ch + spectral_ch, out_ch, 1, bias=False)
    def forward(self, sp_feat, spec_feat):
        s_enh = spec_feat + self.spatial_to_spectral(sp_feat)
        sp_enh = sp_feat + self.spectral_to_spatial(F.interpolate(spec_feat, size=sp_feat.shape[2:], mode='bilinear', align_corners=False))
        min_h, min_w = min(sp_feat.shape[2], spec_feat.shape[2]), min(sp_feat.shape[3], spec_feat.shape[3])
        return self.fuse(torch.cat([F.adaptive_avg_pool2d(sp_enh, (min_h, min_w)), F.adaptive_avg_pool2d(s_enh, (min_h, min_w))], dim=1))

class DamageAttentionRestorationModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(nn.Conv2d(in_channels, in_channels // 4, 3, padding=1, bias=False), nn.GroupNorm(4, in_channels // 4), nn.ReLU(inplace=False), nn.Conv2d(in_channels // 4, 1, 1), nn.Sigmoid())
        self.refine = nn.Sequential(DepthwiseSeparableConv2d(in_channels, in_channels, 3, padding=1), nn.GroupNorm(4, in_channels), nn.ReLU(inplace=False))
    def forward(self, x):
        attn = self.attention(x)
        return self.refine(x * attn) + x, attn

class ContrastColorRecovery(nn.Module):
    def __init__(self, in_ch, out_ch=3):
        super().__init__()
        self.local_conv = nn.Sequential(nn.Conv2d(in_ch, in_ch // 2, 3, padding=1, bias=False), nn.GroupNorm(4, in_ch // 2), nn.ReLU(inplace=False), nn.Conv2d(in_ch // 2, out_ch, 3, padding=1))
        self.global_adjust = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_ch, in_ch // 4, 1, bias=False), nn.ReLU(inplace=False), nn.Conv2d(in_ch // 4, out_ch * 2, 1))
    def forward(self, x, input_img):
        l_ref = self.local_conv(x)
        gain, bias = torch.chunk(self.global_adjust(x), 2, dim=1)
        gain = torch.sigmoid(gain).view(x.shape[0], -1, 1, 1) * 2.0
        bias = torch.tanh(bias).view(x.shape[0], -1, 1, 1) * 0.5
        return torch.clamp(input_img + (l_ref * gain + bias), 0.0, 1.0)

class CleanDecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(in_ch, in_ch // 2, 3, padding=1, bias=False))
        self.conv = nn.Sequential(nn.Conv2d(in_ch // 2 + skip_ch + 1, out_ch, 3, padding=1, bias=False), nn.GroupNorm(4, out_ch), nn.ReLU(inplace=False), DepthwiseSeparableConv2d(out_ch, out_ch, 3, padding=1), nn.GroupNorm(4, out_ch), nn.ReLU(inplace=False))
    def forward(self, x, skip, damage_map):
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        dm = F.interpolate(damage_map, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip, dm], dim=1))

class SpatialEncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.GroupNorm(4, out_ch), nn.ReLU(inplace=False), DepthwiseSeparableConv2d(out_ch, out_ch, 3, padding=1), nn.GroupNorm(4, out_ch), nn.ReLU(inplace=False))
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        feat = self.conv(x)
        return self.pool(feat), feat

class SimpleDecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(in_ch, in_ch // 2, 3, padding=1, bias=False))
        self.conv = nn.Sequential(nn.Conv2d(in_ch // 2 + skip_ch, out_ch, 3, padding=1, bias=False), nn.GroupNorm(4, out_ch), nn.ReLU(inplace=False), DepthwiseSeparableConv2d(out_ch, out_ch, 3, padding=1), nn.GroupNorm(4, out_ch), nn.ReLU(inplace=False))
    def forward(self, x, skip):
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))

# ==============================================================================
# MODELI
# ==============================================================================
class FullCoreModel(nn.Module):
    def __init__(self, base_ch=32):
        super().__init__()
        self.spatial_block1 = SpatialEncoderRestorationBlock(3, base_ch)
        self.spatial_block2 = SpatialEncoderRestorationBlock(base_ch, base_ch * 2)
        self.spatial_block3 = SpatialEncoderRestorationBlock(base_ch * 2, base_ch * 4)
        self.spatial_block4 = SpatialEncoderRestorationBlock(base_ch * 4, base_ch * 8)

        self.spectral_init = nn.Sequential(nn.Conv2d(3, base_ch, 3, padding=1, bias=False), nn.GroupNorm(4, base_ch), nn.ReLU(inplace=False))
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

        self.bottleneck_fuse = nn.Sequential(nn.Conv2d(base_ch * 16, base_ch * 8, 1, bias=False), nn.GroupNorm(4, base_ch * 8), nn.ReLU(inplace=False))
        self.damage_attention = DamageAttentionRestorationModule(base_ch * 8)

        self.decoder4 = CleanDecoderBlock(base_ch * 8, base_ch * 8, base_ch * 4)
        self.decoder3 = CleanDecoderBlock(base_ch * 4, base_ch * 4, base_ch * 2)
        self.decoder2 = CleanDecoderBlock(base_ch * 2, base_ch * 2, base_ch)
        self.decoder1 = CleanDecoderBlock(base_ch, base_ch, base_ch)
        self.contrast_color_recovery = ContrastColorRecovery(base_ch, 3)

    def forward(self, x):
        inp = x
        sp1 = self.spectral_block1(self.spectral_init(x))
        sp2 = self.spectral_block2(self.spec_proj1(self.spectral_pool1(sp1)))
        sp3 = self.spectral_block3(self.spec_proj2(self.spectral_pool2(sp2)))
        sp4 = self.spectral_block4(self.spec_proj3(self.spectral_pool3(sp3)))

        s1, s1_sk = self.spatial_block1(x)
        s2, s2_sk = self.spatial_block2(s1)
        s3, s3_sk = self.spatial_block3(s2)
        s4, s4_sk = self.spatial_block4(s3)

        c1, c2, c3, c4 = self.cross1(s1_sk, sp1), self.cross2(s2_sk, sp2), self.cross3(s3_sk, sp3), self.cross4(s4_sk, sp4)
        sp4_al = F.interpolate(sp4, size=s4.shape[2:], mode='bilinear', align_corners=False)
        s4_enr = s4 + F.adaptive_avg_pool2d(c4, s4.shape[2:])
        bot, dmap = self.damage_attention(self.bottleneck_fuse(torch.cat([s4_enr, sp4_al], dim=1)))

        d4 = self.decoder4(bot, F.interpolate(c4, size=s4_sk.shape[2:], mode='bilinear', align_corners=False), dmap)
        d3 = self.decoder3(d4, F.interpolate(c3, size=s3_sk.shape[2:], mode='bilinear', align_corners=False), dmap)
        d2 = self.decoder2(d3, F.interpolate(c2, size=s2_sk.shape[2:], mode='bilinear', align_corners=False), dmap)
        d1 = self.decoder1(d2, F.interpolate(c1, size=s1_sk.shape[2:], mode='bilinear', align_corners=False), dmap)
        if d1.shape[2:] != inp.shape[2:]:
            d1 = F.interpolate(d1, size=inp.shape[2:], mode='bilinear', align_corners=False)
        return self.contrast_color_recovery(d1, inp)

class NoCrossBridgeModel(nn.Module):
    def __init__(self, base_ch=32):
        super().__init__()
        self.spatial_block1 = SpatialEncoderRestorationBlock(3, base_ch)
        self.spatial_block2 = SpatialEncoderRestorationBlock(base_ch, base_ch * 2)
        self.spatial_block3 = SpatialEncoderRestorationBlock(base_ch * 2, base_ch * 4)
        self.spatial_block4 = SpatialEncoderRestorationBlock(base_ch * 4, base_ch * 8)

        self.spectral_init = nn.Sequential(nn.Conv2d(3, base_ch, 3, padding=1, bias=False), nn.GroupNorm(4, base_ch), nn.ReLU(inplace=False))
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

        self.bottleneck_fuse = nn.Sequential(nn.Conv2d(base_ch * 16, base_ch * 8, 1, bias=False), nn.GroupNorm(4, base_ch * 8), nn.ReLU(inplace=False))
        self.damage_attention = DamageAttentionRestorationModule(base_ch * 8)

        self.decoder4 = CleanDecoderBlock(base_ch * 8, base_ch * 8, base_ch * 4)
        self.decoder3 = CleanDecoderBlock(base_ch * 4, base_ch * 4, base_ch * 2)
        self.decoder2 = CleanDecoderBlock(base_ch * 2, base_ch * 2, base_ch)
        self.decoder1 = CleanDecoderBlock(base_ch, base_ch, base_ch)
        self.contrast_color_recovery = ContrastColorRecovery(base_ch, 3)

    def forward(self, x):
        inp = x
        sp1 = self.spectral_block1(self.spectral_init(x))
        sp2 = self.spectral_block2(self.spec_proj1(self.spectral_pool1(sp1)))
        sp3 = self.spectral_block3(self.spec_proj2(self.spectral_pool2(sp2)))
        sp4 = self.spectral_block4(self.spec_proj3(self.spectral_pool3(sp3)))

        s1, s1_sk = self.spatial_block1(x)
        s2, s2_sk = self.spatial_block2(s1)
        s3, s3_sk = self.spatial_block3(s2)
        s4, s4_sk = self.spatial_block4(s3)

        sp4_al = F.interpolate(sp4, size=s4.shape[2:], mode='bilinear', align_corners=False)
        bot, dmap = self.damage_attention(self.bottleneck_fuse(torch.cat([s4, sp4_al], dim=1)))

        d4 = self.decoder4(bot, s4_sk, dmap)
        d3 = self.decoder3(d4, s3_sk, dmap)
        d2 = self.decoder2(d3, s2_sk, dmap)
        d1 = self.decoder1(d2, s1_sk, dmap)
        if d1.shape[2:] != inp.shape[2:]:
            d1 = F.interpolate(d1, size=inp.shape[2:], mode='bilinear', align_corners=False)
        return self.contrast_color_recovery(d1, inp)

class SpatialOnlyBaseline(nn.Module):
    def __init__(self, base_ch=32):
        super().__init__()
        self.enc1 = SpatialEncoderBlock(3, base_ch)
        self.enc2 = SpatialEncoderBlock(base_ch, base_ch * 2)
        self.enc3 = SpatialEncoderBlock(base_ch * 2, base_ch * 4)
        self.enc4 = SpatialEncoderBlock(base_ch * 4, base_ch * 8)
        self.bottleneck = nn.Sequential(DepthwiseSeparableConv2d(base_ch * 8, base_ch * 8, 3, padding=1), nn.GroupNorm(4, base_ch * 8), nn.ReLU(inplace=False))
        self.dec4 = SimpleDecoderBlock(base_ch * 8, base_ch * 8, base_ch * 4)
        self.dec3 = SimpleDecoderBlock(base_ch * 4, base_ch * 4, base_ch * 2)
        self.dec2 = SimpleDecoderBlock(base_ch * 2, base_ch * 2, base_ch)
        self.dec1 = SimpleDecoderBlock(base_ch, base_ch, base_ch)
        self.contrast_color_recovery = ContrastColorRecovery(base_ch, 3)

    def forward(self, x):
        inp = x
        s1, s1_sk = self.enc1(x)
        s2, s2_sk = self.enc2(s1)
        s3, s3_sk = self.enc3(s2)
        s4, s4_sk = self.enc4(s3)
        b = self.bottleneck(s4)
        d4 = self.dec4(b, s4_sk)
        d3 = self.dec3(d4, s3_sk)
        d2 = self.dec2(d3, s2_sk)
        d1 = self.dec1(d2, s1_sk)
        if d1.shape[2:] != inp.shape[2:]:
            d1 = F.interpolate(d1, size=inp.shape[2:], mode='bilinear', align_corners=False)
        return self.contrast_color_recovery(d1, inp)

# ==============================================================================
# BROJANJE PARAMETARA
# ==============================================================================
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total / 1e6  # u Milionima (M)

# ==============================================================================
# GLAVNI TOK
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--iterations", type=int, default=5, help="Broj iteracija sa permutacijom redosleda")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*90)
    print(f" STATISTIČKI TEST I POREĐENJE SA REFERENCOM (5 ITERACIJA) | Uređaj: {device}")
    print("="*90)
    print(f" Referenca punog modela (sa slike): PSNR = {REF_PSNR} dB | SSIM = {REF_SSIM} | LPIPS = {REF_LPIPS}")
    print(f" Protokol učenja: 25 Epoha (Sepia Pre-train) + 5 Epoha (Fine-Tuning)\n")

    eval_lpips_fn = lpips.LPIPS(net='alex', verbose=False).to(device).eval()

    target_root = os.path.join(args.data_dir, "dataset_target")
    if os.path.exists(os.path.join(target_root, "dataset")):
        target_root = os.path.join(target_root, "dataset")

    val_clean = os.path.join(target_root, "VALIDACIJA", "clean")
    val_deg = os.path.join(target_root, "VALIDACIJA", "degraded")

    ds = RestorationDataset(clean_dir=val_clean, degraded_dir=val_deg, img_size=256)
    print(f"Ukupno validacionih slika po iteraciji: {len(ds)}\n")

    # Učitavanje 3 modela
    models = {
        "1. Full Core Model (Proposed)": (FullCoreModel().to(device), "core_ablation_final.pth"),
        "2. w/o Asymmetric Cross-Bridge": (NoCrossBridgeModel().to(device), "no_crossbridge_final.pth"),
        "3. Vanilla Baseline (Spatial Only)": (SpatialOnlyBaseline().to(device), "spatial_only_final.pth")
    }

    model_stats = {}
    for name, (m, ckpt_name) in models.items():
        ckpt_path = os.path.join(args.output_dir, ckpt_name)
        if os.path.exists(ckpt_path):
            m.load_state_dict(torch.load(ckpt_path, map_location=device))
            m.eval()
            param_m = count_parameters(m)
            model_stats[name] = {"model": m, "params_M": param_m}
            print(f"✓ Učitan: {name:<35} | Parametri: {param_m:.2f} M | Checkpoint: {ckpt_name}")
        else:
            print(f"❌ Nije pronađen čekpoint: {ckpt_path}")

    # Izvršavanje kroz 5 iteracija (Bootstrap sa mešanjem redosleda)
    print(f"\nPokrećem {args.iterations} iteracija evaluacije sa nasumičnim rasporedom...")
    tabela_rezultati = []

    for name, data in model_stats.items():
        m = data["model"]
        iter_psnr, iter_ssim, iter_lpips = [], [], []

        for it in range(args.iterations):
            # Seed za svaku iteraciju garantuje permutaciju redosleda slika
            indices = list(range(len(ds)))
            random.seed(42 + it)
            random.shuffle(indices)

            cur_psnr, cur_ssim, cur_lpips = [], [], []
            with torch.no_grad():
                for idx in indices:
                    d_t, c_t, _ = ds[idx]
                    d_t, c_t = d_t.unsqueeze(0).to(device), c_t.unsqueeze(0).to(device)

                    out_t = torch.clamp(m(d_t), 0.0, 1.0)
                    c_np = c_t.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                    out_np = out_t.squeeze(0).cpu().numpy().transpose(1, 2, 0)

                    out_eval_t = out_t * 2.0 - 1.0
                    c_eval_t = c_t * 2.0 - 1.0

                    p = psnr_metric(c_np, out_np, data_range=1.0)
                    s = ssim_metric(c_np, out_np, channel_axis=2, data_range=1.0)
                    l = eval_lpips_fn(out_eval_t, c_eval_t).item()

                    cur_psnr.append(p)
                    cur_ssim.append(s)
                    cur_lpips.append(l)

            iter_psnr.append(np.mean(cur_psnr))
            iter_ssim.append(np.mean(cur_ssim))
            iter_lpips.append(np.mean(cur_lpips))

        # Statistička analiza u odnosu na referentne vrednosti sa slike
        p_m, p_s = np.mean(iter_psnr), np.std(iter_psnr)
        s_m, s_s = np.mean(iter_ssim), np.std(iter_ssim)
        l_m, l_s = np.mean(iter_lpips), np.std(iter_lpips)

        delta_psnr = p_m - REF_PSNR
        delta_ssim = s_m - REF_SSIM
        delta_lpips = l_m - REF_LPIPS

        # Jednouzorački t-test u odnosu na fiksnu referencu sa slike
        t_stat, p_val = stats.ttest_1samp(iter_psnr, REF_PSNR)

        znacajno = "DA (p < 0.05)" if p_val < 0.05 else "NE (p >= 0.05)"

        tabela_rezultati.append([
            name,
            f"{data['params_M']:.2f} M",
            f"{p_m:.2f} ± {p_s:.3f}",
            f"{s_m:.4f} ± {s_s:.4f}",
            f"{l_m:.4f} ± {l_s:.4f}",
            f"{delta_psnr:+.2f} dB",
            f"{p_val:.4e}" if p_val < 0.001 else f"{p_val:.4f}",
            znacajno
        ])

    print("\n" + "="*115)
    print(f"KONAČNA TABELA (3 MODELA KROZ {args.iterations} ITERACIJA SA TESTOM ZNAČAJNOSTI):")
    print("="*115)
    headers = ["Model / Konfiguracija", "Parametri", "PSNR [↑]", "SSIM [↑]", "LPIPS [↓]", "Delta vs Ref", "p-value", "Stat. Značajno?"]
    
    if tabulate:
        print(tabulate(tabela_rezultati, headers=headers, tablefmt="fancy_grid"))
    else:
        df_out = pd.DataFrame(tabela_rezultati, columns=headers)
        print(df_out.to_string(index=False))

    csv_out = os.path.join(args.output_dir, "tabela_3modela_5iteracija.csv")
    pd.DataFrame(tabela_rezultati, columns=headers).to_csv(csv_out, index=False)
    print(f"\n✓ Tabela sa 3 reda uspešno sačuvana na: {csv_out}\n")

if __name__ == '__main__':
    main()
