#!/usr/bin/env python3
# ==============================================================================
# EKSPERIMENT B: Vanilla Baseline (SAMO Spatial Encoder + Decoder + CCR)
# ==============================================================================

import os
import sys
import zipfile
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

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def log(msg):
    print(msg, flush=True)

class RestorationDataset(Dataset):
    def __init__(self, root_dir=None, clean_dir=None, degraded_dir=None, img_size=256, train=False):
        self.img_size = img_size
        self.train = train
        self.pairs = []

        if root_dir and os.path.exists(root_dir):
            search_dirs = [root_dir]
            for sub in os.listdir(root_dir):
                sub_p = os.path.join(root_dir, sub)
                if os.path.isdir(sub_p):
                    search_dirs.append(sub_p)

            for s_dir in search_dirs:
                f0 = os.path.join(s_dir, '0')
                f1 = os.path.join(s_dir, '1')
                if os.path.exists(f0) and os.path.exists(f1):
                    dmg_files = sorted([f for f in os.listdir(f1) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
                    for dmg_f in dmg_files:
                        parts = dmg_f.split('_')
                        base_name = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else os.path.splitext(dmg_f)[0]
                        clean_name = f"{base_name}_flip.jpg" if "_flip_" in dmg_f else f"{base_name}_clean.jpg"
                        c_p = os.path.join(f0, clean_name)
                        if not os.path.exists(c_p):
                            c_p = os.path.join(f0, dmg_f)
                        if os.path.exists(c_p):
                            self.pairs.append((os.path.join(f1, dmg_f), c_p))
                    if len(self.pairs) > 0:
                        break

        if len(self.pairs) == 0 and clean_dir and degraded_dir and os.path.exists(clean_dir) and os.path.exists(degraded_dir):
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

        log(f"[Dataset] Učitano {len(self.pairs)} validnih parova slika (Train={train}).")
        if len(self.pairs) == 0:
            raise RuntimeError(f"[GREŠKA] Nijedan par slika nije pronađen!")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        d_p, c_p = self.pairs[idx]
        d_img = cv2.resize(cv2.cvtColor(cv2.imread(d_p), cv2.COLOR_BGR2RGB), (self.img_size, self.img_size))
        c_img = cv2.resize(cv2.cvtColor(cv2.imread(c_p), cv2.COLOR_BGR2RGB), (self.img_size, self.img_size))

        d_t = torch.from_numpy(d_img).permute(2, 0, 1).float() / 255.0
        c_t = torch.from_numpy(c_img).permute(2, 0, 1).float() / 255.0

        if self.train:
            if random.random() > 0.5:
                d_t, c_t = torch.flip(d_t, dims=[2]), torch.flip(c_t, dims=[2])
            if random.random() > 0.5:
                d_t, c_t = torch.flip(d_t, dims=[1]), torch.flip(c_t, dims=[1])

        return d_t, c_t, os.path.basename(d_p)

# LOSS FUNKCIJE
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
        channel = img1.size(1)
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
        return torch.clamp(1.0 - ssim_map.mean(), 0.0, 2.0)

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

    def extract_features(self, x: Tensor) -> list:
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
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        error_map = torch.abs(pred - target).mean(dim=1, keepdim=True).detach()
        return 1.0 + torch.sigmoid((error_map - error_map.mean()) / (error_map.std() + 1e-6))

class FrequencyConsistencyLoss(nn.Module):
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

        pred_f = pred.float()
        target_f = target.float()
        pred_fft = torch.fft.rfft2(pred_f, norm='ortho')
        target_fft = torch.fft.rfft2(target_f, norm='ortho')
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

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        loss = self.single_scale_loss(pred, target)
        for scale in [0.5, 0.25]:
            p = F.interpolate(pred, scale_factor=scale, mode='bilinear', align_corners=False)
            t = F.interpolate(target, scale_factor=scale, mode='bilinear', align_corners=False)
            loss += scale * torch.mean(torch.sqrt((p - t) ** 2 + self.eps))
        return loss

# ARHITEKTURA SAMO SPATIAL + CCR
class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(x))

class SpatialEncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
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

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        feat = self.conv(x)
        return self.pool(feat), feat

class SimpleDecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, in_ch // 2, kernel_size=3, padding=1, bias=False)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch // 2 + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(4, out_ch),
            nn.ReLU(inplace=False),
            DepthwiseSeparableConv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.ReLU(inplace=False)
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))

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
        local_refinement = self.local_conv(x)
        gain, bias = torch.chunk(self.global_adjust(x), 2, dim=1)
        gain = torch.sigmoid(gain).view(x.shape[0], -1, 1, 1) * 2.0
        bias = torch.tanh(bias).view(x.shape[0], -1, 1, 1) * 0.5
        adjusted = local_refinement * gain + bias
        return torch.clamp(input_img + adjusted, 0.0, 1.0)

class SpatialOnlyModel(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_ch: int = 32):
        super().__init__()
        # Enkoder
        self.enc1 = SpatialEncoderBlock(in_channels, base_ch)
        self.enc2 = SpatialEncoderBlock(base_ch, base_ch * 2)
        self.enc3 = SpatialEncoderBlock(base_ch * 2, base_ch * 4)
        self.enc4 = SpatialEncoderBlock(base_ch * 4, base_ch * 8)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            DepthwiseSeparableConv2d(base_ch * 8, base_ch * 8, 3, padding=1),
            nn.GroupNorm(4, base_ch * 8),
            nn.ReLU(inplace=False)
        )

        # Dekoder
        self.dec4 = SimpleDecoderBlock(base_ch * 8, base_ch * 8, base_ch * 4)
        self.dec3 = SimpleDecoderBlock(base_ch * 4, base_ch * 4, base_ch * 2)
        self.dec2 = SimpleDecoderBlock(base_ch * 2, base_ch * 2, base_ch)
        self.dec1 = SimpleDecoderBlock(base_ch, base_ch, base_ch)

        # CCR
        self.contrast_color_recovery = ContrastColorRecovery(base_ch, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        input_img = x

        s1, s1_skip = self.enc1(x)
        s2, s2_skip = self.enc2(s1)
        s3, s3_skip = self.enc3(s2)
        s4, s4_skip = self.enc4(s3)

        b = self.bottleneck(s4)

        d4 = self.dec4(b, s4_skip)
        d3 = self.dec3(d4, s3_skip)
        d2 = self.dec2(d3, s2_skip)
        d1 = self.dec1(d2, s1_skip)

        if d1.shape[2:] != input_img.shape[2:]:
            d1 = F.interpolate(d1, size=input_img.shape[2:], mode='bilinear', align_corners=False)

        return self.contrast_color_recovery(d1, input_img)

def evaluate(model, loader, lpips_fn, device):
    model.eval()
    psnr_list, ssim_list, lpips_list = [], [], []
    with torch.no_grad():
        for d_t, c_t, _ in loader:
            d_t, c_t = d_t.to(device), c_t.to(device)
            out_t = torch.clamp(model(d_t), 0.0, 1.0)

            c_np = c_t.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            out_np = out_t.squeeze(0).cpu().numpy().transpose(1, 2, 0)

            out_eval_t = out_t * 2.0 - 1.0
            c_eval_t = c_t * 2.0 - 1.0

            psnr_v = psnr_metric(c_np, out_np, data_range=1.0)
            ssim_v = ssim_metric(c_np, out_np, channel_axis=2, data_range=1.0)
            lpips_v = lpips_fn(out_eval_t, c_eval_t).item()

            psnr_list.append(psnr_v)
            ssim_list.append(ssim_v)
            lpips_list.append(lpips_v)

    return np.mean(psnr_list), np.std(psnr_list), np.mean(ssim_list), np.std(ssim_list), np.mean(lpips_list), np.std(lpips_list)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accum_steps", type=int, default=3)
    parser.add_argument("--epochs_sepia", type=int, default=25)
    parser.add_argument("--epochs_ft", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"==================================================================")
    log(f"[POKRETANJE] Vanilla Baseline (Spatial+CCR) | Uređaj: {device}")
    log(f"==================================================================")

    eval_lpips_fn = lpips.LPIPS(net='alex', verbose=False).to(device).eval()

    sepia_extracted = os.path.join(args.data_dir, "dataset_sepia")
    target_extracted = os.path.join(args.data_dir, "dataset_target")

    sepia_train_path = os.path.join(sepia_extracted, "train") if os.path.exists(os.path.join(sepia_extracted, "train")) else sepia_extracted
    sepia_val_path = os.path.join(sepia_extracted, "val") if os.path.exists(os.path.join(sepia_extracted, "val")) else None

    target_root = target_extracted
    if os.path.exists(os.path.join(target_extracted, "dataset")):
        target_root = os.path.join(target_extracted, "dataset")

    ft_train_clean = os.path.join(target_root, "TRENING", "clean")
    ft_train_deg = os.path.join(target_root, "TRENING", "degraded")
    val_clean = os.path.join(target_root, "VALIDACIJA", "clean")
    val_deg = os.path.join(target_root, "VALIDACIJA", "degraded")

    log("\n[FAZA 1] Učitavanje Sepia trening i validacionog skupa...")
    sepia_train_ds = RestorationDataset(root_dir=sepia_train_path, img_size=256, train=True)
    sepia_train_loader = DataLoader(sepia_train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    sepia_val_loader = None
    if sepia_val_path and os.path.exists(sepia_val_path):
        sepia_val_ds = RestorationDataset(root_dir=sepia_val_path, img_size=256, train=False)
        sepia_val_loader = DataLoader(sepia_val_ds, batch_size=1, shuffle=False)

    model = SpatialOnlyModel(base_ch=32).to(device)
    criterion = RestauracijaLoss().to(device)

    ckpt_sepia = os.path.join(args.output_dir, "spatial_only_sepia25ep.pth")
    ckpt_final = os.path.join(args.output_dir, "spatial_only_final.pth")

    # FAZA 1
    if os.path.exists(ckpt_sepia):
        log(f"✓ Učitavam postojeći Sepia checkpoint: {ckpt_sepia}")
        model.load_state_dict(torch.load(ckpt_sepia, map_location=device))
    else:
        log(f"\n=======================================================")
        log(f" FAZA 1: Trening Spatial Only ({args.epochs_sepia} epoha na Sepia)")
        log(f"=======================================================")
        opt_pretrain = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        sched_pretrain = torch.optim.lr_scheduler.CosineAnnealingLR(opt_pretrain, T_max=args.epochs_sepia, eta_min=1e-6)

        for ep in range(args.epochs_sepia):
            model.train()
            running_loss = 0.0
            opt_pretrain.zero_grad()

            for batch_idx, (d_t, c_t, _) in enumerate(sepia_train_loader):
                d_t, c_t = d_t.to(device), c_t.to(device)
                pred = model(d_t)
                loss = criterion(pred, c_t)
                (loss / args.accum_steps).backward()

                if (batch_idx + 1) % args.accum_steps == 0 or (batch_idx + 1) == len(sepia_train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    opt_pretrain.step()
                    opt_pretrain.zero_grad()

                running_loss += loss.item()

            sched_pretrain.step()
            train_l = running_loss / len(sepia_train_loader)

            val_info = ""
            if sepia_val_loader is not None and (ep + 1) % 5 == 0:
                p_m, _, s_m, _, l_m, _ = evaluate(model, sepia_val_loader, eval_lpips_fn, device)
                val_info = f" | Sepia Val -> PSNR: {p_m:.2f}, SSIM: {s_m:.4f}, LPIPS: {l_m:.4f}"

            log(f" [Sepia Epoha {ep+1:02d}/{args.epochs_sepia:02d}] Loss: {train_l:.4f} | LR: {sched_pretrain.get_last_lr()[0]:.6f}{val_info}")

        torch.save(model.state_dict(), ckpt_sepia)
        log(f"✓ Sačuvan Sepia checkpoint: {ckpt_sepia}")

    # FAZA 2
    if os.path.exists(ckpt_final):
        log(f"✓ Učitavam postojeći Finalni checkpoint: {ckpt_final}")
        model.load_state_dict(torch.load(ckpt_final, map_location=device))
    elif os.path.exists(ft_train_clean) and os.path.exists(ft_train_deg):
        log(f"\n=======================================================")
        log(f" FAZA 2: Fine-Tuning ({args.epochs_ft} epoha na ciljnom skupu)")
        log(f"=======================================================")
        train_ds = RestorationDataset(clean_dir=ft_train_clean, degraded_dir=ft_train_deg, img_size=256, train=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        opt_ft = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
        sched_ft = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ft, T_max=args.epochs_ft, eta_min=1e-6)

        for ep in range(args.epochs_ft):
            model.train()
            running_loss = 0.0
            opt_ft.zero_grad()

            for batch_idx, (d_t, c_t, _) in enumerate(train_loader):
                d_t, c_t = d_t.to(device), c_t.to(device)
                pred = model(d_t)
                loss = criterion(pred, c_t)
                (loss / args.accum_steps).backward()

                if (batch_idx + 1) % args.accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    opt_ft.step()
                    opt_ft.zero_grad()

                running_loss += loss.item()

            sched_ft.step()
            log(f" [Fine-Tune Epoha {ep+1:02d}/{args.epochs_ft:02d}] Loss: {running_loss/len(train_loader):.4f}")

        torch.save(model.state_dict(), ckpt_final)
        log(f"✓ Sačuvan Finalni model: {ckpt_final}")

    # FAZA 3
    log(f"\n=======================================================")
    log(f" FAZA 3: Konačna Evaluacija na VALIDACIJA skupu")
    log(f"=======================================================")
    
    if os.path.exists(val_clean) and os.path.exists(val_deg):
        eval_ds = RestorationDataset(clean_dir=val_clean, degraded_dir=val_deg, img_size=256, train=False)
        eval_loader = DataLoader(eval_ds, batch_size=1, shuffle=False)

        p_m, p_sd, s_m, s_sd, l_m, l_sd = evaluate(model, eval_loader, eval_lpips_fn, device)

        rezultati = [[
            "Vanilla Baseline (Spatial+CCR Only)",
            f"{p_m:.2f} ± {p_sd:.2f}",
            f"{s_m:.4f} ± {s_sd:.4f}",
            f"{l_m:.4f} ± {l_sd:.4f}"
        ]]

        log("\n" + "="*80)
        log("REZULTAT EVALUACIJE NA KONAČNOJ VALIDACIJI:")
        log("="*80)
        if tabulate:
            log(tabulate(rezultati, headers=["Konfiguracija", "PSNR [↑]", "SSIM [↑]", "LPIPS [↓]"], tablefmt="fancy_grid"))
        else:
            log(f"PSNR: {p_m:.2f} ± {p_sd:.2f} | SSIM: {s_m:.4f} ± {s_sd:.4f} | LPIPS: {l_m:.4f} ± {l_sd:.4f}")

        csv_path = os.path.join(args.output_dir, "rezultat_spatial_only.csv")
        pd.DataFrame(rezultati, columns=["Konfiguracija", "PSNR", "SSIM", "LPIPS"]).to_csv(csv_path, index=False)
        log(f"\n✓ CSV izveštaj sačuvan na: {csv_path}\n")

if __name__ == '__main__':
    main()
