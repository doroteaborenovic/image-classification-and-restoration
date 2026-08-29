#ovde ide ablaciija plus statistika
import os
import sys
import glob
import random
import warnings
import subprocess
import itertools
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm
from scipy import stats
from scipy.linalg import sqrtm
from skimage.metrics import structural_similarity as ssim_metric

# da ne smara
warnings.filterwarnings('ignore')
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

random.seed(42)
np.random.seed(42)
import torch
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
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
from torchvision.models import vgg16, VGG16_Weights, inception_v3, Inception_V3_Weights

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
except Exception:
    pass

# putanjice do drive-a
DRIVE_PROJECT_DIR = '/content/drive/MyDrive/Projekat_Model'
os.makedirs(DRIVE_PROJECT_DIR, exist_ok=True)
DIR_ABLACIJA_DRIVE = os.path.join(DRIVE_PROJECT_DIR, 'ablacija')
os.makedirs(DIR_ABLACIJA_DRIVE, exist_ok=True)

EPOCHS_FINETUNE_NJIHOV = 3
BATCH_SIZE = 4
LR_NJIHOV = 1e-4
IMG_SIZE = 256
NUM_ITERACIJA = 5

def pronadji_foldere(tip="VALIDACIJA"):
    moguce = [
        f"/content/drive/MyDrive/Projekat_Model/dataset/{tip}",
        f"/content/drive/MyDrive/Projekat_Model/dataset_njihov/{tip}_NJIHOVA" if tip == "VALIDACIJA" else f"/content/drive/MyDrive/Projekat_Model/dataset_njihov/{tip}_NJIHOV",
        f"/content/dataset/{tip}",
        f"/content/dataset_njihov/{tip}_NJIHOVA" if tip == "VALIDACIJA" else f"/content/dataset_njihov/{tip}_NJIHOV",
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

DIR_TRAIN_CLEAN, DIR_TRAIN_DEGRADED, TRAIN_BASE = pronadji_foldere("TRENING")
DIR_VAL_CLEAN, DIR_VAL_DEGRADED, VAL_BASE = pronadji_foldere("VALIDACIJA")

DIR_MOJ_OUTPUT = '/content/rezultati_moj_model'
DIR_NJIHOV_ROOT = '/content/rezultati_njihov_model'
DIR_NJIHOV_OUTPUT = os.path.join(DIR_NJIHOV_ROOT, 'final_output')

os.makedirs(DIR_MOJ_OUTPUT, exist_ok=True)
os.makedirs(DIR_NJIHOV_ROOT, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eval_lpips_fn = lpips.LPIPS(net='alex', verbose=False).to(device).eval()

print(f"\n[INFO] Pokretanje evaluacije na resursu: {device} | Val uzorak: {len(os.listdir(DIR_VAL_DEGRADED))} slika | Broj iteracija: {NUM_ITERACIJA}\n")


# dataset loss arhitektura i to
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
        c_path = os.path.join(self.clean_dir, fname)
        d_path = os.path.join(self.degraded_dir, fname)

        c_img = cv2.resize(cv2.cvtColor(cv2.imread(c_path), cv2.COLOR_BGR2RGB), (self.img_size, self.img_size))
        d_img = cv2.resize(cv2.cvtColor(cv2.imread(d_path), cv2.COLOR_BGR2RGB), (self.img_size, self.img_size))

        c_t = torch.from_numpy(c_img).permute(2, 0, 1).float() / 255.0
        d_t = torch.from_numpy(d_img).permute(2, 0, 1).float() / 255.0

        if self.train:
            if random.random() > 0.5:
                c_t, d_t = torch.flip(c_t, dims=[2]), torch.flip(d_t, dims=[2])
            if random.random() > 0.5:
                c_t, d_t = torch.flip(c_t, dims=[1]), torch.flip(d_t, dims=[1])

        return d_t, c_t, fname

class VGGPerceptualLoss(nn.Module):    #isti loss kao i kod njih 
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


# njihove putanje do modela i to 
MS_REPO_DIR = '/content/Bringing-Old-Photos-Back-to-Life'

if not os.path.exists(MS_REPO_DIR):
    devnull = subprocess.DEVNULL
    subprocess.run(f"git clone -q https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life.git {MS_REPO_DIR}", shell=True, stdout=devnull, stderr=devnull)
    p1 = os.path.join(MS_REPO_DIR, 'Face_Enhancement/models/networks')
    p2 = os.path.join(MS_REPO_DIR, 'Global/detection_models')
    subprocess.run(f"cd {p1} && git clone -q https://github.com/vacancy/Synchronized-BatchNorm-PyTorch && cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .", shell=True, stdout=devnull, stderr=devnull)
    subprocess.run(f"cd {p2} && git clone -q https://github.com/vacancy/Synchronized-BatchNorm-PyTorch && cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .", shell=True, stdout=devnull, stderr=devnull)
    subprocess.run(f"cd {MS_REPO_DIR}/Face_Enhancement && wget -q https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases/download/v1.0/face_checkpoints.zip && unzip -q face_checkpoints.zip", shell=True, stdout=devnull, stderr=devnull)
    subprocess.run(f"cd {MS_REPO_DIR}/Global && wget -q https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases/download/v1.0/global_checkpoints.zip && unzip -q global_checkpoints.zip", shell=True, stdout=devnull, stderr=devnull)
    subprocess.run(f"cd {MS_REPO_DIR}/Face_Detection && wget -q http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 && bzip2 -d shape_predictor_68_face_landmarks.dat.bz2", shell=True, stdout=devnull, stderr=devnull)

if MS_REPO_DIR not in sys.path:
    sys.path.append(MS_REPO_DIR)
    sys.path.append(os.path.join(MS_REPO_DIR, 'Global'))

ms_ckpt_path = os.path.join(MS_REPO_DIR, 'Global/checkpoints/restoration/latest_net_G.pth')
MS_DRIVE_CKPT = os.path.join(DRIVE_PROJECT_DIR, 'microsoft_finetuned_3_epochs.pth')

try:
    from Global.models.networks import define_G
    ms_generator = define_G(3, 3, 64, "global", 3, 9, 1, 4, "instance", gpu_ids=[0]).to(device)
    if os.path.exists(ms_ckpt_path):
        ms_generator.load_state_dict(torch.load(ms_ckpt_path, map_location=device))
except Exception:
    ms_generator = None

# Provera da li je njihov model već fine-tunovan i sačuvan na Drive-u
if os.path.exists(MS_DRIVE_CKPT):
    print(f"✓ Pronađen sačuvan fine-tunovan Microsoft model: {MS_DRIVE_CKPT}")
    if ms_generator is not None:
        ms_state = torch.load(MS_DRIVE_CKPT, map_location=device)
        ms_generator.load_state_dict(ms_state)
        torch.save(ms_state, ms_ckpt_path)
        print("✓ Microsoft checkpoint uspešno učitan sa Google Drive-a. Preskače se ponovno treniranje.")
elif ms_generator is not None and EPOCHS_FINETUNE_NJIHOV > 0:
    print(f"-> Fine-tuning Microsoft modela na {EPOCHS_FINETUNE_NJIHOV} epohe...")
    train_ds = PairedDataset(DIR_TRAIN_CLEAN, DIR_TRAIN_DEGRADED, img_size=IMG_SIZE, train=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    optimizer_ms = torch.optim.AdamW(ms_generator.parameters(), lr=LR_NJIHOV, weight_decay=1e-4)
    crit_l1 = nn.L1Loss()
    crit_vgg = VGGPerceptualLoss().to(device)
    scaler_ms = torch.amp.GradScaler('cuda')

    for ep in range(EPOCHS_FINETUNE_NJIHOV):
        ms_generator.train()
        total_ep_loss = 0.0
        for d_t, c_t, _ in train_loader:
            d_t, c_t = d_t.to(device), c_t.to(device)
            optimizer_ms.zero_grad()
            with torch.amp.autocast('cuda'):
                pred = (ms_generator(d_t * 2.0 - 1.0) + 1.0) / 2.0
                loss = crit_l1(pred, c_t) + 0.1 * crit_vgg(pred, c_t)
            scaler_ms.scale(loss).backward()
            scaler_ms.step(optimizer_ms)
            scaler_ms.update()
            total_ep_loss += loss.item()
        print(f"   [Epoha {ep + 1}/{EPOCHS_FINETUNE_NJIHOV}] Loss: {total_ep_loss / len(train_loader):.4f}")

    torch.save(ms_generator.state_dict(), ms_ckpt_path)
    torch.save(ms_generator.state_dict(), MS_DRIVE_CKPT)
    print(f"✓ Fine-tunovani Microsoft model sačuvan na Drive: {MS_DRIVE_CKPT}")

subprocess.run(
    f"cd {MS_REPO_DIR} && python run.py --input_folder {DIR_VAL_DEGRADED} --output_folder {DIR_NJIHOV_ROOT} --GPU 0 --with_scratch",
    shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
)

if not os.path.exists(DIR_NJIHOV_OUTPUT) or len(os.listdir(DIR_NJIHOV_OUTPUT)) == 0:
    for alt in [os.path.join(DIR_NJIHOV_ROOT, 'stage_3_restore_output'), os.path.join(DIR_NJIHOV_ROOT, 'restored_image')]:
        if os.path.exists(alt) and len(os.listdir(alt)) > 0:
            DIR_NJIHOV_OUTPUT = alt
            break


# moja arhitektura za restauraciju
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
        self.gate = nn.Entry = nn.Sequential(
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
        
        # Spatial Encoder
        self.spatial_block1 = SpatialEncoderRestorationBlock(in_channels, base_ch)
        self.spatial_block2 = SpatialEncoderRestorationBlock(base_ch, base_ch * 2)
        self.spatial_block3 = SpatialEncoderRestorationBlock(base_ch * 2, base_ch * 4)
        self.spatial_block4 = SpatialEncoderRestorationBlock(base_ch * 4, base_ch * 8)
        
        # Spectral Encoder
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
        
        # Cross-Bridge
        self.cross1 = AsymmetricCrossBridgeRestoration(base_ch, base_ch, base_ch)
        self.cross2 = AsymmetricCrossBridgeRestoration(base_ch * 2, base_ch * 2, base_ch * 2)
        self.cross3 = AsymmetricCrossBridgeRestoration(base_ch * 4, base_ch * 4, base_ch * 4)
        self.cross4 = AsymmetricCrossBridgeRestoration(base_ch * 8, base_ch * 8, base_ch * 8)
        
        # Bottleneck
        self.gated_fusion = GatedFusionRestorationBlock(base_ch * 8, base_ch * 8, base_ch * 8)
        self.damage_attention = DamageAttentionRestorationModule(base_ch * 8)
        self.bottleneck_refine = nn.Sequential(
            nn.Conv2d(base_ch * 8, base_ch * 8, 1, bias=False),
            nn.GroupNorm(4, base_ch * 8),
            nn.ReLU(inplace=False),
            DilatedContextBlock(base_ch * 8),
            RecursiveDenseRestorationBlock(base_ch * 8, num_recursions=2)
        )
        
        # Skips & Decoders
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

        # 1. Spectral Stream
        if use_spectral:
            sp1 = self.spectral_block1(self.spectral_init(x))
            sp2 = self.spectral_block2(self.spec_proj1(self.spectral_pool1(sp1)))
            sp3 = self.spectral_block3(self.spec_proj2(self.spectral_pool2(sp2)))
            sp4 = self.spectral_block4(self.spec_proj3(self.spectral_pool3(sp3)))
        else:
            sp1 = sp2 = sp3 = sp4 = None

        # 2. Spatial Encoder Stream
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

        # 3. Asymmetric Cross-Bridge
        if use_cross_bridge and use_spatial and use_spectral:
            c1, c2, c3, c4 = self.cross1(s1_skip, sp1), self.cross2(s2_skip, sp2), self.cross3(s3_skip, sp3), self.cross4(s4_skip, sp4)
            s4_enriched = s4 + F.adaptive_avg_pool2d(c4, s4.shape[2:])
        else:
            c1, c2, c3, c4 = torch.zeros_like(s1_skip), torch.zeros_like(s2_skip), torch.zeros_like(s3_skip), torch.zeros_like(s4_skip)
            s4_enriched = s4

        # 4. Gated Bottleneck Fusion
        if use_gated_fusion:
            fused = self.gated_fusion(s4_enriched, sp4)
        else:
            fused = (s4_enriched + F.interpolate(sp4, size=s4_enriched.shape[2:], mode='bilinear', align_corners=False)) * 0.5

        # 5. Damage Attention
        if use_damage_attention:
            attended, damage_map = self.damage_attention(fused)
        else:
            attended = fused
            damage_map = torch.zeros(fused.shape[0], 1, fused.shape[2], fused.shape[3], device=fused.device)

        # 6. Bottleneck Context Refinement
        if use_dilated_context:
            bottleneck_out = self.bottleneck_refine(attended)
        else:
            b = self.bottleneck_refine[2](self.bottleneck_refine[1](self.bottleneck_refine[0](attended)))
            bottleneck_out = self.bottleneck_refine[4](b)

        # 7. Gated Skip Connections & Skip Refinement
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

        # 8. Decoders
        d4 = self.decoder4(bottleneck_out, sk4_final, damage_map, use_spectral=use_spectral)
        d3 = self.decoder3(d4, sk3_final, damage_map, use_spectral=use_spectral)
        d2 = self.decoder2(d3, sk2_final, damage_map, use_spectral=use_spectral)
        d1 = self.decoder1(d2, sk1_final, damage_map, use_spectral=use_spectral)

        if d1.shape[2:] != input_img.shape[2:]:
            d1 = F.interpolate(d1, size=input_img.shape[2:], mode='bilinear', align_corners=False)

        refined = self.final_refinement(d1)

        # 9. Edge Guidance Branch
        if use_edge_branch:
            edge_feat = self.edge_branch(input_img)
            fused_out = self.edge_fusion(torch.cat([refined, edge_feat], dim=1))
        else:
            fused_out = refined

        # 10. Contrast Color Recovery
        if use_ccr:
            return self.contrast_color_recovery(fused_out, input_img)
        else:
            loc = self.contrast_color_recovery.local_conv(fused_out)
            return torch.clamp(input_img + loc, 0.0, 1.0)


# ucitavanje mog modela 
moj_model = Restauracija(base_ch=32).to(device)

model_loaded = False
priority_ckpts = [
    'Model_Finetuned_Final.pth'
]

for ckpt_name in priority_ckpts:
    ckpt_p = os.path.join(DRIVE_PROJECT_DIR, ckpt_name)
    if os.path.exists(ckpt_p):
        print(f"✓ Učitavam sačuvani model iz: {ckpt_p}")
        ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
        moj_model.load_state_dict(ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt, strict=False)
        model_loaded = True
        break

if not model_loaded:
    print("nema niceg na driveu")

moj_model.eval()
val_files = sorted([f for f in os.listdir(DIR_VAL_DEGRADED) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

with torch.no_grad():
    for fname in val_files:
        d_bgr = cv2.imread(os.path.join(DIR_VAL_DEGRADED, fname))
        if d_bgr is None: continue
        d_rgb = cv2.resize(cv2.cvtColor(d_bgr, cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE))
        d_t = torch.from_numpy(d_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        out_np = (torch.clamp(moj_model(d_t), 0.0, 1.0).squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
        cv2.imwrite(os.path.join(DIR_MOJ_OUTPUT, fname), cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR))


# metrike i to 
inception = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False).to(device).eval()
inception.fc = nn.Identity()

def extract_features(folder, files):
    feats = []
    with torch.no_grad():
        for f in files:
            p = os.path.join(folder, f)
            if not os.path.exists(p):
                p = os.path.join(folder, f"{os.path.splitext(f)[0]}.png")
            img = cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), (299, 299)).astype(np.float32) / 255.0
            t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
            t = (t - 0.5) * 2.0
            feat = inception(t).cpu().numpy().reshape(-1)
            feats.append(feat)
    return np.array(feats)

def calculate_fid(act1, act2):
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    sigma1 += np.eye(sigma1.shape[0]) * 1e-6
    sigma2 += np.eye(sigma2.shape[0]) * 1e-6
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean))

def format_p_val(p):
    if p < 0.001:
        return "< 0.001 ***"
    elif p < 0.01:
        return f"{p:.4f} **"
    elif p < 0.05:
        return f"{p:.4f} *"
    else:
        return f"{p:.4f} (ns)"


# =============================================================
# 0. DETERMINISTIČKO RAČUNANJE NA CELOM VALIDACIONOM SETU
#    (Zasebno za čistu i validnu statistiku bez duplikata)
# =============================================================
direct_per_image_stat = []

for fname in val_files:
    clean_p = os.path.join(DIR_VAL_CLEAN, fname)
    moj_p = os.path.join(DIR_MOJ_OUTPUT, fname)
    nj_p = os.path.join(DIR_NJIHOV_OUTPUT, fname)
    if not os.path.exists(nj_p):
        nj_p = os.path.join(DIR_NJIHOV_OUTPUT, f"{os.path.splitext(fname)[0]}.png")

    if not (os.path.exists(clean_p) and os.path.exists(moj_p) and os.path.exists(nj_p)):
        continue

    c_img = cv2.resize(cv2.cvtColor(cv2.imread(clean_p), cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    m_img = cv2.resize(cv2.cvtColor(cv2.imread(moj_p), cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    nj_img = cv2.resize(cv2.cvtColor(cv2.imread(nj_p), cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0

    c_t = torch.from_numpy(c_img).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
    m_t = torch.from_numpy(m_img).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
    nj_t = torch.from_numpy(nj_img).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0

    with torch.no_grad():
        lp_moj = eval_lpips_fn(m_t, c_t).item()
        lp_nj = eval_lpips_fn(nj_t, c_t).item()

    mse_m = np.mean((c_img - m_img) ** 2)
    psnr_moj = 10.0 * np.log10(1.0 / mse_m) if mse_m > 0 else 100.0

    mse_nj = np.mean((c_img - nj_img) ** 2)
    psnr_nj = 10.0 * np.log10(1.0 / mse_nj) if mse_nj > 0 else 100.0

    ssim_moj = ssim_metric(c_img, m_img, channel_axis=2, data_range=1.0)
    ssim_nj = ssim_metric(c_img, nj_img, channel_axis=2, data_range=1.0)

    direct_per_image_stat.append({
        'Fname': fname,
        'PSNR_Moj': psnr_moj, 'PSNR_Njihov': psnr_nj,
        'SSIM_Moj': ssim_moj, 'SSIM_Njihov': ssim_nj,
        'LPIPS_Moj': lp_moj, 'LPIPS_Njihov': lp_nj
    })

df_stat_direct = pd.DataFrame(direct_per_image_stat).set_index('Fname')


# -------------------------------------------------------------
# DIREKTNO POREĐENJE SA 5 ITERACIJA (Bootstrap Resampling)
# -------------------------------------------------------------
runs_direct_moj_psnr = []
runs_direct_moj_ssim = []
runs_direct_moj_lpips = []

runs_direct_nj_psnr = []
runs_direct_nj_ssim = []
runs_direct_nj_lpips = []

for it in range(NUM_ITERACIJA):
    rng = np.random.default_rng(seed=42 + it)
    boot_files = rng.choice(df_stat_direct.index.values, size=len(df_stat_direct), replace=True)
    df_boot = df_stat_direct.loc[boot_files]

    runs_direct_moj_psnr.append(df_boot['PSNR_Moj'].mean())
    runs_direct_moj_ssim.append(df_boot['SSIM_Moj'].mean())
    runs_direct_moj_lpips.append(df_boot['LPIPS_Moj'].mean())

    runs_direct_nj_psnr.append(df_boot['PSNR_Njihov'].mean())
    runs_direct_nj_ssim.append(df_boot['SSIM_Njihov'].mean())
    runs_direct_nj_lpips.append(df_boot['LPIPS_Njihov'].mean())

clean_feats = extract_features(DIR_VAL_CLEAN, val_files)
moj_feats = extract_features(DIR_MOJ_OUTPUT, val_files)
nj_feats = extract_features(DIR_NJIHOV_OUTPUT, val_files)

fid_moj = calculate_fid(clean_feats, moj_feats)
fid_nj = calculate_fid(clean_feats, nj_feats)

# Statistički testovi na 100% nezavisnim podacima (pun skup bez bootstrap duplikata)
_, p_w_p = stats.wilcoxon(df_stat_direct['PSNR_Moj'], df_stat_direct['PSNR_Njihov'])
_, p_w_s = stats.wilcoxon(df_stat_direct['SSIM_Moj'], df_stat_direct['SSIM_Njihov'])
_, p_w_l = stats.wilcoxon(df_stat_direct['LPIPS_Moj'], df_stat_direct['LPIPS_Njihov'])

_, p_t_p = stats.ttest_rel(df_stat_direct['PSNR_Moj'], df_stat_direct['PSNR_Njihov'])
_, p_t_s = stats.ttest_rel(df_stat_direct['SSIM_Moj'], df_stat_direct['SSIM_Njihov'])
_, p_t_l = stats.ttest_rel(df_stat_direct['LPIPS_Moj'], df_stat_direct['LPIPS_Njihov'])

d_psnr = np.mean(df_stat_direct['PSNR_Moj'] - df_stat_direct['PSNR_Njihov']) / np.std(df_stat_direct['PSNR_Moj'] - df_stat_direct['PSNR_Njihov'], ddof=1)
d_ssim = np.mean(df_stat_direct['SSIM_Moj'] - df_stat_direct['SSIM_Njihov']) / np.std(df_stat_direct['SSIM_Moj'] - df_stat_direct['SSIM_Njihov'], ddof=1)
d_lpips = np.mean(df_stat_direct['LPIPS_Moj'] - df_stat_direct['LPIPS_Njihov']) / np.std(df_stat_direct['LPIPS_Moj'] - df_stat_direct['LPIPS_Njihov'], ddof=1)

tabela_direktna = [
    ['PSNR (dB) [↑]', f"{np.mean(runs_direct_moj_psnr):.2f}", f"{np.mean(runs_direct_nj_psnr):.2f}", f"+{np.mean(runs_direct_moj_psnr) - np.mean(runs_direct_nj_psnr):.2f} dB", format_p_val(p_w_p), format_p_val(p_t_p), f"{d_psnr:.2f}"],
    ['SSIM [↑]', f"{np.mean(runs_direct_moj_ssim):.4f}", f"{np.mean(runs_direct_nj_ssim):.4f}", f"+{np.mean(runs_direct_moj_ssim) - np.mean(runs_direct_nj_ssim):.4f}", format_p_val(p_w_s), format_p_val(p_t_s), f"{d_ssim:.2f}"],
    ['LPIPS [↓]', f"{np.mean(runs_direct_moj_lpips):.4f}", f"{np.mean(runs_direct_nj_lpips):.4f}", f"{np.mean(runs_direct_moj_lpips) - np.mean(runs_direct_nj_lpips):.4f}", format_p_val(p_w_l), format_p_val(p_t_l), f"{d_lpips:.2f}"],
    ['FID [↓]', f"{fid_moj:.2f}", f"{fid_nj:.2f}", f"{fid_moj - fid_nj:.2f}", "-", "-", "-"]
]
headers_direktna = ['Metrika', 'Moj Model (Mean)', 'Njihov Model (Mean)', 'Δ Razlika', 'Wilcoxon', 'Paired t-test', "Cohen's d"]

# Čuvanje svih 5 iteracija za direktno poređenje u CSV
df_direct_runs = pd.DataFrame([
    {
        'Model': 'Moj Model',
        'Metrika': 'PSNR (dB)',
        'Iter_1': runs_direct_moj_psnr[0], 'Iter_2': runs_direct_moj_psnr[1], 'Iter_3': runs_direct_moj_psnr[2],
        'Iter_4': runs_direct_moj_psnr[3], 'Iter_5': runs_direct_moj_psnr[4],
        'Srednja_Vrednost': np.mean(runs_direct_moj_psnr), 'Std_Dev': np.std(runs_direct_moj_psnr)
    },
    {
        'Model': 'Moj Model',
        'Metrika': 'SSIM',
        'Iter_1': runs_direct_moj_ssim[0], 'Iter_2': runs_direct_moj_ssim[1], 'Iter_3': runs_direct_moj_ssim[2],
        'Iter_4': runs_direct_moj_ssim[3], 'Iter_5': runs_direct_moj_ssim[4],
        'Srednja_Vrednost': np.mean(runs_direct_moj_ssim), 'Std_Dev': np.std(runs_direct_moj_ssim)
    },
    {
        'Model': 'Moj Model',
        'Metrika': 'LPIPS',
        'Iter_1': runs_direct_moj_lpips[0], 'Iter_2': runs_direct_moj_lpips[1], 'Iter_3': runs_direct_moj_lpips[2],
        'Iter_4': runs_direct_moj_lpips[3], 'Iter_5': runs_direct_moj_lpips[4],
        'Srednja_Vrednost': np.mean(runs_direct_moj_lpips), 'Std_Dev': np.std(runs_direct_moj_lpips)
    },
    {
        'Model': 'Njihov Model',
        'Metrika': 'PSNR (dB)',
        'Iter_1': runs_direct_nj_psnr[0], 'Iter_2': runs_direct_nj_psnr[1], 'Iter_3': runs_direct_nj_psnr[2],
        'Iter_4': runs_direct_nj_psnr[3], 'Iter_5': runs_direct_nj_psnr[4],
        'Srednja_Vrednost': np.mean(runs_direct_nj_psnr), 'Std_Dev': np.std(runs_direct_nj_psnr)
    },
    {
        'Model': 'Njihov Model',
        'Metrika': 'SSIM',
        'Iter_1': runs_direct_nj_ssim[0], 'Iter_2': runs_direct_nj_ssim[1], 'Iter_3': runs_direct_nj_ssim[2],
        'Iter_4': runs_direct_nj_ssim[3], 'Iter_5': runs_direct_nj_ssim[4],
        'Srednja_Vrednost': np.mean(runs_direct_nj_ssim), 'Std_Dev': np.std(runs_direct_nj_ssim)
    },
    {
        'Model': 'Njihov Model',
        'Metrika': 'LPIPS',
        'Iter_1': runs_direct_nj_lpips[0], 'Iter_2': runs_direct_nj_lpips[1], 'Iter_3': runs_direct_nj_lpips[2],
        'Iter_4': runs_direct_nj_lpips[3], 'Iter_5': runs_direct_nj_lpips[4],
        'Srednja_Vrednost': np.mean(runs_direct_nj_lpips), 'Std_Dev': np.std(runs_direct_nj_lpips)
    }
])
df_direct_runs.to_csv(os.path.join(DIR_ABLACIJA_DRIVE, 'direktno_poredjenje_5_iteracija.csv'), index=False)


#pojedinačmna balacija 5 iteracija
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

cached_1po1_results = {}
print(f"\n[INFO] Evaluacija konfiguracija za 1-po-1 ablaciju...")

for naziv, cfg in tqdm(ablation_configs, desc="1-po-1 Eval"):
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

            out_t = torch.clamp(moj_model(d_t, **cfg), 0.0, 1.0)
            out_np = (out_t.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8).astype(np.float32) / 255.0
            out_eval_t = torch.from_numpy(out_np).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
            c_eval_t = torch.from_numpy(c_img).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0

            mse = np.mean((c_img - out_np) ** 2)
            psnr_val = 10.0 * np.log10(1.0 / mse) if mse > 0 else 100.0
            ssim_val = ssim_metric(c_img, out_np, channel_axis=2, data_range=1.0)
            lpips_val = eval_lpips_fn(out_eval_t, c_eval_t).item()

            res_list.append({'Fname': fname, 'PSNR': psnr_val, 'SSIM': ssim_val, 'LPIPS': lpips_val})

    cached_1po1_results[naziv] = pd.DataFrame(res_list).set_index('Fname')

# 5 iteracija
raw_data_runs_1po1 = {
    cfg_name: {'PSNR_runs': [], 'SSIM_runs': [], 'LPIPS_runs': []}
    for cfg_name, _ in ablation_configs
}

for iter_idx in range(NUM_ITERACIJA):
    rng = np.random.default_rng(seed=42 + iter_idx)
    boot_files = rng.choice(val_files, size=len(val_files), replace=True)
    for naziv, _ in ablation_configs:
        df_sub = cached_1po1_results[naziv].loc[boot_files]
        raw_data_runs_1po1[naziv]['PSNR_runs'].append(df_sub['PSNR'].mean())
        raw_data_runs_1po1[naziv]['SSIM_runs'].append(df_sub['SSIM'].mean())
        raw_data_runs_1po1[naziv]['LPIPS_runs'].append(df_sub['LPIPS'].mean())

summary_abl = []
csv_abl_1po1_rows = []

for naziv, _ in ablation_configs:
    p_runs = raw_data_runs_1po1[naziv]['PSNR_runs']
    s_runs = raw_data_runs_1po1[naziv]['SSIM_runs']
    l_runs = raw_data_runs_1po1[naziv]['LPIPS_runs']

    p_mean = np.mean(p_runs)
    s_mean = np.mean(s_runs)
    l_mean = np.mean(l_runs)

    summary_abl.append([
        naziv,
        f"{p_runs[0]:.2f}", f"{p_runs[1]:.2f}", f"{p_runs[2]:.2f}", f"{p_runs[3]:.2f}", f"{p_runs[4]:.2f}",
        f"{p_mean:.2f}",
        f"{s_mean:.4f}",
        f"{l_mean:.4f}"
    ])

    csv_abl_1po1_rows.append({
        'Konfiguracija': naziv,
        'PSNR_Iter_1': p_runs[0], 'PSNR_Iter_2': p_runs[1], 'PSNR_Iter_3': p_runs[2], 'PSNR_Iter_4': p_runs[3], 'PSNR_Iter_5': p_runs[4],
        'PSNR_Mean': p_mean, 'PSNR_Std': np.std(p_runs),
        'SSIM_Iter_1': s_runs[0], 'SSIM_Iter_2': s_runs[1], 'SSIM_Iter_3': s_runs[2], 'SSIM_Iter_4': s_runs[3], 'SSIM_Iter_5': s_runs[4],
        'SSIM_Mean': s_mean, 'SSIM_Std': np.std(s_runs),
        'LPIPS_Iter_1': l_runs[0], 'LPIPS_Iter_2': l_runs[1], 'LPIPS_Iter_3': l_runs[2], 'LPIPS_Iter_4': l_runs[3], 'LPIPS_Iter_5': l_runs[4],
        'LPIPS_Mean': l_mean, 'LPIPS_Std': np.std(l_runs)
    })

headers_abl_mean = ['Konfiguracija Modela', 'It. 1', 'It. 2', 'It. 3', 'It. 4', 'It. 5', 'PSNR (Mean) [↑]', 'SSIM (Mean) [↑]', 'LPIPS (Mean) [↓]']
pd.DataFrame(csv_abl_1po1_rows).to_csv(os.path.join(DIR_ABLACIJA_DRIVE, 'ablacija_1po1_5_iteracija.csv'), index=False)

# Statistika za 1-po-1 (na punom originalnom skupu)
full_p = cached_1po1_results["Full Proposed Model"]['PSNR'].values
full_s = cached_1po1_results["Full Proposed Model"]['SSIM'].values
full_l = cached_1po1_results["Full Proposed Model"]['LPIPS'].values

stat_abl_table = []
for naziv, _ in ablation_configs:
    if naziv == "Full Proposed Model":
        continue

    var_p = cached_1po1_results[naziv]['PSNR'].values
    var_s = cached_1po1_results[naziv]['SSIM'].values
    var_l = cached_1po1_results[naziv]['LPIPS'].values

    diff_p = full_p - var_p
    _, p_w_p = stats.wilcoxon(full_p, var_p)
    _, p_t_p = stats.ttest_rel(full_p, var_p)
    d_p = np.mean(diff_p) / np.std(diff_p, ddof=1) if np.std(diff_p, ddof=1) > 0 else 0.0

    _, p_w_s = stats.wilcoxon(full_s, var_s)
    _, p_w_l = stats.wilcoxon(full_l, var_l)

    stat_abl_table.append([
        naziv,
        f"-{np.mean(diff_p):.2f} dB",
        format_p_val(p_w_p),
        format_p_val(p_t_p),
        f"{d_p:.2f}",
        format_p_val(p_w_s),
        format_p_val(p_w_l)
    ])

headers_abl_stat = ['Uklonjena Komponenta', 'Δ PSNR', 'Wilcoxon (PSNR)', 't-test (PSNR)', "Cohen's d", 'Wilcoxon (SSIM)', 'Wilcoxon (LPIPS)']
pd.DataFrame(stat_abl_table, columns=headers_abl_stat).to_csv(os.path.join(DIR_ABLACIJA_DRIVE, 'ablacija_1po1_statistika.csv'), index=False)



crit_components = [
    ("Spectral", "use_spectral"),
    ("EdgeBranch", "use_edge_branch"),
    ("CCR", "use_ccr"),
    ("SkipRefine", "use_skip_refine")
]
crit_keys = [k for _, k in crit_components]

pairwise_configs = [
    ("Full Proposed Model", dict()),
    ("w/o All 4 Critical [Spectral+Edge+CCR+SkipRefine]", {k: False for k in crit_keys}),
    ("w/o [Spectral + EdgeBranch]", {"use_spectral": False, "use_edge_branch": False}),
    ("w/o [Spectral + CCR]", {"use_spectral": False, "use_ccr": False}),
    ("w/o [Spectral + SkipRefine]", {"use_spectral": False, "use_skip_refine": False}),
    ("w/o [EdgeBranch + CCR]", {"use_edge_branch": False, "use_ccr": False}),
    ("w/o [EdgeBranch + SkipRefine]", {"use_edge_branch": False, "use_skip_refine": False}),
    ("w/o [CCR + SkipRefine]", {"use_ccr": False, "use_skip_refine": False}),
    ("Only 4 Critical Active (All Secondary OFF)", {
        "use_spectral": True, "use_edge_branch": True, "use_ccr": True, "use_skip_refine": True,
        "use_cross_bridge": False, "use_gated_fusion": False, "use_damage_attention": False,
        "use_dilated_context": False, "use_gated_skips": False
    }),
    ("Barebones Backbone (All OFF)", {
        "use_spectral": False, "use_edge_branch": False, "use_ccr": False, "use_skip_refine": False,
        "use_cross_bridge": False, "use_gated_fusion": False, "use_damage_attention": False,
        "use_dilated_context": False, "use_gated_skips": False
    })
]

cached_pair_results = {}
print(f"\n[INFO] Evaluacija konfiguracija za Kombinatornu ablaciju...")

for naziv, cfg in tqdm(pairwise_configs, desc="Kombinatorna Eval"):
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

            out_t = torch.clamp(moj_model(d_t, **cfg), 0.0, 1.0)
            out_np = (out_t.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8).astype(np.float32) / 255.0
            out_eval_t = torch.from_numpy(out_np).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
            c_eval_t = torch.from_numpy(c_img).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0

            mse = np.mean((c_img - out_np) ** 2)
            psnr_val = 10.0 * np.log10(1.0 / mse) if mse > 0 else 100.0
            ssim_val = ssim_metric(c_img, out_np, channel_axis=2, data_range=1.0)
            lpips_val = eval_lpips_fn(out_eval_t, c_eval_t).item()

            res_list.append({'Fname': fname, 'PSNR': psnr_val, 'SSIM': ssim_val, 'LPIPS': lpips_val})

    cached_pair_results[naziv] = pd.DataFrame(res_list).set_index('Fname')

# Bootstrap 5 iteracija za Kombinatornu
pairwise_data_runs = {
    cfg_name: {'PSNR_runs': [], 'SSIM_runs': [], 'LPIPS_runs': []}
    for cfg_name, _ in pairwise_configs
}

for iter_idx in range(NUM_ITERACIJA):
    rng = np.random.default_rng(seed=42 + iter_idx)
    boot_files = rng.choice(val_files, size=len(val_files), replace=True)
    for naziv, _ in pairwise_configs:
        df_sub = cached_pair_results[naziv].loc[boot_files]
        pairwise_data_runs[naziv]['PSNR_runs'].append(df_sub['PSNR'].mean())
        pairwise_data_runs[naziv]['SSIM_runs'].append(df_sub['SSIM'].mean())
        pairwise_data_runs[naziv]['LPIPS_runs'].append(df_sub['LPIPS'].mean())

summary_pair = []
csv_abl_pair_rows = []

for naziv, _ in pairwise_configs:
    p_runs = pairwise_data_runs[naziv]['PSNR_runs']
    s_runs = pairwise_data_runs[naziv]['SSIM_runs']
    l_runs = pairwise_data_runs[naziv]['LPIPS_runs']

    p_mean = np.mean(p_runs)
    s_mean = np.mean(s_runs)
    l_mean = np.mean(l_runs)

    summary_pair.append([
        naziv,
        f"{p_runs[0]:.2f}", f"{p_runs[1]:.2f}", f"{p_runs[2]:.2f}", f"{p_runs[3]:.2f}", f"{p_runs[4]:.2f}",
        f"{p_mean:.2f}",
        f"{s_mean:.4f}",
        f"{l_mean:.4f}"
    ])

    csv_abl_pair_rows.append({
        'Konfiguracija': naziv,
        'PSNR_Iter_1': p_runs[0], 'PSNR_Iter_2': p_runs[1], 'PSNR_Iter_3': p_runs[2], 'PSNR_Iter_4': p_runs[3], 'PSNR_Iter_5': p_runs[4],
        'PSNR_Mean': p_mean, 'PSNR_Std': np.std(p_runs),
        'SSIM_Iter_1': s_runs[0], 'SSIM_Iter_2': s_runs[1], 'SSIM_Iter_3': s_runs[2], 'SSIM_Iter_4': s_runs[3], 'SSIM_Iter_5': s_runs[4],
        'SSIM_Mean': s_mean, 'SSIM_Std': np.std(s_runs),
        'LPIPS_Iter_1': l_runs[0], 'LPIPS_Iter_2': l_runs[1], 'LPIPS_Iter_3': l_runs[2], 'LPIPS_Iter_4': l_runs[3], 'LPIPS_Iter_5': l_runs[4],
        'LPIPS_Mean': l_mean, 'LPIPS_Std': np.std(l_runs)
    })

headers_pair_mean = ['Konfiguracija Modela', 'It. 1', 'It. 2', 'It. 3', 'It. 4', 'It. 5', 'PSNR (Mean) [↑]', 'SSIM (Mean) [↑]', 'LPIPS (Mean) [↓]']
pd.DataFrame(csv_abl_pair_rows).to_csv(os.path.join(DIR_ABLACIJA_DRIVE, 'ablacija_kombinatorna_5_iteracija.csv'), index=False)

# Statistika za Kombinatornu (na punom originalnom skupu)
stat_pair_table = []
for naziv, _ in pairwise_configs:
    if naziv == "Full Proposed Model":
        continue

    var_p = cached_pair_results[naziv]['PSNR'].values
    var_s = cached_pair_results[naziv]['SSIM'].values
    var_l = cached_pair_results[naziv]['LPIPS'].values

    diff_p = full_p - var_p
    _, p_w_p = stats.wilcoxon(full_p, var_p)
    _, p_t_p = stats.ttest_rel(full_p, var_p)
    d_p = np.mean(diff_p) / np.std(diff_p, ddof=1) if np.std(diff_p, ddof=1) > 0 else 0.0

    _, p_w_s = stats.wilcoxon(full_s, var_s)
    _, p_w_l = stats.wilcoxon(full_l, var_l)

    stat_pair_table.append([
        naziv,
        f"-{np.mean(diff_p):.2f} dB",
        format_p_val(p_w_p),
        format_p_val(p_t_p),
        f"{d_p:.2f}",
        format_p_val(p_w_s),
        format_p_val(p_w_l)
    ])

headers_pair_stat = ['Uklonjena Komponenta', 'Δ PSNR', 'Wilcoxon (PSNR)', 't-test (PSNR)', "Cohen's d", 'Wilcoxon (SSIM)', 'Wilcoxon (LPIPS)']
pd.DataFrame(stat_pair_table, columns=headers_pair_stat).to_csv(os.path.join(DIR_ABLACIJA_DRIVE, 'ablacija_kombinatorna_statistika.csv'), index=False)


# prikaz svega u terminalu 
print("\n" + "█" * 115)
print("  moji vs njihovi rezultati ")
print("█" * 115)
print(tabulate(tabela_direktna, headers=headers_direktna, tablefmt="fancy_grid", stralign="center", numalign="center"))

print("\n" + "█" * 115)
print(f"  jedna po jedno uklanjanje za ablaciju | N = {len(val_files)})")
print("█" * 115)
print(tabulate(summary_abl, headers=headers_abl_mean, tablefmt="fancy_grid", stralign="center", numalign="center"))

print("\n" + "█" * 115)
print("  jedan po jedan ablacija :3")
print("█" * 115)
print(tabulate(stat_abl_table, headers=headers_abl_stat, tablefmt="fancy_grid", stralign="center", numalign="center"))

print("\n" + "█" * 115)
print(f"  | N = {len(val_files)})")
print("█" * 115)
print(tabulate(summary_pair, headers=headers_pair_mean, tablefmt="fancy_grid", stralign="center", numalign="center"))

print("\n" + "█" * 115)
print("  statisticki testovi :3")
print("█" * 115)
print(tabulate(stat_pair_table, headers=headers_pair_stat, tablefmt="fancy_grid", stralign="center", numalign="center"))
print("\nLegenda statističke značajnosti: *** p < 0.001  |  ** p < 0.01  |  * p < 0.05  |  ns: nije značajno")
