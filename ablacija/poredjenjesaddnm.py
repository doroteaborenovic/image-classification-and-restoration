#ovaj repo  https://github.com/wyhuai/ddnm

import os
import sys
import random
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
import torch.nn as nn
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
normalna_instalacija("blobfile")
normalna_instalacija("pyyaml")

import lpips
from tabulate import tabulate

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
except Exception:
    pass

DRIVE_PROJECT_DIR = '/content/drive/MyDrive/Projekat_Model'
os.makedirs(DRIVE_PROJECT_DIR, exist_ok=True)
DIR_ABLACIJA_DRIVE = os.path.join(DRIVE_PROJECT_DIR, 'ablacija_checkpoints')
DIR_DDNM_DRIVE = os.path.join(DRIVE_PROJECT_DIR, 'rezultati_ddnm_fer_v2')

if os.path.exists(DIR_DDNM_DRIVE):
    shutil.rmtree(DIR_DDNM_DRIVE)
os.makedirs(DIR_DDNM_DRIVE, exist_ok=True)

IMG_SIZE = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eval_lpips_fn = lpips.LPIPS(net='alex', verbose=False).to(device).eval()


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

val_files = sorted([f for f in os.listdir(DIR_VAL_DEGRADED) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
print(f"[INFO] Uređaj: {device} | Validacioni skup: {len(val_files)} slika")


# ==============================================================================
# 1. AUTO-DETEKCIJA DEGRADACIJE (umesto pogađanja tipa)
# ==============================================================================
def detektuj_degradaciju(clean_dir, degraded_dir, files, n_probe=12):
    """
    Meri stvarnu degradaciju na uzorku parova PRE resize-a na IMG_SIZE, da se
    ne izgubi informacija o originalnoj rezoluciji (bitno za detekciju SR faktora).
    Vraća dict sa: is_grayscale, sr_scale, blur_sigma_px, has_mask, mask_fraction.
    """
    probe = files[:min(n_probe, len(files))]
    gray_scores, scale_ratios, blur_ratios, mask_fracs = [], [], [], []

    for fname in probe:
        c_p = os.path.join(clean_dir, fname)
        d_p = os.path.join(degraded_dir, fname)
        c_raw = cv2.imread(c_p)
        d_raw = cv2.imread(d_p)
        if c_raw is None or d_raw is None:
            continue
        c_raw = cv2.cvtColor(c_raw, cv2.COLOR_BGR2RGB)
        d_raw = cv2.cvtColor(d_raw, cv2.COLOR_BGR2RGB)

        # --- grayscale test: R/G/B kanali skoro identični u degraded ---
        d_f = d_raw.astype(np.float32)
        ch_std = np.mean([
            np.std(d_f[..., 0] - d_f[..., 1]),
            np.std(d_f[..., 1] - d_f[..., 2]),
        ])
        gray_scores.append(ch_std)

        # --- rezolucija: originalni fajl (pre bilo kakvog resize-a) ---
        scale_ratios.append(c_raw.shape[0] / max(d_raw.shape[0], 1))

        # --- blur test: odnos Laplasove varijanse (oštrina) na zajedničkoj rezoluciji ---
        c_cmp = cv2.resize(c_raw, (IMG_SIZE, IMG_SIZE))
        d_cmp = cv2.resize(d_raw, (IMG_SIZE, IMG_SIZE))
        c_lap = cv2.Laplacian(cv2.cvtColor(c_cmp, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
        d_lap = cv2.Laplacian(cv2.cvtColor(d_cmp, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
        blur_ratios.append(d_lap / max(c_lap, 1e-6))

        # --- maska: pikseli koji su skoro crni/konstantni u degraded, a nisu u clean ---
        d_gray_cmp = cv2.cvtColor(d_cmp, cv2.COLOR_RGB2GRAY)
        c_gray_cmp = cv2.cvtColor(c_cmp, cv2.COLOR_RGB2GRAY)
        near_black = d_gray_cmp < 8
        clean_not_black = c_gray_cmp > 20
        mask_frac = np.mean(near_black & clean_not_black)
        mask_fracs.append(mask_frac)

    is_grayscale = np.mean(gray_scores) < 2.0
    sr_scale_raw = np.median(scale_ratios)
    # zaokruži na najbliži "razuman" SR faktor
    sr_scale = min([1, 2, 4, 8], key=lambda s: abs(s - sr_scale_raw)) if sr_scale_raw > 1.15 else 1
    blur_ratio = np.median(blur_ratios)
    has_blur = blur_ratio < 0.5  # degraded znatno manje oštra
    mask_fraction = float(np.median(mask_fracs))
    has_mask = mask_fraction > 0.01  # >1% piksela je "izbrisano"

    info = {
        "is_grayscale": bool(is_grayscale),
        "sr_scale": int(sr_scale),
        "has_blur": bool(has_blur),
        "blur_sharpness_ratio": float(blur_ratio),
        "has_mask": bool(has_mask),
        "mask_fraction": mask_fraction,
    }
    return info

deg_info = detektuj_degradaciju(DIR_VAL_CLEAN, DIR_VAL_DEGRADED, val_files)

print("\n" + "=" * 70)
print("  AUTO-DETEKCIJA DEGRADACIJE (proverite vizuelno da li odgovara!)")
print("=" * 70)
for k, v in deg_info.items():
    print(f"   {k}: {v}")
print("=" * 70)


# ==============================================================================
# 2. OPERATORI A / Ap KONSTRUISANI NA OSNOVU DETEKCIJE (kompozicija kao u
#    zvaničnom README-u za "old photo restoration": A = A3(A2(A1(x))) )
# ==============================================================================
def color2gray(x):
    coef = 1.0 / 3.0
    g = x[:, 0:1] * coef + x[:, 1:2] * coef + x[:, 2:3] * coef
    return g.repeat(1, 3, 1, 1)

def gray2color(x):
    coef = 1.0 / 3.0
    base = 3.0 * (coef ** 2)
    ch = x[:, 0:1] * coef / base
    return torch.cat([ch, ch, ch], dim=1)

def make_patch_upsample(scale):
    def f(x):
        n, c, h, w = x.shape
        x_exp = x.view(n, c, h, 1, w, 1).repeat(1, 1, 1, scale, 1, scale)
        return x_exp.view(n, c, scale * h, scale * w)
    return f


class DetektovaniOperator:
    """
    Sklapa A / Ap kao kompoziciju samo onih komponenti koje su STVARNO
    detektovane u podacima. Ako ništa nije detektovano (nema grayscale, nema
    SR, nema blur, nema maske), pada nazad na identitet -> DDNM tada radi
    kao čist "denoising" mod, što je ispravno ako je degradacija generički
    šum/kompresija bez poznatog linearnog modela.
    """
    def __init__(self, info):
        self.info = info
        self.use_mask = info["has_mask"]
        self.use_gray = info["is_grayscale"]
        self.scale = info["sr_scale"] if info["sr_scale"] > 1 else 1
        self.use_sr = self.scale > 1
        if self.use_sr:
            self.pool = nn.AdaptiveAvgPool2d((IMG_SIZE // self.scale, IMG_SIZE // self.scale))
            self.upsample = make_patch_upsample(self.scale)
        # blur: bez poznatog kernela ne možemo tačno invertovati, pa ga NE
        # tretiramo kao deo A (bio bi pogrešan kernel = lažna preciznost).
        # Umesto toga se blur ostavlja da ga apsorbuje sigma_y / difuzioni model.

    def A(self, x, mask=None):
        out = x
        if self.use_mask and mask is not None:
            out = out * mask
        if self.use_gray:
            out = color2gray(out)
        if self.use_sr:
            out = self.pool(out)
        return out

    def Ap(self, y, mask=None):
        out = y
        if self.use_sr:
            out = self.upsample(out)
        if self.use_gray:
            out = gray2color(out)
        if self.use_mask and mask is not None:
            out = out * mask
        return out

    def detektuj_masku(self, y_deg):
        """Maska iz STVARNOG y (ne konstantna jedinica) — regioni bliski nuli."""
        if not self.use_mask:
            return torch.ones_like(y_deg)
        gray = y_deg.mean(dim=1, keepdim=True)
        # y_deg je u [-1,1]; blizu -1 = crno (oštećen region)
        m = (gray > -0.9).float()
        return m.repeat(1, y_deg.shape[1], 1, 1)


op_handler = DetektovaniOperator(deg_info)


# ==============================================================================
# 3. PROCENA sigma_y IZ PODATAKA (umesto proizvoljne konstante)
# ==============================================================================
def proceni_sigma_y(clean_dir, degraded_dir, files, op_handler, n_probe=12):
    """
    sigma_y = std reziduala u prostoru MERE y: A(clean) vs stvarno y.
    Ovo je ono što A ne može da objasni -> šum koji DDNM treba da tretira
    preko sigma_y, umesto da se pogađa napamet.
    """
    residuals = []
    probe = files[:min(n_probe, len(files))]
    for fname in probe:
        c_img = cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(clean_dir, fname)), cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE))
        d_img = cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(degraded_dir, fname)), cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE))
        c_t = torch.from_numpy(c_img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
        d_t = torch.from_numpy(d_img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
        mask = op_handler.detektuj_masku(d_t)
        pred_y = op_handler.A(c_t, mask=mask)
        # uporedi u istom prostoru kao d_t projektovano kroz A (da dimenzije budu iste)
        obs_y = op_handler.A(d_t, mask=mask)
        residuals.append((pred_y - obs_y).abs().flatten())
    residuals = torch.cat(residuals)
    sigma = residuals.std().item()
    return max(sigma, 1e-3)

sigma_y_est = proceni_sigma_y(DIR_VAL_CLEAN, DIR_VAL_DEGRADED, val_files, op_handler)
print(f"\n[INFO] Procenjeno sigma_y (iz podataka): {sigma_y_est:.4f}")


# ==============================================================================
# 4. UČITAVANJE ZVANIČNE OPENAI UNET MREŽE
# ==============================================================================
DDNM_REPO_DIR = '/content/DDNM'
if not os.path.exists(DDNM_REPO_DIR):
    print("-> Kloniram zvanični wyhuai/DDNM repozitorijum...")
    subprocess.run(f"git clone -q https://github.com/wyhuai/DDNM.git {DDNM_REPO_DIR}", shell=True)

if DDNM_REPO_DIR not in sys.path:
    sys.path.insert(0, DDNM_REPO_DIR)

from guided_diffusion.unet import UNetModel

openai_ckpt_path = '/content/256x256_diffusion_uncond.pt'
if not os.path.exists(openai_ckpt_path):
    print("-> Preuzimam zvanični OpenAI pretrained diffusion checkpoint (~550 MB)...")
    subprocess.run(f"wget -q -c https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt -O {openai_ckpt_path}", shell=True)

ddnm_unet = UNetModel(
    image_size=256, in_channels=3, model_channels=256, out_channels=6,
    num_res_blocks=2, attention_resolutions=(8, 16, 32), dropout=0.0,
    channel_mult=(1, 1, 2, 2, 4, 4), num_classes=None, use_checkpoint=False,
    use_fp16=False, num_heads=4, num_head_channels=-1, num_heads_upsample=-1,
    use_scale_shift_norm=True, resblock_updown=True, use_new_attention_order=False
)
openai_state_dict = torch.load(openai_ckpt_path, map_location=device)
ddnm_unet.load_state_dict(openai_state_dict)
ddnm_unet = ddnm_unet.to(device).eval()
print("✓ Zvanični OpenAI UNet model učitan.")


# ==============================================================================
# 5. DDNM+ SAMPLING SA STVARNIM TIME-TRAVEL TRIKOM (Section 3.3)
# ==============================================================================
def ddnm_plus_sampling(unet_model, y_deg, op_handler, num_steps=100, eta=0.85,
                        sigma_y=0.02, travel_length=1, travel_repeat=2):
    """
    Pravi time-travel: posle svakog "unazad" koraka x_t -> x_{t-1}, vraćamo se
    `travel_length` koraka NAPRED (dodavanjem šuma, x_{t-1} -> x_{t-1+L}) i
    ponavljamo taj mini-segment `travel_repeat` puta pre nego što nastavimo
    dalje unazad. Ovo tačno prati RePaint/DDNM time-travel semantiku.
    """
    total_timesteps = 1000
    betas = torch.linspace(1e-4, 0.02, total_timesteps, dtype=torch.float32, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    step_indices = torch.linspace(0, total_timesteps - 1, num_steps, dtype=torch.long, device=device)

    mask = op_handler.detektuj_masku(y_deg)
    y_obs = y_deg  # y JE posmatranje -- A se ne primenjuje ponovo na njega

    def eps_and_x0(xt, idx):
        at = alphas_cumprod[step_indices[idx]]
        t_tensor = torch.tensor([step_indices[idx]], device=device).repeat(xt.shape[0])
        with torch.no_grad():
            out = unet_model(xt, t_tensor)
            eps, _ = torch.split(out, 3, dim=1)
        x0 = torch.clamp((xt - torch.sqrt(1 - at) * eps) / torch.sqrt(at), -1.0, 1.0)
        return eps, x0, at

    def denoise_step(xt, idx):
        """Jedan DDNM+ korak unazad: idx -> idx-1. Vraća x_{idx-1}."""
        eps, x0_t, at = eps_and_x0(xt, idx)
        at_prev = alphas_cumprod[step_indices[idx - 1]] if idx > 0 else torch.tensor(1.0, device=device)

        sigma_t = eta * torch.sqrt((1 - at_prev) / (1 - at) * (1 - at / at_prev)) if idx > 0 else torch.tensor(0.0, device=device)
        a_t_scalar = torch.sqrt(at)

        if sigma_t >= a_t_scalar * sigma_y:
            lambda_t = 1.0
            gamma_t = torch.sqrt(torch.clamp(sigma_t ** 2 - (a_t_scalar * lambda_t * sigma_y) ** 2, min=1e-8))
        else:
            lambda_t = sigma_t / (a_t_scalar * sigma_y + 1e-8)
            gamma_t = torch.tensor(0.0, device=device)

        A_x0 = op_handler.A(x0_t, mask=mask)
        res = y_obs_scaled - A_x0
        x0_t = x0_t + lambda_t * op_handler.Ap(res, mask=mask)

        if idx > 0:
            c1 = torch.sqrt(at_prev)
            c2 = torch.sqrt(torch.clamp(1 - at_prev - sigma_t ** 2, min=0.0))
            noise = torch.randn_like(xt)
            x_prev = c1 * x0_t + c2 * eps + gamma_t * noise
        else:
            x_prev = x0_t
        return x_prev

    def renoise_step(x_low, idx_low, idx_high):
        """Forward: vrati x sa nivoa idx_low nazad na nivo idx_high (idx_high > idx_low)."""
        at_low = alphas_cumprod[step_indices[idx_low]]
        at_high = alphas_cumprod[step_indices[idx_high]]
        noise = torch.randn_like(x_low)
        # q(x_high | x_low) aproksimacija preko x0 rekonstrukcije nije potrebna:
        # standardni DDPM forward re-noising između dva nivoa niza koeficijenata.
        alpha_ratio = at_high / at_low
        x_high = torch.sqrt(alpha_ratio) * x_low + torch.sqrt(torch.clamp(1 - alpha_ratio, min=0.0)) * noise
        return x_high

    y_obs_scaled = y_obs  # y je već u prostoru posmatranja; A se primenjuje samo na x0_t

    xt = torch.randn_like(y_deg)
    i = num_steps - 1
    while i >= 0:
        xt = denoise_step(xt, i)
        i -= 1

        # --- TIME-TRAVEL: na svakih `travel_length` koraka, vrati se nazad i ponovi ---
        if travel_length > 0 and i >= 0 and (num_steps - 1 - i) % max(travel_length, 1) == 0:
            for _rep in range(travel_repeat - 1):
                target_i = min(i + travel_length, num_steps - 1)
                xt = renoise_step(xt, i, target_i)
                j = target_i
                while j > i:
                    xt = denoise_step(xt, j)
                    j -= 1

    return torch.clamp((xt + 1.0) / 2.0, 0.0, 1.0)


# ==============================================================================
# 6. EVALUACIJA
# ==============================================================================
SAMPLE_SIZE = 10  # za konačni rad promenite na len(val_files)

if SAMPLE_SIZE < len(val_files):
    random.seed(SEED)
    eval_files = sorted(random.sample(val_files, SAMPLE_SIZE))
else:
    eval_files = val_files

print(f"\n[INFO] Pokrećem FER POREĐENJE nad {len(eval_files)} slika "
      f"(operator: gray={op_handler.use_gray}, SR x{op_handler.scale if op_handler.use_sr else 1}, "
      f"mask={op_handler.use_mask}, sigma_y={sigma_y_est:.4f}):")

data_input, data_moj, data_ddnm = [], [], []
moj_model.eval()

with torch.no_grad():
    for idx, fname in enumerate(eval_files):
        c_img = cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(DIR_VAL_CLEAN, fname)), cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
        d_img = cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(DIR_VAL_DEGRADED, fname)), cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0

        c_eval_t = torch.from_numpy(c_img).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
        d_eval_t = torch.from_numpy(d_img).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0

        data_input.append({
            'Fname': fname,
            'PSNR': psnr_metric(c_img, d_img, data_range=1.0),
            'SSIM': ssim_metric(c_img, d_img, channel_axis=2, data_range=1.0),
            'LPIPS': eval_lpips_fn(d_eval_t, c_eval_t).item()
        })

        d_t = torch.from_numpy(d_img).permute(2, 0, 1).unsqueeze(0).to(device)
        out_t = torch.clamp(moj_model(d_t), 0.0, 1.0)
        out_np = (out_t.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8).astype(np.float32) / 255.0
        out_eval_t = torch.from_numpy(out_np).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0

        data_moj.append({
            'Fname': fname,
            'PSNR': psnr_metric(c_img, out_np, data_range=1.0),
            'SSIM': ssim_metric(c_img, out_np, channel_axis=2, data_range=1.0),
            'LPIPS': eval_lpips_fn(out_eval_t, c_eval_t).item()
        })

        ddnm_out_t = ddnm_plus_sampling(ddnm_unet, d_eval_t, op_handler, num_steps=100,
                                         eta=0.85, sigma_y=sigma_y_est,
                                         travel_length=1, travel_repeat=2)
        ddnm_np = (ddnm_out_t.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8).astype(np.float32) / 255.0
        ddnm_eval_t = torch.from_numpy(ddnm_np).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0

        cv2.imwrite(os.path.join(DIR_DDNM_DRIVE, fname), cv2.cvtColor((ddnm_np * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR))

        data_ddnm.append({
            'Fname': fname,
            'PSNR': psnr_metric(c_img, ddnm_np, data_range=1.0),
            'SSIM': ssim_metric(c_img, ddnm_np, channel_axis=2, data_range=1.0),
            'LPIPS': eval_lpips_fn(ddnm_eval_t, c_eval_t).item()
        })

        print(f"   [Slika {idx+1}/{len(eval_files)}] -> Vaš PSNR: {data_moj[-1]['PSNR']:.2f} dB | DDNM PSNR: {data_ddnm[-1]['PSNR']:.2f} dB")

df_in = pd.DataFrame(data_input).set_index('Fname')
df_moj = pd.DataFrame(data_moj).set_index('Fname')
df_ddnm = pd.DataFrame(data_ddnm).set_index('Fname')


# ==============================================================================
# 7. STATISTIKA: Wilcoxon/t-test + STVARAN BOOTSTRAP (1000 iteracija)
# ==============================================================================
def bootstrap_ci(a, b, n_iter=1000, ci=95, seed=SEED):
    """Bootstrap 95% CI za srednju razliku (a - b), resampling PO SLIKAMA."""
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = a - b
    boot_means = np.empty(n_iter)
    for k in range(n_iter):
        idx = rng.integers(0, n, size=n)
        boot_means[k] = diffs[idx].mean()
    lo = np.percentile(boot_means, (100 - ci) / 2)
    hi = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return diffs.mean(), lo, hi

def puna_statistika(a, b):
    _, p_w = stats.wilcoxon(a, b) if not np.allclose(a, b) else (None, 1.0)
    _, p_t = stats.ttest_rel(a, b)
    d = np.mean(a - b) / (np.std(a - b, ddof=1) + 1e-8)
    mean_diff, ci_lo, ci_hi = bootstrap_ci(a, b)
    return p_w, p_t, d, mean_diff, ci_lo, ci_hi

def format_p(p):
    return f"{p:.2e}" if p < 1e-4 else f"{p:.4f}"

red_psnr = puna_statistika(df_moj['PSNR'].values, df_ddnm['PSNR'].values)
red_ssim = puna_statistika(df_moj['SSIM'].values, df_ddnm['SSIM'].values)
red_lpips = puna_statistika(df_moj['LPIPS'].values, df_ddnm['LPIPS'].values)

tabela = [
    ['PSNR (dB) [↑]',
     f"{df_in['PSNR'].mean():.2f} ± {df_in['PSNR'].std():.2f}",
     f"{df_moj['PSNR'].mean():.2f} ± {df_moj['PSNR'].std():.2f}",
     f"{df_ddnm['PSNR'].mean():.2f} ± {df_ddnm['PSNR'].std():.2f}",
     f"{df_moj['PSNR'].mean() - df_ddnm['PSNR'].mean():+.2f}",
     f"[{red_psnr[4]:+.2f}, {red_psnr[5]:+.2f}]",
     format_p(red_psnr[0]), format_p(red_psnr[1]), f"{red_psnr[2]:.2f}"],
    ['SSIM [↑]',
     f"{df_in['SSIM'].mean():.4f} ± {df_in['SSIM'].std():.4f}",
     f"{df_moj['SSIM'].mean():.4f} ± {df_moj['SSIM'].std():.4f}",
     f"{df_ddnm['SSIM'].mean():.4f} ± {df_ddnm['SSIM'].std():.4f}",
     f"{df_moj['SSIM'].mean() - df_ddnm['SSIM'].mean():+.4f}",
     f"[{red_ssim[4]:+.4f}, {red_ssim[5]:+.4f}]",
     format_p(red_ssim[0]), format_p(red_ssim[1]), f"{red_ssim[2]:.2f}"],
    ['LPIPS [↓]',
     f"{df_in['LPIPS'].mean():.4f} ± {df_in['LPIPS'].std():.4f}",
     f"{df_moj['LPIPS'].mean():.4f} ± {df_moj['LPIPS'].std():.4f}",
     f"{df_ddnm['LPIPS'].mean():.4f} ± {df_ddnm['LPIPS'].std():.4f}",
     f"{df_moj['LPIPS'].mean() - df_ddnm['LPIPS'].mean():+.4f}",
     f"[{red_lpips[4]:+.4f}, {red_lpips[5]:+.4f}]",
     format_p(red_lpips[0]), format_p(red_lpips[1]), f"{red_lpips[2]:.2f}"],
]

zaglavlja = ['Metrika', 'Ulaz', 'Predloženi Model', 'Zvanični DDNM',
             'Δ (vs DDNM)', '95% Bootstrap CI (Δ)', 'Wilcoxon (p)', 't-test (p)', "Cohen's d"]

print("\n" + "█" * 140)
print(f"  TABELA: FER POREĐENJE (N = {len(eval_files)}) — UPOZORENJE: p-vrednosti/CI su nepouzdani za N < ~30")
print("█" * 140)
print(tabulate(tabela, headers=zaglavlja, tablefmt="fancy_grid", stralign="center", numalign="center"))

csv_izlaz = os.path.join(DRIVE_PROJECT_DIR, "tabela_ddnm_fer_v2.csv")
pd.DataFrame(tabela, columns=zaglavlja).to_csv(csv_izlaz, index=False)
print(f"\n✓ Tabela sačuvana: {csv_izlaz}")
print(f"✓ Detektovana degradacija: {deg_info}")
print(f"✓ Procenjeno sigma_y: {sigma_y_est:.4f}")
