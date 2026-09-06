import numpy as np
import pandas as pd
from scipy import stats
try:
    from statsmodels.stats.multitest import multipletests
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "statsmodels"])
    from statsmodels.stats.multitest import multipletests
from tabulate import tabulate

def cohens_d_paired(x, y):
    """Računa Cohen's d_z za uparene uzorke."""
    diff = x - y
    sd = np.std(diff, ddof=1)
    return np.mean(diff) / (sd + 1e-8)

def format_p(p):
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"

# Učitavanje per-image podataka
csv_path = "/content/drive/MyDrive/Projekat_Model/per_image_ablation_all_variants.csv"
df = pd.read_csv(csv_path)

full_p = df['Full_PSNR'].values
full_s = df['Full_SSIM'].values
full_l = df['Full_LPIPS'].values

# Mapiranje kolona i naziva ablacija
ablation_map = [
    ("1. w/o Spatial Encoder", "Abl_1"),
    ("2. w/o Spectral Encoder", "Abl_2"),
    ("3. w/o Asymmetric Cross-Bridge", "Abl_3"),
    ("4. w/o Gated Bottleneck Fusion", "Abl_4"),
    ("5. w/o Damage Attention", "Abl_5"),
    ("6. w/o Bottleneck Dilated Context", "Abl_6"),
    ("7. w/o Gated Skip Connections", "Abl_7"),
    ("8. w/o Skip Refinement", "Abl_8"),
    ("9. w/o Edge Guidance Branch", "Abl_9"),
    ("10. w/o Contrast Color Recovery", "Abl_10")
]

raw_p_wilcoxon = []
raw_p_wilcoxon_greater = []
stat_rows = []

for name, prefix in ablation_map:
    col_p = f"{prefix}_PSNR"
    if col_p not in df.columns:
        continue
    
    abl_p = df[col_p].values
    diff = abl_p - full_p
    delta_psnr = np.mean(diff)
    
    # Wilcoxon two-sided (da li postoji ikakva razlika)
    try:
        res_two = stats.wilcoxon(abl_p, full_p, zero_method='pratt', alternative='two-sided')
        w_p_two = res_two.pvalue
    except Exception:
        w_p_two = 1.0

    # Wilcoxon one-sided (H1: Full model je bolji, tj. Ablation < Full)
    try:
        res_less = stats.wilcoxon(abl_p, full_p, zero_method='pratt', alternative='less')
        w_p_less = res_less.pvalue
    except Exception:
        w_p_less = 1.0

    d_val = cohens_d_paired(abl_p, full_p)

    raw_p_wilcoxon.append(w_p_less)
    stat_rows.append({
        'name': name,
        'delta_psnr': delta_psnr,
        'p_raw': w_p_less,
        'p_two': w_p_two,
        'd': d_val
    })

# Egzaktna Holm-Bonferroni korekcija preko statsmodels
_, holm_corrected_p, _, _ = multipletests(raw_p_wilcoxon, alpha=0.05, method='holm')

final_table = []
for i, item in enumerate(stat_rows):
    p_adj = holm_corrected_p[i]
    
    # Kvalitativna interpretacija
    if item['delta_psnr'] < -0.10 and p_adj < 0.05:
        zakljucak = "Kritična komponenta (Značajan pad)"
    elif abs(item['delta_psnr']) <= 0.05 or p_adj >= 0.05:
        zakljucak = "Marginalan / Beznačajan uticaj na PSNR"
    elif item['delta_psnr'] > 0.05:
        zakljucak = "Blagi rast PSNR (Perceptivni trade-off)"
    else:
        zakljucak = "Umeren doprinos"

    final_table.append([
        item['name'],
        f"{item['delta_psnr']:+.2f} dB",
        format_p(item['p_raw']),
        format_p(p_adj),
        f"{item['d']:.2f}",
        zakljucak
    ])

headers = ["Uklonjena Komponenta", "Δ PSNR", "Wilcoxon (Raw p, H₁: Abl<Full)", "Wilcoxon (Holm p)", "Cohen's d_z", "Naučni Zaključak"]
print("\n" + "="*115)
print("  TABELA: NAUČNO KORIGOVANA STATISTIČKA ZNAČAJNOST ABLACIJE")
print("="*115)
print(tabulate(final_table, headers=headers, tablefmt="fancy_grid", stralign="center", numalign="center"))
