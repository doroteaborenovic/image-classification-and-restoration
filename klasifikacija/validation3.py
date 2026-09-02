#with roc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    roc_curve,
    balanced_accuracy_score
)
import datetime
import warnings

warnings.filterwarnings('ignore')

drive_test_zip = "/content/drive/MyDrive/Projekat_Model/DATASET_VALIDACIJA.zip"
lokalni_test_path = "/content/DATASET_VALIDACIJA"

if not os.path.exists(lokalni_test_path):
    print("Priprema testnog skupa...")
    if os.path.exists(drive_test_zip):
        print("Pronađena ZIP arhiva na Drive-u, otpakujem...")
        get_ipython().system(f'unzip -q "{drive_test_zip}" -d "{lokalni_test_path}"')
        print("Dataset uspešno raspakovan.")
    else:
        print("UPOZORENJE: dataset.zip nije pronađen na Google Drive-u!")
else:
    print("Testni skup podataka je već spreman lokalno u /content/")


# ==============================================================================
# ARHITEKTURA MODELA (DODINA MREŽA)
# ==============================================================================
class RecursiveDenseMicroBlock(nn.Module):
    def __init__(self, channels: int, num_recursions: int = 3):
        super().__init__()
        self.num_recursions = num_recursions
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.fusion = nn.Conv2d(channels * num_recursions, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        outputs = []
        out = x
        for i in range(self.num_recursions):
            out = F.relu(self.bn(self.conv(out)) + x)
            outputs.append(out)
        merged = torch.cat(outputs, dim=1)
        return self.fusion(merged)


class SpectralDecomposeBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.low_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.high_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, 2, 1),
            nn.Softmax(dim=1)
        )
        self.fuse = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        low = F.interpolate(
            F.avg_pool2d(x, kernel_size=2),
            size=x.shape[2:], mode='bilinear', align_corners=False
        )
        high = x - low
        low_feat = self.low_conv(low)
        high_feat = self.high_conv(high)
        concat = torch.cat([low_feat, high_feat], dim=1)
        w = self.gate(concat)
        fused = w[:, 0:1] * low_feat + w[:, 1:2] * high_feat
        return self.fuse(torch.cat([fused, x], dim=1))


class SpatialBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.dense_micro = RecursiveDenseMicroBlock(out_ch, num_recursions=3)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.conv(x)
        x = self.dense_micro(x)
        pooled = self.pool(x)
        return pooled, x


class AsymmetricCrossBridge(nn.Module):
    def __init__(self, spatial_ch: int, spectral_ch: int, out_ch: int):
        super().__init__()
        self.spatial_to_spectral = nn.Sequential(
            nn.Conv2d(spatial_ch, spectral_ch, 1),
            nn.BatchNorm2d(spectral_ch),
            nn.ReLU(inplace=True)
        )
        self.spectral_to_spatial = nn.Sequential(
            nn.Conv2d(spectral_ch, spatial_ch, 1),
            nn.BatchNorm2d(spatial_ch),
            nn.ReLU(inplace=True)
        )
        self.fuse = nn.Conv2d(spatial_ch + spectral_ch, out_ch, 1)

    def forward(self, spatial_feat: Tensor, spectral_feat: Tensor) -> Tensor:
        spectral_enhanced = spectral_feat + self.spatial_to_spectral(
            F.adaptive_avg_pool2d(spatial_feat, spectral_feat.shape[2:])
        )
        spatial_enhanced = spatial_feat + self.spectral_to_spatial(
            F.interpolate(spectral_feat, size=spatial_feat.shape[2:],
                          mode='bilinear', align_corners=False)
        )
        min_h = min(spatial_feat.shape[2], spectral_feat.shape[2])
        min_w = min(spatial_feat.shape[3], spectral_feat.shape[3])
        s_pooled = F.adaptive_avg_pool2d(spatial_enhanced, (min_h, min_w))
        sp_pooled = F.adaptive_avg_pool2d(spectral_enhanced, (min_h, min_w))
        return self.fuse(torch.cat([s_pooled, sp_pooled], dim=1))


class GatedFusionBlock(nn.Module):
    def __init__(self, spatial_ch: int, spectral_ch: int, out_ch: int):
        super().__init__()
        self.spatial_proj = nn.Conv2d(spatial_ch, out_ch, 1)
        self.spectral_proj = nn.Conv2d(spectral_ch, out_ch, 1)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_ch * 2, out_ch // 4),
            nn.ReLU(inplace=True),
            nn.Linear(out_ch // 4, out_ch * 2),
            nn.Sigmoid()
        )

    def forward(self, spatial: Tensor, spectral: Tensor) -> Tensor:
        s = self.spatial_proj(spatial)
        sp = self.spectral_proj(
            F.interpolate(spectral, size=spatial.shape[2:],
                          mode='bilinear', align_corners=False)
        )
        combined = torch.cat([s, sp], dim=1)
        gates = self.gate(combined).view(combined.shape[0], -1, 1, 1)
        out_ch_val = s.shape[1]
        s_gate = gates[:, :out_ch_val]
        sp_gate = gates[:, out_ch_val:]
        return s_gate * s + sp_gate * sp


class DamageAttentionModule(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        attn_map = self.attention(x)
        attended = x * attn_map
        refined = self.refine(attended) + x
        return refined, attn_map


class DodinaMreza(nn.Module):
    def __init__(self, num_classes: int = 2, in_channels: int = 3):
        super().__init__()
        self.spatial_block1 = SpatialBlock(in_channels, 64)
        self.spatial_block2 = SpatialBlock(64, 128)
        self.spatial_block3 = SpatialBlock(128, 256)

        self.spectral_init = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.spectral_block1 = SpectralDecomposeBlock(64)
        self.spectral_pool1 = nn.MaxPool2d(2)
        self.spec_proj1 = nn.Sequential(
            nn.Conv2d(64, 128, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        self.spectral_block2 = SpectralDecomposeBlock(128)
        self.spectral_pool2 = nn.MaxPool2d(2)
        self.spec_proj2 = nn.Sequential(
            nn.Conv2d(128, 256, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )
        self.spectral_block3 = SpectralDecomposeBlock(256)

        self.cross1 = AsymmetricCrossBridge(64, 64, 64)
        self.cross2 = AsymmetricCrossBridge(128, 128, 128)
        self.cross3 = AsymmetricCrossBridge(256, 256, 256)

        self.gated_fusion = GatedFusionBlock(256, 256, 512)
        self.damage_attention = DamageAttentionModule(512)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        self.damage_map_head = nn.Sequential(
            nn.Conv2d(512, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        s1, s1_skip = self.spatial_block1(x)
        s2, s2_skip = self.spatial_block2(s1)
        s3, s3_skip = self.spatial_block3(s2)

        sp0 = self.spectral_init(x)
        sp1 = self.spectral_block1(sp0)
        sp1_p = self.spec_proj1(self.spectral_pool1(sp1))
        sp2 = self.spectral_block2(sp1_p)
        sp2_p = self.spec_proj2(self.spectral_pool2(sp2))
        sp3 = self.spectral_block3(sp2_p)

        c1 = self.cross1(s1_skip, sp1)
        c2 = self.cross2(s2_skip, sp2)
        c3 = self.cross3(s3_skip, sp3)

        s3_enriched = s3 + F.adaptive_avg_pool2d(c3, s3.shape[2:])

        fused = self.gated_fusion(s3_enriched, sp3)
        attended, damage_map = self.damage_attention(fused)

        logits = self.classifier(attended)
        aux_damage = self.damage_map_head(attended)

        return {
            'logits': logits,
            'damage_map': damage_map,
            'aux_damage': aux_damage
        }


# ==============================================================================
# DATASET
# ==============================================================================
class DamageDataset(Dataset):
    def __init__(self, dataset_dir: str, img_size: int = 128, train: bool = True):
        if train:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                       saturation=0.1, hue=0.05),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ])

        self.samples = []
        for label in [0, 1]:
            folder = os.path.join(dataset_dir, str(label))
            if not os.path.exists(folder):
                continue
            for fname in sorted(os.listdir(folder)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    self.samples.append((os.path.join(folder, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, label, path


# ==============================================================================
# EVALUACIJA SA ROC-AUC I REKALIBRACIJOM (O1)
# ==============================================================================
def evaluiraj_dodinu_mrezu_sa_detaljnim_klasama(model_path: str, test_dataset_dir: str, img_size: int = 128, batch_size: int = 32):
    nested_path = os.path.join(test_dataset_dir, "DATASET_VALIDACIJA")
    if os.path.exists(nested_path) and os.path.isdir(nested_path):
        test_dataset_dir = nested_path

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{'='*70}")
    print(f"Testni dataset: {test_dataset_dir}")
    print(f"{'='*70}\n")

    test_dataset = DamageDataset(test_dataset_dir, img_size=img_size, train=False)
    if len(test_dataset) == 0:
        print(f"Nema slika na putanji {test_dataset_dir} ili folder ne postoji.")
        return None, None

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    print(f"Pronađeno ukupno {len(test_dataset)} slika za testiranje.")
    model = DodinaMreza(num_classes=2).to(device)

    if not os.path.exists(model_path):
        print(f"Model nije pronađen u: {model_path}")
        return None, None

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    has_saved_threshold = False
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'best_threshold' in checkpoint and checkpoint['best_threshold'] is not None:
            best_threshold = checkpoint['best_threshold']
            has_saved_threshold = True
            print(f"✓ Učitan checkpoint (Najbolja tačnost na treningu: {checkpoint.get('best_val_acc', 0.0):.2f}%)")
            print(f"✓ Učitani optimalni prag (Trening prag): {best_threshold:.4f}")
        else:
            best_threshold = 0.70
            print("✓ Učitan checkpoint (Nema sačuvanog praga, postavljen na 0.70 za stabilnost)")
    else:
        model.load_state_dict(checkpoint)
        best_threshold = 0.50
        has_saved_threshold = True
        print("✓ Učitane težine modela. Koristi se podrazumevani prag: 0.50")

    model.eval()

    all_probs = []
    all_labels = []
    all_paths = []

    damage_mapping = {
        'apply_anisotropic_diffusion': 'Vlaga i gubitak detalja',
        'apply_mold_and_decay': 'Bud i bioloska degradacija',
        'apply_chemical_aging': 'Hemijsko starenje i zutilo',
        'apply_fft_lpf': 'Gubitak ostrine (FFT LPF)',
        'apply_cracks': 'Pukotine na platnu',
        'apply_water_stains': 'Vodene mrlje (Coffee-ring)',
        'apply_paint_flaking': 'Ljustenje boje',
        'apply_dust_and_scratches': 'Prasina i ogrebotine',
        'apply_combined_damage': 'Kombinovano ostecenje'
    }

    stats = {name: {'total': 0, 'correct': 0} for name in damage_mapping.values()}
    stats['Bez ostecenja (Ciste slike)'] = {'total': 0, 'correct': 0}

    print("\nPokrećem inferenciju (3-way TTA: Original + HorizFlip + VertFlip)...")
    with torch.no_grad():
        for images, labels, paths in test_loader:
            images = images.to(device)

            # 1. Original
            outputs = model(images)
            probs_orig = F.softmax(outputs['logits'], dim=-1)

            # 2. Horizontal Flip
            images_flipped_h = torch.flip(images, dims=[3])
            outputs_flipped_h = model(images_flipped_h)
            probs_flipped_h = F.softmax(outputs_flipped_h['logits'], dim=-1)

            # 3. Vertical Flip
            images_flipped_v = torch.flip(images, dims=[2])
            outputs_flipped_v = model(images_flipped_v)
            probs_flipped_v = F.softmax(outputs_flipped_v['logits'], dim=-1)

            # 3-way TTA usrednjavanje
            probs_final = (probs_orig + probs_flipped_h + probs_flipped_v) / 3.0

            # Čuvamo sirove verovatnoće za klasu 1 ("oštećeno")
            all_probs.extend(probs_final[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_paths.extend(paths)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Predikcija sa inicijalnim pragom
    all_preds = (all_probs >= best_threshold).astype(int)

    # ==============================================================================
    # ROC-AUC I YOUDEN'S J STATISTIKA (NALAZ O1)
    # ==============================================================================
    auc_val = roc_auc_score(all_labels, all_probs)
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)

    # Youden's J indeks (J = TPR - FPR) za pronalaženje najboljeg praga
    j_scores = tpr - fpr
    best_thr_idx = np.argmax(j_scores)
    best_thr_youden = thresholds[best_thr_idx]

    # Rekalibrisane predikcije
    preds_recalibrated = (all_probs >= best_thr_youden).astype(int)
    
    # Balansirana tačnost
    bal_acc_orig = balanced_accuracy_score(all_labels, all_preds) * 100
    bal_acc_recal = balanced_accuracy_score(all_labels, preds_recalibrated) * 100
    acc_orig = accuracy_score(all_labels, all_preds) * 100
    acc_recal = accuracy_score(all_labels, preds_recalibrated) * 100

    # Razvrstavanje tačnosti po klasama oštećenja (za originalni prag)
    for pred, label, path in zip(all_preds, all_labels, all_paths):
        if label == 0:
            stats['Bez ostecenja (Ciste slike)']['total'] += 1
            if pred == 0:
                stats['Bez ostecenja (Ciste slike)']['correct'] += 1
        else:
            filename = os.path.basename(path).lower()
            found = False
            for func_name, display_name in damage_mapping.items():
                if func_name in filename:
                    stats[display_name]['total'] += 1
                    if pred == 1:
                        stats[display_name]['correct'] += 1
                    found = True
                    break

    # ==============================================================================
    # ISPIS REZULTATA
    # ==============================================================================
    PINK = "\033[38;5;205m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    CYAN = "\033[96m"

    print("\n" + "="*70)
    print(f"{BOLD}REZULTATI EVALUACIJE I ROC-AUC ANALIZE (EKSPERIMENT 3){RESET}")
    print("="*70)
    print(f"Inicijalni prag (Trening prag) : {best_threshold:.4f}")
    print(f"Ukupna tačnost (Accuracy)       : {acc_orig:.2f}%")
    print(f"Balansirana tačnost (Bal. Acc)  : {bal_acc_orig:.2f}%")
    print("-" * 70)
    print(f"{CYAN}{BOLD}ROC-AUC Skor (Nezavisan od praga) : {auc_val:.4f}{RESET}")
    print(f"{GREEN}{BOLD}Optimalni prag (Youden's J)       : {best_thr_youden:.4f}{RESET}")
    print(f"{GREEN}Rekalibrisana Ukupna tačnost     : {acc_recal:.2f}%{RESET}")
    print(f"{GREEN}Rekalibrisana Balansirana tačnost: {bal_acc_recal:.2f}%{RESET}")
    print("="*70 + "\n")

    print(f"{BOLD}Izveštaj klasifikacije (Inicijalni prag):{RESET}")
    report = classification_report(
        all_labels,
        all_preds,
        target_names=['Klasa 0 (bez oštećenja)', 'Klasa 1 (oštećeno)'],
        digits=4
    )
    print(report)

    # Tabela po oštećenjima
    rows = []
    for cat, data in stats.items():
        total = data['total']
        correct = data['correct']
        acc = (correct / total * 100) if total > 0 else 0.0
        rows.append([cat, total, correct, round(acc, 2)])

    df_stats = pd.DataFrame(rows, columns=["tip ostecenja", "broj testiranih", "broj tacnih", "tacnost (%)"])

    top_border = f"{PINK}┌──────────────────────────────────┬────────────┬────────────┬──────────────┐{RESET}"
    mid_border = f"{PINK}├──────────────────────────────────┼────────────┼────────────┼──────────────┤{RESET}"
    bot_border = f"{PINK}└──────────────────────────────────┴────────────┴────────────┴──────────────┘{RESET}"

    print(f"\n{BOLD}Pregled po tipu oštećenja{RESET}")
    print(top_border)
    print(f"{PINK}│{RESET} {BOLD}{'tip ostecenja':<32} {PINK}│{RESET} {BOLD}{'testirano':<10} {PINK}│{RESET} {BOLD}{'tacno':<10} {PINK}│{RESET} {BOLD}{'tacnost (%)':<12} {PINK}│{RESET}")
    print(mid_border)
    for row in rows:
        cat_name = row[0]
        tested = row[1]
        correct = row[2]
        accuracy_val = f"{row[3]:.2f}%"
        print(f"{PINK}│{RESET} {cat_name:<32} {PINK}│{RESET} {tested:<10d} {PINK}│{RESET} {correct:<10d} {PINK}│{RESET} {accuracy_val:<12} {PINK}│{RESET}")
    print(bot_border)

    # ==============================================================================
    # ČUVANJE CSV I GRAFIČKIH ARTEFAKATA
    # ==============================================================================
    results_dir = os.path.dirname(model_path) if os.path.dirname(model_path) != "" else "."
    idx = 1
    while os.path.exists(os.path.join(results_dir, f"tabela_{idx}.csv")):
        idx += 1

    # 1. Tabela po oštećenjima
    csv_path = os.path.join(results_dir, f"tabela_{idx}.csv")
    df_stats.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 2. RAW skorovi po svakoj slici (kljucno za O1 proveru)
    df_raw = pd.DataFrame({
        'path': all_paths,
        'filename': [os.path.basename(p) for p in all_paths],
        'y_true': all_labels,
        'y_score': all_probs,
        'y_pred_orig': all_preds,
        'y_pred_recalibrated': preds_recalibrated
    })
    raw_csv_path = os.path.join(results_dir, f"rezultati_predikcija_raw_{idx}.csv")
    df_raw.to_csv(raw_csv_path, index=False, encoding="utf-8-sig")
    print(f"\n✓ Sirovi skorovi po slikama sačuvani u:\n  {raw_csv_path}")

    # 3. Izveštaj klasifikacije
    report_path = os.path.join(results_dir, f"izvestaj_klasifikacije_{idx}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # 4. Sažetak metrika (uključujući ROC-AUC i Youden J)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    metrics_path = os.path.join(results_dir, f"metrike_{idx}.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"=== METRIKE SA INICIJALNIM PRAGOM ({best_threshold:.4f}) ===\n")
        f.write(f"Accuracy               : {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"Balanced Accuracy      : {bal_acc_orig/100:.4f} ({bal_acc_orig:.2f}%)\n")
        f.write(f"Precision              : {precision:.4f}\n")
        f.write(f"Recall                 : {recall:.4f}\n")
        f.write(f"F1-score               : {f1:.4f}\n\n")
        f.write(f"=== ROC-AUC I REKALIBRACIJA (YOUDEN'S J) ===\n")
        f.write(f"ROC-AUC                : {auc_val:.4f}\n")
        f.write(f"Optimal Threshold (J)  : {best_thr_youden:.4f}\n")
        f.write(f"Recalibrated Accuracy  : {acc_recal/100:.4f} ({acc_recal:.2f}%)\n")
        f.write(f"Recalibrated Bal. Acc  : {bal_acc_recal/100:.4f} ({bal_acc_recal:.2f}%)\n")
    print(f"✓ Sve metrike sačuvane u:\n  {metrics_path}")

    # 5. Matrica konfuzije
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu',
                xticklabels=['Bez oštećenja', 'Oštećeno'],
                yticklabels=['Bez oštećenja', 'Oštećeno'])
    plt.xlabel('Predviđeno (Model)')
    plt.ylabel('Stvarno (Tačna oznaka)')
    plt.title(f'Matrica konfuzije (Prag = {best_threshold:.2f})')
    cm_path = os.path.join(results_dir, f"matrica_konfuzije_{idx}.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 6. ROC KRIVA SA OZNAČENIM YOUDEN'S J PRAGOM (Slika za rad)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='#c2185b', lw=2.5, label=f'ROC Kriva (AUC = {auc_val:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Slučajna klasifikacija (AUC = 0.50)')
    plt.scatter(fpr[best_thr_idx], tpr[best_thr_idx], color='#00796b', marker='o', s=80, zorder=5,
                label=f'Youden Prag = {best_thr_youden:.3f}\n(Bal. Acc = {bal_acc_recal:.1f}%)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=11)
    plt.ylabel('True Positive Rate (TPR)', fontsize=11)
    plt.title('ROC Kriva za Eksperiment 3 (Dodina Mreža)', fontsize=12, fontweight='bold')
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.grid(True, alpha=0.3)
    roc_path = os.path.join(results_dir, f"roc_kriva_{idx}.png")
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ ROC Kriva uspešno iscrtana i sačuvana u:\n  {roc_path}\n")

    return all_labels, all_probs, preds_recalibrated


# ==============================================================================
# POKRETANJE
# ==============================================================================
if __name__ == '__main__':
    putanja_do_modela = "/content/drive/MyDrive/Projekat_Model/dodinamrezajej.pth"
    putanja_do_test_dataseta = "/content/DATASET_VALIDACIJA"

    stvarne_oznake, verovatnoce, rekalibrisane_predikcije = evaluiraj_dodinu_mrezu_sa_detaljnim_klasama(
        model_path=putanja_do_modela,
        test_dataset_dir=putanja_do_test_dataseta,
        img_size=128,
        batch_size=32
    )
