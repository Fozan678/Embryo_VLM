import os
import random
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
EXCLUDE_DIR_NAMES = {'.ipynb_checkpoints', '__pycache__', '.git', 'figures'}

# ============================================================
# ONE SHARED PROJECT FOLDER -- every phase (this script, IQA, and any
# future phase you point at OUTPUT_DIR) writes into this SAME flat folder.
# No per-phase subfolders (processed_data/, embryo_iqa_outputs/, etc.).
# ============================================================
OUTPUT_DIR = "./embryo_project"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def read_csv_smart(path):
    """Read a CSV whose delimiter may be ';' , ',' , tab or '|'. Strips column
    whitespace and drops empty/unnamed trailing columns (e.g. a trailing ';')."""
    with open(path, 'r', encoding='utf-8', errors='replace') as fh:
        sample = fh.readline() + fh.readline()
    counts = {d: sample.count(d) for d in [';', ',', '\t', '|']}
    sep = max(counts, key=counts.get)
    if counts[sep] == 0:
        sep = ','
    df = pd.read_csv(path, sep=sep, engine='python')
    df.columns = [str(c).strip() for c in df.columns]
    keep = [c for c in df.columns if c and not c.lower().startswith('unnamed')]
    return df[keep]

# ============================================================
# EXPLICIT PATH SETUP (Downloads/archive structure)
# ============================================================
INPUT_DIR  = "."
IMAGE_DIR  = "./Downloads/archive/Images/Images"
TRAIN_CSV  = "./Downloads/archive/Gardner_train_silver.csv"
TEST_CSV   = "./Downloads/archive/Gardner_test_gold_onlyGardnerScores.csv"

def locate_input_dir(preferred, train_path):
    if os.path.exists(train_path):
        return preferred
    roots = [preferred, os.getcwd(), os.path.expanduser("~"), "/content", "/data", "/kaggle/input"]
    seen = set()
    for root in roots:
        if not root or root in seen or not os.path.isdir(root):
            continue
        seen.add(root)
        for dp, _, files in os.walk(root):
            if os.path.basename(train_path) in files:
                return dp
    return preferred

INPUT_DIR = locate_input_dir(INPUT_DIR, TRAIN_CSV)
train_csv_path = TRAIN_CSV if os.path.exists(TRAIN_CSV) else os.path.join(INPUT_DIR, os.path.basename(TRAIN_CSV))
test_csv_path = TEST_CSV if os.path.exists(TEST_CSV) else os.path.join(INPUT_DIR, os.path.basename(TEST_CSV))
image_dir_path = IMAGE_DIR if os.path.isdir(IMAGE_DIR) else os.path.join(INPUT_DIR, "Downloads/archive/Images/Images")

print("[PATHS] INPUT_DIR   : {}".format(INPUT_DIR))
print("[PATHS] Train CSV   : {} -> Exists: {}".format(train_csv_path, os.path.exists(train_csv_path)))
print("[PATHS] Test  CSV   : {} -> Exists: {}".format(test_csv_path, os.path.exists(test_csv_path)))
print("[PATHS] Image Dir   : {} -> Exists: {}".format(image_dir_path, os.path.isdir(image_dir_path)))
print("[PATHS] OUTPUT_DIR  : {}  (single shared project folder)\n".format(OUTPUT_DIR))

if not os.path.exists(train_csv_path):
    raise FileNotFoundError("Train CSV not found. Please check paths for {}".format(TRAIN_CSV))

CONFIG = {
    "dataset": {
        "name": "Scientific_Data_Embryo",
        "train_csv": train_csv_path,
        "test_csv": test_csv_path,
        "image_dir": image_dir_path,
        "output_dir": OUTPUT_DIR,
        "val_split_ratio": 0.2,
        "random_seed": 42
    },
    "preprocessing": {
        "image_size": 512,
        "apply_clahe": True,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225]
    },
    "augmentation": {
        "rotation_limit": 15,
        "brightness_limit": 0.2,
        "contrast_limit": 0.2,
        "blur_limit": 3,
        "gamma_limit": [80, 120]
    },
    "training": {
        "batch_size": 32,
        "num_workers": 2,     # Linux/Ubuntu handles multiprocess workers fine (0 was only needed on Win/Mac)
        "pin_memory": True
    }
}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ==========================================
# IMAGE INDEXING & ROBUST PATH RESOLUTION
# ==========================================
def build_image_index(root):
    by_name, by_stem = {}, {}
    if not root or not os.path.exists(root):
        return by_name, by_stem
    for dp, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIR_NAMES and not d.endswith('_outputs') and d != os.path.basename(OUTPUT_DIR)]
        for f in files:
            if f.lower().endswith(IMAGE_EXTS):
                full = os.path.join(dp, f)
                by_name.setdefault(f, full)
                by_name.setdefault(f.lower(), full)
                stem = os.path.splitext(f)[0]
                by_stem.setdefault(stem, full)
                by_stem.setdefault(stem.lower(), full)
    return by_name, by_stem

def resolve_image_path(value, by_name, by_stem):
    if value is None:
        return None
    s = str(value).strip().replace('\\', '/')
    if s == '' or s.lower() == 'nan':
        return None
    base = os.path.basename(s)
    for key in (base, base.lower()):
        if key in by_name:
            return by_name[key]
    stem = os.path.splitext(base)[0]
    for key in (stem, stem.lower()):
        if key in by_stem:
            return by_stem[key]
    for ext in IMAGE_EXTS:
        cand = base + ext
        for key in (cand, cand.lower()):
            if key in by_name:
                return by_name[key]
    return None

def detect_image_column(df, by_name, by_stem, sample=300):
    best_col, best_rate = None, -1.0
    n = min(len(df), sample)
    if n == 0:
        return None, 0.0
    probe = df.head(n)
    for c in df.columns:
        vals = probe[c].tolist()
        hits = sum(1 for v in vals if resolve_image_path(v, by_name, by_stem) is not None)
        rate = hits / n
        if rate > best_rate:
            best_col, best_rate = c, rate
    return best_col, best_rate

def attach_resolved_paths(df, by_name, by_stem, tag=""):
    df = df.copy()
    col, rate = detect_image_column(df, by_name, by_stem)
    if col is None:
        df['resolved_path'] = None
        print("[IMAGES] {}: no usable columns found.".format(tag))
        return df
    df['resolved_path'] = df[col].map(lambda v: resolve_image_path(v, by_name, by_stem))
    resolved = int(df['resolved_path'].notna().sum())
    print("[IMAGES] {}: image column='{}' | resolved {}/{} ({:.1f}%)".format(
        tag, col, resolved, len(df), 100.0 * resolved / max(len(df), 1)))
    return df

# ==========================================
# DATASET
# ==========================================
class EmbryoDataset(Dataset):
    def __init__(self, df, image_dir, is_train=True, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.is_train = is_train
        self.transform = transform
        self.label_cols = {}
        for base in ['EXP', 'ICM', 'TE']:
            for cand in ['{}_silver'.format(base), '{}_gold'.format(base), base]:
                if cand in self.df.columns:
                    self.label_cols[base] = cand
                    break

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = row['resolved_path'] if 'resolved_path' in row else None
        image = None
        if isinstance(img_path, str) and os.path.exists(img_path):
            bgr = cv2.imread(img_path)
            if bgr is not None:
                image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if image is None:
            image = np.zeros((512, 512, 3), dtype=np.uint8)

        if self.transform:
            image = self.transform(image=image)['image']

        def _label(base):
            col = self.label_cols.get(base)
            v = row[col] if (col is not None and col in row) else -1
            if pd.isna(v):
                return torch.tensor(-1, dtype=torch.long), torch.tensor(0.0, dtype=torch.float)
            try:
                v = int(v)
            except (ValueError, TypeError):
                return torch.tensor(-1, dtype=torch.long), torch.tensor(0.0, dtype=torch.float)
            return torch.tensor(v, dtype=torch.long), torch.tensor(1.0, dtype=torch.float)

        exp_t, exp_m = _label('EXP')
        icm_t, icm_m = _label('ICM')
        te_t, te_m = _label('TE')

        labels = {'expansion': exp_t, 'icm': icm_t, 'te': te_t}
        masks = {'expansion': exp_m, 'icm': icm_m, 'te': te_m}
        meta = {
            'image_name': os.path.basename(str(img_path)) if isinstance(img_path, str) else '',
            'image_path': str(img_path),
            'patient_id': str(row.get('patient_id', ''))
        }
        return image, labels, masks, meta

def get_transforms(config):
    s = config['preprocessing']['image_size']
    train_transform = A.Compose([
        A.Resize(s, s),
        A.CLAHE(p=1.0),
        A.Affine(translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
                 scale=(0.95, 1.05),
                 rotate=(-config['augmentation']['rotation_limit'], config['augmentation']['rotation_limit']),
                 p=0.5),
        A.RandomCrop(width=int(s * 0.9), height=int(s * 0.9)),
        A.Resize(s, s),
        A.RandomBrightnessContrast(brightness_limit=config['augmentation']['brightness_limit'],
                                   contrast_limit=config['augmentation']['contrast_limit'], p=0.5),
        A.GaussianBlur(blur_limit=config['augmentation']['blur_limit'], p=0.3),
        A.RandomGamma(gamma_limit=config['augmentation']['gamma_limit'], p=0.3),
        A.Normalize(mean=config['preprocessing']['normalize_mean'], std=config['preprocessing']['normalize_std']),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(s, s),
        A.CLAHE(p=1.0),
        A.Normalize(mean=config['preprocessing']['normalize_mean'], std=config['preprocessing']['normalize_std']),
        ToTensorV2()
    ])
    return train_transform, val_transform

def prepare_data_splits(config):
    train_df = read_csv_smart(config['dataset']['train_csv'])
    if os.path.exists(config['dataset']['test_csv']):
        test_df = read_csv_smart(config['dataset']['test_csv'])
    else:
        test_df = train_df.copy()

    print("[WHOLE DATASET] train_silver rows: {} | test_gold rows: {} (no subsampling applied anywhere)".format(
        len(train_df), len(test_df)))

    by_name, by_stem = build_image_index(config['dataset']['image_dir'])
    print("[IMAGES] indexed {} image files under {}\n".format(len(set(by_name.values())), config['dataset']['image_dir']))

    train_df = attach_resolved_paths(train_df, by_name, by_stem, tag="train")
    if int(train_df['resolved_path'].notna().sum()) == 0:
        for extra in [INPUT_DIR, os.path.expanduser("~")]:
            bn, bs = build_image_index(extra)
            by_name.update(bn); by_stem.update(bs)
        print("[IMAGES] widened search -> {} image files\n".format(len(set(by_name.values()))))
        train_df = attach_resolved_paths(train_df, by_name, by_stem, tag="train (widened)")
    test_df = attach_resolved_paths(test_df, by_name, by_stem, tag="test")

    if 'patient_id' in train_df.columns:
        gss = GroupShuffleSplit(n_splits=1, test_size=config['dataset']['val_split_ratio'], random_state=config['dataset']['random_seed'])
        train_idx, val_idx = next(gss.split(train_df, groups=train_df['patient_id']))
    else:
        train_df['patient_id'] = [str(i) for i in range(len(train_df))]
        ss = ShuffleSplit(n_splits=1, test_size=config['dataset']['val_split_ratio'], random_state=config['dataset']['random_seed'])
        train_idx, val_idx = next(ss.split(train_df))

    train_split = train_df.iloc[train_idx].copy()
    val_split = train_df.iloc[val_idx].copy()
    test_split = test_df.copy()

    train_split['split'] = 'train'
    val_split['split'] = 'val'
    test_split['split'] = 'test'

    combined_df = pd.concat([train_split, val_split, test_split], ignore_index=True)
    # SINGLE flat project folder -- no 'processed_data/' subfolder
    combined_df.to_csv(os.path.join(config['dataset']['output_dir'], 'processed_metadata.csv'), index=False)
    print("[SAVED] processed_metadata.csv -> {} ({} total rows: train={}, val={}, test={})".format(
        config['dataset']['output_dir'], len(combined_df), len(train_split), len(val_split), len(test_split)))
    return train_split, val_split, test_split

def generate_visualizations(df, config):
    # SINGLE flat project folder -- no 'figures/' subfolder
    out_dir = config['dataset']['output_dir']
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='split', hue='split', palette='viridis', legend=False)
    plt.title('Dataset Split Distribution (whole dataset, N={})'.format(len(df)))
    plt.savefig(os.path.join(out_dir, 'dataset_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    cols_to_plot = [('EXP_silver', 'Expansion (EXP)'), ('ICM_silver', 'Inner Cell Mass (ICM)'), ('TE_silver', 'Trophectoderm (TE)')]
    for i, (col, title) in enumerate(cols_to_plot):
        target_col = col if col in df.columns else col.replace('_silver', '_gold')
        if target_col in df.columns:
            sns.countplot(data=df, x=target_col, hue=target_col, ax=axes[i], palette='crest', legend=False)
            axes[i].set_title('Score Distribution: {}'.format(title))
        else:
            axes[i].set_title('Missing: {}'.format(title))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'gardner_score_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("[SAVED] dataset_distribution.png, gardner_score_distribution.png -> {}".format(out_dir))

def build_dataloaders(config):
    set_seed(config['dataset']['random_seed'])
    train_df, val_df, test_df = prepare_data_splits(config)
    generate_visualizations(pd.concat([train_df, val_df, test_df], ignore_index=True), config)

    train_transform, val_transform = get_transforms(config)

    train_dataset = EmbryoDataset(train_df, config['dataset']['image_dir'], is_train=True, transform=train_transform)
    val_dataset = EmbryoDataset(val_df, config['dataset']['image_dir'], is_train=False, transform=val_transform)
    test_dataset = EmbryoDataset(test_df, config['dataset']['image_dir'], is_train=False, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=config['training']['batch_size'],
        shuffle=True, num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'], drop_last=True)
    val_loader = DataLoader(
        val_dataset, batch_size=config['training']['batch_size'],
        shuffle=False, num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'])
    test_loader = DataLoader(
        test_dataset, batch_size=config['training']['batch_size'],
        shuffle=False, num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'])

    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    train_loader, val_loader, test_loader = build_dataloaders(CONFIG)
    print("\nData pipeline compiled (WHOLE dataset). Train batches: {}, Val batches: {}, Test batches: {}".format(
        len(train_loader), len(val_loader), len(test_loader)))

    images, labels, masks, meta = next(iter(train_loader))
    stds = images.view(images.size(0), -1).std(dim=1)
    nonblank = int((stds > 1e-6).sum().item())
    print("Sanity: {}/{} images in first batch are non-blank (real pixels).".format(nonblank, images.size(0)))
    print("Label coverage in first batch -> EXP: {:.0f}, ICM: {:.0f}, TE: {:.0f}".format(
        masks['expansion'].sum().item(), masks['icm'].sum().item(), masks['te'].sum().item()))
    print("\nAll outputs saved directly in: {}".format(os.path.abspath(CONFIG['dataset']['output_dir'])))
