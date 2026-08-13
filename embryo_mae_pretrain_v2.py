import os
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from IPython.display import Image, display
from tqdm.auto import tqdm
import glob
import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
EXCLUDE_DIR_NAMES = {'.ipynb_checkpoints', '__pycache__', '.git', 'processed_data',
                     'embryo_iqa_outputs', 'mae_outputs', 'morphology_outputs', 'probe_outputs',
                     'seg_outputs', 'grader_outputs', 'grounded_morph_outputs',
                     'grounded_morph_probe_outputs', 'relational_graph_outputs',
                     'clinical_multitask_outputs', 'uncertainty_outputs', 'vlm_outputs',
                     'explainability_outputs', 'figures', 'embryo_project'}   # NEW: consolidated project folder

# ============================================================
# PATHS  (LOCAL JUPYTER) — SAME output path as before, so every
# downstream Phase 4-8 script picks up this improved encoder with
# zero changes needed on their end.
# ============================================================
INPUT_DIR  = "."
IMAGE_DIR  = "./Downloads/archive/Images/Images"
TRAIN_CSV  = "Gardner_train_silver.csv"
TEST_CSV   = "Gardner_test_gold_onlyGardnerScores.csv"

def locate_input_dir(preferred, train_name):
    if preferred and os.path.exists(os.path.join(preferred, train_name)):
        return preferred
    for root in [preferred, os.getcwd(), os.path.expanduser("~")]:
        if root and os.path.isdir(root):
            for dp, _, files in os.walk(root):
                if train_name in files:
                    return dp
    return preferred

INPUT_DIR = locate_input_dir(INPUT_DIR, TRAIN_CSV)
train_csv_path = os.path.join(INPUT_DIR, TRAIN_CSV)
test_csv_path = os.path.join(INPUT_DIR, TEST_CSV)
image_dir_path = IMAGE_DIR if (IMAGE_DIR and os.path.isdir(IMAGE_DIR)) else INPUT_DIR

print(f"[PATHS] Train CSV : {train_csv_path} -> Exists: {os.path.exists(train_csv_path)}")
print(f"[PATHS] Test  CSV : {test_csv_path} -> Exists: {os.path.exists(test_csv_path)}")
print(f"[PATHS] Image Dir : {image_dir_path} -> Exists: {os.path.isdir(image_dir_path)}\n")

if not os.path.exists(train_csv_path):
    raise FileNotFoundError(f"Train CSV not found. Set INPUT_DIR to the folder holding {TRAIN_CSV}")

MAE_CONFIG = {
    "dataset": {
        "name": "Scientific_Data_Embryo",
        "train_csv": train_csv_path,
        "test_csv": test_csv_path,
        "image_dir": image_dir_path,
        "image_size": 512,
        "patch_size": 16,
        "in_chans": 3,
        "val_split_ratio": 0.2,
        "random_seed": 42
    },
    "preprocessing": {
        "apply_clahe": True,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225]
    },
    "augmentation": {
        "rotation_limit": 15,
        # ENHANCEMENT: local/zoomed-crop augmentation, applied to a fraction of
        # training samples alongside the existing global whole-embryo view.
        # Rationale: EXP is a global-shape property (mean-pool already captures
        # it fine); ICM/TE are LOCAL sub-structure properties (cell-clump
        # compactness, epithelium regularity) that a whole-embryo resize can
        # wash out. Forcing the encoder to also reconstruct zoomed-in local
        # patches teaches it fine texture, which the graders can then draw on.
        "local_view_prob": 0.5,
        "local_crop_min_frac": 0.35,
        "local_crop_max_frac": 0.65,
    },
    "model": {
        # ENHANCEMENT: real ViT-Base (768-dim, 12 layers) instead of the
        # previous odd "1024-dim, depth=24-labeled-but-8-actual" shape.
        # More parameter-efficient for ~1,600 training images.
        #
        # IMPORTANT: "depth": 36 is NOT a typo. Every downstream script
        # (grader_finetune, grounded_morph_grader, clinical_multitask,
        # uncertainty_module, evidence_vlm, explainability) derives its
        # encoder layer count as `config['model']['depth'] // 3` when it
        # loads this checkpoint. Storing 36 here means those scripts compute
        # 36 // 3 = 12 real layers automatically -> a true ViT-Base loads
        # into every downstream phase with ZERO code changes there.
        "embed_dim": 768,
        "depth": 36,          # 36 // 3 = 12 real transformer layers (see note above)
        "num_heads": 12,
        "decoder_embed_dim": 512,
        "decoder_depth": 8,
        "decoder_num_heads": 16,
        "mask_ratio": 0.75,
        "norm_pix_loss": True,
    },
    "training": {
        "batch_size": 24,      # ViT-Base is lighter than the old config -> room for a bigger batch
        "accum_iter": 2,       # effective batch = 48
        "epochs": 400,
        "lr": 1.5e-4,
        "min_lr": 1e-6,
        "weight_decay": 0.05,
        "warmup_ratio": 0.05,
        "grad_clip": 1.0,
        "num_workers": 2,
        "pin_memory": True,
        "viz_samples": 4,
        "output_dir": "./embryo_project/mae"   # consolidated: embryo_project/mae/{figures,checkpoints}/
    }
}

# ==========================================
# HELPER FUNCTIONS & EMBEDDINGS
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return torch.from_numpy(pos_embed).float()

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.outer(pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)

# ==========================================
# ROBUST CSV + IMAGE PATH RESOLUTION
# ==========================================
def read_csv_smart(path):
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

def build_image_index(root):
    by_name, by_stem = {}, {}
    if not root or not os.path.exists(root):
        return by_name, by_stem
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIR_NAMES and not d.endswith('_outputs')]
        for f in files:
            if f.lower().endswith(IMAGE_EXTS):
                full = os.path.join(dirpath, f)
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
        print(f"[IMAGES] {tag}: no usable columns found.")
        return df
    df['resolved_path'] = df[col].map(lambda v: resolve_image_path(v, by_name, by_stem))
    resolved = int(df['resolved_path'].notna().sum())
    print(f"[IMAGES] {tag}: image column='{col}' | resolved {resolved}/{len(df)} "
          f"({100.0 * resolved / max(len(df), 1):.1f}%)")
    return df

# ==========================================
# DATASET & TRANSFORMS  (global view + local zoomed-crop view)
# ==========================================
class EmbryoDataset(Dataset):
    def __init__(self, df, image_dir, is_train=True, global_tf=None, local_tf=None, aug_cfg=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.is_train = is_train
        self.global_tf = global_tf
        self.local_tf = local_tf
        self.aug_cfg = aug_cfg or {}
        self.label_cols = {}
        for base in ['EXP', 'ICM', 'TE']:
            for cand in [f'{base}_silver', f'{base}_gold', base]:
                if cand in self.df.columns:
                    self.label_cols[base] = cand
                    break

    def __len__(self):
        return len(self.df)

    def _local_zoom_crop(self, image):
        h, w = image.shape[:2]
        frac = random.uniform(self.aug_cfg.get('local_crop_min_frac', 0.35),
                              self.aug_cfg.get('local_crop_max_frac', 0.65))
        ch, cw = max(8, int(h * frac)), max(8, int(w * frac))
        y0 = random.randint(0, max(0, h - ch))
        x0 = random.randint(0, max(0, w - cw))
        return image[y0:y0 + ch, x0:x0 + cw]

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

        if self.is_train and self.local_tf is not None and random.random() < self.aug_cfg.get('local_view_prob', 0.5):
            image = self._local_zoom_crop(image)          # local, zoomed-in sub-region
            image = self.local_tf(image=image)['image']
        elif self.global_tf is not None:
            image = self.global_tf(image=image)['image']  # whole-embryo global view

        def _label(base):
            col = self.label_cols.get(base)
            v = row[col] if (col is not None and col in row) else -1
            if pd.isna(v):
                v = -1
            try:
                v = int(v)
            except (ValueError, TypeError):
                v = -1
            return torch.tensor(v, dtype=torch.long)

        labels = {'expansion': _label('EXP'), 'icm': _label('ICM'), 'te': _label('TE')}
        meta = {'image_path': str(img_path), 'patient_id': str(row.get('patient_id', ''))}
        return image, labels, meta

def get_global_transform(config):
    """Whole-embryo view: mild geometric aug + CLAHE, same convention as before."""
    return A.Compose([
        A.Resize(config['dataset']['image_size'], config['dataset']['image_size']),
        A.CLAHE(p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=config['preprocessing']['normalize_mean'], std=config['preprocessing']['normalize_std']),
        ToTensorV2()
    ])

def get_local_transform(config):
    """Local zoomed-in view: image is already cropped to a small sub-region by
    the Dataset before this runs, so we just resize it back up + light aug."""
    size = config['dataset']['image_size']
    return A.Compose([
        A.Resize(size, size),
        A.CLAHE(p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=config['preprocessing']['normalize_mean'], std=config['preprocessing']['normalize_std']),
        ToTensorV2()
    ])

def get_eval_transforms(config):
    return A.Compose([
        A.Resize(config['dataset']['image_size'], config['dataset']['image_size']),
        A.CLAHE(p=1.0),
        A.Normalize(mean=config['preprocessing']['normalize_mean'], std=config['preprocessing']['normalize_std']),
        ToTensorV2()
    ])

def prepare_data_splits(config):
    train_df = read_csv_smart(config['dataset']['train_csv'])
    test_df = read_csv_smart(config['dataset']['test_csv']) if os.path.exists(config['dataset']['test_csv']) else train_df.copy()
    print(f"[WHOLE DATASET] train_silver rows: {len(train_df)} | test_gold rows: {len(test_df)} (no subsampling)\n")

    img_root = config['dataset']['image_dir']
    by_name, by_stem = build_image_index(img_root if os.path.isdir(img_root) else INPUT_DIR)
    print(f"[IMAGES] indexed {len(set(by_name.values()))} image files\n")

    train_df = attach_resolved_paths(train_df, by_name, by_stem, tag="train")
    if int(train_df['resolved_path'].notna().sum()) == 0:
        for extra in [INPUT_DIR, os.path.expanduser("~")]:
            bn, bs = build_image_index(extra)
            by_name.update(bn); by_stem.update(bs)
        print(f"[IMAGES] widened search -> {len(set(by_name.values()))} image files\n")
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
    output_dir = config['training']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    combined_df.to_csv(os.path.join(output_dir, 'processed_metadata.csv'), index=False)
    return train_split, val_split, test_split

def get_fixed_viz_batch(df, config, n=4):
    ds = EmbryoDataset(df, config['dataset']['image_dir'], is_train=False, global_tf=get_eval_transforms(config))
    imgs = []
    for i in range(len(ds)):
        img, _, meta = ds[i]
        p = meta['image_path']
        if p not in (None, 'None', '') and os.path.exists(str(p)):
            imgs.append(img)
        if len(imgs) == n:
            break
    if len(imgs) == 0:
        for i in range(min(n, len(ds))):
            imgs.append(ds[i][0])
    return torch.stack(imgs)

# ==========================================
# MASKED AUTOENCODER (MAE) ARCHITECTURE  — unchanged structurally, only
# config values (ViT-Base) differ from before.
# ==========================================
class PatchEmbed(nn.Module):
    def __init__(self, img_size=512, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

class MaskedAutoencoderViT(nn.Module):
    def __init__(self, config=MAE_CONFIG):
        super().__init__()
        self.config = config
        self.norm_pix_loss = config['model'].get('norm_pix_loss', False)

        self.patch_embed = PatchEmbed(
            img_size=config['dataset']['image_size'],
            patch_size=config['dataset']['patch_size'],
            in_chans=config['dataset']['in_chans'],
            embed_dim=config['model']['embed_dim']
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config['model']['embed_dim']))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config['model']['embed_dim']), requires_grad=False)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['model']['embed_dim'], nhead=config['model']['num_heads'],
            dim_feedforward=config['model']['embed_dim'] * 4, dropout=0.1,
            activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config['model']['depth'] // 3, enable_nested_tensor=False
        )
        self.encoder_norm = nn.LayerNorm(config['model']['embed_dim'])

        self.decoder_embed = nn.Linear(config['model']['embed_dim'], config['model']['decoder_embed_dim'], bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config['model']['decoder_embed_dim']))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config['model']['decoder_embed_dim']), requires_grad=False)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=config['model']['decoder_embed_dim'], nhead=config['model']['decoder_num_heads'],
            dim_feedforward=config['model']['decoder_embed_dim'] * 4, dropout=0.1,
            activation='gelu', batch_first=True, norm_first=True)
        self.decoder = nn.TransformerEncoder(
            decoder_layer, num_layers=config['model']['decoder_depth'] // 2, enable_nested_tensor=False
        )
        self.decoder_norm = nn.LayerNorm(config['model']['decoder_embed_dim'])

        self.decoder_pred = nn.Linear(
            config['model']['decoder_embed_dim'],
            config['dataset']['patch_size'] ** 2 * config['dataset']['in_chans'], bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** 0.5), cls_token=True)
        self.pos_embed.data.copy_(pos_embed.unsqueeze(0))
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches ** 0.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(decoder_pos_embed.unsqueeze(0))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        p = self.config['dataset']['patch_size']
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        return x.reshape(shape=(imgs.shape[0], h * w, p * p * 3))

    def unpatchify(self, x):
        p = self.config['dataset']['patch_size']
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(shape=(x.shape[0], 3, h * p, w * p))

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.encoder(x)
        x = self.encoder_norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed
        x = self.decoder(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x[:, 1:, :]

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** 0.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

# ==========================================
# LR SCHEDULE + PRETRAINING LOOP
# ==========================================
def adjust_learning_rate(optimizer, epoch_f, config, warmup_epochs):
    base_lr = config['training']['lr']
    min_lr = config['training']['min_lr']
    total = config['training']['epochs']
    if epoch_f < warmup_epochs:
        lr = base_lr * epoch_f / max(1, warmup_epochs)
    else:
        denom = max(1, total - warmup_epochs)
        lr = min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * (epoch_f - warmup_epochs) / denom))
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr

@torch.no_grad()
def visualize_reconstructions(model, images, config, save_path, mask_ratio=0.75, n=4, title=None):
    model.eval()
    device = next(model.parameters()).device
    imgs = images[:n].to(device)
    use_amp = torch.cuda.is_available()
    with torch.amp.autocast('cuda', enabled=use_amp):
        _, pred, mask = model(imgs, mask_ratio)
    pred = pred.float()
    if getattr(model, 'norm_pix_loss', False):
        tgt = model.patchify(imgs.float())
        pm = tgt.mean(dim=-1, keepdim=True)
        ps = (tgt.var(dim=-1, keepdim=True) + 1e-6).sqrt()
        pred = pred * ps + pm
    pred = model.unpatchify(pred).float()

    p_side = config['dataset']['patch_size']
    ch = config['dataset']['in_chans']
    mask_rep = mask.unsqueeze(-1).repeat(1, 1, p_side * p_side * ch)
    mask_full = model.unpatchify(mask_rep).float()

    mean = torch.tensor(config['preprocessing']['normalize_mean'], device=device).view(1, 3, 1, 1)
    std = torch.tensor(config['preprocessing']['normalize_std'], device=device).view(1, 3, 1, 1)

    def denorm(t):
        return torch.clamp(t.float() * std + mean, 0, 1).detach().cpu().numpy().transpose(0, 2, 3, 1)

    orig = denorm(imgs)
    recon = denorm(pred)
    m_img = mask_full.detach().cpu().numpy().transpose(0, 2, 3, 1)
    masked_vis = np.clip(orig * (1 - m_img), 0, 1)
    paste = np.clip(orig * (1 - m_img) + recon * m_img, 0, 1)

    n = min(n, orig.shape[0])
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    if n == 1:
        axes = axes[None, :]
    col_titles = ['Original', f'Masked ({int(mask_ratio*100)}%)', 'MAE Reconstruction', 'Reconstruction + Visible']
    panels = [orig, masked_vis, recon, paste]
    for i in range(n):
        for j in range(4):
            axes[i, j].imshow(panels[j][i])
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(col_titles[j], fontsize=11)
    if title:
        fig.suptitle(title, fontsize=13)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
    else:
        plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    # RECORD: save the real boolean mask array (which patches were masked),
    # not just the rendered picture -- a PNG isn't reanalyzable data.
    mask_path = save_path.replace('.png', '_mask.npy')
    np.save(mask_path, mask_full.detach().cpu().numpy())

def pretrain_mae(model, train_loader, viz_images, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    use_amp = torch.cuda.is_available()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'],
                                  betas=(0.9, 0.95), weight_decay=config['training']['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    epochs = config['training']['epochs']
    accum_iter = config['training'].get('accum_iter', 1)
    grad_clip = config['training'].get('grad_clip', 1.0)
    warmup = max(1, int(config['training'].get('warmup_ratio', 0.1) * epochs))
    steps = max(1, len(train_loader))
    mask_ratio = config['model']['mask_ratio']

    out_dir = config['training']['output_dir']
    fig_dir = os.path.join(out_dir, 'figures')
    ckpt_dir = os.path.join(out_dir, 'checkpoints')
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    viz_every = max(1, epochs // 10)

    real_depth = config['model']['depth'] // 3
    print(f"[TRAIN] ViT-Base | real encoder layers={real_depth} (depth//3) | embed_dim={config['model']['embed_dim']} "
          f"| device={device} | amp={use_amp} | micro-batch={config['training']['batch_size']} "
          f"| accum={accum_iter} | eff-batch={config['training']['batch_size']*accum_iter} "
          f"| steps/epoch={steps} | warmup={warmup}\n")

    visualize_reconstructions(model, viz_images, config,
                              os.path.join(fig_dir, 'recon_epoch_000_random_init.png'),
                              mask_ratio=mask_ratio, n=min(config['training']['viz_samples'], viz_images.shape[0]),
                              title='Before training (random init)')

    history, best = [], float('inf')
    for epoch in range(epochs):
        model.train()
        running = 0.0
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(enumerate(train_loader), total=steps, desc=f"Epoch {epoch+1}/{epochs}")
        for step, batch in pbar:
            lr = adjust_learning_rate(optimizer, epoch + step / steps, config, warmup)
            images = batch[0].to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                loss, _, _ = model(images, mask_ratio=mask_ratio)
                loss_scaled = loss / accum_iter
            scaler.scale(loss_scaled).backward()
            if (step + 1) % accum_iter == 0 or (step + 1) == steps:
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

        avg = running / steps
        history.append(avg)
        print(f"[Epoch {epoch+1:03d}] recon MSE = {avg:.5f}")

        torch.save({'model': model.state_dict(), 'epoch': epoch, 'loss': avg, 'config': config},
                   os.path.join(ckpt_dir, 'mae_last.pth'))
        if avg < best:
            best = avg
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'loss': avg, 'config': config},
                       os.path.join(ckpt_dir, 'mae_best.pth'))

        if (epoch + 1) % viz_every == 0 or (epoch + 1) == epochs:
            visualize_reconstructions(model, viz_images, config,
                                      os.path.join(fig_dir, f'recon_epoch_{epoch+1:03d}.png'),
                                      mask_ratio=mask_ratio,
                                      n=min(config['training']['viz_samples'], viz_images.shape[0]),
                                      title=f'Epoch {epoch+1} (MSE {avg:.4f})')

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), history, marker='o', color='steelblue')
    plt.title('MAE Pretraining Loss (ViT-Base + global/local multi-scale aug)')
    plt.xlabel('Epoch'); plt.ylabel('Masked Reconstruction Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(fig_dir, 'pretrain_loss_curve.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # RECORD: real per-epoch loss values as data (CSV), not just the rendered curve
    pd.DataFrame({'epoch': range(1, epochs + 1), 'recon_loss': history}).to_csv(
        os.path.join(out_dir, 'training_history.csv'), index=False)

    visualize_reconstructions(model, viz_images, config,
                              os.path.join(fig_dir, 'reconstruction_final.png'),
                              mask_ratio=mask_ratio,
                              n=min(config['training']['viz_samples'], viz_images.shape[0]),
                              title='Final trained reconstruction')
    return model, history

# ==========================================
# DATASET / SCORE / LATENT VISUALIZATIONS
# ==========================================
def generate_dataset_visualizations(model, dataloader, combined_df, config):
    fig_dir = os.path.join(config['training']['output_dir'], 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()

    plt.figure(figsize=(8, 5))
    sns.countplot(data=combined_df, x='split', hue='split', palette='viridis', legend=False)
    plt.title('Dataset Split Distribution')
    plt.xlabel('Split Type'); plt.ylabel('Sample Count')
    plt.savefig(os.path.join(fig_dir, 'dataset_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (col, title) in enumerate([('EXP_silver', 'Expansion (EXP)'),
                                       ('ICM_silver', 'Inner Cell Mass (ICM)'),
                                       ('TE_silver', 'Trophectoderm (TE)')]):
        target_col = col if col in combined_df.columns else col.replace('_silver', '_gold')
        if target_col in combined_df.columns:
            sns.countplot(data=combined_df, x=target_col, hue=target_col, ax=axes[i], palette='crest', legend=False)
            axes[i].set_title(f'Score Distribution: {title}')
            axes[i].set_xlabel('Score Value'); axes[i].set_ylabel('Frequency')
        else:
            axes[i].set_title(f'Missing: {title}')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'gardner_score_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    all_embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch[0].to(device)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                latent, _, _ = model.forward_encoder(images, mask_ratio=0.0)
            all_embeddings.append(latent[:, 0, :].float().cpu().numpy())
    embeddings = np.concatenate(all_embeddings, axis=0)

    if len(embeddings) > 5:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, len(embeddings) // 3)))
        emb2d = tsne.fit_transform(embeddings)
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=emb2d[:, 0], y=emb2d[:, 1], color='steelblue', alpha=0.7, s=50)
        plt.title('t-SNE Latent Space Distribution (ViT-Base Encoder Embeddings)')
        plt.xlabel('t-SNE Dimension 1'); plt.ylabel('t-SNE Dimension 2')
        plt.savefig(os.path.join(fig_dir, 'tsne_latent_space.png'), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"[SUCCESS] Dataset/score/latent graphs saved to: {fig_dir}")

if __name__ == '__main__':
    set_seed(MAE_CONFIG['dataset']['random_seed'])

    train_df, val_df, test_df = prepare_data_splits(MAE_CONFIG)
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    global_tf = get_global_transform(MAE_CONFIG)
    local_tf = get_local_transform(MAE_CONFIG)
    train_dataset = EmbryoDataset(train_df, MAE_CONFIG['dataset']['image_dir'], is_train=True,
                                  global_tf=global_tf, local_tf=local_tf, aug_cfg=MAE_CONFIG['augmentation'])
    train_loader = DataLoader(
        train_dataset, batch_size=MAE_CONFIG['training']['batch_size'],
        shuffle=True, num_workers=MAE_CONFIG['training']['num_workers'],
        pin_memory=MAE_CONFIG['training']['pin_memory'], drop_last=True
    )

    viz_source = val_df if len(val_df) >= MAE_CONFIG['training']['viz_samples'] else train_df
    viz_images = get_fixed_viz_batch(viz_source, MAE_CONFIG, n=MAE_CONFIG['training']['viz_samples'])

    model = MaskedAutoencoderViT(MAE_CONFIG)
    print("\nMasked Autoencoder (ViT-Base, ENHANCED w/ multi-scale local+global aug) initialized.\n")

    model, history = pretrain_mae(model, train_loader, viz_images, MAE_CONFIG)

    generate_dataset_visualizations(model, train_loader, combined_df, MAE_CONFIG)
    print(f"\nMAE pretraining (v2, enhanced) complete.\n")
    print("NEXT: re-run embryo_linear_probe.py to check whether this encoder now beats random-init/ImageNet.")

    for fig_path in sorted(glob.glob('./mae_outputs/figures/*.png')):
        print(f"Displaying: {os.path.basename(fig_path)}")
        display(Image(filename=fig_path, width=760))
