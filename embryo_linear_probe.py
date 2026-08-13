import os
import math
import random
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from collections import Counter
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
EXCLUDE_DIR_NAMES = {'.ipynb_checkpoints', '__pycache__', '.git',
                     'processed_data', 'embryo_iqa_outputs',
                     'mae_outputs', 'morphology_outputs', 'probe_outputs', 'figures',
                     'embryo_project'}   # NEW: consolidated project folder

IMNET_MEAN = [0.485, 0.456, 0.406]
IMNET_STD = [0.229, 0.224, 0.225]

# ============================================================
# PATHS + PROBE CONFIG  (LOCAL JUPYTER)
# ============================================================
INPUT_DIR  = "."
IMAGE_DIR  = "./Downloads/archive/Images/Images"
TRAIN_CSV  = "Gardner_train_silver.csv"
TEST_CSV   = "Gardner_test_gold_onlyGardnerScores.csv"

PROBE = {
    "mae_checkpoint": "./embryo_project/mae/checkpoints/mae_best.pth",   # consolidated MAE location
    "size_mae": 512,          # overridden by the size stored in the checkpoint config if present
    "size_imagenet": 224,
    "batch_size": 16,
    "num_workers": 2,
    "targets": ["EXP", "ICM", "TE"],
    "include_imagenet": True,
    "output_dir": "./probe_outputs",
    "random_seed": 42,
}

# Fallback MAE architecture config (used only if the checkpoint lacks 'config')
MAE_CONFIG_DEFAULT = {
    "dataset": {"image_size": 512, "patch_size": 16, "in_chans": 3},
    "model": {"embed_dim": 1024, "depth": 24, "num_heads": 16,
              "decoder_embed_dim": 512, "decoder_depth": 8, "decoder_num_heads": 16,
              "mask_ratio": 0.75, "norm_pix_loss": False},
}

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
image_root = IMAGE_DIR if (IMAGE_DIR and os.path.isdir(IMAGE_DIR)) else INPUT_DIR

# ==========================================
# ROBUST CSV + IMAGE RESOLUTION
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
    for dp, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIR_NAMES and not d.endswith('_outputs')]
        for f in files:
            if f.lower().endswith(IMAGE_EXTS):
                full = os.path.join(dp, f)
                by_name.setdefault(f, full); by_name.setdefault(f.lower(), full)
                stem = os.path.splitext(f)[0]
                by_stem.setdefault(stem, full); by_stem.setdefault(stem.lower(), full)
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
        return None
    probe = df.head(n)
    for c in df.columns:
        hits = sum(1 for v in probe[c].tolist() if resolve_image_path(v, by_name, by_stem) is not None)
        rate = hits / n
        if rate > best_rate:
            best_col, best_rate = c, rate
    return best_col

def build_labeled_df(csv_path, by_name, by_stem, tag):
    df = read_csv_smart(csv_path)
    col = detect_image_column(df, by_name, by_stem)
    df['resolved_path'] = df[col].map(lambda v: resolve_image_path(v, by_name, by_stem)) if col else None
    before = len(df)
    df = df[df['resolved_path'].notna()].reset_index(drop=True)
    print("[IMAGES] {}: image column='{}' | resolved {}/{}".format(tag, col, len(df), before))
    return df

def pick_col(df, base):
    for cand in ['{}_silver'.format(base), '{}_gold'.format(base), base]:
        if cand in df.columns:
            return cand
    return None

# ==========================================
# FEATURE DATASET + TRANSFORM
# ==========================================
def eval_transform(size):
    return A.Compose([
        A.Resize(size, size),
        A.CLAHE(p=1.0),
        A.Normalize(mean=IMNET_MEAN, std=IMNET_STD),
        ToTensorV2()
    ])

class FeatDataset(Dataset):
    def __init__(self, paths, size):
        self.paths = list(paths)
        self.tf = eval_transform(size)
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        bgr = cv2.imread(self.paths[idx])
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else np.zeros((self.size, self.size, 3), np.uint8)
        return self.tf(image=img)['image']

# ==========================================
# MAE ENCODER (architecture required to load the checkpoint)
# ==========================================
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    gh = np.arange(grid_size, dtype=np.float32); gw = np.arange(grid_size, dtype=np.float32)
    grid = np.stack(np.meshgrid(gw, gh), axis=0).reshape([2, 1, grid_size, grid_size])
    def _1d(d, pos):
        omega = np.arange(d // 2, dtype=np.float32); omega /= d / 2.0; omega = 1.0 / 10000 ** omega
        out = np.outer(pos.reshape(-1), omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)
    emb = np.concatenate([_1d(embed_dim // 2, grid[0]), _1d(embed_dim // 2, grid[1])], axis=1)
    if cls_token:
        emb = np.concatenate([np.zeros([1, embed_dim]), emb], axis=0)
    return torch.from_numpy(emb).float()

class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

class MaskedAutoencoderViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        ed = config['model']['embed_dim']
        self.patch_embed = PatchEmbed(config['dataset']['image_size'], config['dataset']['patch_size'],
                                      config['dataset']['in_chans'], ed)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, ed))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, ed), requires_grad=False)
        enc_layer = nn.TransformerEncoderLayer(d_model=ed, nhead=config['model']['num_heads'],
                                               dim_feedforward=ed * 4, dropout=0.1, activation='gelu',
                                               batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=config['model']['depth'] // 3, enable_nested_tensor=False)
        self.encoder_norm = nn.LayerNorm(ed)
        # decoder pieces exist only so load_state_dict finds matching keys
        dd = config['model']['decoder_embed_dim']
        self.decoder_embed = nn.Linear(ed, dd, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dd))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dd), requires_grad=False)
        dec_layer = nn.TransformerEncoderLayer(d_model=dd, nhead=config['model']['decoder_num_heads'],
                                               dim_feedforward=dd * 4, dropout=0.1, activation='gelu',
                                               batch_first=True, norm_first=True)
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=config['model']['decoder_depth'] // 2, enable_nested_tensor=False)
        self.decoder_norm = nn.LayerNorm(dd)
        self.decoder_pred = nn.Linear(dd, config['dataset']['patch_size'] ** 2 * config['dataset']['in_chans'], bias=True)
        pe = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(num_patches ** 0.5), cls_token=True)
        self.pos_embed.data.copy_(pe.unsqueeze(0))

    @torch.no_grad()
    def encode_cls(self, imgs):
        x = self.patch_embed(imgs)
        x = x + self.pos_embed[:, 1:, :]
        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = self.encoder_norm(self.encoder(x))
        return x[:, 0, :]

def cls_extract(model, imgs):
    return model.encode_cls(imgs)

def imnet_extract(model, imgs):
    return model(imgs)

# ==========================================
# FEATURE EXTRACTION
# ==========================================
@torch.no_grad()
def extract_multi(models, df, size, config, device):
    ds = FeatDataset(df['resolved_path'].tolist(), size)
    loader = DataLoader(ds, batch_size=config['batch_size'], shuffle=False,
                        num_workers=config['num_workers'], pin_memory=True)
    out = {k: [] for k in models}
    for imgs in tqdm(loader, desc="features @ {}px".format(size)):
        imgs = imgs.to(device)
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            for name, (m, fn) in models.items():
                out[name].append(fn(m, imgs).float().cpu().numpy())
    return {k: np.concatenate(v, 0) for k, v in out.items()}

# ==========================================
# PROBE
# ==========================================
def run_probe(feats_train, feats_test, train_df, test_df, targets, encoder_names):
    rows = []
    for base in targets:
        ctr, cte = pick_col(train_df, base), pick_col(test_df, base)
        if ctr is None or cte is None:
            print("[PROBE] target {} skipped (no label column)".format(base))
            continue
        ytr = pd.to_numeric(train_df[ctr], errors='coerce')
        yte = pd.to_numeric(test_df[cte], errors='coerce')
        mtr, mte = ytr.notna().values, yte.notna().values
        ytr_v, yte_v = ytr[mtr].astype(int).values, yte[mte].astype(int).values
        if len(np.unique(ytr_v)) < 2 or len(yte_v) == 0:
            print("[PROBE] target {} skipped (degenerate labels)".format(base))
            continue

        maj = Counter(ytr_v).most_common(1)[0][0]
        rows.append({'target': base, 'encoder': 'Majority baseline',
                     'test_acc': float((yte_v == maj).mean()), 'macro_f1': np.nan,
                     'bal_acc': np.nan, 'n_test': int(len(yte_v))})

        for name in encoder_names:
            Xtr, Xte = feats_train[name][mtr], feats_test[name][mte]
            scaler = StandardScaler().fit(Xtr)
            clf = LogisticRegression(max_iter=3000, C=1.0)
            clf.fit(scaler.transform(Xtr), ytr_v)
            pred = clf.predict(scaler.transform(Xte))
            rows.append({'target': base, 'encoder': name,
                         'test_acc': float(accuracy_score(yte_v, pred)),
                         'macro_f1': float(f1_score(yte_v, pred, average='macro', zero_division=0)),
                         'bal_acc': float(balanced_accuracy_score(yte_v, pred)),
                         'n_test': int(len(yte_v))})
    return pd.DataFrame(rows)

def plot_results(res, encoder_names, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    targets = [t for t in res['target'].unique()]
    groups = ['Majority baseline'] + encoder_names
    x = np.arange(len(targets)); w = 0.8 / max(1, len(groups))
    plt.figure(figsize=(1.8 * len(targets) + 4, 5))
    for gi, g in enumerate(groups):
        vals = [res[(res.target == t) & (res.encoder == g)]['test_acc'].values for t in targets]
        vals = [v[0] if len(v) else 0.0 for v in vals]
        plt.bar(x + gi * w, vals, width=w, label=g)
    plt.xticks(x + (len(groups) - 1) * w / 2, targets)
    plt.ylabel('Test accuracy (held-out gold set)')
    plt.title('Linear-probe: frozen encoders vs Gardner grades')
    plt.ylim(0, 1.0); plt.legend(fontsize=8); plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    p = os.path.join(out_dir, 'linear_probe_accuracy.png')
    plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
    return p

# ==========================================
# EXECUTION
# ==========================================
if __name__ == '__main__':
    random.seed(PROBE['random_seed']); np.random.seed(PROBE['random_seed']); torch.manual_seed(PROBE['random_seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(PROBE['output_dir'], exist_ok=True)

    print("[PATHS] train CSV : {} (exists {})".format(train_csv_path, os.path.exists(train_csv_path)))
    print("[PATHS] test  CSV : {} (exists {})".format(test_csv_path, os.path.exists(test_csv_path)))
    print("[PATHS] image root: {}\n".format(image_root))
    if not os.path.exists(PROBE['mae_checkpoint']):
        raise FileNotFoundError("MAE checkpoint not found at {}. Run the MAE pretraining first.".format(PROBE['mae_checkpoint']))

    by_name, by_stem = build_image_index(image_root)
    if len(by_name) == 0:
        for extra in [INPUT_DIR, os.path.expanduser("~")]:
            bn, bs = build_image_index(extra); by_name.update(bn); by_stem.update(bs)
    print("[IMAGES] indexed {} files\n".format(len(set(by_name.values()))))

    train_df = build_labeled_df(train_csv_path, by_name, by_stem, "train")
    test_df = build_labeled_df(test_csv_path, by_name, by_stem, "test")
    print()

    # --- Load encoders (all frozen) ---
    ckpt = torch.load(PROBE['mae_checkpoint'], map_location='cpu')
    mae_cfg = ckpt.get('config', MAE_CONFIG_DEFAULT)
    size_mae = mae_cfg['dataset']['image_size']

    mae_model = MaskedAutoencoderViT(mae_cfg)
    missing = mae_model.load_state_dict(ckpt['model'], strict=False)
    mae_model.to(device).eval()
    print("[ENCODER] MAE loaded from {} (epoch {}, loss {:.4f})".format(
        os.path.basename(PROBE['mae_checkpoint']), ckpt.get('epoch', '?'), ckpt.get('loss', float('nan'))))

    torch.manual_seed(PROBE['random_seed'])
    rand_model = MaskedAutoencoderViT(mae_cfg).to(device).eval()
    print("[ENCODER] Random-init ViT built (same architecture, untrained control)")

    encoder_names = ['MAE (frozen)', 'Random init (frozen)']
    models_512 = {'MAE (frozen)': (mae_model, cls_extract),
                  'Random init (frozen)': (rand_model, cls_extract)}

    imnet_model = None
    if PROBE['include_imagenet']:
        try:
            from torchvision.models import vit_b_16, ViT_B_16_Weights
            imnet_model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            imnet_model.heads = nn.Identity()
            imnet_model.to(device).eval()
            print("[ENCODER] ImageNet ViT-B/16 loaded (external baseline)")
        except Exception as e:
            print("[ENCODER] ImageNet baseline skipped ({}).".format(e))
            imnet_model = None
    print()

    # --- Extract features ---
    ftr = extract_multi(models_512, train_df, size_mae, PROBE, device)
    fte = extract_multi(models_512, test_df, size_mae, PROBE, device)
    if imnet_model is not None:
        m224 = {'ImageNet ViT-B/16 (frozen)': (imnet_model, imnet_extract)}
        ftr.update(extract_multi(m224, train_df, PROBE['size_imagenet'], PROBE, device))
        fte.update(extract_multi(m224, test_df, PROBE['size_imagenet'], PROBE, device))
        encoder_names.append('ImageNet ViT-B/16 (frozen)')
    print("\n[FEATURES] dims: " + " | ".join("{}={}".format(k, v.shape[1]) for k, v in ftr.items()) + "\n")

    # --- Probe + report ---
    res = run_probe(ftr, fte, train_df, test_df, PROBE['targets'], encoder_names)
    res_csv = os.path.join(PROBE['output_dir'], 'linear_probe_results.csv')
    res.to_csv(res_csv, index=False)
    fig = plot_results(res, encoder_names, PROBE['output_dir'])

    print("=" * 74)
    print("LINEAR-PROBE RESULTS (train on train_silver, evaluate on test_gold)")
    print("=" * 74)
    for base in res['target'].unique():
        sub = res[res.target == base]
        print("\nTarget: {}   (n_test = {})".format(base, int(sub['n_test'].iloc[0])))
        print("  {:<30} {:>8} {:>9} {:>9}".format("encoder", "acc", "macroF1", "balAcc"))
        for _, r in sub.iterrows():
            print("  {:<30} {:>8.3f} {:>9} {:>9}".format(
                r['encoder'], r['test_acc'],
                "{:.3f}".format(r['macro_f1']) if pd.notna(r['macro_f1']) else "   -  ",
                "{:.3f}".format(r['bal_acc']) if pd.notna(r['bal_acc']) else "   -  "))

    # --- Verdict ---
    print("\n" + "=" * 74)
    print("VERDICT (does MAE pretraining help?)")
    print("=" * 74)
    for base in res['target'].unique():
        sub = res[res.target == base].set_index('encoder')['test_acc']
        mae_a = sub.get('MAE (frozen)', np.nan)
        rnd_a = sub.get('Random init (frozen)', np.nan)
        line = "  {}: MAE {:.3f}".format(base, mae_a)
        if pd.notna(rnd_a):
            line += " vs random {:.3f} ({:+.3f})".format(rnd_a, mae_a - rnd_a)
        if 'ImageNet ViT-B/16 (frozen)' in sub.index:
            im_a = sub['ImageNet ViT-B/16 (frozen)']
            line += " vs ImageNet {:.3f} ({:+.3f})".format(im_a, mae_a - im_a)
        print(line)
    print("\nIf MAE > random across targets, pretraining learned useful structure.")
    print("If MAE also >= ImageNet, the domain pretraining is worth reporting.")
    print("\nSaved: {}  and  {}".format(os.path.basename(res_csv), os.path.basename(fig)))

    try:
        from IPython.display import Image, display
        display(Image(filename=fig, width=720))
    except Exception:
        pass
