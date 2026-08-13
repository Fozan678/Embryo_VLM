import os, gc, math, random, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import cv2, albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from tqdm.auto import tqdm
import warnings; warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
EXCLUDE_DIR_NAMES = {'.ipynb_checkpoints', '__pycache__', '.git', 'processed_data',
                     'embryo_iqa_outputs', 'mae_outputs', 'morphology_outputs', 'probe_outputs',
                     'seg_outputs', 'grader_outputs', 'grader_v2_outputs', 'grounded_morph_outputs',
                     'grounded_morph_probe_outputs', 'relational_graph_outputs',
                     'clinical_multitask_outputs', 'uncertainty_outputs', 'vlm_outputs',
                     'explainability_outputs', 'figures'}
IMNET_MEAN = [0.485, 0.456, 0.406]; IMNET_STD = [0.229, 0.224, 0.225]

INPUT_DIR  = "."
IMAGE_DIR  = "./Downloads/archive/Images/Images"
TRAIN_CSV  = "Gardner_train_silver.csv"
TEST_CSV   = "Gardner_test_gold_onlyGardnerScores.csv"

CFG = {
    "mae_checkpoint": "./embryo_project/mae/checkpoints/mae_best.pth",   # consolidated MAE location
    "targets": ["EXP", "ICM", "TE"],
    "output_dir": "./embryo_project/grader",   # consolidated project folder
    "seed": 42,

    # --- ENHANCEMENT 1: class-imbalance fixes (targets the per-class-F1-zero collapse) ---
    "focal_gamma": 2.0,                 # focal modulation on top of CORAL; 0 = plain CORAL
    "use_weighted_sampler": True,       # oversample rare EXP/ICM/TE class combinations

    # --- ENHANCEMENT 2: stratified k-fold instead of one noisy split ---
    "n_folds": 3,                       # bump to 5 once you've confirmed this runs cleanly
    "stratify_on": "ICM",               # stratify by the currently-worst target
    "epochs_per_fold": 25,

    # --- ENHANCEMENT 3: fills the still-missing MAE-init vs from-scratch measurement ---
    "run_both_inits": True,             # runs the whole k-fold sweep twice; set False for a quicker single pass

    # --- same multi-scale local/global augmentation used in Phase 1-2 ---
    "local_view_prob": 0.5, "local_crop_min_frac": 0.35, "local_crop_max_frac": 0.65,

    "batch_size": 16, "accum_iter": 1, "encoder_lr": 1e-4, "head_lr": 1e-3, "min_lr": 1e-6,
    "weight_decay": 0.05, "warmup_ratio": 0.1, "grad_clip": 1.0, "num_workers": 2,
}

MAE_CONFIG_DEFAULT = {
    "dataset": {"image_size": 512, "patch_size": 16, "in_chans": 3},
    "model": {"embed_dim": 768, "depth": 36, "num_heads": 12},   # 36 // 3 = 12 real layers
}

def locate_input_dir(preferred, name):
    if preferred and os.path.exists(os.path.join(preferred, name)):
        return preferred
    for r in [preferred, os.getcwd(), os.path.expanduser("~")]:
        if r and os.path.isdir(r):
            for dp, _, files in os.walk(r):
                if name in files:
                    return dp
    return preferred

INPUT_DIR = locate_input_dir(INPUT_DIR, TRAIN_CSV)
train_csv_path = os.path.join(INPUT_DIR, TRAIN_CSV)
test_csv_path = os.path.join(INPUT_DIR, TEST_CSV)
image_root = IMAGE_DIR if (IMAGE_DIR and os.path.isdir(IMAGE_DIR)) else INPUT_DIR

# ============================================================
# ROBUST CSV + IMAGE RESOLUTION  (same proven utilities as every prior script)
# ============================================================
def read_csv_smart(path):
    with open(path, 'r', encoding='utf-8', errors='replace') as fh:
        sample = fh.readline() + fh.readline()
    counts = {d: sample.count(d) for d in [';', ',', '\t', '|']}
    sep = max(counts, key=counts.get); sep = sep if counts[sep] else ','
    df = pd.read_csv(path, sep=sep, engine='python')
    df.columns = [str(c).strip() for c in df.columns]
    return df[[c for c in df.columns if c and not c.lower().startswith('unnamed')]]

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
                st = os.path.splitext(f)[0]; by_stem.setdefault(st, full); by_stem.setdefault(st.lower(), full)
    return by_name, by_stem

def resolve_image_path(value, by_name, by_stem):
    if value is None:
        return None
    s = str(value).strip().replace('\\', '/')
    if s == '' or s.lower() == 'nan':
        return None
    b = os.path.basename(s)
    for k in (b, b.lower()):
        if k in by_name:
            return by_name[k]
    st = os.path.splitext(b)[0]
    for k in (st, st.lower()):
        if k in by_stem:
            return by_stem[k]
    for e in IMAGE_EXTS:
        for k in (b + e, (b + e).lower()):
            if k in by_name:
                return by_name[k]
    return None

def detect_image_column(df, by_name, by_stem, sample=300):
    best, best_rate = None, -1.0
    n = min(len(df), sample)
    if n == 0:
        return None
    probe = df.head(n)
    for c in df.columns:
        hits = sum(1 for v in probe[c].tolist() if resolve_image_path(v, by_name, by_stem) is not None)
        if hits / n > best_rate:
            best, best_rate = c, hits / n
    return best

def build_labeled_df(csv_path, by_name, by_stem, tag):
    df = read_csv_smart(csv_path)
    col = detect_image_column(df, by_name, by_stem)
    df['resolved_path'] = df[col].map(lambda v: resolve_image_path(v, by_name, by_stem)) if col else None
    before = len(df)
    df = df[df['resolved_path'].notna()].reset_index(drop=True)
    print("[IMAGES] {}: column='{}' | resolved {}/{}".format(tag, col, len(df), before))
    return df

def pick_col(df, base):
    for cand in ['{}_silver'.format(base), '{}_gold'.format(base), base]:
        if cand in df.columns:
            return cand
    return None


# ============================================================
# SPARSE-CLASS MERGE  -- confirmed real counts: ICM_silver {0:1305,1:332,
# 2:16,3:391}, TE_silver {0:1081,1:525,2:47,3:391}. Class 2 has far too few
# examples (16 / 47) to learn a distinct visual concept from -- merged into
# the next-higher class per user decision. Result: ICM/TE become 3-class
# {0,1,2} instead of 4-class {0,1,2,3}. NOTE: this redefines the task --
# any checkpoint trained before this merge is incompatible and must be
# retrained, it cannot be reused via shape-inference like a metadata fix.
# ============================================================
SPARSE_MERGE = {'ICM': {2: 3}, 'TE': {2: 3}}

def remap_sparse(target, value):
    return SPARSE_MERGE.get(target, {}).get(value, value)

def build_label_maps(train_df, targets):
    maps = {}
    for t in targets:
        col = pick_col(train_df, t)
        vals = pd.to_numeric(train_df[col], errors='coerce').dropna().astype(int)
        vals = vals.map(lambda v, _t=t: remap_sparse(_t, v))
        maps[t] = {c: i for i, c in enumerate(sorted(vals.unique().tolist()))}
    return maps

def encode_labels(df, targets, maps):
    out = {}
    for t in targets:
        col = pick_col(df, t)
        v = pd.to_numeric(df[col], errors='coerce')
        v = v.map(lambda x, _t=t: remap_sparse(_t, int(x)) if pd.notna(x) else x)
        out[t] = v.map(lambda x: maps[t].get(int(x), -1) if pd.notna(x) else -1).astype(int).values
    return out

# ============================================================
# TRANSFORMS + DATASET  (multi-scale local/global, same as Phase 1-2)
# ============================================================
def post_crop_transform(size):
    return A.Compose([A.Resize(size, size), A.CLAHE(p=1.0), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
                      A.Normalize(mean=IMNET_MEAN, std=IMNET_STD), ToTensorV2()])

def eval_tf(size):
    return A.Compose([A.Resize(size, size), A.CLAHE(p=1.0),
                      A.Normalize(mean=IMNET_MEAN, std=IMNET_STD), ToTensorV2()])

class GradeDataset(Dataset):
    def __init__(self, paths, labels, size, train, aug_cfg=None):
        self.paths = list(paths); self.labels = labels; self.targets = list(labels.keys())
        self.size = size; self.train = train; self.aug_cfg = aug_cfg or {}
        self.tf = post_crop_transform(size) if train else eval_tf(size)
    def __len__(self):
        return len(self.paths)
    def _local_zoom_crop(self, image):
        h, w = image.shape[:2]
        frac = random.uniform(self.aug_cfg.get('local_crop_min_frac', 0.35), self.aug_cfg.get('local_crop_max_frac', 0.65))
        ch, cw = max(8, int(h * frac)), max(8, int(w * frac))
        y0 = random.randint(0, max(0, h - ch)); x0 = random.randint(0, max(0, w - cw))
        return image[y0:y0 + ch, x0:x0 + cw]
    def __getitem__(self, i):
        bgr = cv2.imread(self.paths[i])
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else np.zeros((self.size, self.size, 3), np.uint8)
        if self.train and random.random() < self.aug_cfg.get('local_view_prob', 0.5):
            img = self._local_zoom_crop(img)
        img = self.tf(image=img)['image']
        y = {t: torch.tensor(int(self.labels[t][i]), dtype=torch.long) for t in self.targets}
        return img, y

# ============================================================
# MODEL — identical mean-pool grader architecture as the original
# grader (kept constant on purpose: this run isolates whether TRAINING
# METHODOLOGY, not architecture, fixes ICM/TE).
# ============================================================
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
    def __init__(self, img, ps, ic, ed):
        super().__init__(); self.num_patches = (img // ps) ** 2
        self.proj = nn.Conv2d(ic, ed, ps, ps)
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

class MAEEncoderFT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        ed = cfg['model']['embed_dim']
        self.patch_embed = PatchEmbed(cfg['dataset']['image_size'], cfg['dataset']['patch_size'], cfg['dataset']['in_chans'], ed)
        P = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, ed))
        self.pos_embed = nn.Parameter(torch.zeros(1, P + 1, ed), requires_grad=False)
        layer = nn.TransformerEncoderLayer(d_model=ed, nhead=cfg['model']['num_heads'], dim_feedforward=ed * 4,
                                           dropout=0.1, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg['model']['depth'] // 3, enable_nested_tensor=False)
        self.encoder_norm = nn.LayerNorm(ed)
        pe = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(P ** 0.5), cls_token=True)
        self.pos_embed.data.copy_(pe.unsqueeze(0))
    def forward(self, x):
        x = self.patch_embed(x) + self.pos_embed[:, 1:, :]
        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = self.encoder_norm(self.encoder(x))
        return x[:, 1:, :].mean(dim=1)   # global average pool

class EmbryoGrader(nn.Module):
    def __init__(self, cfg, num_classes, mae_ckpt=None, init_from_mae=True):
        super().__init__()
        self.encoder = MAEEncoderFT(cfg)
        if init_from_mae and mae_ckpt and os.path.exists(mae_ckpt):
            sd = torch.load(mae_ckpt, map_location='cpu')['model']
            self.encoder.load_state_dict(sd, strict=False)
        ed = cfg['model']['embed_dim']
        self.heads = nn.ModuleDict({t: nn.Linear(ed, n) for t, n in num_classes.items()})
    def forward(self, x):
        f = self.encoder(x)
        return {t: h(f) for t, h in self.heads.items()}

# ============================================================
# ENHANCEMENT 1: focal-CORAL loss + class-balanced sampler
# ============================================================
def focal_coral_loss(logits, levels, gamma=2.0):
    if logits.numel() == 0 or levels.numel() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    num_classes = logits.shape[1] + 1
    lv = levels.view(-1, 1)
    val = torch.arange(num_classes - 1, device=logits.device).view(1, -1)
    targets = (lv > val).float()
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)          # prob of the TRUE outcome at each threshold
    focal_w = (1 - p_t).clamp(min=1e-6) ** gamma           # down-weight easy/confident thresholds
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    return (focal_w * bce).sum(dim=1).mean()

def build_weighted_sampler(labels_dict, targets):
    """Per-sample weight = sum of inverse class frequency across all targets ->
    rare EXP/ICM/TE combinations get oversampled during training."""
    n = len(next(iter(labels_dict.values())))
    weights = np.zeros(n, dtype=np.float64)
    for t in targets:
        y = labels_dict[t]
        valid = y >= 0
        counts = np.bincount(y[valid], minlength=max(y[valid].max() + 1, 1)) if valid.any() else np.array([1])
        counts = np.maximum(counts, 1)
        inv = 1.0 / counts
        w = np.where(valid, inv[np.clip(y, 0, len(inv) - 1)], 0.0)
        weights += w
    weights = np.where(weights <= 0, weights[weights > 0].mean() if (weights > 0).any() else 1.0, weights)
    return WeightedRandomSampler(weights=torch.as_tensor(weights, dtype=torch.double), num_samples=n, replacement=True)

def lr_at(ef, base, cfg, epochs):
    warmup = max(1, int(cfg['warmup_ratio'] * epochs))
    if ef < warmup:
        s = ef / warmup
    else:
        s = 0.5 * (1 + math.cos(math.pi * (ef - warmup) / max(1, epochs - warmup)))
    return cfg['min_lr'] + (base - cfg['min_lr']) * s

@torch.no_grad()
def evaluate(model, loader, targets, device):
    model.eval()
    P = {t: [] for t in targets}; T = {t: [] for t in targets}
    for imgs, y in loader:
        imgs = imgs.to(device)
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            logits = model(imgs)
        for t in targets:
            yt = y[t].numpy(); m = yt >= 0
            if not m.any():
                continue
            pred = (torch.sigmoid(logits[t]).cpu().numpy() > 0.5).sum(axis=1)
            P[t].extend(np.asarray(pred)[m].tolist()); T[t].extend(yt[m].tolist())
    out = {}
    for t in targets:
        yt, yp = np.array(T[t]), np.array(P[t])
        if len(yt) == 0:
            continue
        out[t] = {'acc': float((yt == yp).mean()),
                  'macro_f1': float(f1_score(yt, yp, average='macro', zero_division=0)),
                  'qwk': float(cohen_kappa_score(yt, yp, weights='quadratic')) if len(np.unique(yt)) > 1 else 0.0,
                  'n': int(len(yt))}
    return out

# ============================================================
# ONE FOLD, ONE INIT-MODE
# ============================================================
def train_one_fold(train_paths, train_labels, mae_cfg, num_classes, size, cfg, device,
                   te_loader, targets, init_from_mae, fold_idx, tag):
    tr_ds = GradeDataset(train_paths, train_labels, size, train=True, aug_cfg=cfg)
    sampler = build_weighted_sampler(train_labels, targets) if cfg['use_weighted_sampler'] else None
    tr_loader = DataLoader(tr_ds, batch_size=cfg['batch_size'],
                          shuffle=(sampler is None), sampler=sampler,
                          num_workers=cfg['num_workers'], pin_memory=True, drop_last=True)

    model = EmbryoGrader(mae_cfg, num_classes, cfg['mae_checkpoint'], init_from_mae).to(device)
    optim = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': cfg['encoder_lr']},
        {'params': model.heads.parameters(), 'lr': cfg['head_lr']},
    ], betas=(0.9, 0.95), weight_decay=cfg['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    epochs = cfg['epochs_per_fold']; steps = max(1, len(tr_loader))

    for ep in range(epochs):
        model.train()
        pbar = tqdm(enumerate(tr_loader), total=steps, desc="[{}] fold {} ep {}/{}".format(tag, fold_idx, ep + 1, epochs))
        for step, (imgs, y) in pbar:
            ef = ep + step / steps
            optim.param_groups[0]['lr'] = lr_at(ef, cfg['encoder_lr'], cfg, epochs)
            optim.param_groups[1]['lr'] = lr_at(ef, cfg['head_lr'], cfg, epochs)
            imgs = imgs.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                logits = model(imgs)
                loss = sum(focal_coral_loss(logits[t][y[t].to(device) >= 0], y[t][y[t] >= 0].to(device),
                                            gamma=cfg['focal_gamma']) for t in targets) / len(targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optim); torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            scaler.step(optim); scaler.update()
            pbar.set_postfix(loss="{:.4f}".format(float(loss)))

    test_res = evaluate(model, te_loader, targets, device)
    del model, optim
    gc.collect(); torch.cuda.empty_cache()
    return test_res

# ============================================================
# K-FOLD SWEEP (one init mode)
# ============================================================
def run_kfold_sweep(init_from_mae, train_df, maps, targets, mae_cfg, size, cfg, device, te_loader, num_classes):
    tag = "MAE-init" if init_from_mae else "from-scratch"
    y_strat = pd.to_numeric(train_df[pick_col(train_df, cfg['stratify_on'])], errors='coerce').fillna(-1).astype(int).values
    n_folds = cfg['n_folds']

    try:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cfg['seed'])
        folds = list(splitter.split(np.zeros(len(train_df)), y_strat))
    except ValueError:
        print("[KFOLD] StratifiedKFold failed (a class has < n_folds members) -> falling back to plain KFold")
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=cfg['seed'])
        folds = list(splitter.split(np.zeros(len(train_df))))

    paths = train_df['resolved_path'].tolist()
    y_all = encode_labels(train_df, targets, maps)

    fold_results = []
    for fi, (tr_idx, _) in enumerate(folds):
        tr_paths = [paths[i] for i in tr_idx]
        tr_labels = {t: y_all[t][tr_idx] for t in targets}
        res = train_one_fold(tr_paths, tr_labels, mae_cfg, num_classes, size, cfg, device,
                             te_loader, targets, init_from_mae, fi + 1, tag)
        for t in targets:
            if t in res:
                fold_results.append({'init': tag, 'fold': fi + 1, 'target': t, **res[t]})
        print("[{}] fold {}/{} test QWK: {}".format(tag, fi + 1, n_folds,
              {t: round(res[t]['qwk'], 3) for t in targets if t in res}))

    return pd.DataFrame(fold_results)

def main():
    cfg = CFG
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(cfg['output_dir'], exist_ok=True)
    targets = cfg['targets']

    by_name, by_stem = build_image_index(image_root)
    if len(by_name) == 0:
        for extra in [INPUT_DIR, os.path.expanduser("~")]:
            bn, bs = build_image_index(extra); by_name.update(bn); by_stem.update(bs)
    print("[IMAGES] indexed {} files\n".format(len(set(by_name.values()))))

    train_df = build_labeled_df(train_csv_path, by_name, by_stem, "train")
    test_df = build_labeled_df(test_csv_path, by_name, by_stem, "test")
    print("[WHOLE DATASET] train_silver rows: {} | test_gold rows: {} (all folds/evaluation use every row, no subsampling)\n".format(len(train_df), len(test_df)))

    ckpt = torch.load(cfg['mae_checkpoint'], map_location='cpu') if os.path.exists(cfg['mae_checkpoint']) else {}
    mae_cfg = ckpt.get('config', MAE_CONFIG_DEFAULT)
    size = mae_cfg['dataset']['image_size']
    real_depth = mae_cfg['model']['depth'] // 3
    print("[ENCODER] embed_dim={} | real layers={} (depth//3)\n".format(mae_cfg['model']['embed_dim'], real_depth))

    maps = build_label_maps(train_df, targets)
    num_classes = {t: len(maps[t]) for t in targets}
    print("[LABELS] classes per target:", num_classes, "\n")

    y_test = encode_labels(test_df, targets, maps)
    te_loader = DataLoader(GradeDataset(test_df['resolved_path'].tolist(), y_test, size, train=False),
                          batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'], pin_memory=True)

    print("=" * 78 + "\nRUN 1/{}: MAE-init k-fold sweep\n".format(2 if cfg['run_both_inits'] else 1) + "=" * 78)
    res_mae = run_kfold_sweep(True, train_df, maps, targets, mae_cfg, size, cfg, device, te_loader, num_classes)
    res_mae.to_csv(os.path.join(cfg['output_dir'], 'kfold_results_mae_init.csv'), index=False)

    res_scratch = None
    if cfg['run_both_inits']:
        print("\n" + "=" * 78 + "\nRUN 2/2: from-scratch k-fold sweep (fills the missing baseline)\n" + "=" * 78)
        res_scratch = run_kfold_sweep(False, train_df, maps, targets, mae_cfg, size, cfg, device, te_loader, num_classes)
        res_scratch.to_csv(os.path.join(cfg['output_dir'], 'kfold_results_from_scratch.csv'), index=False)

    # ---- summary: mean +/- std QWK per target, per init ----
    def summarize(df, tag):
        rows = []
        for t in targets:
            sub = df[df.target == t]['qwk']
            rows.append({'init': tag, 'target': t, 'mean_qwk': sub.mean(), 'std_qwk': sub.std(), 'n_folds': len(sub)})
        return pd.DataFrame(rows)

    summary = summarize(res_mae, 'MAE-init')
    if res_scratch is not None:
        summary = pd.concat([summary, summarize(res_scratch, 'from-scratch')], ignore_index=True)
    summary.to_csv(os.path.join(cfg['output_dir'], 'kfold_summary.csv'), index=False)

    print("\n" + "=" * 78 + "\nFINAL SUMMARY (mean +/- std QWK across {} folds, test_gold)\n".format(cfg['n_folds']) + "=" * 78)
    for _, r in summary.iterrows():
        print("  {:<12} {:<4} QWK = {:.3f} +/- {:.3f}".format(r['init'], r['target'], r['mean_qwk'], r['std_qwk']))
    mean_qwk_mae = summary[summary.init == 'MAE-init']['mean_qwk'].mean()
    print("\n  MAE-init MEAN QWK (all targets) = {:.3f}   <- compare to original single-split benchmark: 0.295".format(mean_qwk_mae))
    if res_scratch is not None:
        mean_qwk_scratch = summary[summary.init == 'from-scratch']['mean_qwk'].mean()
        print("  from-scratch MEAN QWK           = {:.3f}   (gap = {:+.3f}, this is the MAE pretraining benefit)".format(
            mean_qwk_scratch, mean_qwk_mae - mean_qwk_scratch))

    # bar chart with error bars
    plt.figure(figsize=(8, 5))
    inits = summary['init'].unique()
    x = np.arange(len(targets)); w = 0.8 / len(inits)
    for gi, init in enumerate(inits):
        sub = summary[summary.init == init].set_index('target')
        means = [sub.loc[t, 'mean_qwk'] if t in sub.index else 0 for t in targets]
        stds = [sub.loc[t, 'std_qwk'] if t in sub.index else 0 for t in targets]
        plt.bar(x + gi * w, means, width=w, yerr=stds, capsize=4, label=init)
    plt.xticks(x + (len(inits) - 1) * w / 2, targets)
    plt.ylabel('QWK (mean +/- std across folds, test_gold)'); plt.ylim(0, 1)
    plt.title('Enhanced grader: k-fold QWK by target'); plt.legend(); plt.grid(axis='y', alpha=0.3)
    fig_path = os.path.join(cfg['output_dir'], 'kfold_qwk_summary.png')
    plt.tight_layout(); plt.savefig(fig_path, dpi=200, bbox_inches='tight'); plt.close()
    print("\nSaved: kfold_results_*.csv, kfold_summary.csv, {}".format(os.path.basename(fig_path)))

    try:
        from IPython.display import Image, display
        display(Image(filename=fig_path, width=620))
    except Exception:
        pass

if __name__ == '__main__':
    main()
