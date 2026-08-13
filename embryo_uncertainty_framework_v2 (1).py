import os, math, random, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2, albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
from tqdm.auto import tqdm
import warnings; warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True
IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
EXCLUDE_DIR_NAMES = {'.ipynb_checkpoints', '__pycache__', '.git', 'processed_data',
                     'embryo_iqa_outputs', 'mae_outputs', 'morphology_outputs', 'probe_outputs',
                     'seg_outputs', 'grader_outputs', 'grader_v2_outputs', 'grounded_morph_outputs',
                     'grounded_morph_v2_outputs', 'grounded_morph_probe_outputs', 'relational_graph_outputs',
                     'graph_transformer_outputs', 'clinical_multitask_outputs', 'clinical_multitask_v2_outputs',
                     'uncertainty_outputs', 'uncertainty_framework_v2_outputs', 'vlm_outputs',
                     'explainability_outputs', 'figures', 'embryo_project'}   # NEW: consolidated project folder
IMNET_MEAN = [0.485, 0.456, 0.406]; IMNET_STD = [0.229, 0.224, 0.225]

INPUT_DIR = "."; IMAGE_DIR = "./Downloads/archive/Images/Images"
TRAIN_CSV = "Gardner_train_silver.csv"; TEST_CSV = "Gardner_test_gold_onlyGardnerScores.csv"

CFG = {
    "mae_checkpoint": "./embryo_project/mae/checkpoints/mae_best.pth",   # consolidated MAE location
    "targets": ["EXP", "ICM", "TE"],
    "output_dir": "./embryo_project/uncertainty_framework",   # consolidated
    "seed": 42,
    "batch_size": 8, "epochs": 15, "lr": 1e-4, "min_lr": 1e-6, "weight_decay": 0.05,
    "warmup_ratio": 0.1, "grad_clip": 1.0, "val_ratio": 0.2, "num_workers": 2,
    "dropout_rate": 0.15,
    "mc_samples": 20,           # MC-Dropout stochastic forward passes
    "ensemble_size": 3,         # Deep Ensemble: number of independently-trained models (ACTUALLY used now)
    "coverage_levels": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],  # selective-prediction / reject-option sweep
}
MAE_CONFIG_DEFAULT = {"dataset": {"image_size": 512, "patch_size": 16, "in_chans": 3},
                      "model": {"embed_dim": 768, "depth": 36, "num_heads": 12}}  # 36//3 = 12 real layers

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
# ROBUST CSV + IMAGE RESOLUTION
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

def pick_col(df, base):
    for cand in ['{}_silver'.format(base), '{}_gold'.format(base), base]:
        if cand in df.columns:
            return cand
    return None

def build_labeled_df(csv_path, by_name, by_stem, tag):
    df = read_csv_smart(csv_path)
    col = detect_image_column(df, by_name, by_stem)
    df['resolved_path'] = df[col].map(lambda v: resolve_image_path(v, by_name, by_stem)) if col else None
    before = len(df)
    df = df[df['resolved_path'].notna()].reset_index(drop=True)
    print("[IMAGES] {}: column='{}' | resolved {}/{}".format(tag, col, len(df), before))
    return df


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
# DATASET  (plain, consistent preprocessing -- deliberately no random
# multi-scale augmentation here: uncertainty estimates should reflect
# MODEL uncertainty, not augmentation-induced randomness)
# ============================================================
def train_tf(size):
    return A.Compose([A.Resize(size, size), A.CLAHE(p=1.0), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
                      A.Normalize(mean=IMNET_MEAN, std=IMNET_STD), ToTensorV2()])

def eval_tf(size):
    return A.Compose([A.Resize(size, size), A.CLAHE(p=1.0),
                      A.Normalize(mean=IMNET_MEAN, std=IMNET_STD), ToTensorV2()])

class GradeDataset(Dataset):
    def __init__(self, paths, labels, size, train):
        self.paths = list(paths); self.labels = labels; self.targets = list(labels.keys())
        self.tf = train_tf(size) if train else eval_tf(size); self.size = size
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        bgr = cv2.imread(self.paths[i])
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else np.zeros((self.size, self.size, 3), np.uint8)
        img = self.tf(image=img)['image']
        y = {t: torch.tensor(int(self.labels[t][i]), dtype=torch.long) for t in self.targets}
        return img, y

# ============================================================
# MODEL: shared transformer + (CORAL | Evidential) heads, dropout throughout
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
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

class SharedTransformer(nn.Module):
    def __init__(self, cfg, dropout_rate=0.1):
        super().__init__()
        ed = cfg['model']['embed_dim']
        self.patch_embed = PatchEmbed(cfg['dataset']['image_size'], cfg['dataset']['patch_size'], cfg['dataset']['in_chans'], ed)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, ed))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, ed), requires_grad=False)
        layer = nn.TransformerEncoderLayer(d_model=ed, nhead=cfg['model']['num_heads'], dim_feedforward=ed * 4,
                                          dropout=dropout_rate, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(2, cfg['model']['depth'] // 3), enable_nested_tensor=False)
        self.encoder_norm = nn.LayerNorm(ed)
        pe = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(num_patches ** 0.5), cls_token=True)
        self.pos_embed.data.copy_(pe.unsqueeze(0))
    def forward(self, x):
        B = x.shape[0]
        patches = self.patch_embed(x) + self.pos_embed[:, 1:, :]
        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(B, -1, -1)
        return self.encoder_norm(self.encoder(torch.cat((cls, patches), dim=1)))

class GradeNet(nn.Module):
    """One flexible model class: evidential=False -> CORAL heads (used for
    MC-Dropout and every Deep-Ensemble member); evidential=True -> Dirichlet
    evidence heads (used for the EDL model)."""
    def __init__(self, cfg, num_classes_dict, dropout_rate=0.1, evidential=False):
        super().__init__()
        self.shared = SharedTransformer(cfg, dropout_rate=dropout_rate)
        ed = cfg['model']['embed_dim']
        self.evidential = evidential
        self.heads = nn.ModuleDict()
        for t, n in num_classes_dict.items():
            out_dim = n if evidential else max(1, n - 1)
            self.heads[t] = nn.Sequential(nn.Linear(ed, ed // 2), nn.GELU(),
                                          nn.Dropout(dropout_rate), nn.Linear(ed // 2, out_dim))
    def forward(self, x):
        tokens = self.shared(x)
        cls = tokens[:, 0, :]
        out = {}
        for t, head in self.heads.items():
            o = head(cls)
            out[t] = F.softplus(o) + 1.0 if self.evidential else o
        return out

def enable_mc_dropout(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

# ============================================================
# LOSSES
# ============================================================
def coral_loss(logits, levels):
    if logits.numel() == 0 or levels.numel() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    num_classes = logits.shape[1] + 1
    lv = levels.view(-1, 1)
    val = torch.arange(num_classes - 1, device=logits.device).view(1, -1)
    targets = (lv > val).float()
    return F.binary_cross_entropy_with_logits(logits, targets, reduction='none').sum(dim=1).mean()

def evidential_loss(alpha, y, num_classes, epoch_num, c=10):
    S = torch.sum(alpha, dim=1, keepdim=True)
    y1h = F.one_hot(y, num_classes=num_classes).float()
    A = torch.sum(y1h * (torch.log(S) - torch.log(alpha)), dim=1, keepdim=True)
    def KL(alpha):
        beta = torch.ones_like(alpha)
        Sa = torch.sum(alpha, dim=1, keepdim=True); Sb = torch.sum(beta, dim=1, keepdim=True)
        lnB = torch.lgamma(Sa) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        lnB_beta = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(Sb)
        dg0 = torch.digamma(Sa); dg = torch.digamma(alpha)
        return torch.sum((alpha - beta) * (dg - dg0), dim=1, keepdim=True) + lnB + lnB_beta
    ann = min(1.0, epoch_num / c)
    return torch.mean(A + ann * KL(alpha))

def coral_probs_from_sigmoid(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    N, Km1 = p.shape; K = Km1 + 1
    p_mono = np.minimum.accumulate(p, axis=1)   # FIX: was reversed+accumulate+reversed (wrong direction) -> collapsed everything to the min value
    probs = np.zeros((N, K), dtype=np.float64)
    probs[:, 0] = 1 - p_mono[:, 0]
    for k in range(1, K - 1):
        probs[:, k] = p_mono[:, k - 1] - p_mono[:, k]
    probs[:, K - 1] = p_mono[:, K - 2]
    probs = np.clip(probs, 1e-8, None)
    return probs / probs.sum(axis=1, keepdims=True)

def lr_at(ef, cfg):
    warmup = max(1, int(cfg['warmup_ratio'] * cfg['epochs']))
    if ef < warmup:
        s = ef / warmup
    else:
        s = 0.5 * (1 + math.cos(math.pi * (ef - warmup) / max(1, cfg['epochs'] - warmup)))
    return cfg['min_lr'] + (cfg['lr'] - cfg['min_lr']) * s

@torch.no_grad()
def val_qwk(model, loader, targets, device, evidential):
    model.eval()
    P = {t: [] for t in targets}; T = {t: [] for t in targets}
    for imgs, y in loader:
        imgs = imgs.to(device)
        out = model(imgs)
        for t in targets:
            yt = y[t].numpy(); m = yt >= 0
            if not m.any():
                continue
            if evidential:
                pred = (out[t] / out[t].sum(dim=1, keepdim=True)).argmax(dim=1).cpu().numpy()
            else:
                pred = (torch.sigmoid(out[t]).cpu().numpy() > 0.5).sum(axis=1)
            P[t].extend(np.asarray(pred)[m].tolist()); T[t].extend(yt[m].tolist())
    qwks = [cohen_kappa_score(T[t], P[t], weights='quadratic') for t in targets if len(T[t]) and len(np.unique(T[t])) > 1]
    return float(np.mean(qwks)) if qwks else 0.0

# ============================================================
# GENERIC TRAINER  (used identically for MC-Dropout model, each Deep-
# Ensemble member, and the EDL model -- only seed/evidential/dropout differ)
# ============================================================
def train_single_model(seed, evidential, dropout_rate, mae_cfg, num_classes_dict, mae_ckpt,
                       tr_loader, va_loader, targets, device, cfg, tag):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = GradeNet(mae_cfg, num_classes_dict, dropout_rate=dropout_rate, evidential=evidential).to(device)
    if os.path.exists(mae_ckpt):
        sd = torch.load(mae_ckpt, map_location='cpu').get('model', {})
        enc_sd = {k[len('encoder.'):]: v for k, v in sd.items() if k.startswith('encoder.')}
        model.shared.load_state_dict(enc_sd, strict=False)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    steps = max(1, len(tr_loader))
    best_qwk, best_state = -1.0, None

    for ep in range(cfg['epochs']):
        model.train()
        pbar = tqdm(enumerate(tr_loader), total=steps, desc="[{}] ep {}/{}".format(tag, ep + 1, cfg['epochs']))
        for step, (imgs, y) in pbar:
            for pg in opt.param_groups:
                pg['lr'] = lr_at(ep + step / steps, cfg)
            imgs = imgs.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                out = model(imgs)
                loss = 0.0
                for t in targets:
                    yt = y[t].to(device); vm = yt >= 0
                    if not vm.any():
                        continue
                    if evidential:
                        loss = loss + evidential_loss(out[t][vm], yt[vm], num_classes_dict[t], epoch_num=ep, c=10)
                    else:
                        loss = loss + coral_loss(out[t][vm], yt[vm])
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            scaler.step(opt); scaler.update()
            pbar.set_postfix(loss="{:.4f}".format(float(loss)))
        vq = val_qwk(model, va_loader, targets, device, evidential)
        if vq > best_qwk:
            best_qwk = vq
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    print("[{}] best val mean-QWK = {:.3f}".format(tag, best_qwk))
    return model

# ============================================================
# POST-HOC TEMPERATURE SCALING (MC-Dropout & Ensemble only -- see note
# in the printed output re: why EDL is excluded)
# ============================================================
def fit_temperature(model_or_models, loader, targets, device, iters=50, is_ensemble=False):
    logit_store = {t: [] for t in targets}; true_store = {t: [] for t in targets}
    with torch.no_grad():
        for imgs, y in loader:
            imgs = imgs.to(device)
            if is_ensemble:
                outs = [m(imgs) for m in model_or_models]
                out = {t: torch.stack([o[t] for o in outs], 0).mean(0) for t in targets}
            else:
                model_or_models.eval()
                out = model_or_models(imgs)
            for t in targets:
                yt = y[t].numpy(); m = yt >= 0
                if not m.any():
                    continue
                logit_store[t].append(out[t][torch.from_numpy(m)].cpu())
                true_store[t].append(torch.from_numpy(yt[m]))
    temps = {}
    for t in targets:
        if not logit_store[t]:
            temps[t] = 1.0; continue
        logits = torch.cat(logit_store[t], 0).to(device)
        yt = torch.cat(true_store[t], 0).long().to(device)
        T = torch.nn.Parameter(torch.ones(1, device=device) * 1.5)
        opt = torch.optim.LBFGS([T], lr=0.05, max_iter=iters)
        def closure():
            opt.zero_grad()
            loss = coral_loss(logits / T.clamp(min=0.05), yt)
            loss.backward(); return loss
        opt.step(closure)
        temps[t] = float(T.clamp(min=0.05).item())
    return temps

# ============================================================
# PREDICTION EXTRACTION  -- unified so MC-Dropout and Deep Ensemble share
# one code path (both are "combine multiple predictive samples"; they only
# differ in WHERE the samples come from -- dropout masks vs different
# trained weights). EDL is single-pass with analytic uncertainty.
# ============================================================
@torch.no_grad()
def multi_sample_predict(forward_fns, loader, targets, device, temps=None):
    """forward_fns: list of callables imgs -> {target: logits}. For MC-Dropout
    this is the SAME model called N times (dropout active). For Deep Ensemble
    this is M different trained models called once each."""
    temps = temps or {}
    results = {t: {'probs_samples': [], 'true': []} for t in targets}
    for imgs, y in loader:
        imgs = imgs.to(device)
        per_target_samples = {t: [] for t in targets}
        for fn in forward_fns:
            out = fn(imgs)
            for t in targets:
                logit = out[t] / temps.get(t, 1.0)
                p = coral_probs_from_sigmoid(torch.sigmoid(logit).cpu().numpy())
                per_target_samples[t].append(p)
        for t in targets:
            stack = np.stack(per_target_samples[t], axis=0)   # (n_samples, batch, K)
            results[t]['probs_samples'].append(stack)
            results[t]['true'].append(y[t].numpy())
    out = {}
    for t in targets:
        samples = np.concatenate(results[t]['probs_samples'], axis=1)   # (n_samples, N_total, K)
        true = np.concatenate(results[t]['true'], axis=0)
        mean_p = samples.mean(axis=0)                                    # (N_total, K)
        pred = mean_p.argmax(axis=1)
        confidence = mean_p.max(axis=1)
        entropy = -np.sum(mean_p * np.log(mean_p + 1e-12), axis=-1)
        pred_class_probs = np.take_along_axis(samples, pred[None, :, None], axis=2)[..., 0]  # (n_samples, N_total)
        variance = pred_class_probs.var(axis=0)
        out[t] = {'true': true, 'pred': pred, 'mean_probs': mean_p,
                  'confidence': confidence, 'entropy': entropy, 'variance': variance}
    return out

@torch.no_grad()
def edl_predict(model, loader, targets, device):
    """Single forward pass. Uncertainty comes analytically from the Dirichlet
    parameters -- no sampling needed, the key practical advantage of EDL."""
    model.eval()
    results = {t: {'true': [], 'pred': [], 'mean_probs': [], 'confidence': [], 'entropy': [], 'variance': []} for t in targets}
    for imgs, y in loader:
        imgs = imgs.to(device)
        out = model(imgs)
        for t in targets:
            alpha = out[t].cpu().numpy()
            S = alpha.sum(axis=1, keepdims=True)
            probs = alpha / S
            pred = probs.argmax(axis=1)
            conf = probs.max(axis=1)
            ent = -np.sum(probs * np.log(probs + 1e-12), axis=-1)
            alpha_pred = np.take_along_axis(alpha, pred[:, None], axis=1)[:, 0]
            S_flat = S[:, 0]
            var = alpha_pred * (S_flat - alpha_pred) / (S_flat ** 2 * (S_flat + 1))   # analytic Dirichlet variance
            yt = y[t].numpy()
            results[t]['true'].append(yt); results[t]['pred'].append(pred)
            results[t]['mean_probs'].append(probs); results[t]['confidence'].append(conf)
            results[t]['entropy'].append(ent); results[t]['variance'].append(var)
    for t in targets:
        for k in results[t]:
            results[t][k] = np.concatenate(results[t][k], axis=0)
    return results

# ============================================================
# 2-3. CALIBRATION METRICS
# ============================================================
def expected_calibration_error(confidence, correct, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0; n = len(confidence)
    for i in range(n_bins):
        m = (confidence >= edges[i]) & (confidence < edges[i + 1])
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(correct[m].mean() - confidence[m].mean())
    return float(ece)

# ============================================================
# 4-5-6. CONFIDENCE THRESHOLDING / SELECTIVE PREDICTION / REJECT OPTION
# ============================================================
def risk_coverage_curve(confidence, correct, levels):
    order = np.argsort(-confidence)   # most confident first
    correct_sorted = correct[order]
    n = len(correct)
    coverages, accuracies, thresholds = [], [], []
    for cov in levels:
        k = max(1, int(round(cov * n)))
        acc = correct_sorted[:k].mean()
        coverages.append(k / n); accuracies.append(float(acc))
        thresholds.append(float(np.sort(confidence)[::-1][k - 1]))
    return np.array(coverages), np.array(accuracies), np.array(thresholds)

def area_under_risk_coverage(coverages, accuracies):
    order = np.argsort(coverages)
    return float(np.trapezoid(1 - accuracies[order], coverages[order]))   # area under the ERROR-coverage curve (lower=better)

# ============================================================
# 7. PUBLICATION-QUALITY PLOTS
# ============================================================
def plot_reliability(method_results, targets, out_path):
    fig, axes = plt.subplots(1, len(targets), figsize=(4.5 * len(targets), 4.2))
    if len(targets) == 1:
        axes = [axes]
    for ax, t in zip(axes, targets):
        for name, res in method_results.items():
            r = res[t]; conf = r['confidence']; correct = (r['pred'] == r['true']).astype(float)
            edges = np.linspace(0, 1, 11)
            bc, bconf = [], []
            for i in range(10):
                m = (conf >= edges[i]) & (conf < edges[i + 1])
                if m.sum():
                    bc.append(correct[m].mean()); bconf.append(conf[m].mean())
            ax.plot(bconf, bc, marker='o', label=name, alpha=0.85)
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_title(t); ax.set_xlabel('Confidence'); ax.set_ylabel('Accuracy'); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=220, bbox_inches='tight'); plt.close()

def plot_conf_var_entropy(method_results, targets, out_path):
    fig, axes = plt.subplots(3, len(targets), figsize=(4.3 * len(targets), 10))
    if len(targets) == 1:
        axes = axes.reshape(3, 1)
    for j, t in enumerate(targets):
        for name, res in method_results.items():
            r = res[t]
            axes[0, j].hist(r['confidence'], bins=25, alpha=0.5, label=name, density=True)
            axes[1, j].hist(r['variance'], bins=25, alpha=0.5, label=name, density=True)
            axes[2, j].hist(r['entropy'], bins=25, alpha=0.5, label=name, density=True)
        axes[0, j].set_title('{}: Confidence'.format(t)); axes[1, j].set_title('{}: Variance'.format(t))
        axes[2, j].set_title('{}: Entropy'.format(t))
        for r_ax in (axes[0, j], axes[1, j], axes[2, j]):
            r_ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(out_path, dpi=220, bbox_inches='tight'); plt.close()

def plot_risk_coverage(method_results, targets, levels, out_path):
    fig, axes = plt.subplots(1, len(targets), figsize=(4.5 * len(targets), 4.2))
    if len(targets) == 1:
        axes = [axes]
    for ax, t in zip(axes, targets):
        for name, res in method_results.items():
            r = res[t]; correct = (r['pred'] == r['true']).astype(float)
            cov, acc, _ = risk_coverage_curve(r['confidence'], correct, levels)
            aurc = area_under_risk_coverage(cov, acc)
            ax.plot(cov, acc, marker='o', label='{} (AURC={:.3f})'.format(name, aurc))
        ax.set_xlabel('Coverage (fraction predicted, rest rejected)'); ax.set_ylabel('Accuracy on accepted')
        ax.set_title(t); ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.invert_xaxis()
    plt.tight_layout(); plt.savefig(out_path, dpi=220, bbox_inches='tight'); plt.close()

# ============================================================
# MAIN
# ============================================================
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
    print("[WHOLE DATASET] train_silver rows: {} | test_gold rows: {} (no subsampling)\n".format(
        len(train_df), len(test_df)))
    maps = build_label_maps(train_df, targets)
    num_classes_dict = {t: len(maps[t]) for t in targets}
    print("[LABELS] classes per target:", num_classes_dict, "\n")

    ckpt = torch.load(cfg['mae_checkpoint'], map_location='cpu') if os.path.exists(cfg['mae_checkpoint']) else {}
    mae_cfg = ckpt.get('config', MAE_CONFIG_DEFAULT)
    size = mae_cfg['dataset']['image_size']

    y_train_all = encode_labels(train_df, targets, maps)
    y_test = encode_labels(test_df, targets, maps)
    idx = np.arange(len(train_df)); np.random.shuffle(idx)
    n_val = int(cfg['val_ratio'] * len(idx))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    paths = train_df['resolved_path'].tolist()
    tr_paths = [paths[i] for i in tr_idx]; va_paths = [paths[i] for i in val_idx]
    tr_labels = {t: y_train_all[t][tr_idx] for t in targets}
    va_labels = {t: y_train_all[t][val_idx] for t in targets}

    tr_loader = DataLoader(GradeDataset(tr_paths, tr_labels, size, True), batch_size=cfg['batch_size'],
                          shuffle=True, num_workers=cfg['num_workers'], pin_memory=True, drop_last=True)
    va_loader = DataLoader(GradeDataset(va_paths, va_labels, size, False), batch_size=cfg['batch_size'],
                          shuffle=False, num_workers=cfg['num_workers'], pin_memory=True)
    te_loader = DataLoader(GradeDataset(test_df['resolved_path'].tolist(), y_test, size, False),
                          batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'], pin_memory=True)

    method_results = {}

    # ---------- 1. MC-DROPOUT ----------
    print("=" * 70 + "\nMETHOD 1/3: Monte Carlo Dropout\n" + "=" * 70)
    mc_model = train_single_model(cfg['seed'], False, cfg['dropout_rate'], mae_cfg, num_classes_dict,
                                  cfg['mae_checkpoint'], tr_loader, va_loader, targets, device, cfg, "MC-Dropout")
    mc_temps = fit_temperature(mc_model, va_loader, targets, device)
    print("[MC-Dropout] fitted temperatures:", {k: round(v, 3) for k, v in mc_temps.items()})
    enable_mc_dropout(mc_model)
    forward_fns = [(lambda imgs: mc_model(imgs)) for _ in range(cfg['mc_samples'])]
    method_results['MC-Dropout'] = multi_sample_predict(forward_fns, te_loader, targets, device, temps=mc_temps)
    del mc_model; torch.cuda.empty_cache()

    # ---------- 2. DEEP ENSEMBLE (actually implemented now) ----------
    print("\n" + "=" * 70 + "\nMETHOD 2/3: Deep Ensemble ({} independently-trained models)\n".format(cfg['ensemble_size']) + "=" * 70)
    ensemble_models = []
    for m in range(cfg['ensemble_size']):
        model_m = train_single_model(cfg['seed'] + 1000 + m, False, 0.0, mae_cfg, num_classes_dict,
                                     cfg['mae_checkpoint'], tr_loader, va_loader, targets, device, cfg,
                                     "Ensemble-{}".format(m + 1))
        model_m.eval()
        ensemble_models.append(model_m)
    ens_temps = fit_temperature(ensemble_models, va_loader, targets, device, is_ensemble=True)
    print("[Deep Ensemble] fitted temperatures:", {k: round(v, 3) for k, v in ens_temps.items()})
    forward_fns = [m for m in ensemble_models]   # each member called ONCE = one "sample"
    method_results['Deep Ensemble'] = multi_sample_predict(forward_fns, te_loader, targets, device, temps=ens_temps)
    for m in ensemble_models:
        del m
    torch.cuda.empty_cache()

    # ---------- 3. EVIDENTIAL DEEP LEARNING ----------
    print("\n" + "=" * 70 + "\nMETHOD 3/3: Evidential Deep Learning\n" + "=" * 70)
    edl_model = train_single_model(cfg['seed'] + 2000, True, cfg['dropout_rate'], mae_cfg, num_classes_dict,
                                   cfg['mae_checkpoint'], tr_loader, va_loader, targets, device, cfg, "EDL")
    print("[EDL] NOTE: temperature scaling is a logit-rescaling technique; EDL's Dirichlet")
    print("      evidence isn't a logit, so standard temperature scaling does not directly")
    print("      apply here -- EDL's calibration instead comes from its KL-annealed training loss.")
    method_results['Evidential (EDL)'] = edl_predict(edl_model, te_loader, targets, device)
    del edl_model; torch.cuda.empty_cache()

    # ---------- SUMMARY: accuracy, QWK, ECE, AURC per method per target ----------
    print("\n" + "=" * 70 + "\nSUMMARY -- accuracy / QWK / ECE / AURC per method\n" + "=" * 70)
    rows = []
    for name, res in method_results.items():
        for t in targets:
            r = res[t]
            correct = (r['pred'] == r['true']).astype(float)
            acc = float(correct.mean())
            qwk = float(cohen_kappa_score(r['true'], r['pred'], weights='quadratic')) if len(np.unique(r['true'])) > 1 else 0.0
            ece = expected_calibration_error(r['confidence'], correct)
            cov, accs, _ = risk_coverage_curve(r['confidence'], correct, cfg['coverage_levels'])
            aurc = area_under_risk_coverage(cov, accs)
            rows.append({'method': name, 'target': t, 'acc': acc, 'qwk': qwk, 'ece': ece, 'aurc': aurc,
                        'mean_confidence': float(r['confidence'].mean()), 'mean_entropy': float(r['entropy'].mean()),
                        'mean_variance': float(r['variance'].mean())})
            print("  {:<16} {:<4} acc={:.3f} QWK={:.3f} ECE={:.3f} AURC={:.3f} (lower ECE/AURC = better)".format(
                name, t, acc, qwk, ece, aurc))
    pd.DataFrame(rows).to_csv(os.path.join(cfg['output_dir'], 'uncertainty_summary.csv'), index=False)

    # ---------- REJECT-OPTION TABLE ----------
    reject_rows = []
    for name, res in method_results.items():
        for t in targets:
            r = res[t]; correct = (r['pred'] == r['true']).astype(float)
            cov, accs, thr = risk_coverage_curve(r['confidence'], correct, cfg['coverage_levels'])
            for c, a, th in zip(cov, accs, thr):
                reject_rows.append({'method': name, 'target': t, 'coverage': c, 'reject_rate': 1 - c,
                                    'accuracy_on_accepted': a, 'confidence_threshold': th})
    pd.DataFrame(reject_rows).to_csv(os.path.join(cfg['output_dir'], 'reject_option_table.csv'), index=False)

    # ---------- 7. PLOTS ----------
    plot_reliability(method_results, targets, os.path.join(cfg['output_dir'], 'reliability_diagrams.png'))
    plot_conf_var_entropy(method_results, targets, os.path.join(cfg['output_dir'], 'confidence_variance_entropy.png'))
    plot_risk_coverage(method_results, targets, cfg['coverage_levels'],
                      os.path.join(cfg['output_dir'], 'risk_coverage_curves.png'))

    print("\nSaved: uncertainty_summary.csv, reject_option_table.csv, reliability_diagrams.png,")
    print("       confidence_variance_entropy.png, risk_coverage_curves.png -> {}".format(cfg['output_dir']))

    try:
        from IPython.display import Image, display
        for f in ['reliability_diagrams.png', 'confidence_variance_entropy.png', 'risk_coverage_curves.png']:
            display(Image(filename=os.path.join(cfg['output_dir'], f), width=760))
    except Exception:
        pass

if __name__ == '__main__':
    main()
