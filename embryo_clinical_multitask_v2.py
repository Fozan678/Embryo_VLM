import os, math, random, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2, albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, roc_curve, auc, precision_recall_curve,
                             average_precision_score)
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
                     'grounded_morph_v2_outputs', 'grounded_morph_probe_outputs', 'relational_graph_outputs',
                     'graph_transformer_outputs', 'clinical_multitask_outputs', 'clinical_multitask_v2_outputs',
                     'uncertainty_outputs', 'vlm_outputs', 'explainability_outputs', 'figures', 'embryo_project'}   # NEW: consolidated project folder
IMNET_MEAN = [0.485, 0.456, 0.406]; IMNET_STD = [0.229, 0.224, 0.225]

INPUT_DIR  = "."
IMAGE_DIR  = "./Downloads/archive/Images/Images"
TRAIN_CSV  = "Gardner_train_silver.csv"
TEST_CSV   = "Gardner_test_gold_onlyGardnerScores.csv"

# Set False to run only the full (non-ablated) model once, instead of both --
# requirement 13 (ablation studies) compares WITH vs WITHOUT the clinical
# consistency/correlation constraints, with the FULL visualization suite
# generated for each variant so they can be compared side by side.
RUN_ABLATION = True

CLINICAL_CONFIG = {
    "mae_checkpoint": "./embryo_project/mae/checkpoints/mae_best.pth",   # consolidated MAE location
    "batch_size": 8, "accum_iter": 2, "epochs": 15, "lr": 2e-4, "min_lr": 1e-6,
    "weight_decay": 0.05, "warmup_ratio": 0.1, "grad_clip": 1.0, "val_ratio": 0.2,
    "num_workers": 2, "output_dir": "./embryo_project/clinical_multitask", "seed": 42,   # consolidated
    "lambda_consistency": 0.5, "lambda_correlation": 0.3,
    "quality_exp_thresh": 3, "quality_icm_te_best_k": 2,
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

def derive_quality(df, cfg, icm_good_vals, te_good_vals):
    """Heuristic composite: EXP >= thresh AND ICM/TE in the two best-observed
    classes. NOT a clinically annotated ground-truth label -- verify the
    'best-observed = clinically best' assumption before reporting as an outcome."""
    ecol, icol, tcol = pick_col(df, 'EXP'), pick_col(df, 'ICM'), pick_col(df, 'TE')
    if not (ecol and icol and tcol):
        df['QUALITY'] = np.nan; return df
    exp = pd.to_numeric(df[ecol], errors='coerce')
    icm = pd.to_numeric(df[icol], errors='coerce')
    te = pd.to_numeric(df[tcol], errors='coerce')
    good = (exp >= cfg['quality_exp_thresh']) & icm.isin(icm_good_vals) & te.isin(te_good_vals)
    df['QUALITY'] = np.where(exp.isna() | icm.isna() | te.isna(), np.nan, good.astype(int))
    return df

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
        if col is None:
            maps[t] = {0: 0, 1: 1}; continue
        nums = pd.to_numeric(train_df[col], errors='coerce').dropna().astype(int)
        nums = nums.map(lambda v, _t=t: remap_sparse(_t, v))
        maps[t] = {c: i for i, c in enumerate(sorted(nums.unique().tolist()))}
    return maps

def encode_labels(df, targets, maps):
    out = {}
    for t in targets:
        col = pick_col(df, t)
        if col is None:
            out[t] = np.full(len(df), -1, dtype=int); continue
        v = pd.to_numeric(df[col], errors='coerce')
        v = v.map(lambda x, _t=t: remap_sparse(_t, int(x)) if pd.notna(x) else x)
        out[t] = v.map(lambda x: maps[t].get(int(x), -1) if pd.notna(x) else -1).astype(int).values
    return out

# ============================================================
# TRANSFORMS + DATASET
# ============================================================
def train_tf(size):
    up = int(round(size * 1.15))
    return A.Compose([A.Resize(up, up), A.RandomCrop(height=size, width=size),
                      A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.CLAHE(p=1.0),
                      A.Normalize(mean=IMNET_MEAN, std=IMNET_STD), ToTensorV2()])

def eval_tf(size):
    return A.Compose([A.Resize(size, size), A.CLAHE(p=1.0),
                      A.Normalize(mean=IMNET_MEAN, std=IMNET_STD), ToTensorV2()])

class MultiTaskEmbryoDataset(Dataset):
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
# ARCHITECTURE: SHARED TRANSFORMER + CLINICAL REASONING GRAPH  (unchanged
# from the already-fixed v1 -- this request adds visualizations, not a
# different network)
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

class SharedMorphologyTransformer(nn.Module):
    def __init__(self, cfg, num_morph_tokens=6):
        super().__init__()
        ed = cfg['model']['embed_dim']
        self.patch_embed = PatchEmbed(cfg['dataset']['image_size'], cfg['dataset']['patch_size'],
                                      cfg['dataset']['in_chans'], ed)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, ed))
        self.morph_tokens = nn.Parameter(torch.zeros(1, num_morph_tokens, ed))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, ed), requires_grad=False)
        layer = nn.TransformerEncoderLayer(d_model=ed, nhead=cfg['model']['num_heads'], dim_feedforward=ed * 4,
                                           dropout=0.1, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(2, cfg['model']['depth'] // 3), enable_nested_tensor=False)
        self.encoder_norm = nn.LayerNorm(ed)
        pe = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(num_patches ** 0.5), cls_token=True)
        self.pos_embed.data.copy_(pe.unsqueeze(0))
        nn.init.trunc_normal_(self.morph_tokens, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        patches = self.patch_embed(x) + self.pos_embed[:, 1:, :]
        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(B, -1, -1)
        morph = self.morph_tokens.expand(B, -1, -1)
        tokens = torch.cat((cls, morph, patches), dim=1)
        return self.encoder_norm(self.encoder(tokens))

class ClinicalReasoningGraphNetwork(nn.Module):
    def __init__(self, cfg, num_classes_dict):
        super().__init__()
        self.shared_transformer = SharedMorphologyTransformer(cfg)
        ed = cfg['model']['embed_dim']
        self.num_classes_dict = num_classes_dict
        self.ordinal_heads = nn.ModuleDict()
        for t, num_classes in num_classes_dict.items():
            self.ordinal_heads[t] = nn.Linear(ed, max(1, num_classes - 1))
        num_tasks = len(num_classes_dict)
        self.task_attention = nn.MultiheadAttention(embed_dim=ed, num_heads=4, batch_first=True)
        self.fusion_mlp = nn.Sequential(nn.Linear(ed * 2, ed), nn.GELU(), nn.Dropout(0.1), nn.Linear(ed, ed))
        self.joint_quality_head = nn.Linear(ed * num_tasks, 2)
        self.log_vars = nn.Parameter(torch.zeros(len(num_classes_dict) + 1))

    def forward(self, x):
        tokens = self.shared_transformer(x)
        cls_feat = tokens[:, 0, :]
        num_morph = self.shared_transformer.morph_tokens.shape[1]
        morph_feats = tokens[:, 1:1 + num_morph, :]
        task_queries = morph_feats.mean(dim=1, keepdim=True).repeat(1, len(self.num_classes_dict), 1)
        attn_out, _ = self.task_attention(task_queries, morph_feats, morph_feats)
        task_preds, feats_list = {}, []
        for i, t in enumerate(self.num_classes_dict.keys()):
            fused = self.fusion_mlp(torch.cat([cls_feat, attn_out[:, i, :]], dim=1))
            feats_list.append(fused)
            task_preds[t] = self.ordinal_heads[t](fused)
        task_preds['QUALITY'] = self.joint_quality_head(torch.cat(feats_list, dim=1))
        return task_preds, tokens

# ============================================================
# LOSSES (unchanged)
# ============================================================
def coral_loss(logits, levels):
    if logits.numel() == 0 or levels.numel() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    num_classes = logits.shape[1] + 1
    levels = levels.view(-1, 1)
    val = torch.arange(num_classes - 1, device=logits.device).view(1, -1)
    targets = (levels > val).float()
    return F.binary_cross_entropy_with_logits(logits, targets, reduction='none').sum(dim=1).mean()

def consistency_loss(task_preds):
    loss = 0.0
    if 'EXP' in task_preds and 'ICM' in task_preds:
        exp_prob = torch.sigmoid(task_preds['EXP'][:, 0]); icm_prob = torch.sigmoid(task_preds['ICM'][:, 0])
        loss = loss + torch.relu(icm_prob - exp_prob).mean()
    return loss

def correlation_loss(task_preds):
    loss = 0.0
    if 'QUALITY' in task_preds and 'EXP' in task_preds:
        q_prob = torch.softmax(task_preds['QUALITY'], dim=1)[:, 1]
        exp_score = torch.sigmoid(task_preds['EXP'][:, 0])
        loss = loss + F.mse_loss(q_prob, exp_score)
    return loss

# ============================================================
# CORAL -> proper categorical distribution.
# K-1 independent sigmoid thresholds do NOT sum to 1 -- ROC/PR/calibration
# all require real per-class probabilities, so this conversion is required
# before any of requirements 9/10/11, not optional.
# ============================================================
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

# ============================================================
# ONE PASS -> real probabilities for every visualization
# (avoids 4 separate, redundant forward passes over the test set)
# ============================================================
@torch.no_grad()
def collect_test_probabilities(model, loader, targets, device):
    model.eval()
    raw = {t: {'logits': [], 'true': []} for t in targets}
    for imgs, y in loader:
        imgs = imgs.to(device)
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            task_preds, _ = model(imgs)
        for t in targets:
            raw[t]['logits'].append(task_preds[t].float().cpu().numpy())
            raw[t]['true'].append(y[t].numpy())
    out = {}
    for t in targets:
        logits = np.concatenate(raw[t]['logits'], 0)
        true = np.concatenate(raw[t]['true'], 0)
        m = true >= 0
        if t == 'QUALITY':
            probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)  # already 2-class logits -> softmax
        else:
            probs = coral_probs_from_sigmoid(1 / (1 + np.exp(-logits)))          # CORAL sigmoids -> proper simplex
        out[t] = {'probs': probs[m], 'true': true[m]}
    return out

# ============================================================
# 8. CONFUSION MATRICES  (argmax-decoded from the real categorical probs
# -- same decoding convention as ROC/PR/calibration below, for consistency)
# ============================================================
def plot_confusion_matrices(prob_dict, out_path, tag):
    targets = list(prob_dict.keys())
    fig, axes = plt.subplots(1, len(targets), figsize=(4.2 * len(targets), 4))
    if len(targets) == 1:
        axes = [axes]
    for ax, t in zip(axes, targets):
        y_true = prob_dict[t]['true']; y_pred = prob_dict[t]['probs'].argmax(axis=1)
        cm = confusion_matrix(y_true, y_pred)
        im = ax.imshow(cm, cmap=plt.cm.Blues)
        ax.set_title('{}: {}'.format(tag, t)); ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                       color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=8)
    plt.tight_layout(); plt.savefig(out_path, dpi=200, bbox_inches='tight'); plt.close()

# ============================================================
# 9. ROC CURVES  (one-vs-rest per class per target)
# ============================================================
def plot_roc_curves(prob_dict, out_path, tag):
    targets = list(prob_dict.keys())
    fig, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets), 4.5))
    if len(targets) == 1:
        axes = [axes]
    for ax, t in zip(axes, targets):
        y_true = prob_dict[t]['true']; probs = prob_dict[t]['probs']
        K = probs.shape[1]
        for c in range(K):
            y_bin = (y_true == c).astype(int)
            if len(np.unique(y_bin)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_bin, probs[:, c])
            ax.plot(fpr, tpr, lw=1.8, label='class {} (AUC={:.2f})'.format(c, auc(fpr, tpr)))
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_title('{}: {}'.format(tag, t)); ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
        ax.legend(fontsize=7, loc='lower right')
    plt.tight_layout(); plt.savefig(out_path, dpi=200, bbox_inches='tight'); plt.close()

# ============================================================
# 10. PRECISION-RECALL CURVES  (one-vs-rest per class per target)
# ============================================================
def plot_pr_curves(prob_dict, out_path, tag):
    targets = list(prob_dict.keys())
    fig, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets), 4.5))
    if len(targets) == 1:
        axes = [axes]
    for ax, t in zip(axes, targets):
        y_true = prob_dict[t]['true']; probs = prob_dict[t]['probs']
        K = probs.shape[1]
        for c in range(K):
            y_bin = (y_true == c).astype(int)
            if len(np.unique(y_bin)) < 2:
                continue
            prec, rec, _ = precision_recall_curve(y_bin, probs[:, c])
            ap = average_precision_score(y_bin, probs[:, c])
            ax.plot(rec, prec, lw=1.8, label='class {} (AP={:.2f})'.format(c, ap))
        ax.set_title('{}: {}'.format(tag, t)); ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
        ax.legend(fontsize=7, loc='lower left')
    plt.tight_layout(); plt.savefig(out_path, dpi=200, bbox_inches='tight'); plt.close()

# ============================================================
# 11. CALIBRATION / RELIABILITY CURVES
# ============================================================
def plot_calibration_curves(prob_dict, out_path, tag):
    targets = list(prob_dict.keys())
    fig, axes = plt.subplots(1, len(targets), figsize=(4.2 * len(targets), 4.2))
    if len(targets) == 1:
        axes = [axes]
    for ax, t in zip(axes, targets):
        y_true = prob_dict[t]['true']; probs = prob_dict[t]['probs']
        pred = probs.argmax(axis=1); conf = probs.max(axis=1)
        acc = (pred == y_true)
        edges = np.linspace(0, 1, 11); centers = (edges[:-1] + edges[1:]) / 2
        bin_acc, bin_conf = [], []
        for i in range(10):
            msk = (conf >= edges[i]) & (conf < edges[i + 1])
            bin_acc.append(acc[msk].mean() if msk.sum() else np.nan)
            bin_conf.append(conf[msk].mean() if msk.sum() else centers[i])
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect')
        ax.plot(bin_conf, bin_acc, marker='o', color='b', label='Model')
        ax.set_title('{}: {}'.format(tag, t)); ax.set_xlabel('Confidence'); ax.set_ylabel('Accuracy')
        ax.legend(fontsize=8, loc='lower right'); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=200, bbox_inches='tight'); plt.close()

# ============================================================
# 12. CLASS ACTIVATION MAP  -- real retain_grad()-based Grad-CAM, restricted
# to patch tokens (skip cls+morph), no fabricated values.
# ============================================================
def plot_cam(model, loader, device, out_path, tag):
    model.eval()
    for imgs, y in loader:
        imgs = imgs.to(device)
        task_preds, tokens = model(imgs)
        tokens.retain_grad()
        score = task_preds['QUALITY'][:, 1].sum()
        model.zero_grad(set_to_none=True)
        score.backward()

        num_morph = model.shared_transformer.morph_tokens.shape[1]
        patch_start = 1 + num_morph
        grad_p = tokens.grad[:, patch_start:, :]
        act_p = tokens[:, patch_start:, :].detach()
        weights = grad_p.mean(dim=1, keepdim=True)
        cam = torch.relu((weights * act_p).sum(dim=-1)).cpu().numpy()
        grid = int(round(math.sqrt(cam.shape[1])))

        plt.figure(figsize=(6, 6))
        plt.imshow(cam[0].reshape(grid, grid), cmap='jet')
        plt.title('Grad-CAM ({}) -- QUALITY head'.format(tag)); plt.axis('off')
        plt.savefig(out_path, dpi=200, bbox_inches='tight'); plt.close()
        break

# ============================================================
# TRAIN
# ============================================================
def train_pipeline(ablation_mode=False):
    cfg = CLINICAL_CONFIG
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(cfg['output_dir'], exist_ok=True)
    targets = ["EXP", "ICM", "TE", "QUALITY"]
    ordinal_targets = ["EXP", "ICM", "TE"]

    by_name, by_stem = build_image_index(image_root)
    if len(by_name) == 0:
        for extra in [INPUT_DIR, os.path.expanduser("~")]:
            bn, bs = build_image_index(extra); by_name.update(bn); by_stem.update(bs)
    print("[IMAGES] indexed {} files\n".format(len(set(by_name.values()))))

    train_df = build_labeled_df(train_csv_path, by_name, by_stem, "train")
    test_df = build_labeled_df(test_csv_path, by_name, by_stem, "test")
    print("[WHOLE DATASET] train_silver rows: {} | test_gold rows: {} (no subsampling)\n".format(
        len(train_df), len(test_df)))

    icol, tcol = pick_col(train_df, 'ICM'), pick_col(train_df, 'TE')
    icm_vals = sorted(pd.to_numeric(train_df[icol], errors='coerce').dropna().unique().tolist()) if icol else []
    te_vals = sorted(pd.to_numeric(train_df[tcol], errors='coerce').dropna().unique().tolist()) if tcol else []
    k = cfg['quality_icm_te_best_k']
    icm_good, te_good = set(icm_vals[:k]), set(te_vals[:k])
    print("[QUALITY] heuristic label (NOT clinical ground truth): EXP>={} AND ICM in {} AND TE in {}\n"
          .format(cfg['quality_exp_thresh'], sorted(icm_good), sorted(te_good)))
    train_df = derive_quality(train_df, cfg, icm_good, te_good)
    test_df = derive_quality(test_df, cfg, icm_good, te_good)

    maps = build_label_maps(train_df, targets)
    num_classes_dict = {t: max(2, len(maps.get(t, {}))) for t in targets}
    print("[CONFIG] classes per target:", num_classes_dict)

    y_train_all = encode_labels(train_df, targets, maps)
    y_test = encode_labels(test_df, targets, maps)

    idx = np.arange(len(train_df)); np.random.shuffle(idx)
    n_val = int(cfg['val_ratio'] * len(idx))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    paths = train_df['resolved_path'].tolist()
    tr_paths = [paths[i] for i in tr_idx]; va_paths = [paths[i] for i in val_idx]
    tr_labels = {t: y_train_all[t][tr_idx] for t in targets}
    va_labels = {t: y_train_all[t][val_idx] for t in targets}

    size = 512
    tr_loader = DataLoader(MultiTaskEmbryoDataset(tr_paths, tr_labels, size, True), batch_size=cfg['batch_size'],
                           shuffle=True, num_workers=cfg['num_workers'], pin_memory=True, drop_last=True)
    va_loader = DataLoader(MultiTaskEmbryoDataset(va_paths, va_labels, size, False), batch_size=cfg['batch_size'],
                           shuffle=False, num_workers=cfg['num_workers'], pin_memory=True)
    te_loader = DataLoader(MultiTaskEmbryoDataset(test_df['resolved_path'].tolist(), y_test, size, False),
                           batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'], pin_memory=True)

    model = ClinicalReasoningGraphNetwork(MAE_CONFIG_DEFAULT, num_classes_dict).to(device)
    if os.path.exists(cfg['mae_checkpoint']):
        ckpt = torch.load(cfg['mae_checkpoint'], map_location='cpu')
        sd = ckpt.get('model', ckpt)
        enc_sd = {k[len('encoder.'):]: v for k, v in sd.items() if k.startswith('encoder.')}
        missing, unexpected = model.shared_transformer.load_state_dict(enc_sd, strict=False)
        print("[INIT] MAE backbone loaded (missing {}, unexpected {})".format(len(missing), len(unexpected)))
    else:
        print("[INIT] MAE checkpoint not found at {} -> shared transformer randomly initialized".format(cfg['mae_checkpoint']))

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    steps = max(1, len(tr_loader)); accum = cfg['accum_iter']
    print("\n[TRAIN] Clinical Multi-Task Optimization (ablation={})\n".format(ablation_mode))

    best_qwk_proxy, best_state = -1.0, None
    for epoch in range(cfg['epochs']):
        model.train(); running = 0.0
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(enumerate(tr_loader), total=steps, desc="Epoch {}/{}".format(epoch + 1, cfg['epochs']))
        for step, (imgs, y) in pbar:
            imgs = imgs.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                task_preds, _ = model(imgs)
                total_loss = 0.0
                for t in targets:
                    yt = y[t].to(device); vm = yt >= 0
                    if not vm.any():
                        continue
                    logits = task_preds[t][vm]; tl = yt[vm]
                    loss_t = F.cross_entropy(logits, tl) if t == 'QUALITY' else coral_loss(logits, tl)
                    prec = torch.exp(-model.log_vars[targets.index(t)])
                    total_loss = total_loss + prec * loss_t + model.log_vars[targets.index(t)]
                if not ablation_mode:
                    total_loss = total_loss + cfg['lambda_consistency'] * consistency_loss(task_preds) \
                                            + cfg['lambda_correlation'] * correlation_loss(task_preds)
                loss_s = total_loss / accum
            scaler.scale(loss_s).backward()
            if (step + 1) % accum == 0 or (step + 1) == steps:
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            running += float(total_loss); pbar.set_postfix(loss="{:.4f}".format(float(total_loss)))

        val_probs = collect_test_probabilities(model, va_loader, ordinal_targets, device)
        from sklearn.metrics import cohen_kappa_score
        mean_qwk = float(np.mean([cohen_kappa_score(val_probs[t]['true'], val_probs[t]['probs'].argmax(1), weights='quadratic')
                                  if len(np.unique(val_probs[t]['true'])) > 1 else 0.0 for t in ordinal_targets]))
        print("[Epoch {:02d}] loss {:.4f} | val mean QWK (EXP/ICM/TE) {:.3f}".format(epoch + 1, running / steps, mean_qwk))
        if mean_qwk > best_qwk_proxy:
            best_qwk_proxy = mean_qwk
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    tag = "full" if not ablation_mode else "ablation_no_constraints"
    prob_dict = collect_test_probabilities(model, te_loader, targets, device)

    print("\n" + "=" * 70 + "\nTEST -- Clinical Multi-Task ({}) -- best val mean-QWK {:.3f}\n".format(tag, best_qwk_proxy) + "=" * 70)
    rows = []
    from sklearn.metrics import f1_score, cohen_kappa_score
    for t in targets:
        y_true, probs = prob_dict[t]['true'], prob_dict[t]['probs']
        y_pred = probs.argmax(axis=1)
        acc = float((y_true == y_pred).mean())
        f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        qwk = float(cohen_kappa_score(y_true, y_pred, weights='quadratic')) if (t != 'QUALITY' and len(np.unique(y_true)) > 1) else np.nan
        print("  {:<8} | acc {:.3f} | macroF1 {:.3f}{} (n={})".format(
            t, acc, f1, " | QWK {:.3f}".format(qwk) if not np.isnan(qwk) else "", len(y_true)))
        rows.append({'target': t, 'tag': tag, 'acc': acc, 'macro_f1': f1, 'qwk': qwk, 'n': len(y_true)})
    pd.DataFrame(rows).to_csv(os.path.join(cfg['output_dir'], 'clinical_test_{}.csv'.format(tag)), index=False)

    # 8-12: full visualization suite, generated once per ablation variant
    plot_confusion_matrices(prob_dict, os.path.join(cfg['output_dir'], 'confusion_matrices_{}.png'.format(tag)), tag)
    plot_roc_curves(prob_dict, os.path.join(cfg['output_dir'], 'roc_curves_{}.png'.format(tag)), tag)
    plot_pr_curves(prob_dict, os.path.join(cfg['output_dir'], 'pr_curves_{}.png'.format(tag)), tag)
    plot_calibration_curves(prob_dict, os.path.join(cfg['output_dir'], 'calibration_curves_{}.png'.format(tag)), tag)
    plot_cam(model, te_loader, device, os.path.join(cfg['output_dir'], 'cam_{}.png'.format(tag)), tag)
    print("\n[Artifacts] confusion/ROC/PR/calibration/CAM saved with suffix '_{}':".format(tag))
    print("  Note: QUALITY curves describe fit to the DERIVED heuristic label above, not a clinical outcome.")

    return prob_dict

if __name__ == '__main__':
    print("=== Full Clinically Constrained Multi-Task Model ===")
    full_probs = train_pipeline(ablation_mode=False)
    if RUN_ABLATION:
        print("\n=== Ablation (without clinical consistency/correlation constraints) ===")
        ablation_probs = train_pipeline(ablation_mode=True)
    print("\nDone. Compare *_full.png vs *_ablation_no_constraints.png to see requirement 13 (ablation) directly.")
