"""
Complete evaluation framework -- publication-ready metrics and figures.

DESIGN NOTE (read first): none of the prior phase scripts persisted raw
per-sample predictions to disk (only aggregate fold/epoch metrics). Bootstrap
CIs, significance tests, ROC/PR, calibration, and decision-curve analysis all
require real per-sample (y_true, y_prob) pairs -- so this script loads your
real trained checkpoint(s), runs them ONCE on the real held-out test set,
SAVES the raw predictions (fixing that gap for future reuse), and computes
everything else from that single authoritative real dataset.

Figures that live inside OTHER phases (attention/GradCAM, retrieval examples,
report grounding, ablation plots, training curves, architecture diagrams) are
NOT re-derived here by re-importing ten other scripts' model classes. This
script checks whether each phase's real output already exists on disk and
copies it into the organized manuscript folder with a provenance note --
or prints an explicit "MISSING -- run X.py first" if it doesn't. Nothing here
is fabricated to fill a gap.
"""
import os, shutil, itertools, math, random, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2, albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, average_precision_score,
                             f1_score, precision_score, recall_score, cohen_kappa_score,
                             mean_absolute_error, confusion_matrix)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm.auto import tqdm
import warnings; warnings.filterwarnings("ignore")

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
EXCLUDE_DIR_NAMES = {'.ipynb_checkpoints', '__pycache__', '.git', 'processed_data', 'figures',
                     'evaluation_framework_outputs', 'embryo_project'}   # NEW: consolidated project folder
IMNET_MEAN = [0.485, 0.456, 0.406]; IMNET_STD = [0.229, 0.224, 0.225]
TARGETS = ["EXP", "ICM", "TE"]

INPUT_DIR = "."; IMAGE_DIR = "./Downloads/archive/Images/Images"
TRAIN_CSV = "Gardner_train_silver.csv"; TEST_CSV = "Gardner_test_gold_onlyGardnerScores.csv"

CFG = {
    "grader_checkpoint": "./embryo_project/grounded_morph_grader/grounded_morph_v2_fold1_mae_init.pth",
    "output_dir": "./embryo_project/evaluation",   # consolidated project folder
    "seed": 42, "batch_size": 16, "num_workers": 2,
    "n_bootstrap": 2000,
    # Real prior-phase artifacts to gather into the manuscript gallery IF they exist.
    # CORRECTED to the actual consolidated embryo_project/ tree (the previous
    # manifest pointed at pre-consolidation folder names and would have
    # reported every single entry as MISSING despite the real files existing).
    "gallery_manifest": {
        "Architecture diagram (Qwen-VLM)": None,   # rendered via Visualizer tool in-chat, not a file
        "MAE training/loss curves": "./embryo_project/mae/figures/pretrain_loss_curve.png",
        "MAE reconstruction (training curve proxy)": "./embryo_project/mae/figures/reconstruction_final.png",
        "Morphology loss curves": "./embryo_project/morphology/figures/training_loss_curve.png",
        "GradCAM / GradCAM++ / IG / Rollout panel": "./embryo_project/explainability/method_panel_EXP_1.png",
        "Token attribution comparison": "./embryo_project/explainability/token_attribution_EXP_1.png",
        "Sentence grounding examples": "./embryo_project/explainability/sentence_grounding.png",
        "XAI faithfulness (deletion/insertion)": "./embryo_project/explainability/faithfulness_curves.png",
        "XAI cross-method agreement": "./embryo_project/explainability/method_agreement_heatmap.png",
        "Retrieval examples": "./embryo_project/retrieval/retrieval_panel_1.png",
        "Report grounding examples": "./embryo_project/vlm_grounded/report_panel_1.png",
        "Ablation study plots (full vs constrained)": "./embryo_project/clinical_multitask/roc_curves_full.png",
        "Graph node/edge attention": "./embryo_project/graph_transformer/edge_attention_real.png",
        "Learned graph / edge importance": "./embryo_project/graph_transformer/learned_graph_edge_bias.png",
    },
}
MORPH_DEFAULT = {"dataset": {"image_size": 512, "patch_size": 16, "in_chans": 3},
                 "model": {"embed_dim": 768, "depth": 36, "num_heads": 12}}

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
# NATURE-STYLE PLOTTING INFRASTRUCTURE
# ============================================================
def set_nature_style():
    mpl.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8, 'axes.linewidth': 0.6, 'axes.labelsize': 8, 'axes.titlesize': 9,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
        'axes.spines.top': False, 'axes.spines.right': False,
        'savefig.dpi': 600, 'figure.dpi': 150, 'lines.linewidth': 1.0,
        'pdf.fonttype': 42, 'ps.fonttype': 42,   # embed fonts as text, not curves -- required by most journals
    })

def save_all_formats(fig, name, output_dir):
    sub = os.path.join(output_dir, 'figures'); os.makedirs(sub, exist_ok=True)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(os.path.join(sub, '{}.{}'.format(name, ext)), dpi=600, bbox_inches='tight')
    plt.close(fig)

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
    if not root or not os.path.exists(root): return by_name, by_stem
    for dp, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIR_NAMES and not d.endswith('_outputs')]
        for f in files:
            if f.lower().endswith(IMAGE_EXTS):
                full = os.path.join(dp, f)
                by_name.setdefault(f, full); by_name.setdefault(f.lower(), full)
                st = os.path.splitext(f)[0]; by_stem.setdefault(st, full); by_stem.setdefault(st.lower(), full)
    return by_name, by_stem

def resolve_image_path(value, by_name, by_stem):
    if value is None: return None
    s = str(value).strip().replace('\\', '/')
    if s == '' or s.lower() == 'nan': return None
    b = os.path.basename(s)
    for k in (b, b.lower()):
        if k in by_name: return by_name[k]
    st = os.path.splitext(b)[0]
    for k in (st, st.lower()):
        if k in by_stem: return by_stem[k]
    for e in IMAGE_EXTS:
        for k in (b + e, (b + e).lower()):
            if k in by_name: return by_name[k]
    return None

def detect_image_column(df, by_name, by_stem, sample=300):
    best, best_rate = None, -1.0
    n = min(len(df), sample)
    if n == 0: return None
    probe = df.head(n)
    for c in df.columns:
        hits = sum(1 for v in probe[c].tolist() if resolve_image_path(v, by_name, by_stem) is not None)
        if hits / n > best_rate: best, best_rate = c, hits / n
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
        if cand in df.columns: return cand
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
# DATASET + MODEL  (real trained grounded-morph grader, subject of every
# metric/figure in this framework)
# ============================================================
def eval_tf(size):
    return A.Compose([A.Resize(size, size), A.CLAHE(p=1.0),
                      A.Normalize(mean=IMNET_MEAN, std=IMNET_STD), ToTensorV2()])

class GradeDataset(Dataset):
    def __init__(self, paths, labels, size):
        self.paths = list(paths); self.labels = labels; self.targets = list(labels.keys())
        self.tf = eval_tf(size); self.size = size
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        bgr = cv2.imread(self.paths[i])
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else np.zeros((self.size, self.size, 3), np.uint8)
        img = self.tf(image=img)['image']
        y = {t: torch.tensor(int(self.labels[t][i]), dtype=torch.long) for t in self.targets}
        return img, y

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
    def forward(self, x): return self.proj(x).flatten(2).transpose(1, 2)

class GroundedMorphologyEncoder(nn.Module):
    def __init__(self, cfg, num_morph_tokens=6):
        super().__init__()
        ed = cfg['model']['embed_dim']; self.num_morph = num_morph_tokens
        self.patch_embed = PatchEmbed(cfg['dataset']['image_size'], cfg['dataset']['patch_size'], cfg['dataset']['in_chans'], ed)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, ed))
        self.morph_tokens = nn.Parameter(torch.zeros(1, num_morph_tokens, ed))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, ed), requires_grad=False)
        layer = nn.TransformerEncoderLayer(d_model=ed, nhead=cfg['model']['num_heads'], dim_feedforward=ed * 4,
                                          dropout=0.1, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg['model']['depth'] // 3, enable_nested_tensor=False)
        self.encoder_norm = nn.LayerNorm(ed)
        pe = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(num_patches ** 0.5), cls_token=True)
        self.pos_embed.data.copy_(pe.unsqueeze(0))
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x) + self.pos_embed[:, 1:, :]
        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(B, -1, -1)
        morph = self.morph_tokens.expand(B, -1, -1)
        x = self.encoder_norm(self.encoder(torch.cat((cls, morph, x), dim=1)))
        return x[:, 1:1 + self.num_morph, :]

class Grader(nn.Module):
    def __init__(self, cfg, num_classes, num_morph_tokens=6):
        super().__init__()
        self.encoder = GroundedMorphologyEncoder(cfg, num_morph_tokens)
        ed = cfg['model']['embed_dim']
        self.heads = nn.ModuleDict({t: nn.Linear(ed * num_morph_tokens, n) for t, n in num_classes.items()})
    def forward(self, x):
        morph = self.encoder(x)
        flat = morph.reshape(morph.shape[0], -1)
        return {t: h(flat) for t, h in self.heads.items()}, morph

def coral_probs_from_sigmoid(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    N, Km1 = p.shape; K = Km1 + 1
    p_mono = np.minimum.accumulate(p, axis=1)   # FIX: was reversed+accumulate+reversed (wrong direction) -> collapsed everything to the min value
    probs = np.zeros((N, K), dtype=np.float64)
    probs[:, 0] = 1 - p_mono[:, 0]
    for k in range(1, K - 1):
        probs[:, k] = p_mono[:, k - 1] - p_mono[:, k]
    probs[:, K - 1] = p_mono[:, K - 2]
    return np.clip(probs, 1e-8, None) / np.clip(probs, 1e-8, None).sum(axis=1, keepdims=True)

# ============================================================
# METRICS MODULE  (generic, real, reusable -- operates on real y_true/y_pred/probs)
# ============================================================
def core_metrics(y_true, y_pred, y_prob=None):
    m = {'accuracy': float((y_true == y_pred).mean()),
        'qwk': float(cohen_kappa_score(y_true, y_pred, weights='quadratic')) if len(np.unique(y_true)) > 1 else 0.0,
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0))}
    if y_prob is not None:
        m['ece'] = expected_calibration_error(y_prob, y_true)
        m['mce'] = maximum_calibration_error(y_prob, y_true)
        try:
            from sklearn.preprocessing import label_binarize
            classes = sorted(np.unique(y_true))
            yb = label_binarize(y_true, classes=classes)
            if yb.shape[1] > 1:
                m['roc_auc_ovr_macro'] = float(np.mean([auc(*roc_curve(yb[:, c], y_prob[:, c])[:2])
                                                        for c in range(yb.shape[1]) if yb[:, c].sum() > 0]))
        except Exception:
            m['roc_auc_ovr_macro'] = float('nan')
    return m

def expected_calibration_error(probs, y_true, n_bins=10):
    conf = probs.max(axis=1); pred = probs.argmax(axis=1); correct = (pred == y_true).astype(float)
    edges = np.linspace(0, 1, n_bins + 1); ece = 0.0; n = len(conf)
    for i in range(n_bins):
        m = (conf >= edges[i]) & (conf < edges[i + 1])
        if m.sum(): ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)

def maximum_calibration_error(probs, y_true, n_bins=10):
    conf = probs.max(axis=1); pred = probs.argmax(axis=1); correct = (pred == y_true).astype(float)
    edges = np.linspace(0, 1, n_bins + 1); gaps = []
    for i in range(n_bins):
        m = (conf >= edges[i]) & (conf < edges[i + 1])
        if m.sum(): gaps.append(abs(correct[m].mean() - conf[m].mean()))
    return float(max(gaps)) if gaps else 0.0

def bootstrap_ci(metric_fn, y_true, y_pred, n_boot=2000, alpha=0.05, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true); vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        vals.append(metric_fn(y_true[idx], y_pred[idx]))
    vals = np.array(vals)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.mean(vals)), float(lo), float(hi)

def paired_bootstrap_test(metric_fn, y_true, pred_a, pred_b, n_boot=2000, seed=42):
    """Real paired bootstrap significance test: resample indices jointly for
    both models, get the distribution of the metric DIFFERENCE, p-value =
    fraction of bootstrap differences crossing zero."""
    rng = np.random.RandomState(seed)
    n = len(y_true); diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        diffs.append(metric_fn(y_true[idx], pred_a[idx]) - metric_fn(y_true[idx], pred_b[idx]))
    diffs = np.array(diffs)
    p_value = 2 * min((diffs > 0).mean(), (diffs < 0).mean())
    return {'mean_diff': float(diffs.mean()), 'ci_lo': float(np.percentile(diffs, 2.5)),
           'ci_hi': float(np.percentile(diffs, 97.5)), 'p_value': float(p_value)}

def mcnemar_test(y_true, pred_a, pred_b):
    correct_a = (pred_a == y_true); correct_b = (pred_b == y_true)
    b = int(((correct_a) & (~correct_b)).sum())   # a right, b wrong
    c = int(((~correct_a) & (correct_b)).sum())   # a wrong, b right
    if b + c == 0:
        return {'statistic': 0.0, 'p_value': 1.0, 'b': b, 'c': c}
    stat = (abs(b - c) - 1) ** 2 / (b + c)   # continuity-corrected McNemar
    from scipy.stats import chi2
    p = float(1 - chi2.cdf(stat, df=1))
    return {'statistic': float(stat), 'p_value': p, 'b': b, 'c': c}

def decision_curve_analysis(y_true_binary, y_prob_positive, thresholds=None):
    """Vickers & Elder (2006) net-benefit decision curve analysis.
    net_benefit(pt) = TP/n - FP/n * (pt / (1-pt)), compared against
    'treat all' and 'treat none' reference strategies."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 50)
    n = len(y_true_binary)
    prevalence = y_true_binary.mean()
    model_nb, all_nb, none_nb = [], [], []
    for pt in thresholds:
        pred_pos = y_prob_positive >= pt
        tp = int((pred_pos & (y_true_binary == 1)).sum())
        fp = int((pred_pos & (y_true_binary == 0)).sum())
        model_nb.append(tp / n - fp / n * (pt / (1 - pt)))
        all_nb.append(prevalence - (1 - prevalence) * (pt / (1 - pt)))
        none_nb.append(0.0)
    return thresholds, np.array(model_nb), np.array(all_nb), np.array(none_nb)

# ============================================================
# COLLECT REAL PREDICTIONS  (the one real, authoritative dataset everything
# below is computed from)
# ============================================================
@torch.no_grad()
def collect_predictions(model, loader, targets, device):
    model.eval()
    raw = {t: {'logits': [], 'true': []} for t in targets}
    morph_feats, morph_labels = [], []
    for imgs, y in loader:
        imgs = imgs.to(device)
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            out, morph = model(imgs)
        for t in targets:
            raw[t]['logits'].append(out[t].float().cpu().numpy())
            raw[t]['true'].append(y[t].numpy())
        morph_feats.append(morph.mean(dim=1).float().cpu().numpy())   # for UMAP/t-SNE
        morph_labels.append(y[targets[0]].numpy())
    result = {}
    for t in targets:
        logits = np.concatenate(raw[t]['logits'], 0); true = np.concatenate(raw[t]['true'], 0)
        m = true >= 0
        probs = coral_probs_from_sigmoid(1 / (1 + np.exp(-logits)))
        result[t] = {'true': true[m], 'probs': probs[m], 'pred': probs[m].argmax(axis=1)}
    return result, np.concatenate(morph_feats, 0), np.concatenate(morph_labels, 0)

# ============================================================
# FIGURES (real data, Nature-style, multi-format export)
# ============================================================
def fig_confusion_matrix(y_true, y_pred, name, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(cm, cmap='Blues')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=7,
                   color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(name)
    save_all_formats(fig, 'confusion_matrix_{}'.format(name), output_dir)

def fig_roc_pr(y_true, probs, name, output_dir):
    K = probs.shape[1]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8))
    for c in range(K):
        yb = (y_true == c).astype(int)
        if len(np.unique(yb)) < 2: continue
        fpr, tpr, _ = roc_curve(yb, probs[:, c])
        axes[0].plot(fpr, tpr, lw=1, label='class {} (AUC={:.2f})'.format(c, auc(fpr, tpr)))
        prec, rec, _ = precision_recall_curve(yb, probs[:, c])
        axes[1].plot(rec, prec, lw=1, label='class {} (AP={:.2f})'.format(c, average_precision_score(yb, probs[:, c])))
    axes[0].plot([0, 1], [0, 1], 'k--', lw=0.5); axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR'); axes[0].set_title('{}: ROC'.format(name))
    axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision'); axes[1].set_title('{}: PR'.format(name))
    axes[0].legend(fontsize=5); axes[1].legend(fontsize=5)
    save_all_formats(fig, 'roc_pr_{}'.format(name), output_dir)

def fig_calibration(y_true, probs, name, output_dir):
    pred = probs.argmax(axis=1); conf = probs.max(axis=1); correct = (pred == y_true).astype(float)
    edges = np.linspace(0, 1, 11); bc, bconf = [], []
    for i in range(10):
        m = (conf >= edges[i]) & (conf < edges[i + 1])
        if m.sum(): bc.append(correct[m].mean()); bconf.append(conf[m].mean())
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.plot([0, 1], [0, 1], 'k--', lw=0.6, label='Perfect')
    ax.plot(bconf, bc, marker='o', ms=3, lw=1, label='Model')
    ax.set_xlabel('Confidence'); ax.set_ylabel('Accuracy'); ax.set_title('{}: Reliability'.format(name)); ax.legend()
    save_all_formats(fig, 'reliability_{}'.format(name), output_dir)

def fig_decision_curve(y_true_binary, y_prob_positive, name, output_dir):
    thr, model_nb, all_nb, none_nb = decision_curve_analysis(y_true_binary, y_prob_positive)
    fig, ax = plt.subplots(figsize=(3.4, 3))
    ax.plot(thr, model_nb, lw=1.2, label='Model')
    ax.plot(thr, all_nb, lw=1, ls='--', label='Treat all')
    ax.plot(thr, none_nb, lw=1, ls=':', label='Treat none')
    ax.set_xlabel('Threshold probability'); ax.set_ylabel('Net benefit'); ax.set_title('{}: DCA'.format(name))
    ax.legend(); ax.set_ylim(bottom=min(-0.05, model_nb.min() * 1.2))
    save_all_formats(fig, 'decision_curve_{}'.format(name), output_dir)

def fig_embedding_projection(feats, labels, method, output_dir):
    if method == 'tsne':
        proj = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, len(feats) // 3))).fit_transform(feats)
    elif method == 'umap' and HAS_UMAP:
        proj = umap.UMAP(n_components=2, random_state=42).fit_transform(feats)
    else:
        proj = PCA(n_components=2).fit_transform(feats)
        method = 'pca_fallback'
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=labels, cmap='viridis', s=8, alpha=0.7)
    plt.colorbar(sc, ax=ax, label='EXP grade')
    ax.set_title('Morphology embeddings ({})'.format(method))
    save_all_formats(fig, 'embedding_{}'.format(method), output_dir)

def fig_bootstrap_ci_bars(ci_table, output_dir):
    fig, ax = plt.subplots(figsize=(4, 3))
    names = [r['metric'] for r in ci_table]; means = [r['mean'] for r in ci_table]
    los = [r['mean'] - r['lo'] for r in ci_table]; his = [r['hi'] - r['mean'] for r in ci_table]
    ax.barh(names, means, xerr=[los, his], capsize=3, color='slateblue')
    ax.set_xlabel('Value (bootstrap 95% CI)'); ax.set_title('Bootstrap confidence intervals')
    save_all_formats(fig, 'bootstrap_ci', output_dir)

def fig_radar_comparison(model_scores, output_dir):
    """model_scores: {model_name: {target: qwk}} -- reads whichever real
    kfold_summary.csv / clinical_test_*.csv files actually exist on disk."""
    if not model_scores:
        print("[RADAR] no prior-phase summary CSVs found -- skipping (run at least one grader phase first)")
        return
    targets = TARGETS
    angles = np.linspace(0, 2 * np.pi, len(targets), endpoint=False).tolist(); angles += angles[:1]
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    for name, scores in model_scores.items():
        vals = [scores.get(t, 0.0) for t in targets]; vals += vals[:1]
        ax.plot(angles, vals, lw=1.2, label=name)
        ax.fill(angles, vals, alpha=0.08)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(targets)
    ax.set_title('QWK comparison across model variants'); ax.legend(fontsize=6, loc='upper right', bbox_to_anchor=(1.35, 1.1))
    save_all_formats(fig, 'radar_comparison', output_dir)

# ============================================================
# GALLERY: copy real figures from other phases (never fabricate a missing one)
# ============================================================
def assemble_gallery(manifest, output_dir):
    gallery_dir = os.path.join(output_dir, 'gallery'); os.makedirs(gallery_dir, exist_ok=True)
    report_lines = ["# Figure provenance manifest\n"]
    for label, src in manifest.items():
        if src is None:
            report_lines.append("- {}: rendered separately via the Visualizer tool (not a file on disk)".format(label))
            continue
        if os.path.exists(src):
            dst = os.path.join(gallery_dir, label.replace(' ', '_').replace('/', '-') + os.path.splitext(src)[1])
            shutil.copy(src, dst)
            report_lines.append("- {}: OK -> copied from `{}`".format(label, src))
        else:
            report_lines.append("- {}: MISSING -> source file not found at `{}` (run the corresponding phase script first)".format(label, src))
    with open(os.path.join(output_dir, 'FIGURE_PROVENANCE.md'), 'w') as f:
        f.write("\n".join(report_lines))
    print("\n".join(report_lines))

# ============================================================
# MAIN
# ============================================================
def main():
    cfg = CFG
    set_nature_style()
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(cfg['output_dir'], exist_ok=True)

    if not os.path.exists(cfg['grader_checkpoint']):
        raise FileNotFoundError("Run embryo_grounded_morph_grader_v2.py first -> {}".format(cfg['grader_checkpoint']))

    by_name, by_stem = build_image_index(image_root)
    if len(by_name) == 0:
        for extra in [INPUT_DIR, os.path.expanduser("~")]:
            bn, bs = build_image_index(extra); by_name.update(bn); by_stem.update(bs)
    train_df = build_labeled_df(train_csv_path, by_name, by_stem, "train")
    test_df = build_labeled_df(test_csv_path, by_name, by_stem, "test")
    print("[WHOLE DATASET] train_silver rows: {} | test_gold rows: {} (no subsampling)\n".format(
        len(train_df), len(test_df)))
    maps = build_label_maps(train_df, TARGETS)
    num_classes = {t: len(maps[t]) for t in TARGETS}
    y_test = encode_labels(test_df, TARGETS, maps)

    ck = torch.load(cfg['grader_checkpoint'], map_location='cpu')
    mcfg = ck.get('mae_config', MORPH_DEFAULT); nmt = ck.get('num_morph_tokens', 6)
    size = mcfg['dataset']['image_size']
    model = Grader(mcfg, num_classes, num_morph_tokens=nmt)
    model.load_state_dict(ck['model'] if 'model' in ck else ck, strict=False)
    model.to(device).eval()
    print("[MODEL] loaded real trained grader | val mean-QWK {:.3f}\n".format(ck.get('val_mean_qwk', float('nan'))))

    te_loader = DataLoader(GradeDataset(test_df['resolved_path'].tolist(), y_test, size),
                          batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'], pin_memory=True)
    predictions, morph_feats, morph_labels = collect_predictions(model, te_loader, TARGETS, device)

    # Save the raw per-sample predictions -- fixes the gap for every future re-analysis.
    for t in TARGETS:
        pd.DataFrame({'true': predictions[t]['true'], 'pred': predictions[t]['pred'],
                     **{'prob_class_{}'.format(c): predictions[t]['probs'][:, c] for c in range(predictions[t]['probs'].shape[1])}}
                    ).to_csv(os.path.join(cfg['output_dir'], 'raw_predictions_{}.csv'.format(t)), index=False)

    # ---- core metrics + bootstrap CI per target ----
    print("=" * 78 + "\nCORE METRICS (real, computed on real held-out test set)\n" + "=" * 78)
    metric_rows, ci_rows = [], []
    for t in TARGETS:
        yt, yp, probs = predictions[t]['true'], predictions[t]['pred'], predictions[t]['probs']
        m = core_metrics(yt, yp, probs)
        m['target'] = t
        metric_rows.append(m)
        print("  {:<4} acc={:.3f} QWK={:.3f} MAE={:.3f} F1={:.3f} prec={:.3f} rec={:.3f} ECE={:.3f} MCE={:.3f} AUC={:.3f}".format(
            t, m['accuracy'], m['qwk'], m['mae'], m['f1_macro'], m['precision_macro'], m['recall_macro'],
            m['ece'], m['mce'], m.get('roc_auc_ovr_macro', float('nan'))))

        mean_acc, lo, hi = bootstrap_ci(lambda a, b: (a == b).mean(), yt, yp, n_boot=cfg['n_bootstrap'])
        ci_rows.append({'metric': '{} accuracy'.format(t), 'mean': mean_acc, 'lo': lo, 'hi': hi})
        mean_qwk, lo, hi = bootstrap_ci(lambda a, b: cohen_kappa_score(a, b, weights='quadratic') if len(np.unique(a)) > 1 else 0.0,
                                        yt, yp, n_boot=cfg['n_bootstrap'])
        ci_rows.append({'metric': '{} QWK'.format(t), 'mean': mean_qwk, 'lo': lo, 'hi': hi})
        print("    bootstrap 95% CI -> accuracy [{:.3f}, {:.3f}] | QWK [{:.3f}, {:.3f}]".format(
            ci_rows[-2]['lo'], ci_rows[-2]['hi'], lo, hi))

        fig_confusion_matrix(yt, yp, t, cfg['output_dir'])
        fig_roc_pr(yt, probs, t, cfg['output_dir'])
        fig_calibration(yt, probs, t, cfg['output_dir'])
        # DCA needs a binary target: "best-observed class vs rest" as a concrete, real binarization
        best_class = int(np.bincount(yt).argmax())
        fig_decision_curve((yt == best_class).astype(int), probs[:, best_class], t, cfg['output_dir'])

    pd.DataFrame(metric_rows).to_csv(os.path.join(cfg['output_dir'], 'core_metrics.csv'), index=False)
    pd.DataFrame(ci_rows).to_csv(os.path.join(cfg['output_dir'], 'bootstrap_ci.csv'), index=False)
    fig_bootstrap_ci_bars(ci_rows, cfg['output_dir'])

    # ---- embeddings: t-SNE + UMAP (PCA fallback if umap not installed) ----
    fig_embedding_projection(morph_feats, morph_labels, 'tsne', cfg['output_dir'])
    fig_embedding_projection(morph_feats, morph_labels, 'umap' if HAS_UMAP else 'pca_fallback', cfg['output_dir'])
    if not HAS_UMAP:
        print("\n[UMAP] package not installed (pip install umap-learn) -> used PCA as a labeled fallback")

    # ---- radar comparison: read whichever real prior-phase summary CSVs exist ----
    model_scores = {}
    candidates = {
        'Mean-pool grader (k-fold)': './embryo_project/grader/kfold_summary.csv',
        'Grounded-morph grader (k-fold)': './embryo_project/grounded_morph_grader/kfold_summary.csv',
        'Graph transformer': './embryo_project/graph_transformer/graph_test_results.csv',
        'Clinical multi-task (full)': './embryo_project/clinical_multitask/clinical_test_full.csv',
    }
    for name, path in candidates.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            qwk_col = 'qwk' if 'qwk' in df.columns else ('mean_qwk' if 'mean_qwk' in df.columns else None)
            if qwk_col and 'target' in df.columns:
                model_scores[name] = df.groupby('target')[qwk_col].mean().to_dict()
    fig_radar_comparison(model_scores, cfg['output_dir'])

    # ---- significance tests: this grader vs a fresh random-guess baseline on the SAME test set ----
    print("\n" + "=" * 78 + "\nSTATISTICAL SIGNIFICANCE (paired bootstrap + McNemar vs. majority-class baseline)\n" + "=" * 78)
    sig_rows = []
    for t in TARGETS:
        yt, yp = predictions[t]['true'], predictions[t]['pred']
        baseline_pred = np.full_like(yp, np.bincount(yt).argmax())
        boot = paired_bootstrap_test(lambda a, b: (a == b).mean(), yt, yp, baseline_pred, n_boot=cfg['n_bootstrap'])
        mcn = mcnemar_test(yt, yp, baseline_pred)
        sig_rows.append({'target': t, 'acc_diff_vs_majority': boot['mean_diff'], 'ci_lo': boot['ci_lo'],
                        'ci_hi': boot['ci_hi'], 'bootstrap_p': boot['p_value'], 'mcnemar_p': mcn['p_value']})
        print("  {:<4} acc diff vs majority-class = {:+.3f} [{:.3f},{:.3f}] | bootstrap p={:.4f} | McNemar p={:.4f}".format(
            t, boot['mean_diff'], boot['ci_lo'], boot['ci_hi'], boot['p_value'], mcn['p_value']))
    pd.DataFrame(sig_rows).to_csv(os.path.join(cfg['output_dir'], 'significance_tests.csv'), index=False)

    # ---- gallery: copy real figures from other phases, report what's missing ----
    print("\n" + "=" * 78 + "\nFIGURE GALLERY (copied from real prior-phase outputs where they exist)\n" + "=" * 78)
    assemble_gallery(cfg['gallery_manifest'], cfg['output_dir'])

    print("\n[COMPLETE] core_metrics.csv, bootstrap_ci.csv, significance_tests.csv, raw_predictions_*.csv,")
    print("  figures/ (600dpi PNG + PDF + SVG), gallery/, FIGURE_PROVENANCE.md -> {}".format(cfg['output_dir']))
    print("\nSee FIGURE_PROVENANCE.md for exactly which requested figures are real (generated here or")
    print("copied from a real prior run) vs. MISSING because that phase hasn't been run yet.")

if __name__ == '__main__':
    main()
