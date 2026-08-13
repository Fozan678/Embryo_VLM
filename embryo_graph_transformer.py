import os, math, random, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2, albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import cohen_kappa_score
from tqdm.auto import tqdm
import warnings; warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True
IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
EXCLUDE_DIR_NAMES = {'.ipynb_checkpoints', '__pycache__', '.git', 'processed_data',
                     'embryo_iqa_outputs', 'mae_outputs', 'morphology_outputs', 'probe_outputs',
                     'seg_outputs', 'grader_outputs', 'grader_v2_outputs', 'grounded_morph_outputs',
                     'grounded_morph_v2_outputs', 'grounded_morph_probe_outputs', 'relational_graph_outputs',
                     'graph_transformer_outputs', 'clinical_multitask_outputs', 'uncertainty_outputs',
                     'vlm_outputs', 'explainability_outputs', 'figures', 'embryo_project'}   # NEW: consolidated
IMNET_MEAN = [0.485, 0.456, 0.406]; IMNET_STD = [0.229, 0.224, 0.225]
NODE_NAMES = ['ICM', 'TE', 'Blastocoel', 'Zona', 'Fragmentation', 'Global']
PROPERTY_NAMES = ['Compactness', 'Expansion', 'Continuity', 'BoundaryRegularity', 'Symmetry']

INPUT_DIR = "."; IMAGE_DIR = "./Downloads/archive/Images/Images"
TRAIN_CSV = "Gardner_train_silver.csv"; TEST_CSV = "Gardner_test_gold_onlyGardnerScores.csv"

CFG = {
    # Prefer the v2 grounded-morph checkpoint if it exists, else fall back to the original.
    "morph_checkpoint_candidates": [
        "./embryo_project/grounded_morph_grader/grounded_morph_v2_fold1_mae_init.pth",  # consolidated location
        "./grounded_morph_v2_outputs/grounded_morph_v2_fold1_mae_init.pth",             # legacy fallback
        "./grounded_morph_outputs/grounded_morph_best.pth",
    ],
    "finetune_encoder": False,   # keep frozen: this is a relational-reasoning add-on, not another grader
    "num_nodes": 6, "hidden_dim": 256, "num_heads": 8, "num_layers": 2,
    "lambda_sparsity": 0.02, "lambda_smooth": 0.05,
    "batch_size": 16, "epochs": 60, "lr": 1e-3, "min_lr": 1e-6, "weight_decay": 0.05,
    "warmup_ratio": 0.1, "val_ratio": 0.2, "num_workers": 2, "seed": 42,
    "output_dir": "./embryo_project/graph_transformer",   # consolidated project folder
}
MAE_CONFIG_DEFAULT = {"dataset": {"image_size": 512, "patch_size": 16, "in_chans": 3},
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
# REAL GEOMETRIC PROXY LABELS  (computed once, deterministic, no
# fabrication — classical morphometrics from the actual image mask)
# ============================================================
def segment_embryo(gray):
    """Lightweight, self-contained Otsu segmentation + cleanup (no MAE
    dependency). Less refined than the MAE-feature clustering from
    embryo_mae_segmentation.py, but real, deterministic, and standalone."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if mask.mean() > 127:            # embryo is usually the darker region on a lighter well background
        mask = 255 - mask
    mask = (mask > 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (lab == largest).astype(np.uint8)
    k = max(5, gray.shape[0] // 60)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    return mask

def compute_geometric_proxies(image_rgb):
    """Returns (compactness, continuity, boundary_regularity, symmetry), all in [0,1].
    Definitions (documented, not clinically validated -- geometric proxies only):
      compactness         = isoperimetric ratio 4*pi*Area / Perimeter^2 of the largest contour
      continuity          = largest-connected-component area / total foreground area
      boundary_regularity = convex-hull perimeter / contour perimeter (1.0 = perfectly smooth/convex)
      symmetry            = 1 - normalized pixel disagreement between mask and its horizontal mirror
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    mask_all = segment_embryo(gray)   # already largest-component only, for continuity we need the raw fg too
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, raw_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if raw_mask.mean() > 127:
        raw_mask = 255 - raw_mask
    raw_mask = (raw_mask > 0).astype(np.uint8)

    total_fg = max(1, int(raw_mask.sum()))
    continuity = float(mask_all.sum()) / total_fg
    continuity = float(np.clip(continuity, 0.0, 1.0))

    contours, _ = cv2.findContours(mask_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0, continuity, 0.0, 0.0
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt); perim = cv2.arcLength(cnt, True)
    compactness = 0.0 if perim <= 1e-6 else float(np.clip(4 * math.pi * area / (perim ** 2), 0.0, 1.0))

    hull = cv2.convexHull(cnt)
    hull_perim = cv2.arcLength(hull, True)
    boundary_regularity = 0.0 if perim <= 1e-6 else float(np.clip(hull_perim / perim, 0.0, 1.0))

    flipped = cv2.flip(mask_all, 1)
    denom = max(1, int(mask_all.sum() + flipped.sum()))
    disagreement = int(np.logical_xor(mask_all, flipped).sum())
    symmetry = float(np.clip(1.0 - disagreement / denom, 0.0, 1.0))

    return compactness, continuity, boundary_regularity, symmetry

def build_label_map_ordinal(train_df, col_base):
    col = pick_col(train_df, col_base)
    vals = pd.to_numeric(train_df[col], errors='coerce').dropna().astype(int)
    return {c: i for i, c in enumerate(sorted(vals.unique().tolist()))}

def encode_ordinal(df, col_base, cmap):
    col = pick_col(df, col_base)
    v = pd.to_numeric(df[col], errors='coerce')
    return v.map(lambda x: cmap.get(int(x), -1) if pd.notna(x) else -1).astype(int).values

# ============================================================
# DATASET  (precomputes geometric proxies once -> cheap at train time)
# ============================================================
def eval_tf(size):
    return A.Compose([A.Resize(size, size), A.CLAHE(p=1.0),
                      A.Normalize(mean=IMNET_MEAN, std=IMNET_STD), ToTensorV2()])

class GraphPropDataset(Dataset):
    def __init__(self, paths, exp_labels, size):
        self.paths = list(paths); self.exp_labels = exp_labels; self.size = size
        self.tf = eval_tf(size)
        print("[PROXIES] computing geometric proxy labels for {} images (one-time, cached)...".format(len(paths)))
        self.proxies = np.zeros((len(paths), 4), dtype=np.float32)   # compactness, continuity, boundary_reg, symmetry
        for i, p in enumerate(tqdm(self.paths, desc="geometric proxies")):
            bgr = cv2.imread(p)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb_small = cv2.resize(rgb, (256, 256))   # geometry is scale-invariant-ish; smaller = faster
            self.proxies[i] = compute_geometric_proxies(rgb_small)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        bgr = cv2.imread(self.paths[i])
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else np.zeros((self.size, self.size, 3), np.uint8)
        img = self.tf(image=img)['image']
        exp = torch.tensor(int(self.exp_labels[i]), dtype=torch.long)
        geo = torch.tensor(self.proxies[i], dtype=torch.float32)   # [compactness, continuity, boundary_reg, symmetry]
        return img, exp, geo

# ============================================================
# FROZEN GROUNDED-MORPHOLOGY ENCODER  (produces the 6 morphology tokens)
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

class GroundedMorphologyEncoder(nn.Module):
    def __init__(self, cfg, num_morph_tokens=6):
        super().__init__()
        ed = cfg['model']['embed_dim']; self.num_morph = num_morph_tokens
        self.patch_embed = PatchEmbed(cfg['dataset']['image_size'], cfg['dataset']['patch_size'],
                                      cfg['dataset']['in_chans'], ed)
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

# ============================================================
# 1-2-3. GRAPH ATTENTION TRANSFORMER  (multi-head + a LEARNABLE per-layer
# edge-bias matrix = the "learned graph" / "spatial relationships" between
# the 6 morphology nodes -- a real trained parameter, not attention alone).
# ============================================================
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, num_nodes):
        super().__init__()
        assert out_dim % num_heads == 0
        self.num_heads = num_heads; self.head_dim = out_dim // num_heads
        self.q = nn.Linear(in_dim, out_dim); self.k = nn.Linear(in_dim, out_dim); self.v = nn.Linear(in_dim, out_dim)
        self.edge_bias = nn.Parameter(torch.zeros(num_heads, num_nodes, num_nodes))  # THE learned graph
        self.attn_dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.res_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        B, N, _ = x.shape
        q = self.q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + self.edge_bias.unsqueeze(0)          # inject the learned graph structure
        attn = self.attn_dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, -1)
        out = self.norm(self.out_proj(out) + self.res_proj(x))
        return out, attn.mean(dim=1)   # node features, head-averaged attention (N,N)

class GraphMorphologyTransformer(nn.Module):
    def __init__(self, cfg, num_exp_classes):
        super().__init__()
        in_dim = cfg['model']['embed_dim'] if 'model' in cfg else cfg['embed_dim']
        hid = CFG['hidden_dim']; heads = CFG['num_heads']; nodes = CFG['num_nodes']
        self.layers = nn.ModuleList()
        d_in = in_dim
        for _ in range(CFG['num_layers']):
            self.layers.append(GraphAttentionLayer(d_in, hid, heads, nodes))
            d_in = hid
        self.pool = nn.Linear(hid, 1)                              # learned attention-pool over 6 nodes
        self.exp_head = nn.Linear(hid, max(1, num_exp_classes - 1)) # CORAL ordinal head for the real EXP grade
        self.geo_head = nn.Sequential(nn.Linear(hid, 64), nn.GELU(), nn.Linear(64, 4), nn.Sigmoid())  # 4 geometric proxies

    def forward(self, morph_tokens):
        x = morph_tokens
        attn_maps = []
        for layer in self.layers:
            x, attn = layer(x)
            attn_maps.append(attn)
        w = torch.softmax(self.pool(x), dim=1)
        pooled = (w * x).sum(dim=1)
        exp_logits = self.exp_head(pooled)
        geo_pred = self.geo_head(pooled)     # [compactness, continuity, boundary_reg, symmetry] in (0,1)
        return exp_logits, geo_pred, x, attn_maps

# ============================================================
# 5. GRAPH REGULARIZATION
# ============================================================
def graph_regularization(model, node_feats, attn_maps, lambda_sparsity, lambda_smooth):
    sparsity = sum(layer.edge_bias.abs().mean() for layer in model.layers)
    smooth = 0.0
    diff = (node_feats.unsqueeze(2) - node_feats.unsqueeze(1)).pow(2).sum(-1)   # (B,N,N) pairwise feature distance
    A = attn_maps[-1]                                                          # last layer's attention (B,N,N)
    smooth = (A * diff).mean()   # Dirichlet-energy-style: connected (high-attention) nodes should be similar
    return lambda_sparsity * sparsity + lambda_smooth * smooth, sparsity, smooth

def coral_loss(logits, levels):
    if logits.numel() == 0 or levels.numel() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    num_classes = logits.shape[1] + 1
    lv = levels.view(-1, 1)
    val = torch.arange(num_classes - 1, device=logits.device).view(1, -1)
    targets = (lv > val).float()
    return F.binary_cross_entropy_with_logits(logits, targets, reduction='none').sum(dim=1).mean()

def coral_decode(logits):
    return (torch.sigmoid(logits).detach().cpu().numpy() > 0.5).sum(axis=1)

def lr_at(ef, cfg):
    warmup = max(1, int(cfg['warmup_ratio'] * cfg['epochs']))
    if ef < warmup:
        s = ef / warmup
    else:
        s = 0.5 * (1 + math.cos(math.pi * (ef - warmup) / max(1, cfg['epochs'] - warmup)))
    return cfg['min_lr'] + (cfg['lr'] - cfg['min_lr']) * s

# ============================================================
# 6-7-8. VISUALIZATIONS  (all computed from REAL trained parameters /
# REAL test-set forward passes -- no dummy batches, no random noise)
# ============================================================
def plot_learned_graph(model, out_dir):
    """'Learned graph' + 'Edge importance': the static, input-INdependent
    edge-bias matrix each GraphAttentionLayer actually trained."""
    n_layers = len(model.layers)
    fig, axes = plt.subplots(1, n_layers, figsize=(7 * n_layers, 6))
    if n_layers == 1:
        axes = [axes]
    for li, layer in enumerate(model.layers):
        mat = layer.edge_bias.mean(dim=0).detach().cpu().numpy()   # average over heads
        sns.heatmap(mat, annot=True, fmt=".2f", xticklabels=NODE_NAMES, yticklabels=NODE_NAMES,
                   cmap='coolwarm', center=0, ax=axes[li])
        axes[li].set_title("Learned graph -- layer {} (edge bias)".format(li + 1))
    plt.tight_layout()
    p = os.path.join(out_dir, 'learned_graph_edge_bias.png')
    plt.savefig(p, dpi=250, bbox_inches='tight'); plt.close()
    return p

@torch.no_grad()
def plot_edge_attention(model, loader, device, out_dir, n_batches=8):
    """'Graph attention' + 'Edge importance': REAL input-dependent attention,
    averaged over many real test images (not a single dummy batch)."""
    model.eval()
    accum = None; count = 0
    for bi, (tokens, _, _) in enumerate(loader):
        if bi >= n_batches:
            break
        tokens = tokens.to(device)
        _, _, _, attn_maps = model(tokens)
        A = attn_maps[-1].mean(dim=0).cpu().numpy()
        accum = A if accum is None else accum + A
        count += 1
    mean_attn = accum / max(1, count)
    plt.figure(figsize=(7, 6))
    sns.heatmap(mean_attn, annot=True, fmt=".2f", xticklabels=NODE_NAMES, yticklabels=NODE_NAMES, cmap='Blues')
    plt.title("Graph attention -- averaged over {} real test batches".format(count))
    plt.xlabel("Target node"); plt.ylabel("Source node")
    p = os.path.join(out_dir, 'edge_attention_real.png')
    plt.tight_layout(); plt.savefig(p, dpi=250, bbox_inches='tight'); plt.close()
    return p

@torch.no_grad()
def plot_embedding_clusters(model, loader, device, out_dir, max_items=400):
    """'Embedding clusters': PCA + t-SNE of the graph transformer's final
    node embeddings, computed on REAL test images."""
    model.eval()
    feats, node_ids = [], []
    seen = 0
    for tokens, _, _ in loader:
        if seen >= max_items:
            break
        tokens = tokens.to(device)
        _, _, node_feats, _ = model(tokens)
        B, N, D = node_feats.shape
        feats.append(node_feats.reshape(B * N, D).cpu().numpy())
        node_ids.extend(list(range(N)) * B)
        seen += B
    X = np.concatenate(feats, axis=0)
    labels = [NODE_NAMES[i] for i in node_ids]

    pca = PCA(n_components=2).fit_transform(X)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=pca[:, 0], y=pca[:, 1], hue=labels, palette='tab10', s=60, edgecolor='k')
    plt.title("Graph node embeddings (PCA) -- real test images")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    p1 = os.path.join(out_dir, 'embedding_clusters_pca.png')
    plt.tight_layout(); plt.savefig(p1, dpi=250, bbox_inches='tight'); plt.close()

    if X.shape[0] > 10:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, X.shape[0] // 10))).fit_transform(X)
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=tsne[:, 0], y=tsne[:, 1], hue=labels, palette='tab10', s=60, edgecolor='k')
        plt.title("Graph node embeddings (t-SNE) -- real test images")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        p2 = os.path.join(out_dir, 'embedding_clusters_tsne.png')
        plt.tight_layout(); plt.savefig(p2, dpi=250, bbox_inches='tight'); plt.close()
        return p1, p2
    return p1, None

# ============================================================
# TRAIN + EVAL
# ============================================================
class TokenCache(Dataset):
    """Pre-extracts frozen morphology tokens once, so graph-transformer
    training doesn't repeatedly run the (frozen) ViT encoder every epoch."""
    def __init__(self, tokens, exp_labels, geo):
        self.tokens = tokens; self.exp = exp_labels; self.geo = geo
    def __len__(self):
        return len(self.tokens)
    def __getitem__(self, i):
        return self.tokens[i], self.exp[i], self.geo[i]

@torch.no_grad()
def extract_tokens(encoder, ds, device, batch_size=16, num_workers=2):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    toks, exps, geos = [], [], []
    for imgs, exp, geo in tqdm(loader, desc="extracting frozen morphology tokens"):
        imgs = imgs.to(device)
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            t = encoder(imgs)
        toks.append(t.float().cpu()); exps.append(exp); geos.append(geo)
    return torch.cat(toks, 0), torch.cat(exps, 0), torch.cat(geos, 0)

@torch.no_grad()
def evaluate_exp(model, loader, device):
    model.eval()
    P, T = [], []
    for tokens, exp, _ in loader:
        tokens = tokens.to(device)
        logits, _, _, _ = model(tokens)
        yt = exp.numpy(); m = yt >= 0
        if not m.any():
            continue
        pred = coral_decode(logits)
        P.extend(np.asarray(pred)[m].tolist()); T.extend(yt[m].tolist())
    if len(T) == 0 or len(np.unique(T)) < 2:
        return 0.0
    return float(cohen_kappa_score(T, P, weights='quadratic'))

def main():
    cfg = CFG
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(cfg['output_dir'], exist_ok=True)

    ckpt_path = next((p for p in cfg['morph_checkpoint_candidates'] if os.path.exists(p)), None)
    if ckpt_path is None:
        raise FileNotFoundError("No grounded-morph checkpoint found. Run embryo_grounded_morph_grader_v2.py "
                                "(or the original embryo_grounded_morph_grader.py) first.")
    print("[ENCODER] using checkpoint: {}".format(ckpt_path))

    by_name, by_stem = build_image_index(image_root)
    if len(by_name) == 0:
        for extra in [INPUT_DIR, os.path.expanduser("~")]:
            bn, bs = build_image_index(extra); by_name.update(bn); by_stem.update(bs)
    print("[IMAGES] indexed {} files\n".format(len(set(by_name.values()))))

    train_df = build_labeled_df(train_csv_path, by_name, by_stem, "train")
    test_df = build_labeled_df(test_csv_path, by_name, by_stem, "test")
    print("[WHOLE DATASET] train_silver rows: {} | test_gold rows: {} (no subsampling)\n".format(
        len(train_df), len(test_df)))

    exp_map = build_label_map_ordinal(train_df, 'EXP')
    num_exp_classes = len(exp_map)
    print("[LABELS] Expansion classes: {}\n".format(num_exp_classes))
    print("[NOTE] Compactness / Continuity / BoundaryRegularity / Symmetry are geometric proxies")
    print("       computed from real image masks -- NOT clinically annotated ground truth.\n")

    ck = torch.load(ckpt_path, map_location='cpu')
    mae_cfg = ck.get('mae_config', MAE_CONFIG_DEFAULT)
    nmt = ck.get('num_morph_tokens', cfg['num_nodes'])
    size = mae_cfg['dataset']['image_size']

    encoder = GroundedMorphologyEncoder(mae_cfg, num_morph_tokens=nmt)
    encoder.load_state_dict(ck['encoder'] if 'encoder' in ck else
                            {k[8:]: v for k, v in ck['model'].items() if k.startswith('encoder.')}, strict=False)
    encoder.to(device).eval()
    if not cfg['finetune_encoder']:
        for p in encoder.parameters():
            p.requires_grad_(False)

    idx = np.arange(len(train_df)); np.random.shuffle(idx)
    n_val = int(cfg['val_ratio'] * len(idx))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    exp_all = encode_ordinal(train_df, 'EXP', exp_map)
    tr_paths = [train_df['resolved_path'].iloc[i] for i in tr_idx]
    va_paths = [train_df['resolved_path'].iloc[i] for i in val_idx]
    tr_exp = exp_all[tr_idx]; va_exp = exp_all[val_idx]
    te_exp = encode_ordinal(test_df, 'EXP', exp_map)

    tr_ds = GraphPropDataset(tr_paths, tr_exp, size)
    va_ds = GraphPropDataset(va_paths, va_exp, size)
    te_ds = GraphPropDataset(test_df['resolved_path'].tolist(), te_exp, size)

    tr_tok, tr_e, tr_g = extract_tokens(encoder, tr_ds, device)
    va_tok, va_e, va_g = extract_tokens(encoder, va_ds, device)
    te_tok, te_e, te_g = extract_tokens(encoder, te_ds, device)

    tr_loader = DataLoader(TokenCache(tr_tok, tr_e, tr_g), batch_size=cfg['batch_size'], shuffle=True)
    va_loader = DataLoader(TokenCache(va_tok, va_e, va_g), batch_size=cfg['batch_size'], shuffle=False)
    te_loader = DataLoader(TokenCache(te_tok, te_e, te_g), batch_size=cfg['batch_size'], shuffle=False)

    model = GraphMorphologyTransformer(mae_cfg, num_exp_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    steps = max(1, len(tr_loader))

    print("\n[TRAIN] Graph Attention Transformer over {} morphology nodes | epochs={}\n".format(cfg['num_nodes'], cfg['epochs']))
    best_qwk, best_state = -1.0, None
    for ep in range(cfg['epochs']):
        model.train(); run = 0.0
        for step, (tokens, exp, geo) in enumerate(tr_loader):
            for pg in opt.param_groups:
                pg['lr'] = lr_at(ep + step / steps, cfg)
            tokens, exp, geo = tokens.to(device), exp.to(device), geo.to(device)
            exp_logits, geo_pred, node_feats, attn_maps = model(tokens)
            m = exp >= 0
            exp_loss = coral_loss(exp_logits[m], exp[m]) if m.any() else torch.tensor(0.0, device=device)
            geo_loss = F.smooth_l1_loss(geo_pred, geo)
            reg_loss, sparsity, smooth = graph_regularization(model, node_feats, attn_maps,
                                                               cfg['lambda_sparsity'], cfg['lambda_smooth'])
            loss = exp_loss + geo_loss + reg_loss
            opt.zero_grad(); loss.backward(); opt.step()
            run += float(loss)
        vq = evaluate_exp(model, va_loader, device)
        if (ep + 1) % 10 == 0 or ep == 0:
            print("[Epoch {:02d}] loss {:.4f} | val Expansion QWK {:.3f}".format(ep + 1, run / steps, vq))
        if vq > best_qwk:
            best_qwk = vq
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({'model': model.state_dict(), 'val_expansion_qwk': best_qwk}, os.path.join(cfg['output_dir'], 'graph_transformer_best.pth'))

    test_qwk = evaluate_exp(model, te_loader, device)
    print("\n" + "=" * 66 + "\nTEST (test_gold) Expansion QWK = {:.3f}  (best val = {:.3f})\n".format(test_qwk, best_qwk) + "=" * 66)

    # geometric-proxy fit quality (how well the graph predicts its own computed proxies)
    model.eval()
    with torch.no_grad():
        preds, trues = [], []
        for tokens, _, geo in te_loader:
            tokens = tokens.to(device)
            _, geo_pred, _, _ = model(tokens)
            preds.append(geo_pred.cpu().numpy()); trues.append(geo.numpy())
    preds = np.concatenate(preds, 0); trues = np.concatenate(trues, 0)
    mae_per_prop = np.abs(preds - trues).mean(axis=0)
    for name, err in zip(['Compactness', 'Continuity', 'BoundaryRegularity', 'Symmetry'], mae_per_prop):
        print("  {:<20} test MAE (proxy target) = {:.4f}".format(name, err))

    fig1 = plot_learned_graph(model, cfg['output_dir'])
    fig2 = plot_edge_attention(model, te_loader, device, cfg['output_dir'])
    fig3, fig4 = plot_embedding_clusters(model, te_loader, device, cfg['output_dir'])
    print("\nSaved: {}".format(", ".join(os.path.basename(p) for p in [fig1, fig2, fig3, fig4] if p)))

    try:
        from IPython.display import Image, display
        for p in [fig1, fig2, fig3, fig4]:
            if p:
                display(Image(filename=p, width=700))
    except Exception:
        pass

if __name__ == '__main__':
    main()
