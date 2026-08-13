import os, math, random, itertools, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.stats import spearmanr
from tqdm.auto import tqdm
import warnings; warnings.filterwarnings("ignore")

try:
    import networkx as nx
except ImportError:
    raise ImportError("networkx is required for Attention Flow. Install: pip install networkx")

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
EXCLUDE_DIR_NAMES = {'.ipynb_checkpoints', '__pycache__', '.git', 'processed_data', 'figures',
                     'embryo_project'}   # consolidated project folder excluded from image indexing
IMNET_MEAN = [0.485, 0.456, 0.406]; IMNET_STD = [0.229, 0.224, 0.225]
TARGETS = ["EXP", "ICM", "TE"]

INPUT_DIR = "."; IMAGE_DIR = "./Downloads/archive/Images/Images"
TEST_CSV = "Gardner_test_gold_onlyGardnerScores.csv"

CFG = {
    "grader_checkpoint": "./embryo_project/grounded_morph_grader/grounded_morph_v2_fold1_mae_init.pth",
    "output_dir": "./embryo_project/explainability",   # consolidated project folder
    "targets": ["EXP", "ICM", "TE"],   # ENHANCEMENT: cover all real targets, not just EXP -- ICM/TE are the
                                       # harder, more clinically important open questions in this whole project
    "num_demo_images": 3,      # qualitative panels per target (kept small -- for looking at, not statistics)
    "num_eval_images": 20,     # ENHANCEMENT: real sample size for the NEW quantitative faithfulness/
                               # agreement analysis below (was previously n=1, printed once per demo image)
    "ig_steps": 30,
    "deletion_steps": 15,      # ENHANCEMENT: deletion/insertion faithfulness curve resolution
    "seed": 42,
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

INPUT_DIR = locate_input_dir(INPUT_DIR, TEST_CSV)
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

def real_grade(df, i, base):
    col = pick_col(df, base)
    if col is None: return None
    v = pd.to_numeric(df[col].iloc[i], errors='coerce')
    return None if pd.isna(v) else int(v)

def eval_tf(size):
    return A.Compose([A.Resize(size, size), A.CLAHE(p=1.0),
                      A.Normalize(mean=IMNET_MEAN, std=IMNET_STD), ToTensorV2()])

def load_tensor(path, size, device):
    bgr = cv2.imread(path)
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else np.zeros((size, size, 3), np.uint8)
    t = eval_tf(size)(image=img)['image'].unsqueeze(0).to(device)
    return t

# ============================================================
# MODEL: real trained grounded-morphology grader (subject of every
# explanation below -- an UNTRAINED model would make all of this vacuous).
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
        return x[:, 1:1 + self.num_morph, :], x[:, 1 + self.num_morph:, :]   # morph tokens, patch tokens

class Grader(nn.Module):
    def __init__(self, cfg, num_classes, num_morph_tokens=6):
        super().__init__()
        self.encoder = GroundedMorphologyEncoder(cfg, num_morph_tokens)
        ed = cfg['model']['embed_dim']
        self.heads = nn.ModuleDict({t: nn.Linear(ed * num_morph_tokens, n) for t, n in num_classes.items()})
    def forward(self, x):
        morph, patch = self.encoder(x)
        flat = morph.reshape(morph.shape[0], -1)
        return {t: h(flat) for t, h in self.heads.items()}, morph, patch

def coral_prob(logits):
    p = torch.sigmoid(logits)
    K = p.shape[1] + 1
    p_mono = torch.cummin(p, dim=1).values   # FIX: was flip+cummin+flip (wrong direction) -> collapsed everything to the min value
    probs = torch.zeros(p.shape[0], K, device=p.device)
    probs[:, 0] = 1 - p_mono[:, 0]
    for k in range(1, K - 1):
        probs[:, k] = p_mono[:, k - 1] - p_mono[:, k]
    probs[:, K - 1] = p_mono[:, K - 2]
    return probs.clamp(min=1e-8)

# ============================================================
# 1. GradCAM  (real gradients, retain_grad on the actual patch tokens)
# ============================================================
def grad_cam(model, x, target_task):
    model.eval()
    captured = {}
    h = model.encoder.encoder_norm.register_forward_hook(
        lambda m, i, o: captured.__setitem__('tokens', o) or o.retain_grad())
    preds, _, _ = model(x)
    h.remove()
    tokens = captured['tokens']
    score = preds[target_task].sum()
    model.zero_grad(set_to_none=True)
    score.backward()
    grad = tokens.grad; act = tokens.detach()
    num_morph = model.encoder.num_morph
    grad_p = grad[:, 1 + num_morph:, :]; act_p = act[:, 1 + num_morph:, :]
    weights = grad_p.mean(dim=1, keepdim=True)
    cam = torch.relu((weights * act_p).sum(dim=-1))
    grid = int(round(math.sqrt(cam.shape[1])))
    m = cam[0].reshape(grid, grid).detach().cpu().numpy()
    return (m - m.min()) / (m.max() - m.min() + 1e-8)

# ============================================================
# 2. GradCAM++  (Chattopadhyay et al. 2018 -- real 2nd/3rd-order gradient
# weighting, DIFFERENT math from vanilla GradCAM above, not a relabeling)
# ============================================================
def grad_cam_pp(model, x, target_task):
    model.eval()
    captured = {}
    h = model.encoder.encoder_norm.register_forward_hook(
        lambda m, i, o: captured.__setitem__('tokens', o) or o.retain_grad())
    preds, _, _ = model(x)
    h.remove()
    tokens = captured['tokens']
    score = preds[target_task].sum()
    model.zero_grad(set_to_none=True)
    score.backward()

    grad = tokens.grad; act = tokens.detach()
    num_morph = model.encoder.num_morph
    g = grad[:, 1 + num_morph:, :]
    a = act[:, 1 + num_morph:, :]

    g2 = g ** 2; g3 = g ** 3
    alpha_num = g2
    alpha_denom = 2 * g2 + (a * g3).sum(dim=1, keepdim=True)
    alpha_denom = torch.where(alpha_denom != 0, alpha_denom, torch.ones_like(alpha_denom))
    alpha = alpha_num / alpha_denom
    weights = (alpha * torch.relu(g)).sum(dim=1)
    cam = torch.relu((weights.unsqueeze(1) * a).sum(dim=-1))
    grid = int(round(math.sqrt(cam.shape[1])))
    m = cam[0].reshape(grid, grid).detach().cpu().numpy()
    return (m - m.min()) / (m.max() - m.min() + 1e-8)

# ============================================================
# 3. Integrated Gradients  (Sundararajan et al. 2017, batched)
# ============================================================
def integrated_gradients(model, x, target_task, steps=30):
    model.eval()
    baseline = torch.zeros_like(x)
    alphas = torch.linspace(0, 1, steps + 1, device=x.device).view(-1, 1, 1, 1)
    scaled = (baseline + alphas * (x - baseline)).clone().detach().requires_grad_(True)
    preds, _, _ = model(scaled)
    score = preds[target_task].sum()
    model.zero_grad(set_to_none=True)
    score.backward()
    avg_grad = scaled.grad.mean(dim=0, keepdim=True)
    ig = (x.detach() - baseline.detach()) * avg_grad
    m = ig.abs().mean(dim=1)[0].cpu().numpy()   # pixel resolution (size x size)
    # FIX: pool down to the ViT's own patch-grid resolution instead of leaving
    # this at pixel resolution -- GradCAM/GradCAM++/Rollout are all naturally
    # patch-grid already, and the mismatch (this was the only one upsampled to
    # pixel space) is exactly what broke deletion/insertion faithfulness scoring.
    grid = int(round(math.sqrt(model.encoder.patch_embed.num_patches)))
    patch_size = x.shape[-1] // grid
    m_crop = m[:grid * patch_size, :grid * patch_size]
    m_patch = m_crop.reshape(grid, patch_size, grid, patch_size).mean(axis=(1, 3))
    return (m_patch - m_patch.min()) / (m_patch.max() - m_patch.min() + 1e-8)

# ============================================================
# 4. Attention Rollout  (Abnar & Zuidema 2020 -- matrix multiplication
# across layers, real attention weights via hook + forced need_weights)
# ============================================================
def _capture_all_layer_attn(model, x):
    captured = []
    orig_forwards = []
    for layer in model.encoder.encoder.layers:
        orig = layer.self_attn.forward
        orig_forwards.append(orig)
        def wrapped(*args, __orig=orig, **kwargs):
            kwargs['need_weights'] = True; kwargs['average_attn_weights'] = True
            return __orig(*args, **kwargs)
        layer.self_attn.forward = wrapped
    hooks = [layer.self_attn.register_forward_hook(lambda m, i, o: captured.append(o[1].detach()))
             for layer in model.encoder.encoder.layers]
    with torch.no_grad():
        model(x)
    for layer, orig in zip(model.encoder.encoder.layers, orig_forwards):
        layer.self_attn.forward = orig
    for h in hooks:
        h.remove()
    return captured

def attention_rollout(model, x, num_morph):
    layer_attn = _capture_all_layer_attn(model, x)
    seq = layer_attn[0].shape[-1]
    rollout = torch.eye(seq, device=x.device).unsqueeze(0).repeat(x.shape[0], 1, 1)
    for A in layer_attn:
        A = A + torch.eye(seq, device=A.device).unsqueeze(0)
        A = A / A.sum(dim=-1, keepdim=True)
        rollout = torch.bmm(A, rollout)
    cls_row = rollout[:, 0, :]
    patch_map = cls_row[:, 1 + num_morph:]
    grid = int(round(math.sqrt(patch_map.shape[1])))
    m = patch_map[0].reshape(grid, grid).cpu().numpy()
    return (m - m.min()) / (m.max() - m.min() + 1e-8)

# ============================================================
# 5. Attention Flow  (scoped to the 6-morphology-token sub-graph across
# layers -- real, exact max-flow, stated simplification vs full-resolution)
# ============================================================
def attention_flow_morph_tokens(model, x, num_morph):
    layer_attn = _capture_all_layer_attn(model, x)
    L = len(layer_attn)
    morph_slice = slice(1, 1 + num_morph)
    G = nx.DiGraph()
    src = "src"; G.add_node(src)
    for l in range(L + 1):
        for t in range(num_morph):
            G.add_node((l, t))
    for t in range(num_morph):
        G.add_edge(src, (0, t), capacity=1.0)
    for l in range(L):
        A = layer_attn[l][0, morph_slice, morph_slice].cpu().numpy()
        for i in range(num_morph):
            for j in range(num_morph):
                w = float(A[i, j])
                if w > 1e-4:
                    G.add_edge((l, i), (l + 1, j), capacity=w)
    flow_to_final = np.zeros(num_morph)
    for t in range(num_morph):
        sink = (L, t)
        try:
            val, _ = nx.maximum_flow(G, src, sink)
        except Exception:
            val = 0.0
        flow_to_final[t] = val
    total = flow_to_final.sum()
    return flow_to_final / total if total > 0 else flow_to_final

# ============================================================
# 6. Token Attribution  (ablation)
# ============================================================
@torch.no_grad()
def token_attribution_ablation(model, x, target_task, num_morph=6):
    preds, morph, patch = model(x)
    probs = coral_prob(preds[target_task])
    pred_class = probs.argmax(dim=1).item()
    base_p = probs[0, pred_class].item()
    attributions = np.zeros(num_morph)
    for t in range(num_morph):
        morph_ablated = morph.clone()
        morph_ablated[:, t, :] = morph.mean(dim=1)
        flat = morph_ablated.reshape(morph_ablated.shape[0], -1)
        logits_ablated = model.heads[target_task](flat)
        p_ablated = coral_prob(logits_ablated)[0, pred_class].item()
        attributions[t] = base_p - p_ablated
    return attributions, pred_class, base_p

# ============================================================
# 7. SHAP  (exact Shapley values, 6 players -> 64 coalitions)
# ============================================================
@torch.no_grad()
def exact_shapley_values(model, x, target_task, num_morph=6):
    preds, morph, patch = model(x)
    probs = coral_prob(preds[target_task])
    pred_class = probs.argmax(dim=1).item()
    mean_token = morph.mean(dim=1, keepdim=True)

    def value_of_coalition(present_mask):
        m = morph.clone()
        for t in range(num_morph):
            if not present_mask[t]:
                m[:, t, :] = mean_token[:, 0, :]
        flat = m.reshape(m.shape[0], -1)
        logits = model.heads[target_task](flat)
        return coral_prob(logits)[0, pred_class].item()

    all_players = list(range(num_morph))
    shapley = np.zeros(num_morph)
    coalition_cache = {}
    for subset_size in range(num_morph + 1):
        for subset in itertools.combinations(all_players, subset_size):
            mask = tuple(i in subset for i in range(num_morph))
            coalition_cache[mask] = value_of_coalition(mask)

    for i in range(num_morph):
        others = [p for p in all_players if p != i]
        total = 0.0
        n = len(others)
        for r in range(n + 1):
            for subset in itertools.combinations(others, r):
                mask_without = tuple(p in subset for p in range(num_morph))
                mask_with = tuple((p in subset) or (p == i) for p in range(num_morph))
                weight = math.factorial(r) * math.factorial(n - r) / math.factorial(n + 1)
                total += weight * (coalition_cache[mask_with] - coalition_cache[mask_without])
        shapley[i] = total
    return shapley, pred_class

# ============================================================
# 8. LIME  (real local-surrogate algorithm)
# ============================================================
@torch.no_grad()
def lime_token_explanation(model, x, target_task, num_morph=6, n_samples=500, seed=42):
    rng = np.random.RandomState(seed)
    preds, morph, patch = model(x)
    probs = coral_prob(preds[target_task])
    pred_class = probs.argmax(dim=1).item()
    mean_token = morph.mean(dim=1, keepdim=True)

    Z = rng.randint(0, 2, size=(n_samples, num_morph)).astype(np.float64)
    Z[0, :] = 1.0
    y = np.zeros(n_samples)
    for s in range(n_samples):
        m = morph.clone()
        for t in range(num_morph):
            if Z[s, t] == 0:
                m[:, t, :] = mean_token[:, 0, :]
        flat = m.reshape(m.shape[0], -1)
        logits = model.heads[target_task](flat)
        y[s] = coral_prob(logits)[0, pred_class].item()

    distances = (Z.shape[1] - Z.sum(axis=1))
    kernel_width = 1.0
    weights = np.sqrt(np.exp(-(distances ** 2) / (kernel_width ** 2)))
    Zw = Z * weights[:, None]; yw = y * weights
    lam = 1.0
    coef = np.linalg.solve(Z.T @ Zw + lam * np.eye(num_morph), Z.T @ yw)
    return coef, pred_class

# ============================================================
# ENHANCEMENT A: DELETION/INSERTION FAITHFULNESS (Petsiuk et al. 2018)
# Turns "these 4 heatmaps look reasonable" into an objective, comparable
# score: how fast does confidence actually drop/rise when the heatmap's
# claimed most-important patches are removed/revealed?
# ============================================================
def gaussian_baseline(x):
    """Heavily blurred version of the input -- a standard 'no information'
    baseline for deletion/insertion (avoids the hard edge artifacts a flat
    black/gray baseline would introduce)."""
    img = x[0].permute(1, 2, 0).cpu().numpy()
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=25)
    return torch.from_numpy(blurred).permute(2, 0, 1).unsqueeze(0).to(x.device)

def heatmap_to_patch_order(heatmap, grid, patch_size):
    """heatmap is now ALWAYS at the ViT's native patch-grid resolution
    (every cam method returns grid x grid consistently) -- just rank patches,
    no pooling needed."""
    assert heatmap.shape == (grid, grid), (
        "heatmap_to_patch_order expects patch-grid resolution {}x{}, got {}. "
        "A cam method is returning the wrong resolution.".format(grid, grid, heatmap.shape))
    order = np.argsort(-heatmap.flatten())
    return order

def mask_patches(x, baseline, patch_indices, grid, patch_size):
    x_out = x.clone()
    for idx in patch_indices:
        r, c = divmod(int(idx), grid)
        sl_r = slice(r * patch_size, (r + 1) * patch_size)
        sl_c = slice(c * patch_size, (c + 1) * patch_size)
        x_out[:, :, sl_r, sl_c] = baseline[:, :, sl_r, sl_c]
    return x_out

@torch.no_grad()
def deletion_insertion_auc(model, x, heatmap, target_task, pred_class, patch_size, steps):
    grid = x.shape[-1] // patch_size
    num_patches = grid * grid
    order = heatmap_to_patch_order(heatmap, grid, patch_size)
    baseline = gaussian_baseline(x)
    step_sizes = np.linspace(0, num_patches, steps + 1).astype(int)

    del_probs, ins_probs = [], []
    for k in step_sizes:
        # deletion: remove the top-k most-important patches (replace with baseline)
        x_del = mask_patches(x, baseline, order[:k], grid, patch_size)
        p_del = coral_prob(model(x_del)[0][target_task])[0, pred_class].item()
        del_probs.append(p_del)
        # insertion: start from baseline, reveal the top-k most-important patches
        x_ins = mask_patches(baseline, x, order[:k], grid, patch_size)
        p_ins = coral_prob(model(x_ins)[0][target_task])[0, pred_class].item()
        ins_probs.append(p_ins)

    frac = step_sizes / num_patches
    del_auc = float(np.trapezoid(del_probs, frac))   # lower = more faithful (confidence collapses fast)
    ins_auc = float(np.trapezoid(ins_probs, frac))   # higher = more faithful (confidence recovers fast)
    return del_auc, ins_auc, frac, np.array(del_probs), np.array(ins_probs)

# ============================================================
# 3. CONFIDENCE
# ============================================================
@torch.no_grad()
def predicted_confidence(model, x, target_task):
    preds, _, _ = model(x)
    probs = coral_prob(preds[target_task])
    pred_class = int(probs.argmax(dim=1).item())
    return pred_class, float(probs[0, pred_class].item())

# ============================================================
# 2. SENTENCE GROUNDING
# ============================================================
def build_caption_and_ground(model, x, targets, num_morph=6):
    sentences, grounding_maps = [], {}
    for t in targets:
        cls, conf = predicted_confidence(model, x, t)
        sentences.append("{} grade {} predicted with confidence {:.2f}.".format(t, cls, conf))
        grounding_maps[t] = grad_cam(model, x, t)
    return sentences, grounding_maps

# ============================================================
# 5. PUBLICATION-QUALITY FIGURES
# ============================================================
def overlay(ax, img, heatmap, title):
    # FIX: heatmap is patch-grid resolution (e.g. 32x32); explicitly upsample
    # to the display image's resolution rather than relying on matplotlib's
    # default per-array extent, which silently misaligns two different-shaped
    # imshow calls without ever raising an error.
    if heatmap.shape[0] != img.shape[0] or heatmap.shape[1] != img.shape[1]:
        heatmap = cv2.resize(heatmap.astype(np.float32), (img.shape[1], img.shape[0]))
    ax.imshow(img); ax.imshow(heatmap, cmap='jet', alpha=0.5); ax.set_title(title, fontsize=10); ax.axis('off')

def plot_full_method_panel(img, maps_dict, target_task, pred_class, conf, out_path):
    n = len(maps_dict) + 1
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(3.4 * ((n + 1) // 2), 7))
    axes = axes.flatten()
    axes[0].imshow(img); axes[0].axis('off'); axes[0].set_title('Original', fontsize=10)
    for i, (name, m) in enumerate(maps_dict.items(), start=1):
        overlay(axes[i], img, m, name)
    for j in range(n, len(axes)):
        axes[j].axis('off')
    fig.suptitle('Explaining {} = {} (confidence {:.2f})'.format(target_task, pred_class, conf), fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=250, bbox_inches='tight'); plt.close()

def plot_token_importance_comparison(ablation, shapley, lime_coef, flow, out_path):
    methods = {'Ablation': ablation, 'Shapley (exact)': shapley, 'LIME': lime_coef, 'Attn. flow': flow}
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (name, vals) in zip(axes, methods.items()):
        ax.bar(range(len(vals)), vals, color='slateblue')
        ax.set_xticks(range(len(vals))); ax.set_xticklabels(['t{}'.format(i) for i in range(len(vals))])
        ax.set_title(name, fontsize=11); ax.axhline(0, color='gray', lw=0.5)
    fig.suptitle('Token-level attribution: do independent methods agree?', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=250, bbox_inches='tight'); plt.close()

def plot_sentence_grounding(img, sentences, grounding_maps, targets, out_path):
    n = len(sentences)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.6))
    if n == 1:
        axes = [axes]
    for i, (sent, t) in enumerate(zip(sentences, targets)):
        overlay(axes[i], img, grounding_maps[t], t)
        axes[i].text(0.5, -0.12, sent, transform=axes[i].transAxes, ha='center', fontsize=8, wrap=True)
    plt.tight_layout(); plt.savefig(out_path, dpi=220, bbox_inches='tight'); plt.close()

def plot_faithfulness_curves(agg_curves, out_path):
    """ENHANCEMENT A output: mean deletion/insertion curves per method,
    averaged over num_eval_images real test images, with AUC in the legend."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, d in agg_curves.items():
        axes[0].plot(d['frac'], d['del_mean'], marker='o', ms=3, label='{} (AUC={:.3f})'.format(name, d['del_auc']))
        axes[1].plot(d['frac'], d['ins_mean'], marker='o', ms=3, label='{} (AUC={:.3f})'.format(name, d['ins_auc']))
    axes[0].set_title('Deletion (lower AUC = more faithful)'); axes[0].set_xlabel('Fraction of patches removed')
    axes[1].set_title('Insertion (higher AUC = more faithful)'); axes[1].set_xlabel('Fraction of patches revealed')
    for ax in axes:
        ax.set_ylabel('Predicted-class probability'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=250, bbox_inches='tight'); plt.close()

def plot_method_agreement(corr_matrix, method_names, out_path):
    """ENHANCEMENT B output: real pairwise Spearman rank correlation between
    the 4 token-attribution methods, averaged over num_eval_images images."""
    import seaborn as sns
    plt.figure(figsize=(5.5, 5))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", xticklabels=method_names, yticklabels=method_names,
               cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Cross-method token-attribution agreement\n(mean Spearman rank correlation, n={} images)'.format(CFG['num_eval_images']))
    plt.tight_layout(); plt.savefig(out_path, dpi=250, bbox_inches='tight'); plt.close()

# ============================================================
# MAIN
# ============================================================
def main():
    cfg = CFG
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(cfg['output_dir'], exist_ok=True)

    if not os.path.exists(cfg['grader_checkpoint']):
        raise FileNotFoundError("Run embryo_grounded_morph_grader_v2.py first -> {}".format(cfg['grader_checkpoint']))

    by_name, by_stem = build_image_index(image_root)
    if len(by_name) == 0:
        for extra in [INPUT_DIR, os.path.expanduser("~")]:
            bn, bs = build_image_index(extra); by_name.update(bn); by_stem.update(bs)
    test_df = build_labeled_df(test_csv_path, by_name, by_stem, "test")
    print("[WHOLE DATASET] test_gold rows available: {} (eval sample uses {} of them)\n".format(
        len(test_df), min(cfg['num_eval_images'], len(test_df))))

    ck = torch.load(cfg['grader_checkpoint'], map_location='cpu')
    mcfg = ck.get('mae_config', MORPH_DEFAULT)
    nmt = ck.get('num_morph_tokens', 6)
    size = mcfg['dataset']['image_size']; patch_size = mcfg['dataset']['patch_size']
    sd = ck['model'] if 'model' in ck else ck

    # FIX: infer each head's real class count directly from its saved weight
    # shape rather than trusting a 'label_maps' key (this checkpoint's training
    # script never actually saved one -- see embryo_grounded_morph_grader_v2.py).
    # Correct by construction: can't drift out of sync with the checkpoint the
    # way a hardcoded fallback guess can (EXP genuinely has 5 classes, not 4).
    num_classes = {}
    for t in TARGETS:
        key = 'heads.{}.weight'.format(t)
        if key in sd:
            num_classes[t] = sd[key].shape[0]
        else:
            num_classes[t] = 4
            print("[WARNING] no saved head found for target '{}' -> defaulting to 4 classes".format(t))
    print("[MODEL] inferred real class counts from checkpoint weights:", num_classes)

    model = Grader(mcfg, num_classes, num_morph_tokens=nmt)
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    print("[MODEL] loaded real trained grader | val mean-QWK {:.3f}\n".format(ck.get('val_mean_qwk', float('nan'))))

    rng = random.Random(cfg['seed'])
    eval_idx = rng.sample(range(len(test_df)), min(cfg['num_eval_images'], len(test_df)))
    demo_idx = eval_idx[:cfg['num_demo_images']]

    # ================= qualitative demo panels, per target =================
    for target in cfg['targets']:
        print("=" * 70 + "\nQualitative demo panels for target: {}\n".format(target) + "=" * 70)
        for k, i in enumerate(demo_idx):
            path = test_df['resolved_path'].iloc[i]
            x = load_tensor(path, size, device)
            img = cv2.resize(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB), (size, size))
            true_grade = real_grade(test_df, i, target)
            pred_class, conf = predicted_confidence(model, x, target)
            print("  Image {}/{} | predicted={} true={} | confidence={:.2f}".format(
                k + 1, len(demo_idx), pred_class, true_grade, conf))

            maps = {
                'GradCAM': grad_cam(model, x, target),
                'GradCAM++': grad_cam_pp(model, x, target),
                'Integrated Gradients': integrated_gradients(model, x, target, steps=cfg['ig_steps']),
                'Attention Rollout': attention_rollout(model, x, nmt),
            }
            plot_full_method_panel(img, maps, target, pred_class, conf,
                                  os.path.join(cfg['output_dir'], 'method_panel_{}_{}.png'.format(target, k + 1)))

            flow = attention_flow_morph_tokens(model, x, nmt)
            ablation, _, _ = token_attribution_ablation(model, x, target, nmt)
            shapley, _ = exact_shapley_values(model, x, target, nmt)
            lime_coef, _ = lime_token_explanation(model, x, target, nmt)
            plot_token_importance_comparison(ablation, shapley, lime_coef, flow,
                                            os.path.join(cfg['output_dir'], 'token_attribution_{}_{}.png'.format(target, k + 1)))

        sentences, grounding_maps = build_caption_and_ground(model, load_tensor(test_df['resolved_path'].iloc[demo_idx[0]], size, device), cfg['targets'], nmt)
        img0 = cv2.resize(cv2.cvtColor(cv2.imread(test_df['resolved_path'].iloc[demo_idx[0]]), cv2.COLOR_BGR2RGB), (size, size))
        plot_sentence_grounding(img0, sentences, grounding_maps, cfg['targets'],
                               os.path.join(cfg['output_dir'], 'sentence_grounding.png'))
        break   # sentence grounding covers all targets at once; only needs generating once

    # ================= ENHANCEMENT A: real faithfulness (deletion/insertion) =================
    print("\n" + "=" * 70 + "\nQuantifying faithfulness (deletion/insertion, n={} real test images)\n".format(
        len(eval_idx)) + "=" * 70)
    cam_methods = {
        'GradCAM': grad_cam, 'GradCAM++': grad_cam_pp,
        'Integrated Gradients': lambda m, x, t: integrated_gradients(m, x, t, steps=cfg['ig_steps']),
        'Attention Rollout': lambda m, x, t: attention_rollout(m, x, nmt),
    }
    faith_target = cfg['targets'][0]   # run faithfulness on the first target (fast; extend the loop for all 3 if desired)
    agg_curves = {name: {'del': [], 'ins': [], 'frac': None} for name in cam_methods}
    for i in tqdm(eval_idx, desc="faithfulness eval"):
        path = test_df['resolved_path'].iloc[i]
        x = load_tensor(path, size, device)
        pred_class, _ = predicted_confidence(model, x, faith_target)
        for name, fn in cam_methods.items():
            heat = fn(model, x, faith_target)
            del_auc, ins_auc, frac, del_curve, ins_curve = deletion_insertion_auc(
                model, x, heat, faith_target, pred_class, patch_size, cfg['deletion_steps'])
            agg_curves[name]['del'].append(del_curve); agg_curves[name]['ins'].append(ins_curve)
            agg_curves[name]['frac'] = frac

    faith_rows = []
    for name in cam_methods:
        del_mean = np.mean(agg_curves[name]['del'], axis=0)
        ins_mean = np.mean(agg_curves[name]['ins'], axis=0)
        frac = agg_curves[name]['frac']
        del_auc = float(np.trapezoid(del_mean, frac)); ins_auc = float(np.trapezoid(ins_mean, frac))
        agg_curves[name].update({'del_mean': del_mean, 'ins_mean': ins_mean, 'del_auc': del_auc, 'ins_auc': ins_auc})
        faith_rows.append({'method': name, 'deletion_auc': del_auc, 'insertion_auc': ins_auc,
                          'faithfulness_score': ins_auc - del_auc})
        print("  {:<22} deletion_AUC={:.3f} (lower=better) | insertion_AUC={:.3f} (higher=better)".format(
            name, del_auc, ins_auc))
    faith_df = pd.DataFrame(faith_rows).sort_values('faithfulness_score', ascending=False)
    faith_df.to_csv(os.path.join(cfg['output_dir'], 'faithfulness_scores.csv'), index=False)
    plot_faithfulness_curves(agg_curves, os.path.join(cfg['output_dir'], 'faithfulness_curves.png'))
    print("\n  Most faithful method (highest insertion-minus-deletion AUC): {}".format(faith_df.iloc[0]['method']))

    # ================= ENHANCEMENT B: cross-method token-attribution agreement =================
    print("\n" + "=" * 70 + "\nQuantifying token-attribution agreement (Spearman rank correlation, n={} images)\n".format(
        len(eval_idx)) + "=" * 70)
    method_names = ['Ablation', 'Shapley', 'LIME', 'AttnFlow']
    agree_target = cfg['targets'][0]
    all_rankings = {m: [] for m in method_names}
    for i in tqdm(eval_idx, desc="token-attribution agreement eval"):
        path = test_df['resolved_path'].iloc[i]
        x = load_tensor(path, size, device)
        ablation, _, _ = token_attribution_ablation(model, x, agree_target, nmt)
        shapley, _ = exact_shapley_values(model, x, agree_target, nmt)
        lime_coef, _ = lime_token_explanation(model, x, agree_target, nmt)
        flow = attention_flow_morph_tokens(model, x, nmt)
        for name, vals in zip(method_names, [ablation, shapley, lime_coef, flow]):
            all_rankings[name].append(np.argsort(-np.abs(vals)))   # rank order, most-important first

    corr_matrix = np.eye(len(method_names))
    for a in range(len(method_names)):
        for b in range(a + 1, len(method_names)):
            corrs = [spearmanr(all_rankings[method_names[a]][k], all_rankings[method_names[b]][k]).correlation
                    for k in range(len(eval_idx))]
            corrs = [c for c in corrs if not np.isnan(c)]
            mean_corr = float(np.mean(corrs)) if corrs else 0.0
            corr_matrix[a, b] = corr_matrix[b, a] = mean_corr
    pd.DataFrame(corr_matrix, index=method_names, columns=method_names).to_csv(
        os.path.join(cfg['output_dir'], 'method_agreement.csv'))
    plot_method_agreement(corr_matrix, method_names, os.path.join(cfg['output_dir'], 'method_agreement_heatmap.png'))
    print("  Mean pairwise Spearman correlation across all method pairs: {:.3f}".format(
        (corr_matrix.sum() - len(method_names)) / (len(method_names) ** 2 - len(method_names))))
    print("  (Near 0 = methods disagree on token importance; near 1 = strong agreement. Either is a")
    print("   real finding about this model, not something to force into agreement.)")

    print("\n[COMPLETE] Saved qualitative panels + faithfulness_scores.csv + faithfulness_curves.png +")
    print("           method_agreement.csv + method_agreement_heatmap.png -> {}".format(cfg['output_dir']))

if __name__ == '__main__':
    main()
