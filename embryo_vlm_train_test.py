"""
Evidence-grounded VLM on the REAL train/test Gardner data (train_silver as
the retrieval database, test_gold as queries WITH real ground truth --
unseen-data inference is dropped per instruction).

CRITICAL DESIGN CHOICE: the grader-loading and inference code below is taken
DIRECTLY from check_grader_collapse.py, which was independently verified (by
your own pasted output) to produce genuinely diverse, reasonable predictions
on test_gold -- NOT rebuilt from embryo_evidence_grounded_vlm.py's more
complex, multiply-patched code path, since that's where the recurring
"every embryo identical" symptom lives despite the underlying checkpoint
being fine in isolation.

SANITY GATE: before any Qwen2-VL work happens, this script predicts on a
real sample of test_gold and checks for genuine class diversity. If it
collapses here too, the script HALTS with a clear error instead of silently
generating confidently-wrong VLM reports -- this is the direct fix for the
recurring symptom, not another guess.
"""
import os, re, random, math, numpy as np, pandas as pd, torch, torch.nn as nn
import cv2, albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from collections import Counter
from tqdm.auto import tqdm
import warnings; warnings.filterwarnings("ignore")

try:
    import faiss
except ImportError:
    faiss = None

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
EXCLUDE_DIR_NAMES = {'.ipynb_checkpoints', '__pycache__', '.git', 'embryo_project'}
IMNET_MEAN = [0.485, 0.456, 0.406]; IMNET_STD = [0.229, 0.224, 0.225]

CFG = {
    "grader_checkpoint": "./embryo_project/grounded_morph_grader/grounded_morph_v2_fold1_mae_init.pth",
    "mae_checkpoint": "./embryo_project/mae/checkpoints/mae_best.pth",
    "train_csv": "./Downloads/archive/Gardner_train_silver.csv",
    "test_csv": "./Downloads/archive/Gardner_test_gold_onlyGardnerScores.csv",
    "image_root": "./Downloads/archive/Images/Images",
    "retrieval_db_embeddings": "./embryo_project/retrieval/db_embeddings.npy",
    "qwen_model_id": "Qwen/Qwen2-VL-2B-Instruct",
    "lora_adapter_path": "./embryo_project/vlm_grounded/qwen_lora_adapter",
    "output_dir": "./embryo_project/vlm_train_test",
    "sanity_check_n": 30,          # real test_gold images checked BEFORE any VLM work
    "num_demo_reports": 10,
    "top_k_retrieval": 5,
    "seed": 42,
}

GARDNER_LABEL_MAPS = {
    'ICM': {0: 'A', 1: 'B', 2: 'C', 3: 'D'},
    'TE':  {0: 'A', 1: 'B', 2: 'C', 3: 'D'},
    'EXP': {0: 'Stage 1 (early blastocyst)', 1: 'Stage 2 (blastocyst)',
            2: 'Stage 3 (full blastocyst)', 3: 'Stage 4 (expanded blastocyst)',
            4: 'Stage 5 (hatching blastocyst)'},
}
def format_grade(target, value):
    if value is None: return "unknown"
    return GARDNER_LABEL_MAPS.get(target, {}).get(value, str(value))

# ============================================================
# CSV + IMAGE RESOLUTION
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
    return None

def detect_image_column(df, by_name, by_stem, sample=200):
    best, best_rate = None, -1.0
    n = min(len(df), sample)
    for c in df.columns:
        hits = sum(1 for v in df[c].head(n).tolist() if resolve_image_path(v, by_name, by_stem) is not None)
        if n and hits / n > best_rate: best, best_rate = c, hits / n
    return best

def build_labeled_df(csv_path, by_name, by_stem, tag):
    df = read_csv_smart(csv_path)
    col = detect_image_column(df, by_name, by_stem)
    df['resolved_path'] = df[col].map(lambda v: resolve_image_path(v, by_name, by_stem))
    n0 = len(df); df = df[df['resolved_path'].notna()].reset_index(drop=True)
    print("[IMAGES] {}: column='{}' | resolved {}/{}".format(tag, col, len(df), n0))
    return df

def real_grade(df, i, base):
    col = base + '_gold' if base + '_gold' in df.columns else (base + '_silver' if base + '_silver' in df.columns else base)
    if col not in df.columns: return None
    v = pd.to_numeric(df[col].iloc[i], errors='coerce')
    return None if pd.isna(v) else int(v)

# ============================================================
# GRADER  -- identical to check_grader_collapse.py (verified correct)
# ============================================================
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    gh = np.arange(grid_size, dtype=np.float32); gw = np.arange(grid_size, dtype=np.float32)
    grid = np.stack(np.meshgrid(gw, gh), axis=0).reshape([2, 1, grid_size, grid_size])
    def _1d(d, pos):
        omega = np.arange(d // 2, dtype=np.float32); omega /= d / 2.0; omega = 1.0 / 10000 ** omega
        out = np.outer(pos.reshape(-1), omega); return np.concatenate([np.sin(out), np.cos(out)], axis=1)
    emb = np.concatenate([_1d(embed_dim // 2, grid[0]), _1d(embed_dim // 2, grid[1])], axis=1)
    if cls_token: emb = np.concatenate([np.zeros([1, embed_dim]), emb], axis=0)
    return torch.from_numpy(emb).float()

class PatchEmbed(nn.Module):
    def __init__(self, s, p, c, e):
        super().__init__(); self.num_patches = (s // p) ** 2
        self.proj = nn.Conv2d(c, e, p, p)
    def forward(self, x): return self.proj(x).flatten(2).transpose(1, 2)

class Encoder(nn.Module):
    def __init__(self, cfg, nmt=6):
        super().__init__(); ed = cfg['model']['embed_dim']; self.nmt = nmt
        self.patch_embed = PatchEmbed(cfg['dataset']['image_size'], cfg['dataset']['patch_size'], cfg['dataset']['in_chans'], ed)
        P = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, ed)); self.morph_tokens = nn.Parameter(torch.zeros(1, nmt, ed))
        self.pos_embed = nn.Parameter(torch.zeros(1, P + 1, ed), requires_grad=False)
        l = nn.TransformerEncoderLayer(ed, cfg['model']['num_heads'], ed * 4, 0.1, 'gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(l, cfg['model']['depth'] // 3, enable_nested_tensor=False)
        self.encoder_norm = nn.LayerNorm(ed)
        self.pos_embed.data.copy_(get_2d_sincos_pos_embed(ed, int(P ** 0.5), True).unsqueeze(0))
    def forward(self, x):
        B = x.shape[0]; x = self.patch_embed(x) + self.pos_embed[:, 1:, :]
        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(B, -1, -1)
        m = self.morph_tokens.expand(B, -1, -1)
        x = self.encoder_norm(self.encoder(torch.cat((cls, m, x), 1)))
        return x[:, 1:1 + self.nmt, :]

class Grader(nn.Module):
    def __init__(self, cfg, nc, nmt=6):
        super().__init__(); self.encoder = Encoder(cfg, nmt); ed = cfg['model']['embed_dim']
        self.heads = nn.ModuleDict({t: nn.Linear(ed * nmt, n) for t, n in nc.items()})
    def forward(self, x):
        m = self.encoder(x); return {t: h(m.reshape(m.shape[0], -1)) for t, h in self.heads.items()}

def coral_decode(logits):
    return (torch.sigmoid(logits).cpu().numpy() > 0.5).sum(axis=1)

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

def load_grader(cfg, device):
    ck = torch.load(cfg['grader_checkpoint'], map_location='cpu')
    mcfg = ck.get('mae_config', {"dataset": {"image_size": 512, "patch_size": 16, "in_chans": 3},
                                 "model": {"embed_dim": 768, "depth": 36, "num_heads": 12}})
    nmt = ck.get('num_morph_tokens', 6); sd = ck['model'] if 'model' in ck else ck
    nc = {t: sd['heads.{}.weight'.format(t)].shape[0] for t in ['EXP', 'ICM', 'TE'] if 'heads.{}.weight'.format(t) in sd}
    model = Grader(mcfg, nc, nmt)
    result = model.load_state_dict(sd, strict=False)
    print("[GRADER] load: {}/{} params matched, {} missing".format(
        len(model.state_dict()) - len(result.missing_keys), len(model.state_dict()), len(result.missing_keys)))
    model.to(device).eval()
    return model, mcfg['dataset']['image_size']

def eval_tf(size):
    return A.Compose([A.Resize(size, size), A.CLAHE(p=1.0), A.Normalize(mean=IMNET_MEAN, std=IMNET_STD), ToTensorV2()])

@torch.no_grad()
def predict_grades(model, image_path, size, device):
    bgr = cv2.imread(image_path)
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else np.zeros((size, size, 3), np.uint8)
    x = eval_tf(size)(image=img)['image'].unsqueeze(0).to(device)
    logits = model(x)
    out = {}
    for t in logits:
        probs = coral_prob(logits[t])
        pred = int(probs.argmax(dim=1).item())
        out[t] = (pred, float(probs[0, pred].item()))
    return out

# ============================================================
# SANITY GATE  -- run BEFORE any VLM work; halts loudly on collapse instead
# of silently generating misleading reports.
# ============================================================
def sanity_gate(model, size, device, test_df, n_check):
    print("\n[SANITY GATE] predicting on {} real test_gold images before any VLM work...".format(n_check))
    idx = random.sample(range(len(test_df)), min(n_check, len(test_df)))
    preds = {t: [] for t in ['EXP', 'ICM', 'TE']}
    for i in idx:
        p = predict_grades(model, test_df['resolved_path'].iloc[i], size, device)
        for t in preds:
            if t in p: preds[t].append(p[t][0])
    collapsed = []
    for t, vals in preds.items():
        dist = Counter(vals)
        print("  {}: predicted distribution over {} images = {}".format(t, len(vals), dict(dist)))
        if len(dist) <= 1:
            collapsed.append(t)
    if collapsed:
        raise SystemExit(
            "[SANITY GATE FAILED] {} target(s) predicted only ONE class across {} real, different "
            "test images: {}. This means something is wrong RIGHT NOW (not fixed by the earlier "
            "training changes) -- do not proceed to generate VLM reports from this. Compare this "
            "output to check_grader_collapse.py's result on the SAME checkpoint: if that script "
            "still shows diversity but this one doesn't, the bug is in image loading/preprocessing "
            "in THIS script, not the model.".format(collapsed, n_check, collapsed))
    print("[SANITY GATE PASSED] real diversity confirmed -> proceeding.\n")

# ============================================================
# RETRIEVAL (train_silver database)
# ============================================================
class MAEEncoderFT(nn.Module):
    def __init__(self, cfg):
        super().__init__(); ed = cfg['model']['embed_dim']
        self.patch_embed = PatchEmbed(cfg['dataset']['image_size'], cfg['dataset']['patch_size'], cfg['dataset']['in_chans'], ed)
        P = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, ed)); self.pos_embed = nn.Parameter(torch.zeros(1, P + 1, ed), requires_grad=False)
        l = nn.TransformerEncoderLayer(ed, cfg['model']['num_heads'], ed * 4, 0.1, 'gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(l, cfg['model']['depth'] // 3, enable_nested_tensor=False)
        self.encoder_norm = nn.LayerNorm(ed)
        self.pos_embed.data.copy_(get_2d_sincos_pos_embed(ed, int(P ** 0.5), True).unsqueeze(0))
    def forward(self, x):
        x = self.patch_embed(x) + self.pos_embed[:, 1:, :]
        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1); x = self.encoder_norm(self.encoder(x))
        return x[:, 1:, :].mean(dim=1)

@torch.no_grad()
def embed_query(encoder, image_path, size, device):
    bgr = cv2.imread(image_path)
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else np.zeros((size, size, 3), np.uint8)
    x = eval_tf(size)(image=img)['image'].unsqueeze(0).to(device)
    emb = encoder(x).float().cpu().numpy().astype('float32')
    if faiss is not None: faiss.normalize_L2(emb)
    return emb

def load_retrieval_db(cfg, train_df):
    if faiss is None or not os.path.exists(cfg['retrieval_db_embeddings']):
        print("[RETRIEVAL] cached embeddings or faiss missing -> retrieval evidence skipped")
        return None
    emb = np.load(cfg['retrieval_db_embeddings'])
    n = min(len(emb), len(train_df))
    meta = [{'EXP': real_grade(train_df, i, 'EXP'), 'ICM': real_grade(train_df, i, 'ICM'), 'TE': real_grade(train_df, i, 'TE')}
           for i in range(n)]
    index = faiss.IndexFlatIP(emb.shape[1]); index.add(emb[:n])
    print("[RETRIEVAL] loaded real database: {} embedded cases".format(n))
    return {'index': index, 'paths': train_df['resolved_path'].tolist()[:n], 'meta': meta}

def retrieve_neighbors(db, query_emb, k=5):
    if db is None: return []
    q = query_emb.reshape(1, -1).astype('float32'); faiss.normalize_L2(q)
    sims, idxs = db['index'].search(q, k)
    return [{'path': db['paths'][idx], 'similarity': float(sim), **db['meta'][idx]}
           for idx, sim in zip(idxs[0], sims[0]) if idx >= 0]

# ============================================================
# EVIDENCE + REPORT
# ============================================================
class EvidenceBundle:
    def __init__(self, exp, icm, te, exp_conf, icm_conf, te_conf, retrieved_neighbors, true_exp=None, true_icm=None, true_te=None):
        self.exp, self.icm, self.te = exp, icm, te
        self.exp_conf, self.icm_conf, self.te_conf = exp_conf, icm_conf, te_conf
        self.retrieved_neighbors = retrieved_neighbors
        self.true_exp, self.true_icm, self.true_te = true_exp, true_icm, true_te

    def to_facts_dict(self):
        facts = {'EXP': self.exp, 'ICM': self.icm, 'TE': self.te,
                'EXP_confidence': round(self.exp_conf, 2), 'ICM_confidence': round(self.icm_conf, 2),
                'TE_confidence': round(self.te_conf, 2)}
        for i, n in enumerate(self.retrieved_neighbors):
            facts['neighbor_{}_similarity'.format(i)] = round(n['similarity'], 2)
        return facts

    def to_prompt_text(self):
        lines = ["Clinical model predictions (real, from a trained grader; Gardner notation):",
                "  Expansion: {} (confidence {:.2f})".format(format_grade('EXP', self.exp), self.exp_conf),
                "  ICM grade: {} (confidence {:.2f})".format(format_grade('ICM', self.icm), self.icm_conf),
                "  TE grade: {} (confidence {:.2f})".format(format_grade('TE', self.te), self.te_conf),
                "", "Most similar previously-graded embryos (real FAISS retrieval, cosine similarity):"]
        for i, n in enumerate(self.retrieved_neighbors):
            lines.append("  #{}: similarity {:.2f}, graded EXP={} ICM={} TE={}".format(
                i + 1, n['similarity'], format_grade('EXP', n['EXP']), format_grade('ICM', n['ICM']), format_grade('TE', n['TE'])))
        return "\n".join(lines)

def build_structured_prompt(ev):
    return ("You are assisting an embryologist. Using ONLY the real facts below (do not invent "
           "any grade not listed here), write a short grounded report using standard Gardner "
           "terminology. Include one sentence for EACH of Expansion, ICM, and TE. Cite sources "
           "in brackets, e.g. [grader], [retrieval#1].\n\n" + ev.to_prompt_text())

def ensure_all_grades_stated(report_text, ev):
    low = report_text.lower()
    if ('exp' in low or 'expansion' in low) and ('icm' in low) and ('te' in low or 'trophectoderm' in low):
        return report_text
    return report_text + ("\n\n[Structured summary -- guaranteed complete, Gardner notation]\n"
                         "Expansion: {} (confidence {:.2f}) [grader]\n"
                         "ICM: grade {} (confidence {:.2f}) [grader]\n"
                         "TE: grade {} (confidence {:.2f}) [grader]").format(
        format_grade('EXP', ev.exp), ev.exp_conf, format_grade('ICM', ev.icm), ev.icm_conf, format_grade('TE', ev.te), ev.te_conf)

NUMBER_RE = re.compile(r'-?\d+\.?\d*')
def verify_report(report_text, facts_dict, tol=0.05):
    stated = [float(x) for x in NUMBER_RE.findall(report_text)]
    real_vals = [v for v in facts_dict.values() if isinstance(v, (int, float)) and v is not None]
    flagged = [n for n in stated if not any(abs(n - rv) <= tol for rv in real_vals)]
    return {'verified': len(flagged) == 0, 'flagged_numbers': flagged}

def estimate_report_confidence(ev, verification):
    base = float(np.mean([ev.exp_conf, ev.icm_conf, ev.te_conf]))
    return float(np.clip(base - 0.15 * len(verification['flagged_numbers']), 0.0, 1.0))

@torch.no_grad()
def generate_grounded_report(model, processor, image_path, ev, max_new_tokens=200):
    from qwen_vl_utils import process_vision_info
    prompt = build_structured_prompt(ev)
    messages = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
    gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.batch_decode(gen_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]

def plot_report_panel(image_path, report_text, confidence, true_summary, out_path):
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={'width_ratios': [1, 1.3]})
    axes[0].imshow(img); axes[0].axis('off'); axes[0].set_title('Test embryo (real ground truth known)')
    axes[1].axis('off')
    axes[1].text(0.0, 0.98, "Grounded report (confidence: {:.2f}) | TRUE: {}".format(confidence, true_summary),
                fontsize=10, fontweight='bold', transform=axes[1].transAxes, va='top', wrap=True)
    axes[1].text(0.0, 0.85, report_text, fontsize=9, transform=axes[1].transAxes, va='top', wrap=True)
    plt.tight_layout(); plt.savefig(out_path, dpi=200, bbox_inches='tight'); plt.close()

# ============================================================
# MAIN
# ============================================================
def main():
    cfg = CFG
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(cfg['output_dir'], exist_ok=True)

    by_name, by_stem = build_image_index(cfg['image_root'])
    train_df = build_labeled_df(cfg['train_csv'], by_name, by_stem, "train")
    test_df = build_labeled_df(cfg['test_csv'], by_name, by_stem, "test")
    print("[WHOLE DATASET] train_silver: {} | test_gold: {}\n".format(len(train_df), len(test_df)))

    if not os.path.exists(cfg['grader_checkpoint']):
        raise FileNotFoundError("Run embryo_grounded_morph_grader_v2.py first.")
    grader, size = load_grader(cfg, device)

    # HARD GATE: stop here if collapse reproduces, before any Qwen2-VL cost is spent.
    sanity_gate(grader, size, device, test_df, cfg['sanity_check_n'])

    mae_encoder, mae_size, db = None, size, None
    if os.path.exists(cfg['mae_checkpoint']):
        mae_ck = torch.load(cfg['mae_checkpoint'], map_location='cpu')
        mcfg = mae_ck.get('config', {"dataset": {"image_size": 512, "patch_size": 16, "in_chans": 3},
                                     "model": {"embed_dim": 768, "depth": 36, "num_heads": 12}})
        mae_encoder = MAEEncoderFT(mcfg); mae_encoder.load_state_dict(mae_ck['model'], strict=False)
        mae_encoder.to(device).eval(); mae_size = mcfg['dataset']['image_size']
        db = load_retrieval_db(cfg, train_df)

    # ---- predict for the WHOLE real test_gold set (fast, real accuracy computable) ----
    all_rows = []
    for i in tqdm(range(len(test_df)), desc="predicting on whole test_gold"):
        path = test_df['resolved_path'].iloc[i]
        preds = predict_grades(grader, path, size, device)
        row = {'path': path}
        for t in ['EXP', 'ICM', 'TE']:
            true_v = real_grade(test_df, i, t)
            row[t] = preds.get(t, (None, None))[0]; row[t + '_conf'] = preds.get(t, (None, None))[1]
            row['true_' + t] = true_v
        all_rows.append(row)
    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(os.path.join(cfg['output_dir'], 'test_predictions_all.csv'), index=False)
    for t in ['EXP', 'ICM', 'TE']:
        valid = all_df.dropna(subset=[t, 'true_' + t])
        acc = (valid[t] == valid['true_' + t]).mean() if len(valid) else float('nan')
        print("[REAL ACCURACY] {}: {:.3f} over {} test images".format(t, acc, len(valid)))

    # ---- Qwen2-VL + previously fine-tuned LoRA (load, don't retrain) ----
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(cfg['qwen_model_id'], torch_dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(cfg['qwen_model_id'])
    if os.path.exists(cfg['lora_adapter_path']):
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, cfg['lora_adapter_path'])
        print("[LoRA] loaded fine-tuned adapter -> {}".format(cfg['lora_adapter_path']))
    else:
        model = base_model
        print("[LoRA] WARNING: no adapter found -> using base Qwen2-VL (not fine-tuned on your real facts).")
    model.eval()

    demo_idx = random.sample(range(len(test_df)), min(cfg['num_demo_reports'], len(test_df)))
    for k, i in enumerate(demo_idx):
        path = test_df['resolved_path'].iloc[i]
        preds = predict_grades(grader, path, size, device)
        exp, exp_conf = preds['EXP']; icm, icm_conf = preds['ICM']; te, te_conf = preds['TE']
        true_exp, true_icm, true_te = real_grade(test_df, i, 'EXP'), real_grade(test_df, i, 'ICM'), real_grade(test_df, i, 'TE')
        neighbors = []
        if db is not None:
            qemb = embed_query(mae_encoder, path, mae_size, device)
            neighbors = retrieve_neighbors(db, qemb, k=cfg['top_k_retrieval'])
        ev = EvidenceBundle(exp, icm, te, exp_conf, icm_conf, te_conf, neighbors, true_exp, true_icm, true_te)

        # SIMPLIFIED: dropped the independent Qwen morphology-detection call --
        # not needed to verify the SFT notation fix, and it doubled generation
        # cost per case for no benefit toward that specific question.
        report = generate_grounded_report(model, processor, path, ev)
        report = ensure_all_grades_stated(report, ev)
        facts = ev.to_facts_dict()
        verification = verify_report(report, facts)
        confidence = estimate_report_confidence(ev, verification)

        true_summary = "EXP={} ICM={} TE={}".format(format_grade('EXP', true_exp), format_grade('ICM', true_icm), format_grade('TE', true_te))
        print("=" * 70)
        print("Test image {}/{} | TRUE: {}".format(k + 1, len(demo_idx), true_summary))
        print("Predicted: EXP={} (conf {:.2f}) ICM={} (conf {:.2f}) TE={} (conf {:.2f})".format(
            format_grade('EXP', exp), exp_conf, format_grade('ICM', icm), icm_conf, format_grade('TE', te), te_conf))
        print(report)
        # always print the verification result (not just on failure) -- this is
        # the direct signal for whether the SFT notation fix actually worked
        print("[HALLUCINATION CHECK] verified={} | flagged numbers: {}".format(
            verification['verified'], verification['flagged_numbers']))

        plot_report_panel(path, report, confidence, true_summary, os.path.join(cfg['output_dir'], 'test_report_{}.png'.format(k + 1)))

    print("\n[COMPLETE] Real accuracy over the whole test set + {} demo reports -> {}".format(len(demo_idx), cfg['output_dir']))

if __name__ == '__main__':
    main()
