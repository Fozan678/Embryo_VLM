import os, math, random, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2, albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
import warnings; warnings.filterwarnings("ignore")

try:
    import faiss
except ImportError:
    raise ImportError(
        "faiss is not installed. Run in a notebook cell first:\n"
        "  import sys; !{sys.executable} -m pip install faiss-gpu-cu12\n"
        "  (fallback: !{sys.executable} -m pip install faiss-cpu)")

torch.backends.cudnn.benchmark = True
IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
EXCLUDE_DIR_NAMES = {'.ipynb_checkpoints', '__pycache__', '.git', 'processed_data',
                     'embryo_iqa_outputs', 'mae_outputs', 'morphology_outputs', 'probe_outputs',
                     'seg_outputs', 'grader_outputs', 'grader_v2_outputs', 'grounded_morph_outputs',
                     'grounded_morph_v2_outputs', 'grounded_morph_probe_outputs', 'relational_graph_outputs',
                     'graph_transformer_outputs', 'clinical_multitask_outputs', 'clinical_multitask_v2_outputs',
                     'uncertainty_outputs', 'uncertainty_framework_v2_outputs', 'vlm_outputs',
                     'explainability_outputs', 'retrieval_outputs', 'figures', 'embryo_project'}   # NEW: consolidated project folder
IMNET_MEAN = [0.485, 0.456, 0.406]; IMNET_STD = [0.229, 0.224, 0.225]

INPUT_DIR = "."; IMAGE_DIR = "./Downloads/archive/Images/Images"
TRAIN_CSV = "Gardner_train_silver.csv"; TEST_CSV = "Gardner_test_gold_onlyGardnerScores.csv"

CFG = {
    "mae_checkpoint": "./embryo_project/mae/checkpoints/mae_best.pth",   # consolidated MAE location
    "output_dir": "./embryo_project/retrieval",   # consolidated project folder
    "top_k": 5,
    "num_demo_queries": 6,
    "batch_size": 16, "num_workers": 2, "seed": 42,
    "cache_embeddings": True,
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

def real_grades(df):
    """Real EXP/ICM/TE grades (raw values, 'ND'/missing -> None) for metadata display."""
    out = []
    ecol, icol, tcol = pick_col(df, 'EXP'), pick_col(df, 'ICM'), pick_col(df, 'TE')
    for i in range(len(df)):
        def g(col):
            if col is None:
                return None
            v = pd.to_numeric(df[col].iloc[i], errors='coerce')
            return None if pd.isna(v) else int(v)
        out.append({'EXP': g(ecol), 'ICM': g(icol), 'TE': g(tcol)})
    return out

# ============================================================
# 1. EMBEDDING EXTRACTION  (real trained MAE encoder, mean-pooled patch
# tokens, L2-normalized -> cosine similarity via inner-product FAISS index)
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
        return x[:, 1:, :].mean(dim=1)   # mean-pooled patch embedding, same representation the grader used

def eval_tf(size):
    return A.Compose([A.Resize(size, size), A.CLAHE(p=1.0),
                      A.Normalize(mean=IMNET_MEAN, std=IMNET_STD), ToTensorV2()])

class ImgDataset(Dataset):
    def __init__(self, paths, size):
        self.paths = list(paths); self.tf = eval_tf(size); self.size = size
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        bgr = cv2.imread(self.paths[i])
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else np.zeros((self.size, self.size, 3), np.uint8)
        return self.tf(image=img)['image']

@torch.no_grad()
def extract_embeddings(encoder, paths, size, cfg, device, cache_path):
    if cfg['cache_embeddings'] and os.path.exists(cache_path):
        print("[EMBED] loading cached embeddings -> {}".format(cache_path))
        return np.load(cache_path)
    loader = DataLoader(ImgDataset(paths, size), batch_size=cfg['batch_size'], shuffle=False,
                        num_workers=cfg['num_workers'], pin_memory=True)
    feats = []
    for imgs in tqdm(loader, desc="extracting embeddings"):
        imgs = imgs.to(device)
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            f = encoder(imgs)
        feats.append(f.float().cpu().numpy())
    emb = np.concatenate(feats, 0).astype('float32')
    faiss.normalize_L2(emb)   # L2-normalize -> inner product = cosine similarity
    if cfg['cache_embeddings']:
        np.save(cache_path, emb)
    return emb

# ============================================================
# 2. FAISS INDEX  +  6. METADATA  +  7. EMBEDDING SEARCH API
# ============================================================
class EmbryoRetrievalEngine:
    """Clean, reusable search API over a FAISS index of embryo embeddings."""
    def __init__(self, embeddings, metadata_records, image_paths):
        assert len(embeddings) == len(metadata_records) == len(image_paths)
        self.dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)     # inner product on normalized vectors = cosine similarity
        self.index.add(embeddings)
        self.metadata = metadata_records
        self.paths = image_paths

    def search(self, query_embedding, k=5):
        """query_embedding: (dim,) or (1,dim) float32, L2-normalized.
        Returns list of dicts: {rank, path, similarity, metadata}."""
        q = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(q)
        sims, idxs = self.index.search(q, k)
        results = []
        for rank, (idx, sim) in enumerate(zip(idxs[0], sims[0])):
            if idx < 0:
                continue
            results.append({'rank': rank + 1, 'path': self.paths[idx], 'similarity': float(sim),
                           'metadata': self.metadata[idx]})
        return results

    def search_by_index(self, i, k=5):
        return self.search(self._get_raw(i), k=k)

    def _get_raw(self, i):
        v = np.zeros(self.dim, dtype='float32')
        self.index.reconstruct(i, v)
        return v

    def size(self):
        return self.index.ntotal

# ---- optional: wrap the engine as an actual HTTP API (not auto-started) ----
def build_flask_api(engine, encoder, size, device, host="127.0.0.1", port=5000):
    """Optional literal web API. Call this yourself in a separate cell/process
    if you want HTTP access -- it is NOT started automatically by this script.
        from embryo_faiss_retrieval import build_flask_api
        build_flask_api(engine, encoder, size, device)   # then POST an image to /search
    """
    from flask import Flask, request, jsonify
    app = Flask(__name__)

    @app.route("/search", methods=["POST"])
    def search_route():
        file = request.files['image']
        arr = np.frombuffer(file.read(), np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = eval_tf(size)(image=rgb)['image'].unsqueeze(0).to(device)
        with torch.no_grad():
            emb = encoder(tensor).float().cpu().numpy()
        k = int(request.args.get('k', 5))
        results = engine.search(emb, k=k)
        return jsonify([{'rank': r['rank'], 'similarity': r['similarity'], 'metadata': r['metadata']} for r in results])

    print("Starting embedding search API at http://{}:{}/search (POST an image file, field name 'image')".format(host, port))
    app.run(host=host, port=port)

# ============================================================
# 4-5-8. RETRIEVAL DISPLAY + SIMILARITY VISUALIZATION (publication-quality)
# ============================================================
def load_disp(path, size=256):
    bgr = cv2.imread(path)
    if bgr is None:
        return np.zeros((size, size, 3), dtype=np.uint8)
    return cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (size, size))

def plot_retrieval_panel(query_path, query_meta, results, out_path):
    n = len(results)
    fig = plt.figure(figsize=(3.2 * (n + 1), 3.6))
    gs = gridspec.GridSpec(1, n + 1, wspace=0.15)

    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(load_disp(query_path)); ax0.axis('off')
    ax0.set_title("QUERY\nEXP={} ICM={} TE={}".format(query_meta['EXP'], query_meta['ICM'], query_meta['TE']),
                 fontsize=9, fontweight='bold')
    for spine in ax0.spines.values():
        pass

    for i, r in enumerate(results):
        ax = fig.add_subplot(gs[i + 1])
        ax.imshow(load_disp(r['path'])); ax.axis('off')
        m = r['metadata']
        ax.set_title("#{} sim={:.3f}\nEXP={} ICM={} TE={}".format(r['rank'], r['similarity'], m['EXP'], m['ICM'], m['TE']),
                    fontsize=8.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches='tight'); plt.close()

def plot_embedding_space(db_emb, db_meta, query_idx_list, query_emb, results_list, out_path):
    """Global 2D (PCA) view of the database, colored by real EXP grade, with
    query points and their retrieved neighbors highlighted."""
    proj = PCA(n_components=2).fit_transform(db_emb)
    exp_vals = np.array([m['EXP'] if m['EXP'] is not None else -1 for m in db_meta])

    plt.figure(figsize=(8, 7))
    sc = plt.scatter(proj[:, 0], proj[:, 1], c=exp_vals, cmap='viridis', s=18, alpha=0.5)
    plt.colorbar(sc, label='EXP grade (database)')

    q_proj = PCA(n_components=2).fit(db_emb).transform(query_emb)
    for qi, (qp, results) in enumerate(zip(q_proj, results_list)):
        plt.scatter(*qp, marker='*', s=260, color='red', edgecolor='k', zorder=5,
                   label='Query' if qi == 0 else None)
        for r in results:
            idx = r.get('_db_index')
            if idx is not None:
                plt.plot([qp[0], proj[idx, 0]], [qp[1], proj[idx, 1]], color='red', alpha=0.4, lw=1, zorder=4)
    plt.title("Embedding space (PCA) -- database colored by real EXP grade\nred stars = queries, lines = retrieved top-{} neighbors".format(CFG['top_k']))
    plt.xlabel('PC 1'); plt.ylabel('PC 2'); plt.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=250, bbox_inches='tight'); plt.close()

def plot_similarity_distribution(all_top1_sims, out_path):
    plt.figure(figsize=(7, 5))
    plt.hist(all_top1_sims, bins=30, color='slateblue', alpha=0.8)
    plt.xlabel('Top-1 cosine similarity (query vs nearest database match)')
    plt.ylabel('Number of queries')
    plt.title('Retrieval quality: top-1 similarity distribution across all test queries')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=220, bbox_inches='tight'); plt.close()

# ============================================================
# MAIN
# ============================================================
def main():
    cfg = CFG
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(cfg['output_dir'], exist_ok=True)

    by_name, by_stem = build_image_index(image_root)
    if len(by_name) == 0:
        for extra in [INPUT_DIR, os.path.expanduser("~")]:
            bn, bs = build_image_index(extra); by_name.update(bn); by_stem.update(bs)
    print("[IMAGES] indexed {} files\n".format(len(set(by_name.values()))))

    # Database = train_silver (never seen as a query). Queries = test_gold
    # (never in the database) -> guarantees the top match is never the query itself.
    db_df = build_labeled_df(train_csv_path, by_name, by_stem, "database (train_silver)")
    query_df = build_labeled_df(test_csv_path, by_name, by_stem, "queries (test_gold)")
    print("[WHOLE DATASET] database (train_silver) rows: {} | queries (test_gold) rows: {} "
          "(retrieval runs for EVERY query, no subsampling)\n".format(len(db_df), len(query_df)))

    db_meta = real_grades(db_df)
    query_meta = real_grades(query_df)

    if not os.path.exists(cfg['mae_checkpoint']):
        raise FileNotFoundError("MAE checkpoint not found at {}. Run MAE pretraining first.".format(cfg['mae_checkpoint']))
    ckpt = torch.load(cfg['mae_checkpoint'], map_location='cpu')
    mae_cfg = ckpt.get('config', MAE_CONFIG_DEFAULT)
    size = mae_cfg['dataset']['image_size']

    encoder = MAEEncoderFT(mae_cfg)
    encoder.load_state_dict(ckpt['model'], strict=False)
    encoder.to(device).eval()
    print("[ENCODER] loaded MAE encoder | embed_dim={} | real layers={}\n".format(
        mae_cfg['model']['embed_dim'], mae_cfg['model']['depth'] // 3))

    db_emb = extract_embeddings(encoder, db_df['resolved_path'].tolist(), size, cfg, device,
                                os.path.join(cfg['output_dir'], 'db_embeddings.npy'))
    query_emb = extract_embeddings(encoder, query_df['resolved_path'].tolist(), size, cfg, device,
                                   os.path.join(cfg['output_dir'], 'query_embeddings.npy'))
    print("\n[FAISS] database size = {} | embedding dim = {}".format(db_emb.shape[0], db_emb.shape[1]))

    # 2. Build FAISS index + 7. search API
    engine = EmbryoRetrievalEngine(db_emb, db_meta, db_df['resolved_path'].tolist())
    print("[FAISS] index built | ntotal = {}\n".format(engine.size()))

    # 3-4. Retrieve top-k for every query; save a full results table
    all_rows = []
    all_top1_sims = []
    demo_results, demo_query_idx = [], []
    rng = random.Random(cfg['seed'])
    demo_ids = rng.sample(range(len(query_df)), min(cfg['num_demo_queries'], len(query_df)))

    for qi in tqdm(range(len(query_df)), desc="retrieving top-{} for every query".format(cfg['top_k'])):
        results = engine.search(query_emb[qi], k=cfg['top_k'])
        for r in results:
            r['_db_index'] = db_df.index[db_df['resolved_path'] == r['path']].tolist()
            r['_db_index'] = r['_db_index'][0] if r['_db_index'] else None
            all_rows.append({'query_path': query_df['resolved_path'].iloc[qi], 'query_EXP': query_meta[qi]['EXP'],
                            'query_ICM': query_meta[qi]['ICM'], 'query_TE': query_meta[qi]['TE'],
                            'rank': r['rank'], 'match_path': r['path'], 'similarity': r['similarity'],
                            'match_EXP': r['metadata']['EXP'], 'match_ICM': r['metadata']['ICM'], 'match_TE': r['metadata']['TE']})
        if results:
            all_top1_sims.append(results[0]['similarity'])
        if qi in demo_ids:
            demo_results.append(results); demo_query_idx.append(qi)

    pd.DataFrame(all_rows).to_csv(os.path.join(cfg['output_dir'], 'retrieval_results_full.csv'), index=False)
    print("\n[SUCCESS] Full retrieval table ({} rows) saved.".format(len(all_rows)))

    # 4-8. Per-query retrieval panels (publication-quality)
    for k, qi in enumerate(demo_query_idx):
        out_path = os.path.join(cfg['output_dir'], 'retrieval_panel_{}.png'.format(k + 1))
        plot_retrieval_panel(query_df['resolved_path'].iloc[qi], query_meta[qi], demo_results[k], out_path)
        print("[Artifacts] {}".format(out_path))

    # 5. Similarity visualizations
    plot_similarity_distribution(all_top1_sims, os.path.join(cfg['output_dir'], 'similarity_distribution.png'))
    plot_embedding_space(db_emb, db_meta, demo_query_idx, query_emb[demo_query_idx],
                        demo_results, os.path.join(cfg['output_dir'], 'embedding_space_pca.png'))

    print("\n[COMPLETE] Retrieval system artifacts in: {}".format(cfg['output_dir']))
    print("Mean top-1 similarity across all {} queries: {:.3f}".format(len(all_top1_sims), np.mean(all_top1_sims)))
    print("(This measures visual/perceptual retrieval quality, not grading accuracy --")
    print(" a high similarity score does not guarantee the retrieved embryo shares the same real grade.)")

    try:
        from IPython.display import Image, display
        for k in range(len(demo_query_idx)):
            display(Image(filename=os.path.join(cfg['output_dir'], 'retrieval_panel_{}.png'.format(k + 1)), width=900))
        display(Image(filename=os.path.join(cfg['output_dir'], 'similarity_distribution.png'), width=600))
        display(Image(filename=os.path.join(cfg['output_dir'], 'embedding_space_pca.png'), width=700))
    except Exception:
        pass

    return engine, encoder, size, device   # handy if calling build_flask_api(...) afterward in the same session

if __name__ == '__main__':
    main()
