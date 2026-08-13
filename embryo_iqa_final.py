import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore")

try:
    import pyiqa
except ImportError:
    raise ImportError("pyiqa is required. Install it first:\n    pip install pyiqa")

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
EXCLUDE_DIR_NAMES = {'.ipynb_checkpoints', '__pycache__', '.git', 'figures'}

# ============================================================
# SAME SINGLE SHARED PROJECT FOLDER as embryo_data_pipeline_final.py --
# everything accumulates in one place, no separate embryo_iqa_outputs/.
# ============================================================
OUTPUT_DIR = "./embryo_project"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# EXPLICIT PATH SETUP (Downloads/archive structure)
# ============================================================
INPUT_DIR  = "."
IMAGE_DIR  = "./Downloads/archive/Images/Images"
TRAIN_CSV  = "./Downloads/archive/Gardner_train_silver.csv"
TEST_CSV   = "./Downloads/archive/Gardner_test_gold_onlyGardnerScores.csv"

def locate_input_dir(preferred, train_path):
    if os.path.exists(train_path):
        return preferred
    for root in [preferred, os.getcwd(), os.path.expanduser("~")]:
        if root and os.path.isdir(root):
            for dp, _, files in os.walk(root):
                if os.path.basename(train_path) in files:
                    return dp
    return preferred

INPUT_DIR = locate_input_dir(INPUT_DIR, TRAIN_CSV)
train_csv_path = TRAIN_CSV if os.path.exists(TRAIN_CSV) else os.path.join(INPUT_DIR, os.path.basename(TRAIN_CSV))
test_csv_path = TEST_CSV if os.path.exists(TEST_CSV) else os.path.join(INPUT_DIR, os.path.basename(TEST_CSV))

print("[PATHS] INPUT_DIR  : {}".format(INPUT_DIR))
print("[PATHS] Train CSV  : {} -> Exists: {}".format(train_csv_path, os.path.exists(train_csv_path)))
print("[PATHS] Test  CSV  : {} -> Exists: {}".format(test_csv_path, os.path.exists(test_csv_path)))
print("[PATHS] IMAGE_DIR  : {} -> Exists: {}".format(IMAGE_DIR, os.path.isdir(IMAGE_DIR) if IMAGE_DIR else False))
print("[PATHS] OUTPUT_DIR : {}  (single shared project folder)\n".format(OUTPUT_DIR))

if not os.path.exists(train_csv_path):
    raise FileNotFoundError("Train CSV not found. Please check paths for {}".format(TRAIN_CSV))

CONFIG = {
    "image_size": 512,               # restored to full resolution (256 was a GPU-OOM workaround; the real
                                     # cause was zombie processes holding VRAM, not this script -- see note below)
    "batch_size": 16,
    "num_workers": 2,
    "metrics": ["brisque", "niqe"],
    "low_quality_percentile": 95,
    "output_dir": OUTPUT_DIR,
}

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
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIR_NAMES and not d.endswith('_outputs') and d != os.path.basename(OUTPUT_DIR)]
        for f in files:
            if f.lower().endswith(IMAGE_EXTS):
                full = os.path.join(dp, f)
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
        hits = sum(1 for v in probe[c].tolist() if resolve_image_path(v, by_name, by_stem) is not None)
        rate = hits / n
        if rate > best_rate:
            best_col, best_rate = c, rate
    return best_col, best_rate

def collect_all_images():
    train_df = read_csv_smart(train_csv_path)
    test_df = read_csv_smart(test_csv_path) if os.path.exists(test_csv_path) else train_df.copy()
    print("[WHOLE DATASET] train_silver rows: {} | test_gold rows: {} (no subsampling applied anywhere)".format(
        len(train_df), len(test_df)))

    if IMAGE_DIR and os.path.isdir(IMAGE_DIR):
        by_name, by_stem = build_image_index(IMAGE_DIR)
    else:
        by_name, by_stem = build_image_index(INPUT_DIR)
        if len(by_name) == 0:
            for extra in [os.getcwd(), os.path.expanduser("~")]:
                bn, bs = build_image_index(extra)
                by_name.update(bn); by_stem.update(bs)
    print("[IMAGES] indexed {} image files\n".format(len(set(by_name.values()))))

    paths, names, seen = [], [], set()
    for df, tag in [(train_df, "train"), (test_df, "test")]:
        col, rate = detect_image_column(df, by_name, by_stem)
        resolved = 0
        if col is not None:
            for v in df[col].tolist():
                rp = resolve_image_path(v, by_name, by_stem)
                if rp:
                    resolved += 1
                    if rp not in seen:
                        seen.add(rp); paths.append(rp); names.append(os.path.basename(rp))
        print("[IMAGES] {}: column='{}' | resolved {}/{} ({:.1f}%)".format(
            tag, col, resolved, len(df), 100.0 * resolved / max(len(df), 1)))

    if paths:
        located = os.path.dirname(paths[0])
        print("[IMAGES] embryo images located in: {}  (unique images to score: {})\n".format(located, len(paths)))
    else:
        print("[IMAGES] WARNING: no CSV image names resolved. Set IMAGE_DIR to the real embryo .png folder.\n")
    return paths, names

# ==========================================
# IQA DATASET
# ==========================================
class EmbryoIQADataset(Dataset):
    def __init__(self, paths, names, size):
        self.paths = paths
        self.names = names
        self.tf = A.Compose([
            A.Resize(size, size),
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        bgr = cv2.imread(self.paths[idx])
        if bgr is None:
            img = np.zeros((self.tf[0].height, self.tf[0].width, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = self.tf(image=img)['image']
        return img, self.names[idx]

# ==========================================
# COMPUTE NO-REFERENCE IQA (BRISQUE + NIQE)
# GPU-first, with a genuine (not blanket) OOM fallback to CPU. The earlier
# forced-CPU version was a workaround for a zombie-process OOM, not a real
# memory limitation of this script -- see the GPU memory note printed below.
# ==========================================
def compute_iqa(paths, names, config):
    print("[GPU CHECK] free/total VRAM before starting:")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print("  {:.2f} GiB free / {:.2f} GiB total".format(free / 1e9, total / 1e9))
        if free / total < 0.15:
            print("  WARNING: <15% VRAM free. If this OOMs, run `nvidia-smi`, kill stale kernel")
            print("  processes, THEN retry on GPU rather than assuming this script needs less memory.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Computing no-reference IQA (device: {})...".format(device))

    metrics = {}
    for m in config['metrics']:
        try:
            metrics[m] = pyiqa.create_metric(m, device=device)
        except RuntimeError as e:
            print("  [FALLBACK] could not init '{}' on {} ({}) -> retrying on CPU".format(m, device, e))
            device = torch.device('cpu')
            metrics[m] = pyiqa.create_metric(m, device=device)
        print("  loaded metric '{}' (lower_better={})".format(m, getattr(metrics[m], 'lower_better', 'NA')))

    ds = EmbryoIQADataset(paths, names, config['image_size'])
    loader = DataLoader(ds, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    records = {m: [] for m in config['metrics']}
    all_names = []
    with torch.no_grad():
        for imgs, nm in tqdm(loader, desc="Scoring images (whole dataset)"):
            try:
                imgs_dev = imgs.to(device)
                for m in config['metrics']:
                    val = metrics[m](imgs_dev).detach().flatten().cpu().numpy()
                    records[m].extend(np.atleast_1d(val).tolist())
            except torch.cuda.OutOfMemoryError:
                print("\n  [OOM] genuine GPU out-of-memory mid-run -> falling back to CPU for remaining batches.")
                torch.cuda.empty_cache()
                device = torch.device('cpu')
                for m in config['metrics']:
                    metrics[m] = pyiqa.create_metric(m, device=device)
                imgs_dev = imgs.to(device)
                for m in config['metrics']:
                    val = metrics[m](imgs_dev).detach().flatten().cpu().numpy()
                    records[m].extend(np.atleast_1d(val).tolist())
            all_names.extend(list(nm))

    df = pd.DataFrame({'image': all_names})
    for m in config['metrics']:
        df[m] = records[m]
    return df

# ==========================================
# VISUALIZATION
# ==========================================
def load_rgb(path, size):
    bgr = cv2.imread(path)
    if bgr is None:
        return np.zeros((size, size, 3), dtype=np.uint8)
    return cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (size, size))

def generate_iqa_visualizations(df, name_to_path, config):
    # SINGLE flat project folder -- no 'embryo_iqa_outputs/' subfolder
    out = config['output_dir']
    sns.set_theme(style="whitegrid")

    for m in config['metrics']:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[m], kde=True, bins=30, color='teal')
        plt.title("Embryo No-Reference IQA: {} distribution, N={} (lower = better)".format(m.upper(), len(df)))
        plt.xlabel("{} score".format(m.upper())); plt.ylabel("Number of images")
        plt.savefig(os.path.join(out, "iqa_{}_histogram.png".format(m)), dpi=300, bbox_inches='tight')
        plt.close()

    if len(config['metrics']) >= 2:
        a, b = config['metrics'][0], config['metrics'][1]
        r = np.corrcoef(df[a], df[b])[0, 1]
        plt.figure(figsize=(7, 6))
        plt.scatter(df[a], df[b], alpha=0.5, c='slateblue', edgecolors='none', s=18)
        plt.xlabel("{} (lower=better)".format(a.upper())); plt.ylabel("{} (lower=better)".format(b.upper()))
        plt.title("IQA metric agreement: {} vs {}  (r = {:.2f})".format(a.upper(), b.upper(), r))
        plt.savefig(os.path.join(out, "iqa_metric_agreement.png"), dpi=300, bbox_inches='tight')
        plt.close()

    key = config['metrics'][0]
    n_show = min(4, len(df))
    if n_show >= 1:
        d = df.sort_values(key)
        best = d.head(n_show)
        worst = d.tail(n_show).iloc[::-1]
        fig, axes = plt.subplots(2, n_show, figsize=(3.4 * n_show, 7))
        if n_show == 1:
            axes = axes.reshape(2, 1)
        for j, (_, row) in enumerate(best.iterrows()):
            axes[0, j].imshow(load_rgb(name_to_path.get(row['image'], ''), config['image_size']))
            axes[0, j].set_title("{} {:.1f}".format(key.upper(), row[key]), fontsize=10)
            axes[0, j].axis('off')
        for j, (_, row) in enumerate(worst.iterrows()):
            axes[1, j].imshow(load_rgb(name_to_path.get(row['image'], ''), config['image_size']))
            axes[1, j].set_title("{} {:.1f}".format(key.upper(), row[key]), fontsize=10)
            axes[1, j].axis('off')
        fig.text(0.5, 0.98, "Best quality (low {})".format(key.upper()), ha='center', fontsize=12)
        fig.text(0.5, 0.50, "Worst quality (high {})".format(key.upper()), ha='center', fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(out, "iqa_best_vs_worst.png"), dpi=200, bbox_inches='tight')
        plt.close()

    print("[SUCCESS] IQA figures saved to: {}".format(out))

# ==========================================
# EXECUTION
# ==========================================
if __name__ == '__main__':
    paths, names = collect_all_images()
    if not paths:
        raise SystemExit("No images resolved - fix IMAGE_DIR before running IQA.")

    name_to_path = {n: p for n, p in zip(names, paths)}

    scores = compute_iqa(paths, names, CONFIG)

    # SINGLE flat project folder -- everything lands directly in OUTPUT_DIR
    scores_csv = os.path.join(CONFIG['output_dir'], "embryo_iqa_scores.csv")
    scores.to_csv(scores_csv, index=False)

    key = CONFIG['metrics'][0]
    thr = np.percentile(scores[key], CONFIG['low_quality_percentile'])
    flagged = scores[scores[key] >= thr].sort_values(key, ascending=False)
    flagged_csv = os.path.join(CONFIG['output_dir'], "embryo_iqa_low_quality.csv")
    flagged.to_csv(flagged_csv, index=False)

    generate_iqa_visualizations(scores, name_to_path, CONFIG)

    print("\n--- Embryo No-Reference IQA Summary (WHOLE dataset) ---")
    print("Images scored           : {}".format(len(scores)))
    for m in CONFIG['metrics']:
        print("  {:<8} lower=better | mean {:.2f} | median {:.2f} | min {:.2f} | max {:.2f}".format(
            m.upper(), scores[m].mean(), scores[m].median(), scores[m].min(), scores[m].max()))
    print("Low-quality flagged     : {} images (>= {}th pct {} = {:.2f}) -> {}".format(
        len(flagged), CONFIG['low_quality_percentile'], key.upper(), thr, os.path.basename(flagged_csv)))
    print("Per-image scores saved  : {}".format(os.path.basename(scores_csv)))
    print("These are real, training-free NR-IQA measurements (BRISQUE/NIQE), citable as such.")
    print("\nAll outputs saved directly in: {}".format(os.path.abspath(CONFIG['output_dir'])))

    try:
        from IPython.display import Image, display
        for fig_path in sorted(glob.glob(os.path.join(CONFIG['output_dir'], "iqa_*.png"))):
            print("Displaying: {}".format(os.path.basename(fig_path)))
            display(Image(filename=fig_path, width=780))
    except Exception:
        pass
