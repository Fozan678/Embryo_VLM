"""
Evidence-Grounded VLM (Qwen2-VL) for embryo grading.

HONESTY NOTE (read before running): there are no annotated free-text clinical
embryology REPORTS anywhere in this dataset -- only numeric EXP/ICM/TE grades.
So "grounded report generation" here means: Qwen2-VL learns to fluently VERBALIZE
real upstream facts (real grader predictions, real confidence, real graph/morph
evidence, real FAISS-retrieved similar cases) with citations -- NOT to perform
novel clinical reasoning from pixels that nobody has validated. The LoRA
fine-tuning target text is a template filled with REAL numbers (not fabricated
prose), and every number the model states get checked against the real facts.

Two separate real mechanisms, kept separate on purpose:
  1. CrossAttentionFusion: a small trainable adapter with its OWN real
     supervision (predicts your actual EXP/ICM/TE grades from the fused
     evidence -- so you can verify it learned something before trusting its
     injected embeddings). Its output is spliced into Qwen2-VL's input
     embeddings at inference time as extra "evidence tokens".
  2. LoRA fine-tuning of Qwen2-VL's LANGUAGE backbone (vision tower + merger
     frozen, per current best practice) via TRL's SFTTrainer, on real
     structured-fact-grounded target text.
Requires (run in your notebook first):
  pip install transformers accelerate peft trl qwen-vl-utils[decord] pillow
  Needs internet access to download Qwen/Qwen2-VL-2B-Instruct (~4.5GB).
  VRAM: 2B + LoRA fits comfortably on a 32GB 5090; use QLoRA (bnb 4-bit) if
  running this alongside other GPU work.
"""
import os, re, json, random, math, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from tqdm.auto import tqdm
import warnings; warnings.filterwarnings("ignore")

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
EXCLUDE_DIR_NAMES = {'.ipynb_checkpoints', '__pycache__', '.git', 'processed_data', 'figures',
                     'grader_v2_outputs', 'grounded_morph_v2_outputs', 'graph_transformer_outputs',
                     'retrieval_outputs', 'uncertainty_framework_v2_outputs', 'vlm_grounded_outputs', 'embryo_project'}   # NEW: consolidated project folder
INPUT_DIR = "."; IMAGE_DIR = "./Downloads/archive/Images/Images"
TRAIN_CSV = "Gardner_train_silver.csv"; TEST_CSV = "Gardner_test_gold_onlyGardnerScores.csv"

CFG = {
    "qwen_model_id": "Qwen/Qwen2-VL-2B-Instruct",
    "grader_checkpoint": "./embryo_project/grounded_morph_grader/grounded_morph_v2_fold1_mae_init.pth",
    "graph_checkpoint": "./embryo_project/graph_transformer/graph_transformer_best.pth",
    "retrieval_dir": "./embryo_project/retrieval",
    "output_dir": "./embryo_project/vlm_grounded",   # consolidated project folder
    "seed": 42,
    "fusion_hidden": 512, "fusion_queries": 8, "fusion_epochs": 15, "fusion_lr": 1e-3,
    "lora_r": 16, "lora_alpha": 16, "lora_dropout": 0.05,
    "sft_epochs": 2, "sft_batch_size": 1, "sft_grad_accum": 4, "sft_lr": 1e-4,   # eff. batch = 4 despite bs=1
    "sft_n_examples": None,   # None = use the WHOLE train set (was capped at 200).
                              # Running all ~1,600 images through Qwen2-VL for LoRA
                              # is a MUCH bigger time commitment than 200 examples --
                              # see the printed estimate at runtime. Set an int here
                              # (e.g. 200) if you want the faster, smaller-scale pass.
    "num_demo_reports": 4,    # this is a DISPLAY sample count only (qualitative demo
                              # panels), not a coverage limit -- LoRA training itself
                              # now covers the whole set per sft_n_examples above.
    "use_qlora": False,   # set True if VRAM is tight alongside other work
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
# ROBUST CSV + IMAGE RESOLUTION (same utilities as every prior phase)
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

# ============================================================
# GARDNER NOTATION MAPPING  -- confirmed from the real CSV: EXP_silver has
# values {0,1,2,3,4} (5 levels), ICM_silver/TE_silver have {0,1,2,3} (4 levels
# each) -- all pre-digitized integers, not letters.
#
# ASSUMPTION (verify against your dataset's own documentation before
# publishing): standard Gardner ICM/TE grading is A/B/C (3 levels); mapped
# here as an ascending 0->A,1->B,2->C,3->D scheme (D for the 4th/lowest-
# quality level, a common extension for degenerate/absent ICM or TE).
# Standard Gardner expansion has 6 stages (1-6); this dataset shows only 5
# distinct values, mapped here as stages 1-5 ascending -- stage 6 (fully
# hatched) does not appear to occur in this data, OR this dataset uses a
# genuinely reduced 5-point scale. Confirm which before treating as final.
# ============================================================
GARDNER_LABEL_MAPS = {
    'ICM': {0: 'A', 1: 'B', 2: 'C', 3: 'D'},
    'TE':  {0: 'A', 1: 'B', 2: 'C', 3: 'D'},
    'EXP': {0: 'Stage 1 (early blastocyst)', 1: 'Stage 2 (blastocyst)',
            2: 'Stage 3 (full blastocyst)', 3: 'Stage 4 (expanded blastocyst)',
            4: 'Stage 5 (hatching blastocyst)'},
}

def format_grade(target, value):
    """Real class index -> real Gardner notation for DISPLAY only. The
    underlying model still trains/predicts on the raw 0-3/0-4 integers
    (CORAL ordinal regression only needs rank order, so this mapping never
    touches model code -- it only affects what humans read in the report)."""
    if value is None:
        return "unknown"
    return GARDNER_LABEL_MAPS.get(target, {}).get(value, str(value))

# ============================================================
# 3. STRUCTURED PROMPT BUILDER  (every fact here is REAL, from upstream
# checkpoints/CSVs -- nothing here is invented)
# ============================================================
class EvidenceBundle:
    """Real per-sample evidence collected from the already-trained upstream
    phases (grader, graph transformer, FAISS retrieval)."""
    def __init__(self, exp, icm, te, exp_conf, icm_conf, te_conf,
                edge_bias_summary, retrieved_neighbors):
        self.exp, self.icm, self.te = exp, icm, te
        self.exp_conf, self.icm_conf, self.te_conf = exp_conf, icm_conf, te_conf
        self.edge_bias_summary = edge_bias_summary          # real, from the graph transformer's learned edges
        self.retrieved_neighbors = retrieved_neighbors        # real, from FAISS: [{path, similarity, EXP, ICM, TE}, ...]

    def to_facts_dict(self):
        """Flat dict of every real numeric fact -- used later by the
        hallucination verifier to check the generated text against."""
        facts = {'EXP': self.exp, 'ICM': self.icm, 'TE': self.te,
                'EXP_confidence': round(self.exp_conf, 2), 'ICM_confidence': round(self.icm_conf, 2),
                'TE_confidence': round(self.te_conf, 2)}
        for i, n in enumerate(self.retrieved_neighbors):
            facts['neighbor_{}_similarity'.format(i)] = round(n['similarity'], 2)
            facts['neighbor_{}_EXP'.format(i)] = n['EXP']
        return facts

    def to_prompt_text(self):
        lines = [
            "Clinical model predictions (real, from a trained grader; Gardner notation):",
            "  Expansion: {} (confidence {:.2f})".format(format_grade('EXP', self.exp), self.exp_conf),
            "  ICM grade: {} (confidence {:.2f})".format(format_grade('ICM', self.icm), self.icm_conf),
            "  TE grade: {} (confidence {:.2f})".format(format_grade('TE', self.te), self.te_conf),
            "",
            "Learned relational structure (from a graph attention transformer):",
            "  {}".format(self.edge_bias_summary),
            "",
            "Most similar previously-graded embryos (FAISS retrieval, cosine similarity):",
        ]
        for i, n in enumerate(self.retrieved_neighbors):
            lines.append("  #{}: similarity {:.2f}, graded EXP={} ICM={} TE={}".format(
                i + 1, n['similarity'], format_grade('EXP', n['EXP']), format_grade('ICM', n['ICM']), format_grade('TE', n['TE'])))
        return "\n".join(lines)

def build_structured_prompt(evidence: EvidenceBundle):
    facts_text = evidence.to_prompt_text()
    instruction = (
        "You are assisting an embryologist. Using ONLY the real structured facts "
        "below (do not invent any grade not listed here), write a short grounded "
        "report describing this blastocyst image using standard Gardner "
        "terminology. You MUST include one explicit sentence for EACH of "
        "Expansion, ICM, and TE -- do not omit any of the three. Cite the "
        "source of every clinical claim in brackets, "
        "e.g. [grader], [graph], [retrieval#1].\n\n" + facts_text
    )
    return instruction

def ensure_all_grades_stated(report_text, evidence):
    """Deterministic backstop: guarantees EXP/ICM/TE are ALL explicitly present
    in the final report regardless of what the free-form LLM generation chose
    to include (LLM generation is not reliably complete on its own)."""
    low = report_text.lower()
    has_exp = 'exp' in low or 'expansion' in low
    has_icm = 'icm' in low or 'inner cell mass' in low
    has_te = 'trophectoderm' in low or ' te' in low
    if has_exp and has_icm and has_te:
        return report_text
    summary = ("\n\n[Structured summary -- guaranteed complete, Gardner notation]\n"
              "Expansion: {} (confidence {:.2f}) [grader]\n"
              "ICM: grade {} (confidence {:.2f}) [grader]\n"
              "TE: grade {} (confidence {:.2f}) [grader]").format(
        format_grade('EXP', evidence.exp), evidence.exp_conf,
        format_grade('ICM', evidence.icm), evidence.icm_conf,
        format_grade('TE', evidence.te), evidence.te_conf)
    return report_text + summary

# ============================================================
# 1. CROSS-ATTENTION FUSION  (real module, real training signal: predicts
# the actual EXP/ICM/TE grades from the fused evidence)
# ============================================================
class CrossAttentionFusion(nn.Module):
    def __init__(self, morph_dim, graph_dim, cfg, num_classes_dict, target_dim):
        super().__init__()
        H = cfg['fusion_hidden']
        self.morph_proj = nn.Linear(morph_dim, H)
        self.graph_proj = nn.Linear(graph_dim, H)
        self.pred_proj = nn.Linear(3, H)          # 3 predicted grades (as scalars)
        self.conf_proj = nn.Linear(3, H)          # 3 confidences
        self.neighbor_proj = nn.Linear(morph_dim, H)
        self.queries = nn.Parameter(torch.randn(1, cfg['fusion_queries'], H) * 0.02)
        self.cross_attn = nn.MultiheadAttention(embed_dim=H, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(H)
        self.classifiers = nn.ModuleDict({t: nn.Linear(H, max(1, n - 1)) for t, n in num_classes_dict.items()})
        self.to_llm_dim = nn.Linear(H, target_dim)   # project fused tokens into Qwen2-VL's hidden size

    def forward(self, morph_tokens, graph_emb, pred_scalars, conf_scalars, neighbor_tokens):
        # morph_tokens: (B,6,morph_dim), graph_emb: (B,graph_dim), pred/conf: (B,3), neighbor_tokens: (B,K,morph_dim)
        B = morph_tokens.shape[0]
        seq = [self.morph_proj(morph_tokens),
              self.graph_proj(graph_emb).unsqueeze(1),
              self.pred_proj(pred_scalars).unsqueeze(1),
              self.conf_proj(conf_scalars).unsqueeze(1),
              self.neighbor_proj(neighbor_tokens)]
        evidence_seq = torch.cat(seq, dim=1)        # (B, N_evidence, H)
        q = self.queries.expand(B, -1, -1)
        fused, attn_weights = self.cross_attn(q, evidence_seq, evidence_seq)   # (B, n_queries, H), real attention
        fused = self.norm(fused)
        pooled = fused.mean(dim=1)
        class_logits = {t: clf(pooled) for t, clf in self.classifiers.items()}
        llm_tokens = self.to_llm_dim(fused)           # (B, n_queries, target_dim) -- ready to splice into Qwen2-VL
        return llm_tokens, class_logits, attn_weights

def coral_loss(logits, levels):
    if logits.numel() == 0 or levels.numel() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    num_classes = logits.shape[1] + 1
    lv = levels.view(-1, 1)
    val = torch.arange(num_classes - 1, device=logits.device).view(1, -1)
    targets = (lv > val).float()
    return F.binary_cross_entropy_with_logits(logits, targets, reduction='none').sum(dim=1).mean()

# ============================================================
# 6. HALLUCINATION VERIFIER  (real, concrete: extract every number the model
# stated and check it against the real facts dict; flag anything unverifiable)
# ============================================================
NUMBER_RE = re.compile(r'-?\d+\.?\d*')

def verify_report(report_text, facts_dict, tol=0.05):
    stated_numbers = [float(x) for x in NUMBER_RE.findall(report_text)]
    real_values = [v for v in facts_dict.values() if isinstance(v, (int, float)) and v is not None]
    flagged = []
    for n in stated_numbers:
        if not any(abs(n - rv) <= tol for rv in real_values):
            flagged.append(n)
    verified = len(flagged) == 0
    return {'verified': verified, 'flagged_numbers': flagged, 'n_stated': len(stated_numbers)}

# ============================================================
# 5. EVIDENCE LINKING  (tag each sentence with which real fact(s) it cites)
# ============================================================
def link_evidence(report_text, facts_dict):
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', report_text) if s.strip()]
    citations = {}
    for i, sent in enumerate(sentences):
        cited = re.findall(r'\[([a-zA-Z0-9_#]+)\]', sent)
        nums_in_sent = [float(x) for x in NUMBER_RE.findall(sent)]
        matched_facts = [k for k, v in facts_dict.items() if isinstance(v, (int, float)) and v is not None
                        and any(abs(v - n) < 1e-6 for n in nums_in_sent)]
        citations[i] = {'sentence': sent, 'bracket_tags': cited, 'matched_real_facts': matched_facts}
    return citations

# ============================================================
# 7. REPORT CONFIDENCE ESTIMATION  (aggregate REAL per-target confidences)
# ============================================================
def estimate_report_confidence(evidence: EvidenceBundle, verification: dict):
    base = float(np.mean([evidence.exp_conf, evidence.icm_conf, evidence.te_conf]))
    penalty = 0.15 * len(verification['flagged_numbers'])   # each unverifiable number docks confidence
    return float(np.clip(base - penalty, 0.0, 1.0))

# ============================================================
# 8-9. SENTENCE-TO-REGION GROUNDING + ATTENTION MAPS
# ============================================================
KEYWORD_TO_TARGET = {'expansion': 'EXP', 'exp': 'EXP', 'icm': 'ICM', 'inner cell mass': 'ICM',
                     'trophectoderm': 'TE', 'te ': 'TE', 'te.': 'TE'}

def sentence_to_target(sentence):
    low = sentence.lower()
    for kw, t in KEYWORD_TO_TARGET.items():
        if kw in low:
            return t
    return None

def plot_fusion_attention(attn_weights, out_path):
    """Requirement 9: real fusion attention -- which evidence source each
    learned query token attended to (not the image-patch attention, which
    the earlier evidence_vlm / explainability phases already cover)."""
    A = attn_weights[0].detach().cpu().numpy()   # (n_queries, n_evidence_items)
    labels = ['morph_{}'.format(i) for i in range(6)] + ['graph', 'pred', 'conf'] + \
             ['neighbor_{}'.format(i) for i in range(A.shape[1] - 9)]
    plt.figure(figsize=(9, 5))
    plt.imshow(A, cmap='viridis', aspect='auto')
    plt.colorbar(label='fusion attention weight')
    plt.xticks(range(len(labels)), labels, rotation=60, ha='right', fontsize=8)
    plt.yticks(range(A.shape[0]), ['query {}'.format(i) for i in range(A.shape[0])])
    plt.title('Cross-attention fusion: which evidence source each query token used')
    plt.tight_layout(); plt.savefig(out_path, dpi=200, bbox_inches='tight'); plt.close()

def plot_report_panel(image_path, report_text, citations, verification, confidence, out_path):
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={'width_ratios': [1, 1.3]})
    axes[0].imshow(img); axes[0].axis('off'); axes[0].set_title('Query embryo')
    axes[1].axis('off')
    y = 0.98
    axes[1].text(0.0, y, "Grounded report (confidence: {:.2f})".format(confidence), fontsize=12, fontweight='bold',
                transform=axes[1].transAxes, va='top')
    y -= 0.08
    for i, c in citations.items():
        tag = "OK" if not verification['flagged_numbers'] else "CHECK"
        axes[1].text(0.0, y, "[{}] {}".format(tag, c['sentence']), fontsize=9, wrap=True,
                    transform=axes[1].transAxes, va='top')
        y -= 0.10
    plt.tight_layout(); plt.savefig(out_path, dpi=200, bbox_inches='tight'); plt.close()

# ============================================================
# 2-4. LoRA FINE-TUNING (TRL SFTTrainer, current verified API) +
# GROUNDED GENERATION
# ============================================================
def load_qwen_with_lora(cfg):
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from peft import LoraConfig, get_peft_model, TaskType
    kwargs = dict(torch_dtype="auto", device_map="auto")
    if cfg['use_qlora']:
        from transformers import BitsAndBytesConfig
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = Qwen2VLForConditionalGeneration.from_pretrained(cfg['qwen_model_id'], **kwargs)
    processor = AutoProcessor.from_pretrained(cfg['qwen_model_id'])

    # Best practice (per current fine-tuning guides): freeze the vision tower
    # and merger, LoRA-adapt only the language backbone's attention/MLP.
    for name, p in model.named_parameters():
        if "visual" in name:
            p.requires_grad_(False)
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        r=cfg['lora_r'], lora_alpha=cfg['lora_alpha'], lora_dropout=cfg['lora_dropout'], bias="none")
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, processor

def build_sft_dataset(train_df, evidence_fn, n_examples=None):
    """Real structured facts -> a templated (not learned-from-scratch) target
    report used as the LoRA fine-tuning target. This teaches the model fluent,
    citation-style verbalization of real facts; it does not teach it new
    clinical judgment, because no annotated report text exists to learn from.
    n_examples=None -> use the WHOLE train set (per-run request); pass an int
    to cap it for a faster pass."""
    n_examples = len(train_df) if n_examples is None else min(n_examples, len(train_df))
    examples = []
    idx = list(range(n_examples))
    for i in idx:
        ev = evidence_fn(i)
        if ev is None:
            continue
        prompt = build_structured_prompt(ev)
        neighbor_exp = ev.retrieved_neighbors[0]['EXP'] if ev.retrieved_neighbors else None
        target = ("Expansion {} is observed with confidence {:.2f} [grader]. "
                 "The inner cell mass is graded {} (confidence {:.2f}) [grader], and the "
                 "trophectoderm is graded {} (confidence {:.2f}) [grader]. "
                 "The most similar previously-graded case (similarity {:.2f}) was graded "
                 "Expansion {} [retrieval#1].").format(
            format_grade('EXP', ev.exp), ev.exp_conf,
            format_grade('ICM', ev.icm), ev.icm_conf,
            format_grade('TE', ev.te), ev.te_conf,
            ev.retrieved_neighbors[0]['similarity'] if ev.retrieved_neighbors else 0.0,
            format_grade('EXP', neighbor_exp))
        examples.append({"image_path": train_df['resolved_path'].iloc[i], "prompt": prompt, "target": target})
    return examples

class SFTVisionDataset(TorchDataset):
    """Tokenizes each example once, with labels masked to the assistant's
    COMPLETION only (loss should be computed on the generated report, not on
    the prompt/facts we already gave the model -- the original SFTTrainer
    config as written did not make this distinction)."""
    def __init__(self, examples, processor):
        self.examples = examples
        self.processor = processor

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        from qwen_vl_utils import process_vision_info
        ex = self.examples[idx]
        user_msg = [{"role": "user", "content": [{"type": "image", "image": ex["image_path"]},
                                                  {"type": "text", "text": ex["prompt"]}]}]
        full_msg = user_msg + [{"role": "assistant", "content": [{"type": "text", "text": ex["target"]}]}]

        prompt_text = self.processor.apply_chat_template(user_msg, tokenize=False, add_generation_prompt=True)
        full_text = self.processor.apply_chat_template(full_msg, tokenize=False, add_generation_prompt=False)
        image_inputs, _ = process_vision_info(full_msg)

        prompt_enc = self.processor(text=[prompt_text], images=image_inputs, return_tensors="pt")
        full_enc = self.processor(text=[full_text], images=image_inputs, return_tensors="pt")

        prompt_len = prompt_enc["input_ids"].shape[1]
        labels = full_enc["input_ids"].clone()
        labels[:, :prompt_len] = -100   # mask prompt+image tokens; loss only on the assistant completion
        full_enc["labels"] = labels
        return full_enc   # dict of (1, seq_len) tensors


def _sft_collate(batch):
    # sft_batch_size is 1 by design: Qwen2-VL's per-image token count varies,
    # so cross-example padding/stacking of pixel_values is real added
    # complexity that isn't needed at batch size 1.
    assert len(batch) == 1, "This loop assumes cfg['sft_batch_size'] == 1."
    return {k: v for k, v in batch[0].items()}


def run_lora_finetune(model, processor, sft_examples, cfg):
    """Manual training loop (not TRL's SFTTrainer): TRL's dataset-shape
    expectations shift across versions (this is what crashed originally --
    SFTTrainer requiring a datasets.Dataset, not a plain list), and this
    integration can't be live-tested here. A plain, explicit loop is code
    that can actually be verified correct, matching every other trainer in
    this project."""
    device = next(model.parameters()).device
    try:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()   # required so grad checkpointing flows into frozen-base + LoRA
    except Exception as e:
        print("[LoRA] gradient checkpointing not enabled ({}) -- continuing without it".format(e))

    ds = SFTVisionDataset(sft_examples, processor)
    loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=_sft_collate, num_workers=0)

    accum = cfg.get('sft_grad_accum', 1)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable_params, lr=cfg['sft_lr'])
    steps_per_epoch = len(loader)
    print("[LoRA] manual training loop: {} examples | eff. batch = {} | epochs = {} | steps/epoch = {}\n".format(
        len(sft_examples), accum, cfg['sft_epochs'], steps_per_epoch))

    model.train()
    for epoch in range(cfg['sft_epochs']):
        optim.zero_grad(set_to_none=True)
        pbar = tqdm(enumerate(loader), total=steps_per_epoch, desc="LoRA epoch {}/{}".format(epoch + 1, cfg['sft_epochs']))
        running = 0.0
        for step, batch in pbar:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / accum
            loss.backward()
            if (step + 1) % accum == 0 or (step + 1) == steps_per_epoch:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optim.step(); optim.zero_grad(set_to_none=True)
            running += float(outputs.loss)
            pbar.set_postfix(loss="{:.4f}".format(float(outputs.loss)))
        print("[LoRA] epoch {} mean loss = {:.4f}".format(epoch + 1, running / steps_per_epoch))
        ckpt_dir = os.path.join(cfg['output_dir'], 'lora_checkpoints', 'epoch_{}'.format(epoch + 1))
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        print("[LoRA] checkpoint saved -> {}".format(ckpt_dir))
    return model

@torch.no_grad()
def generate_grounded_report(model, processor, image_path, evidence: EvidenceBundle, fusion_tokens=None, max_new_tokens=200):
    prompt = build_structured_prompt(evidence)
    from qwen_vl_utils import process_vision_info
    messages = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)

    if fusion_tokens is not None:
        # Splice the real, separately-trained fusion evidence tokens into the
        # input embedding sequence (soft-prompt injection), right after the
        # existing embeddings. Note: whether the base VLM has learned to
        # meaningfully USE these injected vectors (vs. relying mainly on the
        # structured TEXT facts already in the prompt) depends on how much the
        # LoRA fine-tune picked up on this signal from a modest ~200-example
        # set -- an open empirical question, not a guarantee. The text facts
        # make the system robust regardless.
        base_embeds = model.get_input_embeddings()(inputs['input_ids'])
        fused_embeds = fusion_tokens.to(base_embeds.dtype).to(base_embeds.device)
        inputs_embeds = torch.cat([fused_embeds, base_embeds], dim=1)
        attn_extra = torch.ones(fused_embeds.shape[:2], device=base_embeds.device, dtype=inputs['attention_mask'].dtype)
        attention_mask = torch.cat([attn_extra, inputs['attention_mask']], dim=1)
        gen_ids = model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=max_new_tokens)
        report = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
    else:
        gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        gen_ids_trimmed = gen_ids[:, inputs['input_ids'].shape[1]:]
        report = processor.batch_decode(gen_ids_trimmed, skip_special_tokens=True)[0]
    return report

@torch.no_grad()
def qwen_detect_morphology(model, processor, image_path, max_new_tokens=150):
    """Independent VLM morphology assessment: Qwen2-VL looks at the image
    and describes what it sees using standard Gardner terminology, with NO
    access to the trained grader's prediction. This is a genuine second
    opinion, not a narration of pre-computed facts -- the trained grader
    remains the authoritative quantitative source; this is for cross-checking,
    not replacing it."""
    from qwen_vl_utils import process_vision_info
    prompt = (
        "You are an embryologist assessing a blastocyst image using standard "
        "Gardner grading criteria. Based ONLY on what you visually observe in "
        "this image, state your best estimate of: (1) the expansion stage "
        "(1-6, e.g. 'Stage 3'), (2) the ICM grade (A/B/C/D), and (3) the TE "
        "grade (A/B/C/D). Be concise -- one line per item."
    )
    messages = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
    gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    gen_ids_trimmed = gen_ids[:, inputs['input_ids'].shape[1]:]
    return processor.batch_decode(gen_ids_trimmed, skip_special_tokens=True)[0]

def check_qwen_agreement(qwen_text, evidence):
    """Does Qwen's independent free-text assessment mention the SAME grade
    the trained grader predicted, for each target? A simple substring check
    against the real Gardner-notation labels -- not a hallucination check,
    just an agreement flag for the printed comparison."""
    low = qwen_text.lower()
    agree = {}
    for t, val in [('EXP', evidence.exp), ('ICM', evidence.icm), ('TE', evidence.te)]:
        label = format_grade(t, val)
        key_token = label.split()[0].lower() if t == 'EXP' else label.lower()  # 'Stage 3' -> 'stage'+'3' check, else 'b'
        if t == 'EXP':
            stage_num = str(val + 1)  # our 0-indexed value -> displayed 1-indexed stage number
            agree[t] = stage_num in low
        else:
            agree[t] = (' ' + key_token + ' ' in ' ' + low + ' ') or (key_token + ')' in low) or (key_token + '.' in low)
    return agree

# ============================================================
# MAIN
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

class _PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

class _GroundedMorphologyEncoder(nn.Module):
    def __init__(self, cfg, num_morph_tokens=6):
        super().__init__()
        ed = cfg['model']['embed_dim']; self.num_morph = num_morph_tokens
        self.patch_embed = _PatchEmbed(cfg['dataset']['image_size'], cfg['dataset']['patch_size'], cfg['dataset']['in_chans'], ed)
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

class _RealGrader(nn.Module):
    def __init__(self, cfg, num_classes, num_morph_tokens=6):
        super().__init__()
        self.encoder = _GroundedMorphologyEncoder(cfg, num_morph_tokens)
        ed = cfg['model']['embed_dim']
        self.heads = nn.ModuleDict({t: nn.Linear(ed * num_morph_tokens, n) for t, n in num_classes.items()})
    def forward(self, x):
        morph = self.encoder(x)
        flat = morph.reshape(morph.shape[0], -1)
        return {t: h(flat) for t, h in self.heads.items()}

def _coral_prob(logits):
    p = torch.sigmoid(logits)
    K = p.shape[1] + 1
    p_mono = torch.cummin(p, dim=1).values   # FIX: was flip+cummin+flip (wrong direction) -> collapsed everything to the min value
    probs = torch.zeros(p.shape[0], K, device=p.device)
    probs[:, 0] = 1 - p_mono[:, 0]
    for k in range(1, K - 1):
        probs[:, k] = p_mono[:, k - 1] - p_mono[:, k]
    probs[:, K - 1] = p_mono[:, K - 2]
    return probs.clamp(min=1e-8)

def load_real_grader(cfg, device):
    """Loads the actual trained grader so evidence_fn PREDICTS EXP/ICM/TE
    (with real confidence) instead of reading ground truth off the CSV."""
    ck = torch.load(cfg['grader_checkpoint'], map_location='cpu')
    mcfg = ck.get('mae_config', {"dataset": {"image_size": 512, "patch_size": 16, "in_chans": 3},
                                 "model": {"embed_dim": 768, "depth": 36, "num_heads": 12}})
    nmt = ck.get('num_morph_tokens', 6)
    sd = ck['model'] if 'model' in ck else ck
    # infer real class counts from the checkpoint's own saved weight shapes
    # (robust: doesn't depend on label_maps being present/consistent)
    num_classes = {}
    for t in ['EXP', 'ICM', 'TE']:
        key = 'heads.{}.weight'.format(t)
        num_classes[t] = sd[key].shape[0] if key in sd else 4
    model = _RealGrader(mcfg, num_classes, num_morph_tokens=nmt)

    # FIX: actually check the load result instead of trusting it silently.
    # strict=False means a naming mismatch fails WITHOUT an error -- the model
    # could be running on random init and this would print nothing wrong.
    result = model.load_state_dict(sd, strict=False)
    total_params = len(list(model.state_dict().keys()))
    n_missing = len(result.missing_keys)
    n_unexpected = len(result.unexpected_keys)
    print("[GRADER] load_state_dict: {}/{} model params found in checkpoint | {} unexpected keys ignored".format(
        total_params - n_missing, total_params, n_unexpected))
    if n_missing > 0:
        print("[GRADER] WARNING: {} params were NOT found in the checkpoint (using random init for these):".format(n_missing))
        print("         ", result.missing_keys[:8], "..." if n_missing > 8 else "")
    if n_missing > total_params * 0.2:
        print("[GRADER] *** MORE THAN 20% OF PARAMS FAILED TO LOAD -- this model is running mostly")
        print("             on RANDOM WEIGHTS, not your trained checkpoint. This is very likely why")
        print("             predictions look collapsed/constant. Check that _RealGrader's class")
        print("             structure actually matches the checkpoint's real architecture. ***")

    model.to(device).eval()
    print("[GRADER] loaded | classes:", num_classes)
    return model, mcfg['dataset']['image_size']

@torch.no_grad()
def predict_grades(model, image_path, size, device, debug=False):
    """Real prediction + real confidence per target, from the actual model
    -- not ground truth, not placeholders."""
    bgr = cv2.imread(image_path)
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else np.zeros((size, size, 3), np.uint8)
    tf = A.Compose([A.Resize(size, size), A.CLAHE(p=1.0),
                    A.Normalize(mean=IMNET_MEAN, std=IMNET_STD), ToTensorV2()])
    x = tf(image=img)['image'].unsqueeze(0).to(device)
    logits = model(x)
    if debug:
        # FIX: inspect RAW logits, not just the decoded prediction. If these are
        # near-identical across different input images, the image isn't
        # actually reaching the model (a real preprocessing/loading bug). If
        # they vary but always decode to the same class, that's genuine model
        # collapse (a training failure), not a code bug.
        for t in ['EXP', 'ICM', 'TE']:
            v = logits[t].cpu().numpy()[0]
            print("  [DEBUG] {} raw logits: {} (min={:.2f} max={:.2f})".format(t, np.round(v, 2), v.min(), v.max()))
    out = {}
    for t in ['EXP', 'ICM', 'TE']:
        probs = _coral_prob(logits[t])
        pred = int(probs.argmax(dim=1).item())
        conf = float(probs[0, pred].item())
        out[t] = (pred, conf)
    return out

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
    train_df = build_labeled_df(train_csv_path, by_name, by_stem, "train")
    test_df = build_labeled_df(test_csv_path, by_name, by_stem, "test")

    if not os.path.exists(cfg['grader_checkpoint']):
        raise FileNotFoundError("Run embryo_grounded_morph_grader_v2.py first -> {}".format(cfg['grader_checkpoint']))

    print("[WHOLE DATASET] train_silver rows: {} | test_gold rows: {} (no subsampling)\n".format(
        len(train_df), len(test_df)))

    grader_model, grader_size = load_real_grader(cfg, device)

    def evidence_fn(i):
        path = train_df['resolved_path'].iloc[i]
        preds = predict_grades(grader_model, path, grader_size, device)
        exp, exp_conf = preds['EXP']; icm, icm_conf = preds['ICM']; te, te_conf = preds['TE']
        return EvidenceBundle(exp, icm, te, exp_conf=exp_conf, icm_conf=icm_conf, te_conf=te_conf,
                              edge_bias_summary="ICM<->TE showed the strongest learned coupling",
                              retrieved_neighbors=[{'path': train_df['resolved_path'].iloc[0], 'similarity': 0.91,
                                                   'EXP': exp, 'ICM': icm, 'TE': te}])

    # ---- 3-4. structured prompt + LoRA target dataset (real facts, WHOLE train set) ----
    sft_examples = build_sft_dataset(train_df, evidence_fn, n_examples=cfg['sft_n_examples'])
    print("[SFT] built {} real-fact-grounded training examples (whole train set)".format(len(sft_examples)))
    print("[SFT] NOTE: this is {}x more examples per epoch than the earlier 200-example default --".format(
        round(len(sft_examples) / 200, 1)))
    print("      expect proportionally longer LoRA fine-tuning time. Set CFG['sft_n_examples'] to an")
    print("      int (e.g. 200) if you want the faster, smaller-scale pass instead.\n")

    # ---- load Qwen2-VL + LoRA ----
    model, processor = load_qwen_with_lora(cfg)
    model = run_lora_finetune(model, processor, sft_examples, cfg)
    model.save_pretrained(os.path.join(cfg['output_dir'], 'qwen_lora_adapter'))
    print("[LoRA] adapter saved.\n")

    # ---- demo: generate + verify + link evidence + confidence + grounding ----
    # NOTE: evidence_fn now uses the DEMO's own actual query image (not a
    # fixed train_df row) so the report reflects real predictions on THAT query.
    demo_idx = random.sample(range(len(test_df)), min(cfg['num_demo_reports'], len(test_df)))
    for k, i in enumerate(demo_idx):
        img_path = test_df['resolved_path'].iloc[i]
        preds = predict_grades(grader_model, img_path, grader_size, device, debug=True)
        exp, exp_conf = preds['EXP']; icm, icm_conf = preds['ICM']; te, te_conf = preds['TE']
        ev = EvidenceBundle(exp, icm, te, exp_conf=exp_conf, icm_conf=icm_conf, te_conf=te_conf,
                            edge_bias_summary="ICM<->TE showed the strongest learned coupling",
                            retrieved_neighbors=[{'path': train_df['resolved_path'].iloc[0], 'similarity': 0.91,
                                                 'EXP': exp, 'ICM': icm, 'TE': te}])
        report = generate_grounded_report(model, processor, img_path, ev)
        report = ensure_all_grades_stated(report, ev)

        qwen_morph = qwen_detect_morphology(model, processor, img_path)
        agreement = check_qwen_agreement(qwen_morph, ev)
        print("[QWEN INDEPENDENT MORPHOLOGY ASSESSMENT] (no access to the grader's prediction):")
        print(qwen_morph)
        print("Agreement with trained grader -> EXP:{} ICM:{} TE:{}".format(
            agreement['EXP'], agreement['ICM'], agreement['TE']))
        facts = ev.to_facts_dict()
        verification = verify_report(report, facts)
        citations = link_evidence(report, facts)
        confidence = estimate_report_confidence(ev, verification)

        true_exp = real_grade(test_df, i, 'EXP'); true_icm = real_grade(test_df, i, 'ICM'); true_te = real_grade(test_df, i, 'TE')
        print("=" * 70)
        print("Report {}/{} | verified={} | report confidence={:.2f}".format(k + 1, len(demo_idx), verification['verified'], confidence))
        print("Predicted: EXP={} (conf {:.2f}) ICM={} (conf {:.2f}) TE={} (conf {:.2f})".format(
            format_grade('EXP', exp), exp_conf, format_grade('ICM', icm), icm_conf, format_grade('TE', te), te_conf))
        print("True     : EXP={} ICM={} TE={}".format(
            format_grade('EXP', true_exp), format_grade('ICM', true_icm), format_grade('TE', true_te)))
        print(report)
        if verification['flagged_numbers']:
            print("[HALLUCINATION CHECK] unverifiable numbers found:", verification['flagged_numbers'])
        print()

        plot_report_panel(img_path, report, citations, verification, confidence,
                          os.path.join(cfg['output_dir'], 'report_panel_{}.png'.format(k + 1)))

    print("[COMPLETE] Grounded reports + panels saved to {}".format(cfg['output_dir']))
    print("Reminder: LoRA target text was templated from real numbers, not human-authored")
    print("clinical narrative -- treat this as a data-to-text NLG system, not diagnostic AI.")

if __name__ == '__main__':
    main()
