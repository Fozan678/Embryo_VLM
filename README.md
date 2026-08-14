# FEMI-VLM: Evidence-Grounded Vision–Language Assessment of Blastocyst Morphology

A research pipeline for Gardner-criteria blastocyst grading, combining self-supervised pretraining, ordinal grading, uncertainty quantification, explainability, and evidence-grounded report generation with Qwen2-VL.

> **Status: research code.** This is not a clinical tool and has not been validated for clinical use. Several components produced negative results, documented honestly below rather than omitted.

---

## Key Findings (read before running)

| Target | QWK (MAE-init) | QWK (from scratch) | Assessment |
|---|---|---|---|
| Expansion (EXP) | **0.86** | 0.78 | Reliably learnable; pretraining helps |
| Trophectoderm (TE) | 0.21 | 0.07 | Pretraining helps ~3× in relative terms; low absolute |
| Inner Cell Mass (ICM) | 0.15 | 0.17 | No reliable pretraining benefit; unresolved |

**Two results that shaped everything downstream:**

1. **Severe class sparsity.** ICM grade 2 has **16** training examples (0.8%); TE grade 2 has **47** (2.3%). Class rebalancing, focal loss, and weighted sampling did *not* fix ICM — imbalance handling was not the operative bottleneck. Sparse classes were subsequently merged into their adjacent higher class.

2. **The morphology-token architecture did not differentiate (negative result).** Five independent measurements converge: token attribution magnitudes ≈ 1×10⁻⁶, attention flow exactly uniform at 1/6, PCA/t-SNE node embeddings fully superimposed, learned graph edge biases ≈ 10⁻⁴, and the diversity loss collapsing to zero within 30 of 250 epochs. Token names (`ICM`, `TE`, `Blastocoel`, `Zona`, `Fragmentation`, `Global`) are **display conventions, not validated semantic assignments.**

See `FEMI_VLM_Results_Report.tex` for the full write-up including a Threats to Validity section.

---

## Setup

```bash
conda create -n embryo python=3.12 && conda activate embryo
pip install torch --index-url https://download.pytorch.org/whl/cu128   # RTX 50-series / sm_120
pip install "numpy<2" opencv-python albumentations pandas scikit-learn matplotlib seaborn tqdm
pip install pyiqa networkx umap-learn scipy
pip install faiss-gpu-cu12          # fallback: faiss-cpu
pip install transformers accelerate peft "qwen-vl-utils[decord]" pillow
```

**Expected layout:**

```
project_root/
├── Downloads/archive/
│   ├── Gardner_train_silver.csv                    # semicolon-delimited
│   ├── Gardner_test_gold_onlyGardnerScores.csv
│   └── Images/Images/                              # 2,344 PNGs
└── embryo_project/                                 # all outputs land here
```

CSVs are **semicolon-delimited** and grades are stored as **integers**, not Gardner letters (EXP `{0–4}`, ICM/TE `{0–3}`). Every script auto-detects the delimiter.

---

## Run Order

Each stage depends on checkpoints from earlier ones. Run sequentially.

### Stage 1 — Data Foundation

```bash
python embryo_data_pipeline_final.py    # 1. splits, metadata, distribution figures
python embryo_iqa_final.py              # 2. BRISQUE/NIQE no-reference quality scoring
```
Independent of everything downstream; safe to run in parallel or skip if only reproducing model results.

### Stage 2 — Self-Supervised Pretraining

```bash
python embryo_mae_pretrain_v2.py        # 3. MAE, ViT-Base, 400 epochs  → mae/checkpoints/mae_best.pth
python embryo_linear_probe.py           # 4. validates MAE vs random-init / ImageNet
python embryo_morph_grounded_v2.py      # 5. morphology-token SSL branch (see negative result above)
```
Step 3 is the **critical dependency** — nothing after Stage 2 runs without `mae_best.pth`. Step 5 is a side branch that feeds nothing downstream.

### Stage 3 — Supervised Grading

```bash
python embryo_grader_finetune_v2.py         # 6. mean-pool grader, k-fold, MAE vs scratch
python embryo_grounded_morph_grader_v2.py   # 7. 6-token grader → saves ALL folds
python check_grader_collapse.py             # 8. ⚠️ VERIFY before proceeding
```

**Step 8 is mandatory, not optional.** It runs the checkpoint over all 300 test images and prints the predicted-class distribution. Expected healthy output:

```
EXP: predicted = {2: 73, 3: 159, 0: 20, 1: 32, 4: 16} | true = {2: 86, 3: 153, 0: 23, 1: 31, 4: 5}
```

If any target predicts only **one** class, stop — that checkpoint is unusable. Step 7 saves every fold (`grounded_morph_v2_fold{1,2,3}_mae_init.pth`); check `kfold_summary.csv` and point downstream configs at whichever fold has real QWK.

### Stage 4 — Reasoning, Retrieval, Explainability

```bash
python embryo_graph_transformer.py          # 9.  graph attention over 6 morphology nodes
python embryo_faiss_retrieval.py            # 10. FAISS index (train=DB, test=queries)
python embryo_explainability_complete.py    # 11. GradCAM/++/IG/Rollout/Flow/Shapley/LIME
                                            #     + deletion-insertion faithfulness
python embryo_clinical_multitask_v2.py      # 12. multi-task + ablation
python embryo_uncertainty_framework_v2.py   # 13. MC-Dropout / Deep Ensemble / EDL
```

Step 11 quantifies *which* explanation method is trustworthy via deletion/insertion AUC — not just which produces attractive heatmaps. Step 13 trains 5 models total (1 + 3 ensemble + 1 EDL); reduce `ensemble_size` or `epochs` for a faster pass.

### Stage 5 — Evidence-Grounded VLM

```bash
python embryo_evidence_grounded_vlm.py         # 14. LoRA fine-tune → qwen_lora_adapter
python embryo_vlm_train_test.py                # 15. grounded reports w/ sanity gate
python embryo_vlm_counterfactual_dialogue.py   # (optional) counterfactual + multi-turn
```

**Step 14 must precede 15** — it produces the adapter that 15 loads. Downloads Qwen2-VL-2B (~4.5 GB); set `use_qlora: True` if VRAM is tight.

Step 15 runs a **hard sanity gate** before any VLM generation and halts if the grader collapses, rather than producing fluent, confidently-wrong reports.

### Stage 6 — Evaluation

```bash
python embryo_evaluation_framework.py       # 16. metrics, bootstrap CI, significance, gallery
```
Saves per-sample raw predictions (previously missing), bootstrap CIs, McNemar/paired-bootstrap tests, and figures at 600 dpi in PNG/PDF/SVG. Check `FIGURE_PROVENANCE.md` — it states which requested figures are real versus **MISSING** because a stage wasn't run.

---

## Optional / Diagnostic

| Script | Purpose |
|---|---|
| `check_test_distribution.py` | Prints test-set class counts |
| `embryo_vlm_unseen_inference.py` | Inference on unlabeled data with MC-Dropout + retrieval-similarity OOD flagging |

**On out-of-distribution data:** preliminary application to a different image source produced confidently incorrect, low-diversity predictions — consistent with known neural overconfidence under distribution shift. The OOD script therefore flags trust using signals *independent* of the model's own confidence, which is not reliable here.

---

## Output Structure

```
embryo_project/
├── mae/{figures,checkpoints}/          ├── retrieval/
├── morphology/{figures,checkpoints}/   ├── explainability/
├── grader/                             ├── vlm_grounded/
├── grounded_morph_grader/              ├── vlm_train_test/
├── graph_transformer/                  ├── vlm_counterfactual_dialogue/
├── clinical_multitask/                 └── evaluation/{figures,gallery}/
├── uncertainty_framework/
```

---

## Known Issues

- **Table 2 / k-fold QWK predate the sparse-class merge.** Pre- and post-merge kappa are not directly comparable (merging changes the ordinal threshold count). Re-run Stage 3 under the merged formulation before citing.
- **`embryo_vlm_counterfactual_dialogue.py` renders raw integers**, not Gardner notation — the SFT notation fix reached the main VLM scripts but not this one.
- **`k=3` folds with no paired significance test.** Error bars summarise three observations; treat "clear benefit" as descriptive, not inferential. Use `n_folds=5` before publication.
- **Silver train / gold test label asymmetry** was never quantified — an unknown share of the ICM/TE deficit may be training-label noise.
- **No human inter-rater baseline**, so kappa cannot be positioned against the achievable ceiling.
- A **CORAL→categorical probability decoding defect** (monotonicity enforced in the wrong direction) was found and fixed. It never affected training or rank-based kappa, but any calibration/entropy/confidence figure generated before the fix must be regenerated.

---

## Method Notes

- **Ordinal regression:** CORAL throughout; only rank order matters, so integer-vs-letter grade encoding does not affect training.
- **Counterfactuals:** gradient-based search (Wachter et al., 2017) on the real differentiable grader — the VLM verbalizes these, it does not invent them.
- **Hallucination checking:** every number in generated text is extracted and verified against the real fact set, on *every* dialogue turn.
- **Geometric proxies** (compactness, continuity, boundary regularity, symmetry) in the graph transformer are classical morphometrics computed from image masks — real measurements, but **not** clinically validated labels.

