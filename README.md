# Substep 1: Step Temporal Localization → Step Embeddings

Uses **ActionFormer** + **EgoVLP** features to localize procedural steps in CaptainCook4D videos and produce per-step embeddings for downstream mistake detection (Substeps 2–4).

## Overview

**What is the goal?**
Given a raw cooking video, find *when* each step happens and represent it as a fixed-size vector. These vectors (step embeddings) are the input to all downstream substeps.

**What are the two building blocks?**

- **EgoVLP** — A vision-language model pre-trained on Ego4D egocentric video. We use it as a frozen feature extractor: it converts each second of video into a 256-dim vector that is semantically aligned with natural language action descriptions. These features are extracted offline and never fine-tuned.

- **ActionFormer** — A Transformer-based temporal action detector. It takes the EgoVLP feature sequence as input and predicts *where* (start/end time) and *what* (step class) each step is. We train this on CaptainCook4D.

**Why not use GT boundaries directly?**
GT boundaries are unavailable at test time. ActionFormer learns to approximate them from video features alone, enabling the full pipeline to run without human annotation at inference.

**What does the output look like?**
```
step_embeddings/person_ep030.npz
  └─ "1_7"  → np.array (15, 256)   # 15 predicted steps, each 256-dim
  └─ "2_3"  → np.array (12, 256)
  └─ ...                            # 384 videos total
```

## Pipeline Overview

```
EgoVLP features (256-dim, 1fps)
        ↓
ActionFormer (temporal action localization)
        ↓
Predicted step boundaries (t_start, t_end, score)
        ↓
compute_step_embeddings.py (Gaussian-weighted aggregation)
        ↓
Step embeddings: { video_id → (N_steps, 256) }
```

---

## Setup

```bash
conda activate aml
pip install -r requirements.txt
```

**Required data:**
- `egovlp/` — EgoVLP frame features (`{video_id}_360p_224.mp4_1s_1s.npz`, shape `T×256`)
- `captaincook_actionformer_annotations/combined/` — split JSON files (`person.json`, `recordings.json`)
- `captaincook/annotation_json/step_annotations.json` — GT step annotations

---

## Step 1: Train ActionFormer

```bash
python train.py configs/captaincook_egovlp.yaml \
  --backbone egovlp \
  --feat_folder ./egovlp \
  --num_frames 1 --stride 1 \
  --division_type person \
  --output reproduce
```

Checkpoints saved to `ckpt/error/egovlp_person_reproduce/` every 5 epochs.

**Key config (`configs/captaincook_egovlp.yaml`):**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `backbone_arch` | `[2, 2, 7]` | Transformer depth |
| `regression_range` | 8 levels | Step duration range (0–10000s) |
| `epochs` | 50 | Training epochs (+5 warmup) |
| `batch_size` | 2 | Per-GPU batch size |
| `learning_rate` | 1e-4 | AdamW optimizer |

---

## Step 2: Find Best Checkpoint

Evaluate all checkpoints on the validation set:

```bash
for epoch in 005 010 015 020 025 030 035 040 045 050; do
    echo -n "Epoch $epoch: "
    python eval.py configs/captaincook_egovlp.yaml reproduce \
        --backbone egovlp --feat_folder ./egovlp \
        --num_frames 1 --stride 1 \
        --division_type person \
        -epoch $epoch 2>/dev/null | grep "Average mAP"
done
```

**Best result:** `person`, epoch 30 → **Average mAP 14.25% (val set)**

> **Evaluation metric:** Average mAP computed by `ANETdetection` (`actionformer/libs/utils/metrics.py`, adapted from the [ActivityNet official evaluation code](https://github.com/activitynet/ActivityNet), also used in EPIC-Kitchens). Requires both correct label and tIoU ≥ threshold to count as a true positive. Averaged over tIoU = [0.1, 0.2, 0.3, 0.4, 0.5].

> Note: mAP peaks around epoch 30 and degrades beyond that due to overfitting.

---

## Step 3: Generate Predictions for All 384 Videos

```bash
python eval.py configs/captaincook_egovlp_infer.yaml reproduce \
  --backbone egovlp --feat_folder ./egovlp \
  --num_frames 1 --stride 1 \
  --division_type person \
  -epoch 30 --saveonly \
  --output_pkl ./predictions/person_ep030_all.pkl
```

`captaincook_egovlp_infer.yaml` sets `val_split: ['training', 'validation', 'test']` to cover all 384 videos.

---

## Step 4: Generate Step Embeddings

```bash
python compute_step_embeddings.py \
  --pred_pkl ./predictions/person_ep030_all.pkl \
  --egovlp_folder ./egovlp \
  --output_dir ./step_embeddings/person_ep030 \
  --score_threshold 0.27
```

**`--score_threshold 0.27`** keeps predictions with confidence ≥ 0.27, yielding ~15 steps/video on average (close to GT mean of 14.8). If no predictions pass the threshold, the script falls back to lower thresholds automatically.

**Output:**
- `step_embeddings/person_ep030.npz` — `{ video_id → np.array(N_steps, 256) }`
- `step_embeddings/person_ep030.json` — step boundaries and metadata

---

## Step 5: Evaluate Boundary Quality (Optional)

Evaluate on the **test set only** for honest reporting:

```bash
python eval_boundary_quality.py \
  --pred ./step_embeddings/person_ep030.json \
  --gt ./captaincook/annotation_json/step_annotations.json \
  --split_json ./captaincook_actionformer_annotations/combined/person.json \
  --subset test
```

| Metric | person ep030 (test set) |
|--------|------------------------|
| Mean TIoU | 40.71% |
| Recall@0.1 | 62.21% |
| Recall@0.5 | 27.03% |
| F1@0.1 (strict) | 12.13% |
| F1@0.5 (strict) | 7.63% |

---

## Division Type Comparison

All metrics evaluated on the **test set** for each split.

| Split | Best Epoch | mAP (val) | mAP (test) | Boundary TIoU (test) | Downstream F1 (Substep 2/4) |
|-------|-----------|-----------|-----------|----------------------|------------------------------|
| **person** | **30** | **14.25%** | 10.62% | 40.71% | **~0.53** |
| recordings | 30 | 12.81% | **11.10%** | **42.14%** | ~0.45 |

On the test set, `recordings` achieves slightly higher mAP and boundary TIoU than `person`. Despite this, `person` consistently outperforms on downstream Substep 2/4 tasks. The most likely explanation is that `person` split enforces subject-level generalisation during training — no person appears in both train and val — forcing the model to learn person-agnostic step features. This produces embeddings with more consistent semantic content across different subjects, which is more useful for downstream mistake detection than raw boundary accuracy.
