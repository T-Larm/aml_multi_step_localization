# Substep 1: Step Temporal Localization → Step Embeddings

Uses **ActionFormer** + **EgoVLP** features to localize procedural steps in CaptainCook4D videos and produce per-step embeddings for downstream mistake detection (Substeps 2–4).

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
  --division_type recordings \
  --output reproduce
```

Checkpoints saved to `ckpt/error/egovlp_recordings_reproduce/` every 5 epochs.

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
for epoch in 005 010 015 020 025 030 035 040 045 050 055; do
    echo -n "Epoch $epoch: "
    python eval.py configs/captaincook_egovlp.yaml reproduce \
        --backbone egovlp --feat_folder ./egovlp \
        --num_frames 1 --stride 1 \
        --division_type recordings \
        -epoch $epoch 2>/dev/null | grep "Average mAP"
done
```

**Best result:** `recordings`, epoch 30 → **Average mAP 12.81%**, boundary Mean TIoU **54.51%**

> Note: mAP peaks around epoch 30; training beyond 50 epochs causes overfitting.

---

## Step 3: Generate Predictions for All 384 Videos

```bash
python eval.py configs/captaincook_egovlp_infer.yaml reproduce \
  --backbone egovlp --feat_folder ./egovlp \
  --num_frames 1 --stride 1 \
  --division_type recordings \
  -epoch 30 --saveonly \
  --output_pkl ./predictions/recordings_ep030_all.pkl
```

`captaincook_egovlp_infer.yaml` sets `val_split: ['training', 'validation', 'test']` to cover all 384 videos.

---

## Step 4: Generate Step Embeddings

```bash
python compute_step_embeddings.py \
  --pred_pkl ./predictions/recordings_ep030_all.pkl \
  --egovlp_folder ./egovlp \
  --output_dir ./step_embeddings/recordings_ep030 \
  --score_threshold 0.27
```

**`--score_threshold 0.27`** keeps predictions with confidence ≥ 0.27, yielding ~15 steps/video on average (close to GT mean of 14.8). If no predictions pass the threshold, the script falls back to lower thresholds automatically.

**Output:**
- `step_embeddings/recordings_ep030.npz` — `{ video_id → np.array(N_steps, 256) }`
- `step_embeddings/recordings_ep030.json` — step boundaries and metadata

---

## Step 5: Evaluate Boundary Quality (Optional)

```bash
python eval_boundary_quality.py \
  --pred ./step_embeddings/recordings_ep030.json \
  --gt ./captaincook/annotation_json/step_annotations.json
```

| Metric | recordings ep030 |
|--------|-----------------|
| Mean TIoU | 54.51% |
| Recall@0.1 | 65.59% |
| Recall@0.5 | 43.21% |
| F1@0.1 (strict) | 29.38% |
| F1@0.5 (strict) | 26.39% |

---

## Division Type Comparison

| Split | Best Epoch | ActionFormer mAP | Boundary TIoU |
|-------|-----------|-----------------|---------------|
| person | 30 | 14.25% | 53.47% |
| **recordings** | **30** | **12.81%** | **54.51%** |

`recordings` is preferred for downstream substeps as it produces better temporal boundaries, which directly determine embedding quality.
