"""
Substep 1: Compute step-level EgoVLP embeddings using ActionFormer predicted boundaries.

Pipeline:
  1. Load ActionFormer inference results (.pkl) containing predicted (start, end, label, score)
     for each video.
  2. For each video, keep predictions above --score_threshold. If none pass, fall back through
     lower thresholds (0.25 → 0.20 → 0.15 → 0.1 → 0.05 → 0.01 → top-1).
  3. Sort selected predictions by start time.
  4. For each predicted segment, extract the corresponding EgoVLP frame features (1 fps, 256-dim)
     and aggregate them into a single embedding via Gaussian-weighted average (center frames
     weighted more), then L2-normalize.
  5. Save results.

Input:
  --pred_pkl      : ActionFormer eval_results.pkl  (from eval.py --saveonly)
  --egovlp_folder : folder containing {video_id}_360p_224.mp4_1s_1s.npz  (shape: T x 256)

Output (saved to --output_dir):
  <output_dir>.npz  : { video_id -> np.array (N_steps, 256) }  step embeddings
  <output_dir>.json : per-video step metadata (step_id, start_time, end_time, embeddings_shape)
"""

import os
import json
import argparse
import numpy as np
import pickle


def compute_step_embeddings(pred_pkl, egovlp_folder, output_dir, score_threshold=0.0):
    # Load ActionFormer predictions
    with open(pred_pkl, 'rb') as f:
        preds = pickle.load(f)
    # preds keys: 'video-id', 't-start', 't-end', 'label', 'score'

    video_ids    = np.array(preds['video-id'])
    t_starts     = preds['t-start']   # (N,)
    t_ends       = preds['t-end']     # (N,)
    labels       = preds['label']     # (N,)
    scores       = preds['score']     # (N,)

    # Group by video (before filtering, to keep all videos)
    unique_videos = np.unique(video_ids)
    print(f"Total videos: {len(unique_videos)}")

    step_embeddings = {}  # video_id -> (N_steps, 256)
    step_segments   = {}  # video_id -> (N_steps, 2)
    step_labels     = {}  # video_id -> (N_steps,)

    missing = []
    for vid in unique_videos:
        egovlp_path = os.path.join(egovlp_folder, f"{vid}_360p_224.mp4_1s_1s.npz")
        if not os.path.exists(egovlp_path):
            missing.append(vid)
            continue

        feats = np.load(egovlp_path)['arr_0'].astype(np.float32)  # (T, 256)
        T = feats.shape[0]

        vid_mask = video_ids == vid
        starts = t_starts[vid_mask]
        ends   = t_ends[vid_mask]
        lbls   = labels[vid_mask]
        scrs   = scores[vid_mask]

        # Per-video threshold: if no prediction passes, fallback to lower thresholds
        thresh_mask = scrs >= score_threshold
        if thresh_mask.sum() == 0:
            fallback_thresholds = [0.25, 0.20, 0.15, 0.1, 0.05, 0.01]
            for fb_thresh in fallback_thresholds:
                thresh_mask = scrs >= fb_thresh
                if thresh_mask.sum() > 0:
                    print(f"Warning: {vid} has no predictions above {score_threshold}, using fallback threshold {fb_thresh}")
                    break
            else:
                print(f"Warning: {vid} has no predictions above any threshold, using top-1")
                thresh_mask = np.zeros(len(scrs), dtype=bool)
                thresh_mask[np.argmax(scrs)] = True
        starts = starts[thresh_mask]
        ends   = ends[thresh_mask]
        lbls   = lbls[thresh_mask]

        # Sort by start time
        order  = np.argsort(starts)
        starts = starts[order]
        ends   = ends[order]
        lbls   = lbls[order]

        embs = []
        for s, e in zip(starts, ends):
            s_idx = max(0, int(s))           # 1s stride -> index = seconds
            e_idx = min(T, max(int(e), s_idx + 1))
            n = e_idx - s_idx
            if n > 1:
                # Gaussian-weighted average: center frames weighted more
                weights = np.exp(-0.5 * ((np.arange(n) - n / 2.0) / (n / 4.0 + 1e-8)) ** 2)
                weights /= weights.sum()
                emb = (feats[s_idx:e_idx] * weights[:, None]).sum(axis=0)
            else:
                emb = feats[s_idx:e_idx].mean(axis=0)
            # L2 normalize
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            embs.append(emb)

        step_embeddings[vid] = np.stack(embs)                    # (N_steps, 256)
        step_segments[vid]   = np.stack([starts, ends], axis=1)  # (N_steps, 2)
        step_labels[vid]     = lbls                              # (N_steps,)

    if missing:
        print(f"Warning: {len(missing)} videos missing EgoVLP features: {missing[:5]}...")

    os.makedirs(os.path.dirname(os.path.abspath(output_dir)), exist_ok=True)

    # Save all video embeddings into a single .npz file
    npz_path = output_dir if output_dir.endswith('.npz') else output_dir + '.npz'
    np.savez_compressed(npz_path, **step_embeddings)

    # Save JSON with step boundaries, next to the .npz
    json_out = {}
    for vid in step_embeddings:
        segs = step_segments[vid]   # (N_steps, 2)
        lbls = step_labels[vid]     # (N_steps,)
        json_out[vid] = {
            "recording_id": vid,
            "steps": [
                {
                    "step_id": int(lbls[i]),
                    "start_time": float(segs[i][0]),
                    "end_time": float(segs[i][1]),
                }
                for i in range(len(lbls))
            ],
            "embeddings_shape": list(step_embeddings[vid].shape),
        }
    json_path = npz_path.replace('.npz', '.json')
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)

    print(f"Saved {len(step_embeddings)} video embeddings to {npz_path}")
    print(f"Saved step boundaries to {json_path}")
    sample_vid = list(step_embeddings.keys())[0]
    print(f"Sample '{sample_vid}': {step_embeddings[sample_vid].shape}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_pkl',      required=True,
                        help='ActionFormer eval_results.pkl (from eval.py --saveonly)')
    parser.add_argument('--egovlp_folder', default='./egovlp',
                        help='Folder with EgoVLP .npz files')
    parser.add_argument('--output_dir',    default='./step_embeddings',
                        help='Output path (will be saved as <output_dir>.npz)')
    parser.add_argument('--score_threshold', type=float, default=0.0,
                        help='Min score to keep a predicted segment (fallback to lower thresholds if none pass)')
    args = parser.parse_args()

    compute_step_embeddings(
        args.pred_pkl,
        args.egovlp_folder,
        args.output_dir,
        args.score_threshold,
    )
