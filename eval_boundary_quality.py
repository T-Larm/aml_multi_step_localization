"""
Compare ActionFormer predicted steps vs GT step_annotations.json.

Method 1 (HiERO-style): time overlap only, no step_id matching
  - Mean TIoU: average best-match IoU per GT segment across videos
  - mAP: recall@tIoU averaged across thresholds [0.3, 0.5, 0.75], then across videos

Method 2 (Strict): step_id must match + tIoU >= threshold
  - Precision, Recall, F1 @ tIoU thresholds [0.1, 0.2, 0.3, 0.4, 0.5]
"""

import json
import argparse
import numpy as np


def tiou(s1, e1, s2, e2):
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return inter / union if union > 0 else 0.0


def method1_hiero(preds, gts, common_videos, iou_thresholds=(0.1, 0.2, 0.3, 0.4, 0.5)):
    """HiERO-style: time overlap only, per-video mean TIoU and recall@tIoU."""
    all_mean_ious = []
    all_recalls = {t: [] for t in iou_thresholds}

    for vid in sorted(common_videos):
        pred_segs = [(s['start_time'], s['end_time']) for s in preds[vid]['steps']]
        gt_segs   = [(s['start_time'], s['end_time']) for s in gts[vid]['steps']]

        # Mean TIoU: for each GT find best matching pred
        best_ious = []
        for gs, ge in gt_segs:
            best = max((tiou(ps, pe, gs, ge) for ps, pe in pred_segs), default=0.0)
            if best > 0:
                best_ious.append(best)

        if best_ious:
            all_mean_ious.append(np.mean(best_ious))

        # Recall @ each threshold
        for thresh in iou_thresholds:
            if not gt_segs:
                continue
            n_detected = sum(
                1 for gs, ge in gt_segs
                if max((tiou(ps, pe, gs, ge) for ps, pe in pred_segs), default=0.0) >= thresh
            )
            all_recalls[thresh].append(n_detected / len(gt_segs))

    print("\n=== Method 1: HiERO-style (time overlap only) ===")
    print(f"Overall Mean TIoU: {np.mean(all_mean_ious)*100:.2f}%  (over {len(all_mean_ious)} videos)")
    print(f"\n{'tIoU':>6}  {'Recall (=mAP)':>14}")
    print("-" * 25)
    recalls = []
    for thresh in iou_thresholds:
        r = np.mean(all_recalls[thresh]) if all_recalls[thresh] else 0.0
        recalls.append(r)
        print(f"{thresh:>6.2f}  {r*100:>13.2f}%")
    print(f"\nAverage mAP (mean Recall): {np.mean(recalls)*100:.2f}%")

    print(f"\n--- Per-video details ---")
    print(f"{'Video':<12}  {'Mean TIoU':>10}  {'R@0.1':>7}  {'R@0.3':>7}  {'R@0.5':>7}")
    print("-" * 52)
    for vid in sorted(common_videos):
        pred_segs = [(s['start_time'], s['end_time']) for s in preds[vid]['steps']]
        gt_segs   = [(s['start_time'], s['end_time']) for s in gts[vid]['steps']]
        best_ious = [max((tiou(ps, pe, gs, ge) for ps, pe in pred_segs), default=0.0) for gs, ge in gt_segs]
        mean_iou  = np.mean([v for v in best_ious if v > 0]) if any(v > 0 for v in best_ious) else 0.0
        recalls_vid = []
        for thresh in iou_thresholds:
            r = sum(1 for v in best_ious if v >= thresh) / len(gt_segs) if gt_segs else 0.0
            recalls_vid.append(r)
        print(f"{vid:<12}  {mean_iou*100:>9.2f}%  {recalls_vid[0]*100:>6.2f}%  {recalls_vid[2]*100:>6.2f}%  {recalls_vid[4]*100:>6.2f}%")


def method2_strict(preds, gts, common_videos, iou_thresholds=(0.1, 0.2, 0.3, 0.4, 0.5)):
    """Strict: step_id must match + tIoU >= threshold."""
    print("\n=== Method 2: Strict (step_id + tIoU must match) ===")

    # Per-video details at tIoU=0.5
    print(f"\n--- Per-video details (tIoU=0.5) ---")
    print(f"{'Video':<12}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
    print("-" * 44)
    for vid in sorted(common_videos):
        pred_segs = [(s['start_time'], s['end_time'], s['step_id']) for s in preds[vid]['steps']]
        gt_segs   = [(s['start_time'], s['end_time'], s['step_id']) for s in gts[vid]['steps']]
        matched_gt = set()
        tp = 0
        for ps, pe, pl in pred_segs:
            best_iou, best_j = 0.0, -1
            for j, (gs, ge, gl) in enumerate(gt_segs):
                if j in matched_gt or gl != pl:
                    continue
                iou = tiou(ps, pe, gs, ge)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= 0.5 and best_j >= 0:
                tp += 1
                matched_gt.add(best_j)
        p = tp / len(pred_segs) if pred_segs else 0.0
        r = tp / len(gt_segs)   if gt_segs   else 0.0
        f = 2*p*r/(p+r) if (p+r) > 0 else 0.0
        print(f"{vid:<12}  {p*100:>9.2f}%  {r*100:>7.2f}%  {f*100:>7.2f}%")

    print(f"\n--- Overall across thresholds ---")
    print(f"\n{'tIoU':>6}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
    print("-" * 40)

    for thresh in iou_thresholds:
        tp_total = fp_total = fn_total = 0

        for vid in common_videos:
            pred_segs = [(s['start_time'], s['end_time'], s['step_id']) for s in preds[vid]['steps']]
            gt_segs   = [(s['start_time'], s['end_time'], s['step_id']) for s in gts[vid]['steps']]

            matched_gt = set()
            tp = 0
            for ps, pe, pl in pred_segs:
                best_iou, best_j = 0.0, -1
                for j, (gs, ge, gl) in enumerate(gt_segs):
                    if j in matched_gt or gl != pl:
                        continue
                    iou = tiou(ps, pe, gs, ge)
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_iou >= thresh and best_j >= 0:
                    tp += 1
                    matched_gt.add(best_j)

            fp_total += len(pred_segs) - tp
            fn_total += len(gt_segs) - tp
            tp_total += tp

        p = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        r = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        print(f"{thresh:>6.2f}  {p*100:>9.2f}%  {r*100:>7.2f}%  {f*100:>7.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', required=True, help='step_embeddings JSON file')
    parser.add_argument('--gt',   required=True, help='step_annotations.json')
    parser.add_argument('--split_json', default='', help='ActionFormer split JSON to filter by subset')
    parser.add_argument('--subset', default='test', choices=['training', 'validation', 'test'],
                        help='Which subset to evaluate (default: test)')
    args = parser.parse_args()

    with open(args.pred) as f:
        preds = json.load(f)
    with open(args.gt) as f:
        gts = json.load(f)

    if args.split_json:
        with open(args.split_json) as f:
            split_data = json.load(f)['database']
        subset_vids = set(v for v, info in split_data.items()
                         if info['subset'].lower() == args.subset.lower())
        preds = {v: preds[v] for v in subset_vids if v in preds}
        print(f"Filtered to {args.subset} subset: {len(preds)} videos")

    common = set(preds.keys()) & set(gts.keys())
    print(f"Videos — pred: {len(preds)}, GT: {len(gts)}, common: {len(common)}")

    method1_hiero(preds, gts, common)
    method2_strict(preds, gts, common)
