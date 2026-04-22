"""
Evaluate predicted step boundaries against GT step_annotations.json.
Computes Precision, Recall, F1 @ tIoU thresholds (ignoring step_id).
"""

import json
import argparse
import numpy as np


def tiou(s1, e1, s2, e2):
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return inter / union if union > 0 else 0.0


def eval_boundaries(pred_json, gt_json, tiou_thresholds):
    with open(pred_json) as f:
        preds = json.load(f)
    with open(gt_json) as f:
        gts = json.load(f)

    common_videos = set(preds.keys()) & set(gts.keys())
    print(f"Videos in pred: {len(preds)}, in GT: {len(gts)}, common: {len(common_videos)}")

    results = {}
    for thresh in tiou_thresholds:
        tp_total, fp_total, fn_total = 0, 0, 0

        for vid in common_videos:
            pred_segs = [(s['start_time'], s['end_time'], s['step_id']) for s in preds[vid]['steps']]
            gt_segs   = [(s['start_time'], s['end_time'], s['step_id']) for s in gts[vid]['steps']]

            matched_gt = set()
            tp = 0
            for ps, pe, pl in pred_segs:
                best_iou = 0.0
                best_j = -1
                for j, (gs, ge, gl) in enumerate(gt_segs):
                    if j in matched_gt or gl != pl:
                        continue
                    iou = tiou(ps, pe, gs, ge)
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j
                if best_iou >= thresh and best_j >= 0:
                    tp += 1
                    matched_gt.add(best_j)

            fp = len(pred_segs) - tp
            fn = len(gt_segs)  - tp
            tp_total += tp
            fp_total += fp
            fn_total += fn

        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        recall    = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results[thresh] = (precision, recall, f1)

    print(f"\n{'tIoU':>6}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
    print("-" * 40)
    for thresh, (p, r, f) in results.items():
        print(f"{thresh:>6.2f}  {p*100:>9.2f}%  {r*100:>7.2f}%  {f*100:>7.2f}%")

    avg_f1 = np.mean([v[2] for v in results.values()])
    print(f"\nAverage F1: {avg_f1*100:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred',  required=True, help='Predicted JSON (from compute_step_embeddings)')
    parser.add_argument('--gt',    required=True, help='GT step_annotations.json')
    parser.add_argument('--tiou',  nargs='+', type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5])
    args = parser.parse_args()

    eval_boundaries(args.pred, args.gt, args.tiou)
