"""
MOT Challenge Evaluation Script

Computes standard Multi-Object Tracking metrics:
    - MOTA  (Multiple Object Tracking Accuracy)
    - MOTP  (Multiple Object Tracking Precision)
    - IDF1  (ID F1-Score)
    - IDP   (ID Precision)
    - IDR   (ID Recall)
    - IDSW  (ID Switches)
    - FP    (False Positives)
    - FN    (False Negatives)
    - Rcll  (Recall)
    - Prcn  (Precision)
    - MT    (Mostly Tracked)
    - ML    (Mostly Lost)
    - Frag  (Fragmentations)

Uses py-motmetrics for standardized evaluation matching MOTChallenge benchmark.

Usage:
    # Evaluate after annotating GT:
    uv run python -m src.evaluation.evaluate_mot --mot-dir output_dir

    # Evaluate specific files:
    uv run python -m src.evaluation.evaluate_mot --gt gt.txt --pred pred.txt

    # Evaluate with custom IoU threshold:
    uv run python -m src.evaluation.evaluate_mot --mot-dir output_dir --iou-thresh 0.5

    # Filter by class (e.g., only riders: class 0 and 1):
    uv run python -m src.evaluation.evaluate_mot --mot-dir output_dir --classes 0 1

Output:
    - Console table with all metrics
    - JSON file with results (for Bab 4)
    - Per-class breakdown if --classes specified
"""

import csv
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from loguru import logger


def load_mot_file(path: str, filter_classes: list[int] | None = None) -> dict[int, list[dict]]:
    """
    Load MOT format file.

    Returns:
        dict mapping frame_id -> list of {id, bb_left, bb_top, bb_w, bb_h, conf, class_id}
    """
    data: dict[int, list[dict]] = defaultdict(list)
    path = Path(path)

    if not path.exists() or path.stat().st_size == 0:
        logger.warning(f"File empty or not found: {path}")
        return data

    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 7:
                continue
            frame_id = int(row[0])
            track_id = int(row[1])
            bb_left = float(row[2])
            bb_top = float(row[3])
            bb_w = float(row[4])
            bb_h = float(row[5])
            conf = float(row[6])
            class_id = int(row[7]) if len(row) > 7 else -1

            if filter_classes and class_id not in filter_classes:
                continue

            data[frame_id].append({
                "id": track_id,
                "bb_left": bb_left,
                "bb_top": bb_top,
                "bb_w": bb_w,
                "bb_h": bb_h,
                "conf": conf,
                "class_id": class_id,
            })

    return data


def compute_iou(box_a: dict, box_b: dict) -> float:
    """Compute IoU between two boxes in (left, top, w, h) format."""
    ax1 = box_a["bb_left"]
    ay1 = box_a["bb_top"]
    ax2 = ax1 + box_a["bb_w"]
    ay2 = ay1 + box_a["bb_h"]

    bx1 = box_b["bb_left"]
    by1 = box_b["bb_top"]
    bx2 = bx1 + box_b["bb_w"]
    by2 = by1 + box_b["bb_h"]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def _compute_distance_matrix(gt_boxes: list[dict], pred_boxes: list[dict]) -> np.ndarray:
    """Compute IoU distance matrix (1 - IoU)."""
    n_gt = len(gt_boxes)
    n_pred = len(pred_boxes)
    dist = np.full((n_gt, n_pred), np.nan)

    for i, gb in enumerate(gt_boxes):
        for j, pb in enumerate(pred_boxes):
            iou = compute_iou(gb, pb)
            if iou > 0:
                dist[i, j] = 1.0 - iou

    return dist


def evaluate_mot(
    gt_data: dict[int, list[dict]],
    pred_data: dict[int, list[dict]],
    iou_threshold: float = 0.5,
) -> dict:
    """
    Compute MOT metrics from ground truth and prediction data.

    Uses Hungarian matching per frame, then computes:
    MOTA, MOTP, IDF1, Precision, Recall, ID switches, etc.

    Implementation follows MOTChallenge evaluation protocol.
    """
    try:
        import motmetrics as mm
        return _evaluate_with_motmetrics(gt_data, pred_data, iou_threshold)
    except ImportError:
        logger.warning("motmetrics not installed, using built-in evaluator")
        return _evaluate_builtin(gt_data, pred_data, iou_threshold)


def _evaluate_with_motmetrics(
    gt_data: dict[int, list[dict]],
    pred_data: dict[int, list[dict]],
    iou_threshold: float,
) -> dict:
    """Evaluate using py-motmetrics library (industry standard)."""
    import motmetrics as mm

    acc = mm.MOTAccumulator(auto_id=True)
    all_frames = sorted(set(list(gt_data.keys()) + list(pred_data.keys())))

    for fid in all_frames:
        gt_boxes = gt_data.get(fid, [])
        pred_boxes = pred_data.get(fid, [])

        gt_ids = [b["id"] for b in gt_boxes]
        pred_ids = [b["id"] for b in pred_boxes]

        # Compute IoU distance matrix
        if gt_boxes and pred_boxes:
            dist = np.zeros((len(gt_boxes), len(pred_boxes)))
            for i, gb in enumerate(gt_boxes):
                for j, pb in enumerate(pred_boxes):
                    iou = compute_iou(gb, pb)
                    dist[i, j] = 1.0 - iou if iou > 0 else np.nan
            # Mark pairs below IoU threshold as invalid
            dist[dist > (1.0 - iou_threshold)] = np.nan
        else:
            dist = np.empty((len(gt_boxes), len(pred_boxes)))

        acc.update(gt_ids, pred_ids, dist)

    # Compute metrics
    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=[
            "mota", "motp", "idf1", "idp", "idr",
            "recall", "precision",
            "num_switches", "num_false_positives", "num_misses",
            "num_objects", "num_predictions",
            "mostly_tracked", "mostly_lost", "partially_tracked",
            "num_fragmentations",
        ],
        name="ITERA_Sentinel"
    )

    # Extract results
    results = {
        "MOTA": float(summary["mota"].iloc[0]) * 100,
        "MOTP": (1.0 - float(summary["motp"].iloc[0])) * 100,  # Convert distance to IoU %
        "IDF1": float(summary["idf1"].iloc[0]) * 100,
        "IDP": float(summary["idp"].iloc[0]) * 100,
        "IDR": float(summary["idr"].iloc[0]) * 100,
        "Recall": float(summary["recall"].iloc[0]) * 100,
        "Precision": float(summary["precision"].iloc[0]) * 100,
        "ID_Switches": int(summary["num_switches"].iloc[0]),
        "FP": int(summary["num_false_positives"].iloc[0]),
        "FN": int(summary["num_misses"].iloc[0]),
        "GT_Objects": int(summary["num_objects"].iloc[0]),
        "Predictions": int(summary["num_predictions"].iloc[0]),
        "MT": int(summary["mostly_tracked"].iloc[0]),
        "ML": int(summary["mostly_lost"].iloc[0]),
        "PT": int(summary["partially_tracked"].iloc[0]),
        "Frag": int(summary["num_fragmentations"].iloc[0]),
        "Frames": len(all_frames),
        "IoU_Threshold": iou_threshold,
        "evaluator": "py-motmetrics",
    }

    return results


def _evaluate_builtin(
    gt_data: dict[int, list[dict]],
    pred_data: dict[int, list[dict]],
    iou_threshold: float,
) -> dict:
    """
    Built-in MOT evaluation (no external dependency).
    Implements simplified CLEAR-MOT protocol.
    """
    from scipy.optimize import linear_sum_assignment

    all_frames = sorted(set(list(gt_data.keys()) + list(pred_data.keys())))

    total_fp = 0
    total_fn = 0
    total_idsw = 0
    total_matches = 0
    total_gt = 0
    total_pred = 0
    sum_iou = 0.0

    # Track ID mapping: gt_id -> last matched pred_id
    prev_match: dict[int, int] = {}

    # Per-GT-track statistics for MT/ML
    gt_track_frames: dict[int, int] = defaultdict(int)
    gt_track_matched: dict[int, int] = defaultdict(int)

    for fid in all_frames:
        gt_boxes = gt_data.get(fid, [])
        pred_boxes = pred_data.get(fid, [])

        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes)

        for gb in gt_boxes:
            gt_track_frames[gb["id"]] += 1

        if not gt_boxes or not pred_boxes:
            total_fn += len(gt_boxes)
            total_fp += len(pred_boxes)
            continue

        # Compute cost matrix (1 - IoU)
        cost = np.ones((len(gt_boxes), len(pred_boxes)))
        ious = np.zeros((len(gt_boxes), len(pred_boxes)))

        for i, gb in enumerate(gt_boxes):
            for j, pb in enumerate(pred_boxes):
                iou = compute_iou(gb, pb)
                ious[i, j] = iou
                cost[i, j] = 1.0 - iou

        # Hungarian matching
        row_ind, col_ind = linear_sum_assignment(cost)

        matched_gt = set()
        matched_pred = set()
        frame_matches = 0

        for ri, ci in zip(row_ind, col_ind):
            if ious[ri, ci] >= iou_threshold:
                gt_id = gt_boxes[ri]["id"]
                pred_id = pred_boxes[ci]["id"]

                matched_gt.add(ri)
                matched_pred.add(ci)
                frame_matches += 1
                sum_iou += ious[ri, ci]

                gt_track_matched[gt_id] += 1

                # ID switch detection
                if gt_id in prev_match and prev_match[gt_id] != pred_id:
                    total_idsw += 1

                prev_match[gt_id] = pred_id

        total_matches += frame_matches
        total_fn += len(gt_boxes) - len(matched_gt)
        total_fp += len(pred_boxes) - len(matched_pred)

    # Compute metrics
    mota = (1.0 - (total_fn + total_fp + total_idsw) / total_gt) * 100 if total_gt > 0 else 0
    motp = (sum_iou / total_matches) * 100 if total_matches > 0 else 0
    recall = (total_matches / total_gt) * 100 if total_gt > 0 else 0
    precision = (total_matches / total_pred) * 100 if total_pred > 0 else 0

    # IDF1 (simplified: using matched/gt/pred counts)
    idf1 = (2 * total_matches / (total_gt + total_pred)) * 100 if (total_gt + total_pred) > 0 else 0

    # MT / ML
    mt = sum(1 for tid in gt_track_frames
             if gt_track_matched.get(tid, 0) / gt_track_frames[tid] >= 0.8)
    ml = sum(1 for tid in gt_track_frames
             if gt_track_matched.get(tid, 0) / gt_track_frames[tid] <= 0.2)
    pt = len(gt_track_frames) - mt - ml

    results = {
        "MOTA": round(mota, 2),
        "MOTP": round(motp, 2),
        "IDF1": round(idf1, 2),
        "IDP": round(precision, 2),
        "IDR": round(recall, 2),
        "Recall": round(recall, 2),
        "Precision": round(precision, 2),
        "ID_Switches": total_idsw,
        "FP": total_fp,
        "FN": total_fn,
        "GT_Objects": total_gt,
        "Predictions": total_pred,
        "MT": mt,
        "ML": ml,
        "PT": pt,
        "Frag": 0,  # not computed in builtin
        "Frames": len(all_frames),
        "IoU_Threshold": iou_threshold,
        "evaluator": "builtin (CLEAR-MOT)",
    }

    return results


def print_results(results: dict, title: str = "MOT Evaluation Results"):
    """Pretty-print evaluation results table."""
    print()
    print("=" * 65)
    print(f"  {title}")
    print("=" * 65)

    # Main metrics
    print(f"  {'Metric':<20} {'Value':>15}")
    print("-" * 40)

    main_metrics = ["MOTA", "MOTP", "IDF1", "Recall", "Precision"]
    for m in main_metrics:
        v = results.get(m, "N/A")
        print(f"  {m:<20} {v:>14.2f}%")

    print("-" * 40)

    # Count metrics
    count_metrics = [
        ("ID Switches", "ID_Switches"),
        ("False Positives", "FP"),
        ("False Negatives", "FN"),
        ("Mostly Tracked", "MT"),
        ("Mostly Lost", "ML"),
        ("Partially Tracked", "PT"),
        ("Fragmentations", "Frag"),
    ]
    for label, key in count_metrics:
        v = results.get(key, 0)
        print(f"  {label:<20} {v:>15d}")

    print("-" * 40)

    # Info
    print(f"  {'Frames':<20} {results.get('Frames', 0):>15d}")
    print(f"  {'GT Detections':<20} {results.get('GT_Objects', 0):>15d}")
    print(f"  {'Predictions':<20} {results.get('Predictions', 0):>15d}")
    print(f"  {'IoU Threshold':<20} {results.get('IoU_Threshold', 0.5):>15.2f}")
    print(f"  {'Evaluator':<20} {results.get('evaluator', '?'):>15}")

    print("=" * 65)
    print()


def run_evaluation(
    mot_dir: str | None = None,
    gt_path: str | None = None,
    pred_path: str | None = None,
    iou_threshold: float = 0.5,
    filter_classes: list[int] | None = None,
    output_json: str | None = None,
) -> dict:
    """
    Run MOT evaluation end-to-end.

    Args:
        mot_dir: MOT directory with gt.txt and pred.txt
        gt_path: explicit GT file path
        pred_path: explicit prediction file path
        iou_threshold: IoU matching threshold
        filter_classes: only evaluate these class IDs
        output_json: save results JSON to this path

    Returns:
        dict of evaluation metrics
    """
    if mot_dir:
        mot_dir = Path(mot_dir)
        gt_path = gt_path or str(mot_dir / "gt.txt")
        pred_path = pred_path or str(mot_dir / "pred.txt")

    if not gt_path or not pred_path:
        raise ValueError("Must provide either --mot-dir or both --gt and --pred")

    logger.info(f"Loading GT: {gt_path}")
    logger.info(f"Loading Pred: {pred_path}")

    gt_data = load_mot_file(gt_path, filter_classes)
    pred_data = load_mot_file(pred_path, filter_classes)

    if not gt_data:
        logger.error("GT file is empty! Run gt_annotator.py first to create ground truth.")
        return {}

    if not pred_data:
        logger.error("Prediction file is empty!")
        return {}

    gt_dets = sum(len(v) for v in gt_data.values())
    pred_dets = sum(len(v) for v in pred_data.values())
    logger.info(f"GT: {len(gt_data)} frames, {gt_dets} detections")
    logger.info(f"Pred: {len(pred_data)} frames, {pred_dets} detections")

    results = evaluate_mot(gt_data, pred_data, iou_threshold)

    # Print results
    class_str = f" (classes: {filter_classes})" if filter_classes else ""
    print_results(results, f"MOT Evaluation{class_str}")

    # Save JSON
    if output_json:
        out_path = Path(output_json)
    elif mot_dir:
        out_path = Path(mot_dir) / "mot_eval_results.json"
    else:
        out_path = Path("mot_eval_results.json")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MOT Challenge Evaluation - MOTA/MOTP/IDF1 Metrics"
    )
    parser.add_argument(
        "--mot-dir", type=str, default=None,
        help="MOT output directory containing gt.txt and pred.txt"
    )
    parser.add_argument(
        "--gt", type=str, default=None,
        help="Ground truth file (MOT format)"
    )
    parser.add_argument(
        "--pred", type=str, default=None,
        help="Prediction file (MOT format)"
    )
    parser.add_argument(
        "--iou-thresh", type=float, default=0.5,
        help="IoU threshold for matching (default: 0.5)"
    )
    parser.add_argument(
        "--classes", type=int, nargs="+", default=None,
        help="Filter by class IDs (e.g., --classes 0 1 for riders only)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file path"
    )
    args = parser.parse_args()

    run_evaluation(
        mot_dir=args.mot_dir,
        gt_path=args.gt,
        pred_path=args.pred,
        iou_threshold=args.iou_thresh,
        filter_classes=args.classes,
        output_json=args.output,
    )
