"""
Debug Display Module — Step-by-step Pipeline Visualization

Provides separate debug view modes for each stage of the pipeline,
designed for thesis screenshots and analysis (Bab 4).

Modes:
    1. "detection"  – Raw YOLOv8 output: boxes, class names, confidence scores
    2. "tracking"   – ByteTrack IDs, track trails, re-ID visualization
    3. "ioa"        – IoA spatial association: rider<>motorcycle overlap %
    4. "violation"  – N-frame confirmation logic: streak, patience, status
    5. "full"       – Combined view of all stages (default)

Usage:
    uv run python -m src.pipeline.main --source video.mp4 --debug detection --no-kafka
    uv run python -m src.pipeline.main --source video.mp4 --debug tracking --no-kafka
    uv run python -m src.pipeline.main --source video.mp4 --debug ioa --no-kafka
    uv run python -m src.pipeline.main --source video.mp4 --debug violation --no-kafka
    uv run python -m src.pipeline.main --source video.mp4 --debug full --no-kafka
"""

import cv2
import numpy as np
import supervision as sv
from collections import defaultdict

from config.settings import settings
from src.detector.violation import compute_ioa, compute_iou


# ── Colour Palette (BGR) ────────────────────────────────────────────
CLR_HELMET = (80, 200, 80)        # green  - helmet OK
CLR_NO_HELMET = (60, 60, 230)     # red    - no helmet
CLR_MOTORCYCLE = (200, 180, 60)   # teal   - motorcycle
CLR_WHITE = (255, 255, 255)
CLR_BLACK = (0, 0, 0)
CLR_DARK_BG = (30, 30, 30)
CLR_ORANGE = (0, 165, 255)        # orange - suspect / streak
CLR_YELLOW = (0, 230, 255)        # yellow - info
CLR_CYAN = (230, 200, 60)         # cyan   - tracking
CLR_GREY = (140, 140, 140)
CLR_LIGHT_GREY = (200, 200, 200)
CLR_CONFIRMED = (60, 60, 230)     # red    - confirmed violation
CLR_STREAK = (0, 165, 255)        # orange - building streak
CLR_SAFE = (80, 200, 80)          # green  - safe / helmet
CLR_NO_ASSOC = (100, 160, 200)    # warm   - no motorcycle assoc

# Class ID -> colour
CLASS_COLORS = {0: CLR_HELMET, 1: CLR_NO_HELMET, 2: CLR_MOTORCYCLE}
CLASS_LABELS = {0: "HELMET", 1: "NO_HELMET", 2: "MOTORCYCLE"}


class DebugDisplay:
    """
    Debug visualizer for each pipeline stage.

    Stores per-track history for trail drawing and maintains references
    to the violation checker state for status overlay.
    """

    def __init__(self, violation_checker=None):
        self.violation_checker = violation_checker
        self._track_trails: dict[int, list[tuple[int, int]]] = defaultdict(list)
        self._max_trail_length = 60
        self._frame_count = 0
        self._fps = 0.0
        self._total_violations = 0

    def set_stats(self, frame_count: int, fps: float, total_violations: int):
        """Update stats from the pipeline loop."""
        self._frame_count = frame_count
        self._fps = fps
        self._total_violations = total_violations

    # ==================================================================
    # MODE 1: DETECTION - Raw YOLOv8 Output
    # ==================================================================

    def draw_detection(
        self, frame: np.ndarray, detections: sv.Detections
    ) -> np.ndarray:
        """Raw YOLOv8 detections BEFORE tracking. Boxes + class + confidence."""
        out = frame.copy()
        h, w = out.shape[:2]

        if len(detections) == 0:
            self._draw_header(out, "DETECTION (YOLOv8)", "No detections")
            return out

        class_ids = detections.class_id
        xyxy = detections.xyxy
        confidences = detections.confidence
        class_names = detections.data.get("class_name", [None] * len(detections))

        counts = {0: 0, 1: 0, 2: 0}
        for cid in class_ids:
            counts[int(cid)] = counts.get(int(cid), 0) + 1

        for i in range(len(detections)):
            x1, y1, x2, y2 = xyxy[i].astype(int)
            cid = int(class_ids[i])
            conf = float(confidences[i])
            cls_name = class_names[i] if class_names[i] is not None else CLASS_LABELS.get(cid, "?")
            color = CLASS_COLORS.get(cid, CLR_GREY)

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # Confidence bar (top edge of box)
            bar_w = int((x2 - x1) * conf)
            cv2.rectangle(out, (x1, y1), (x1 + bar_w, y1 + 4), color, -1)

            # Label
            label = f"{cls_name} {conf:.0%}"
            _draw_pill(out, label, x1, y1 - 4, color, font_scale=0.40)

        # Header
        subtitle = (
            f"Total: {len(detections)} | "
            f"Helmet: {counts.get(0, 0)} | NoHelmet: {counts.get(1, 0)} | "
            f"Motor: {counts.get(2, 0)}"
        )
        self._draw_header(out, "DETECTION (YOLOv8)", subtitle)

        # Config panel
        self._draw_info_panel(out, [
            f"Model: {settings.model_path}",
            f"Conf: {settings.model_confidence}",
            f"NMS IoU: {settings.model_iou_threshold}",
            f"ImgSz: {settings.model_imgsz}px",
            f"Device: {settings.model_device}",
        ], x=w - 280, y=55)

        self._draw_class_legend(out, h)
        return out

    # ==================================================================
    # MODE 2: TRACKING - ByteTrack Visualization
    # ==================================================================

    def draw_tracking(
        self, frame: np.ndarray, raw_detections: sv.Detections, tracked: sv.Detections
    ) -> np.ndarray:
        """ByteTrack tracking: track IDs, centroid trails."""
        out = frame.copy()
        h, w = out.shape[:2]

        # Update trails
        if len(tracked) > 0 and tracked.tracker_id is not None:
            for i in range(len(tracked)):
                tid = int(tracked.tracker_id[i])
                box = tracked.xyxy[i]
                cx = int((box[0] + box[2]) / 2)
                cy = int((box[1] + box[3]) / 2)
                self._track_trails[tid].append((cx, cy))
                if len(self._track_trails[tid]) > self._max_trail_length:
                    self._track_trails[tid].pop(0)

        if len(tracked) == 0:
            self._draw_header(out, "TRACKING (ByteTrack)", "No tracked objects")
            return out

        class_ids = tracked.class_id
        xyxy = tracked.xyxy
        tracker_ids = tracked.tracker_id
        confidences = tracked.confidence
        class_names = tracked.data.get("class_name", [None] * len(tracked))

        # Draw trails (fading)
        for tid, trail in self._track_trails.items():
            if len(trail) < 2:
                continue
            for j in range(1, len(trail)):
                alpha = j / len(trail)
                color = tuple(int(c * alpha) for c in CLR_CYAN)
                thickness = max(1, int(2 * alpha))
                cv2.line(out, trail[j - 1], trail[j], color, thickness, cv2.LINE_AA)

        # Draw boxes
        unique_ids = set()
        for i in range(len(tracked)):
            x1, y1, x2, y2 = xyxy[i].astype(int)
            cid = int(class_ids[i])
            conf = float(confidences[i])
            tid = int(tracker_ids[i]) if tracker_ids is not None else None
            cls_name = class_names[i] if class_names[i] is not None else CLASS_LABELS.get(cid, "?")
            color = CLASS_COLORS.get(cid, CLR_GREY)

            if tid is not None:
                unique_ids.add(tid)

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # Centroid dot
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(out, (cx, cy), 4, CLR_CYAN, -1)

            # Label: ID + class + confidence
            id_str = f"#{tid} " if tid is not None else ""
            label = f"{id_str}{cls_name} {conf:.0%}"
            _draw_pill(out, label, x1, y1 - 4, color, font_scale=0.40)

        # Header
        raw_count = len(raw_detections) if raw_detections is not None else 0
        subtitle = (
            f"Raw: {raw_count} -> Tracked: {len(tracked)} | "
            f"IDs: {len(unique_ids)} | Trails: {len(self._track_trails)}"
        )
        self._draw_header(out, "TRACKING (ByteTrack)", subtitle)

        # Config panel
        self._draw_info_panel(out, [
            f"Activation: {settings.tracker_activation_threshold}",
            f"Lost Buffer: {settings.tracker_lost_buffer}F",
            f"Matching: {settings.tracker_matching_threshold}",
            f"Min Consec: {settings.tracker_min_consecutive_frames}",
            f"FPS: {settings.tracker_frame_rate} (0=auto)",
        ], x=w - 280, y=55)

        return out

    # ==================================================================
    # MODE 3: IoA - Spatial Association (CLEAN)
    # ==================================================================

    def draw_ioa(
        self, frame: np.ndarray, tracked: sv.Detections
    ) -> np.ndarray:
        """
        IoA spatial association - clean layout.

        Only the BEST motorcycle match per rider is shown.
        Compact inline IoA label on connecting line.
        Detailed table in the side panel (no floating boxes).
        """
        out = frame.copy()
        h, w = out.shape[:2]

        if len(tracked) == 0:
            self._draw_header(out, "IoA SPATIAL ASSOCIATION", "No detections")
            return out

        class_ids = tracked.class_id
        xyxy = tracked.xyxy
        tracker_ids = tracked.tracker_id
        confidences = tracked.confidence
        class_names = tracked.data.get("class_name", [None] * len(tracked))

        rider_mask = (class_ids == 0) | (class_ids == 1)
        motorbike_mask = (class_ids == 2)
        rider_indices = np.where(rider_mask)[0]
        motorbike_indices = np.where(motorbike_mask)[0]
        ioa_threshold = settings.association_iou_threshold

        # -- Find BEST motorcycle match per rider (not all NxM) --
        best_match: dict[int, dict] = {}
        for r_idx in rider_indices:
            r_box = xyxy[r_idx]
            best = {"m_idx": -1, "ioa": 0.0, "iou": 0.0, "passed": False}
            for m_idx in motorbike_indices:
                m_box = xyxy[m_idx]
                ioa_val = compute_ioa(r_box, m_box)
                if ioa_val > best["ioa"]:
                    iou_val = compute_iou(r_box, m_box)
                    best = {
                        "m_idx": int(m_idx),
                        "ioa": float(ioa_val),
                        "iou": float(iou_val),
                        "passed": ioa_val >= ioa_threshold,
                    }
            best_match[int(r_idx)] = best

        passed_count = sum(1 for b in best_match.values() if b["passed"])

        # -- Draw motorcycle boxes (translucent fill) --
        for m_idx in motorbike_indices:
            x1, y1, x2, y2 = xyxy[m_idx].astype(int)
            m_tid = _get_tid(tracker_ids, m_idx)
            overlay = out.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), CLR_MOTORCYCLE, -1)
            cv2.addWeighted(overlay, 0.12, out, 0.88, 0, out)
            cv2.rectangle(out, (x1, y1), (x2, y2), CLR_MOTORCYCLE, 2)
            id_str = f"#{m_tid} " if m_tid is not None else ""
            _draw_pill(out, f"{id_str}MOTORCYCLE", x1, y1 - 4, CLR_MOTORCYCLE, font_scale=0.38)

        # -- Draw riders + association lines (best match only) --
        for r_idx_int, match in best_match.items():
            r_box = xyxy[r_idx_int]
            x1, y1, x2, y2 = r_box.astype(int)
            r_cid = int(class_ids[r_idx_int])
            r_conf = float(confidences[r_idx_int])
            r_tid = _get_tid(tracker_ids, r_idx_int)
            cls_name = class_names[r_idx_int] if class_names[r_idx_int] is not None else CLASS_LABELS.get(r_cid, "?")

            passed = match["passed"]
            ioa_val = match["ioa"]
            m_idx = match["m_idx"]

            # Box colour based on association result
            if passed:
                box_color = CLR_SAFE
            elif m_idx >= 0 and ioa_val > 0:
                box_color = CLR_ORANGE
            else:
                box_color = CLR_NO_ASSOC

            cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2)

            # Rider label
            id_str = f"#{r_tid} " if r_tid is not None else ""
            _draw_pill(out, f"{id_str}{cls_name} {r_conf:.0%}", x1, y1 - 4, box_color, font_scale=0.38)

            # Connection line + intersection (only for best match)
            if m_idx >= 0:
                m_box = xyxy[m_idx]
                r_cx = int((r_box[0] + r_box[2]) / 2)
                r_cy = int((r_box[1] + r_box[3]) / 2)
                m_cx = int((m_box[0] + m_box[2]) / 2)
                m_cy = int((m_box[1] + m_box[3]) / 2)

                line_color = CLR_SAFE if passed else CLR_ORANGE
                cv2.line(out, (r_cx, r_cy), (m_cx, m_cy), line_color, 2, cv2.LINE_AA)
                cv2.circle(out, (r_cx, r_cy), 4, box_color, -1)
                cv2.circle(out, (m_cx, m_cy), 4, CLR_MOTORCYCLE, -1)

                # Intersection highlight
                ix1 = int(max(r_box[0], m_box[0]))
                iy1 = int(max(r_box[1], m_box[1]))
                ix2 = int(min(r_box[2], m_box[2]))
                iy2 = int(min(r_box[3], m_box[3]))
                if ix1 < ix2 and iy1 < iy2:
                    fill_color = CLR_SAFE if passed else CLR_ORANGE
                    overlay = out.copy()
                    cv2.rectangle(overlay, (ix1, iy1), (ix2, iy2), fill_color, -1)
                    cv2.addWeighted(overlay, 0.25, out, 0.75, 0, out)
                    cv2.rectangle(out, (ix1, iy1), (ix2, iy2), fill_color, 1)

                # Compact IoA label at midpoint of the line
                mid_x = (r_cx + m_cx) // 2
                mid_y = (r_cy + m_cy) // 2
                status = "OK" if passed else "X"
                _draw_pill(out, f"IoA:{ioa_val:.0%}[{status}]", mid_x - 30, mid_y, line_color, font_scale=0.36)

        # -- Header --
        subtitle = (
            f"Riders: {len(rider_indices)} | Motors: {len(motorbike_indices)} | "
            f"Assoc: {passed_count}/{len(rider_indices)} | "
            f"Thresh: {ioa_threshold:.0%}"
        )
        self._draw_header(out, "IoA SPATIAL ASSOCIATION", subtitle)

        # -- Side panel: detailed IoA table --
        table_lines = [
            f"IoA Threshold: {ioa_threshold:.0%}",
            f"Method: Intersect/RiderArea",
            "---",
        ]
        for r_idx_int in sorted(best_match.keys()):
            match = best_match[r_idx_int]
            r_tid = _get_tid(tracker_ids, r_idx_int)
            m_tid = _get_tid(tracker_ids, match["m_idx"]) if match["m_idx"] >= 0 else None
            r_box = xyxy[r_idx_int]
            area_r = int((r_box[2] - r_box[0]) * (r_box[3] - r_box[1]))
            p = "OK" if match["passed"] else "X"
            r_str = f"R#{r_tid}" if r_tid is not None else "R?"
            m_str = f"M#{m_tid}" if m_tid is not None else "---"
            table_lines.append(f"[{p}] {r_str} <> {m_str}")
            table_lines.append(f"  IoA={match['ioa']:.0%} IoU={match['iou']:.0%} A={area_r}")

        self._draw_info_panel(out, table_lines, x=w - 260, y=55)

        # Legend
        self._draw_legend(out, [
            ("Green = PASS (riding)", CLR_SAFE),
            ("Orange = FAIL (<thresh)", CLR_ORANGE),
            ("Blue = No motorcycle", CLR_NO_ASSOC),
            ("Teal = Motorcycle box", CLR_MOTORCYCLE),
        ], h)

        return out

    # ==================================================================
    # MODE 4: VIOLATION - N-Frame Confirmation Logic
    # ==================================================================

    def draw_violation(
        self, frame: np.ndarray, tracked: sv.Detections
    ) -> np.ndarray:
        """
        Violation confirmation: streak counters, patience, progress bars.
        Detailed per-track table in side panel (not floating boxes).
        """
        out = frame.copy()
        h, w = out.shape[:2]

        vc = self.violation_checker
        if vc is None:
            self._draw_header(out, "VIOLATION LOGIC", "No violation checker!")
            return out

        if len(tracked) == 0:
            self._draw_header(out, "VIOLATION LOGIC (N-Frame)", "No detections")
            return out

        class_ids = tracked.class_id
        xyxy = tracked.xyxy
        tracker_ids = tracked.tracker_id
        confidences = tracked.confidence
        class_names = tracked.data.get("class_name", [None] * len(tracked))

        confirm_needed = vc.confirm_frames
        patience = vc.patience_frames
        streak_map = vc._no_helmet_streak
        missing_map = vc._missing_count
        violated_ids = vc._violated_ids
        conf_accum = vc._confidence_accumulator

        # IoA for rider-motorcycle association
        rider_mask = (class_ids == 0) | (class_ids == 1)
        motorbike_mask = (class_ids == 2)
        rider_indices = np.where(rider_mask)[0]
        motorbike_indices = np.where(motorbike_mask)[0]
        ioa_threshold = settings.association_iou_threshold

        riding_set: set[int] = set()
        for r_idx in rider_indices:
            for m_idx in motorbike_indices:
                if compute_ioa(xyxy[r_idx], xyxy[m_idx]) >= ioa_threshold:
                    riding_set.add(int(r_idx))
                    break

        # -- Draw detections with status --
        rider_info_lines = []

        for i in range(len(tracked)):
            x1, y1, x2, y2 = xyxy[i].astype(int)
            cid = int(class_ids[i])
            conf = float(confidences[i])
            tid = _get_tid(tracker_ids, i)
            cls_name = class_names[i] if class_names[i] is not None else CLASS_LABELS.get(cid, "?")
            is_rider = cid in (0, 1)
            has_motor = i in riding_set

            streak = streak_map.get(tid, 0) if tid is not None else 0
            missing = missing_map.get(tid, 0) if tid is not None else 0
            is_violated = tid in violated_ids if tid is not None else False

            # Colour and status
            if cid == 2:
                color, status_text = CLR_MOTORCYCLE, "MOTOR"
            elif is_violated:
                color, status_text = CLR_CONFIRMED, "VIOLATED"
            elif streak > 0 and has_motor:
                color, status_text = CLR_STREAK, f"STREAK {streak}/{confirm_needed}"
            elif is_rider and not has_motor:
                color, status_text = CLR_NO_ASSOC, "NO MOTOR"
            elif cid == 0:
                color, status_text = CLR_SAFE, "SAFE"
            else:
                color, status_text = CLR_GREY, "WATCH"

            # Box
            thick = 3 if is_violated else 2
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)

            if is_violated:
                _draw_corner_accents(out, x1, y1, x2, y2, color)

            # Pill labels
            id_str = f"#{tid} " if tid is not None else ""
            _draw_pill(out, f"{id_str}{cls_name} {conf:.0%}", x1, y1 - 4, color, font_scale=0.38)
            _draw_pill(out, status_text, x1, y1 - 24, color, font_scale=0.38, bold=True)

            # Progress bar (below box) for active streaks
            if is_rider and streak > 0 and not is_violated and has_motor:
                progress = min(streak / confirm_needed, 1.0)
                bar_w = max(x2 - x1, 60)
                bar_h = 8
                bar_y = y2 + 4
                cv2.rectangle(out, (x1, bar_y), (x1 + bar_w, bar_y + bar_h), CLR_GREY, 1)
                fill_w = int(bar_w * progress)
                fill_color = CLR_CONFIRMED if progress >= 1.0 else CLR_STREAK
                cv2.rectangle(out, (x1, bar_y), (x1 + fill_w, bar_y + bar_h), fill_color, -1)
                cv2.putText(out, f"{progress:.0%}", (x1 + bar_w + 4, bar_y + 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.30, CLR_WHITE, 1, cv2.LINE_AA)

            # Collect info for side panel
            if is_rider and tid is not None:
                avg_conf_list = conf_accum.get(tid, [])
                avg_c = float(np.mean(avg_conf_list)) if avg_conf_list else conf
                rider_info_lines.append(
                    f"#{tid} {status_text}"
                )
                rider_info_lines.append(
                    f"  s={streak}/{confirm_needed} m={missing}/{patience} c={avg_c:.0%}"
                )

        # -- Header --
        total_streaks = sum(1 for v in streak_map.values() if v > 0)
        subtitle = (
            f"F:{self._frame_count} | Streaks: {total_streaks} | "
            f"Confirmed: {len(violated_ids)} | "
            f"N={confirm_needed} Pat={patience}"
        )
        self._draw_header(out, "VIOLATION LOGIC (N-Frame)", subtitle)

        # -- Side panel --
        panel_lines = [
            f"Confirm Frames: {confirm_needed}",
            f"Patience: {patience}",
            f"IoA Thresh: {ioa_threshold:.0%}",
            f"Camera: {vc.camera_id}",
            f"Violated: {len(violated_ids)}",
            "---",
        ]
        if rider_info_lines:
            panel_lines.extend(rider_info_lines)
        else:
            panel_lines.append("(no riders)")

        self._draw_info_panel(out, panel_lines, x=w - 280, y=55)

        # Legend
        self._draw_legend(out, [
            ("Green = Helmet OK", CLR_SAFE),
            ("Orange = Streak", CLR_STREAK),
            ("Red = CONFIRMED", CLR_CONFIRMED),
            ("Blue = No motorcycle", CLR_NO_ASSOC),
            ("Teal = Motorcycle", CLR_MOTORCYCLE),
        ], h)

        return out

    # ==================================================================
    # MODE 5: FULL - Combined Debug View
    # ==================================================================

    def draw_full(
        self, frame: np.ndarray, raw_detections: sv.Detections, tracked: sv.Detections
    ) -> np.ndarray:
        """Combined view: detection + tracking trails + IoA lines + violation."""
        out = frame.copy()
        h, w = out.shape[:2]

        vc = self.violation_checker

        if len(tracked) == 0:
            self._draw_header(out, "FULL DEBUG", "No detections")
            return out

        class_ids = tracked.class_id
        xyxy = tracked.xyxy
        tracker_ids = tracked.tracker_id
        confidences = tracked.confidence
        class_names = tracked.data.get("class_name", [None] * len(tracked))

        # Update trails
        if tracker_ids is not None:
            for i in range(len(tracked)):
                tid = int(tracker_ids[i])
                box = xyxy[i]
                cx = int((box[0] + box[2]) / 2)
                cy = int((box[1] + box[3]) / 2)
                self._track_trails[tid].append((cx, cy))
                if len(self._track_trails[tid]) > self._max_trail_length:
                    self._track_trails[tid].pop(0)

        # IoA (best match only)
        rider_mask = (class_ids == 0) | (class_ids == 1)
        motorbike_mask = (class_ids == 2)
        rider_indices = np.where(rider_mask)[0]
        motorbike_indices = np.where(motorbike_mask)[0]
        ioa_threshold = settings.association_iou_threshold

        ioa_map: dict[int, tuple[int, float, bool]] = {}
        for r_idx in rider_indices:
            best_ioa = 0.0
            best_m = -1
            for m_idx in motorbike_indices:
                v = compute_ioa(xyxy[r_idx], xyxy[m_idx])
                if v > best_ioa:
                    best_ioa = v
                    best_m = int(m_idx)
            ioa_map[int(r_idx)] = (best_m, best_ioa, best_ioa >= ioa_threshold)

        # Draw trails
        for tid, trail in self._track_trails.items():
            if len(trail) < 2:
                continue
            for j in range(1, len(trail)):
                alpha = j / len(trail)
                color = tuple(int(c * alpha) for c in CLR_CYAN)
                thickness = max(1, int(2 * alpha))
                cv2.line(out, trail[j - 1], trail[j], color, thickness, cv2.LINE_AA)

        # Violation state
        streak_map = vc._no_helmet_streak if vc else {}
        violated_ids = vc._violated_ids if vc else set()
        confirm_needed = vc.confirm_frames if vc else 3

        # IoA association lines (best match only)
        for r_idx_int, (m_idx, best_ioa, passed) in ioa_map.items():
            if m_idx < 0:
                continue
            r_box = xyxy[r_idx_int]
            m_box = xyxy[m_idx]
            r_cx = int((r_box[0] + r_box[2]) / 2)
            r_cy = int((r_box[1] + r_box[3]) / 2)
            m_cx = int((m_box[0] + m_box[2]) / 2)
            m_cy = int((m_box[1] + m_box[3]) / 2)

            line_color = CLR_SAFE if passed else CLR_GREY
            cv2.line(out, (r_cx, r_cy), (m_cx, m_cy), line_color, 1, cv2.LINE_AA)

            mid_x = (r_cx + m_cx) // 2
            mid_y = (r_cy + m_cy) // 2
            _draw_pill(out, f"IoA:{best_ioa:.0%}", mid_x - 20, mid_y, line_color, font_scale=0.30)

            if passed:
                ix1 = int(max(r_box[0], m_box[0]))
                iy1 = int(max(r_box[1], m_box[1]))
                ix2 = int(min(r_box[2], m_box[2]))
                iy2 = int(min(r_box[3], m_box[3]))
                if ix1 < ix2 and iy1 < iy2:
                    overlay = out.copy()
                    cv2.rectangle(overlay, (ix1, iy1), (ix2, iy2), CLR_SAFE, -1)
                    cv2.addWeighted(overlay, 0.18, out, 0.82, 0, out)

        # Draw detections
        for i in range(len(tracked)):
            x1, y1, x2, y2 = xyxy[i].astype(int)
            cid = int(class_ids[i])
            conf = float(confidences[i])
            tid = _get_tid(tracker_ids, i)
            cls_name = class_names[i] if class_names[i] is not None else CLASS_LABELS.get(cid, "?")
            is_rider = cid in (0, 1)
            has_motor = int(i) in ioa_map and ioa_map[int(i)][2]

            streak = streak_map.get(tid, 0) if tid is not None else 0
            is_violated = tid in violated_ids if tid is not None else False

            if cid == 2:
                color = CLR_MOTORCYCLE
            elif is_violated:
                color = CLR_CONFIRMED
            elif streak > 0 and has_motor:
                color = CLR_STREAK
            elif is_rider and not has_motor:
                color = CLR_NO_ASSOC
            else:
                color = CLR_SAFE

            thick = 3 if is_violated else 2
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)

            id_str = f"#{tid} " if tid is not None else ""
            _draw_pill(out, f"{id_str}{cls_name} {conf:.0%}", x1, y1 - 4, color, font_scale=0.36)

            # Status badge
            if is_violated:
                _draw_pill(out, "VIOLATION", x1, y1 - 22, CLR_CONFIRMED, font_scale=0.36, bold=True)
            elif streak > 0 and has_motor:
                _draw_pill(out, f"S:{streak}/{confirm_needed}", x1, y1 - 22, CLR_STREAK, font_scale=0.34, bold=True)
            elif is_rider and has_motor:
                ioa_val = ioa_map.get(int(i), (-1, 0, False))[1]
                _draw_pill(out, f"RIDING IoA:{ioa_val:.0%}", x1, y1 - 22, CLR_SAFE, font_scale=0.32)
            elif is_rider and not has_motor:
                _draw_pill(out, "NO MOTOR", x1, y1 - 22, CLR_NO_ASSOC, font_scale=0.32)

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(out, (cx, cy), 3, CLR_CYAN, -1)

        # Header
        subtitle = (
            f"F:{self._frame_count} | FPS:{self._fps:.1f} | "
            f"Viol:{len(violated_ids)} | "
            f"Det:{len(raw_detections) if raw_detections is not None else 0} -> "
            f"Trk:{len(tracked)}"
        )
        self._draw_header(out, "FULL DEBUG PIPELINE", subtitle)

        self._draw_legend(out, [
            ("Green = Safe", CLR_SAFE),
            ("Red = NoHelmet/Viol", CLR_NO_HELMET),
            ("Teal = Motorcycle", CLR_MOTORCYCLE),
            ("Blue = No motor", CLR_NO_ASSOC),
            ("Orange = Streak", CLR_STREAK),
            ("Cyan = Trail", CLR_CYAN),
        ], h)

        return out

    # ==================================================================
    # SHARED HELPERS
    # ==================================================================

    def _draw_header(self, frame: np.ndarray, title: str, subtitle: str):
        """Draw a top header bar."""
        h, w = frame.shape[:2]
        bar_h = 44

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), CLR_DARK_BG, -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

        cv2.putText(frame, title, (10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, CLR_WHITE, 2, cv2.LINE_AA)
        cv2.putText(frame, subtitle, (10, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, CLR_YELLOW, 1, cv2.LINE_AA)

        fps_text = f"F:{self._frame_count} FPS:{self._fps:.1f}"
        (tw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.putText(frame, fps_text, (w - tw - 10, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, CLR_LIGHT_GREY, 1, cv2.LINE_AA)

    def _draw_info_panel(self, frame: np.ndarray, lines: list[str], x: int, y: int):
        """Draw a semi-transparent info panel."""
        if not lines:
            return

        line_h = 16
        max_tw = 0
        for line in lines:
            if line.startswith("---"):
                continue
            (tw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.33, 1)
            max_tw = max(max_tw, tw)
        panel_w = max_tw + 18
        panel_h = len(lines) * line_h + 12

        h, w = frame.shape[:2]
        x = max(0, min(x, w - panel_w))

        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), CLR_DARK_BG, -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

        for i, line in enumerate(lines):
            ty = y + 13 + i * line_h
            if line.startswith("---"):
                cv2.line(frame, (x + 4, ty - 3), (x + panel_w - 4, ty - 3), CLR_GREY, 1)
                continue
            cv2.putText(frame, line, (x + 6, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, CLR_LIGHT_GREY, 1, cv2.LINE_AA)

    def _draw_class_legend(self, frame: np.ndarray, frame_h: int):
        """Draw class colour legend."""
        self._draw_legend(frame, [
            ("DRIVER_HELMET (0)", CLR_HELMET),
            ("DRIVER_NO_HELMET (1)", CLR_NO_HELMET),
            ("MOTORCYCLE (2)", CLR_MOTORCYCLE),
        ], frame_h)

    def _draw_legend(self, frame: np.ndarray, items: list[tuple[str, tuple]], frame_h: int):
        """Draw a compact legend in bottom-left."""
        line_h = 17
        legend_h = len(items) * line_h + 8
        legend_w = 0
        for text, _ in items:
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)
            legend_w = max(legend_w, tw + 26)
        ly = frame_h - legend_h

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, ly), (legend_w, frame_h), CLR_DARK_BG, -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        for idx, (text, color) in enumerate(items):
            ty = ly + 13 + idx * line_h
            cv2.circle(frame, (10, ty - 3), 4, color, -1)
            cv2.putText(frame, text, (20, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, color, 1, cv2.LINE_AA)


# =====================================================================
# Module-level drawing utilities
# =====================================================================

def _get_tid(tracker_ids, idx: int):
    """Safely extract tracker_id as int or None."""
    if tracker_ids is None:
        return None
    tid = tracker_ids[idx]
    return int(tid) if tid is not None else None


def _draw_pill(
    img: np.ndarray,
    text: str,
    x: int, y: int,
    color: tuple,
    font_scale: float = 0.40,
    bold: bool = False,
):
    """Draw a compact pill label with semi-transparent background."""
    thick = 2 if bold else 1
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
    pad_x, pad_y = 4, 2
    bx1 = x
    by1 = max(y - th - 2 * pad_y, 0)
    bx2 = x + tw + 2 * pad_x
    by2 = y

    overlay = img.copy()
    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), color, -1)
    cv2.addWeighted(overlay, 0.72, img, 0.28, 0, img)

    text_y = by2 - pad_y
    cv2.putText(img, text, (x + pad_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, CLR_WHITE, thick, cv2.LINE_AA)


def _draw_corner_accents(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: tuple,
):
    """Draw corner accent lines for confirmed violations."""
    accent_len = min(18, (x2 - x1) // 3, (y2 - y1) // 3)
    for cx, cy, dx, dy in [
        (x1, y1, 1, 1), (x2, y1, -1, 1),
        (x1, y2, 1, -1), (x2, y2, -1, -1),
    ]:
        cv2.line(img, (cx, cy), (cx + dx * accent_len, cy), color, 3)
        cv2.line(img, (cx, cy), (cx, cy + dy * accent_len), color, 3)
