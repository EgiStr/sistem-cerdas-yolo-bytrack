"""
Ground Truth Annotator — MOT Format

OpenCV-based GUI to annotate/correct ground truth for MOT evaluation.
Loads prediction file (pred.txt) as starting point, lets user:
  - Navigate frames (arrow keys, trackbar slider)
  - Correct bounding boxes (drag to move, drag corners to resize)
  - Change/assign track IDs
  - Add new bounding boxes
  - Delete false positive boxes
  - Save corrected GT to gt.txt

Usage:
    uv run python -m src.evaluation.gt_annotator --mot-dir output_dir

Keyboard Controls:
    D / Right Arrow  : Next frame
    A / Left Arrow   : Previous frame
    W               : Jump +10 frames
    S               : Jump -10 frames
    N               : Add new bounding box mode (drag to draw)
    X               : Delete selected box
    C               : Change class ID of selected box (cycle 0->1->2)
    I               : Set track ID of selected box (type in terminal)
    R               : Propagate selected box to next N frames (type N in terminal)
    Space           : Toggle play/pause auto-advance
    Ctrl+S          : Save GT file
    Q / Esc         : Quit (prompts to save)
    H               : Show help overlay

Mouse:
    Left click       : Select nearest box
    Left drag on box : Move selected box
    Left drag corner : Resize selected box
    Left drag (N)    : Draw new bounding box (when in New Box mode)
"""

import cv2
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict
from loguru import logger


# ── Colours (BGR) ───────────────────────────────────────────────────
COLORS = [
    (80, 200, 80),    # 0: HELMET - green
    (60, 60, 230),    # 1: NO_HELMET - red
    (200, 180, 60),   # 2: MOTORCYCLE - teal
]
CLR_SELECTED = (0, 255, 255)   # yellow highlight
CLR_NEW_BOX  = (255, 150, 50)  # bright blue - new box preview
CLR_TEXT     = (255, 255, 255)
CLR_BG       = (30, 30, 30)
CLR_HINT     = (180, 180, 180)
CLASS_NAMES  = {0: "HELMET", 1: "NO_HELMET", 2: "MOTORCYCLE"}

# ── Key codes (from cv2.waitKeyEx) ──────────────────────────────────
KEY_ESC         = 27
KEY_SPACE       = 32
KEY_LEFT        = 65361
KEY_RIGHT       = 65363
KEY_UP          = 65362
KEY_DOWN        = 65364
KEY_DELETE       = 65535


class BBox:
    """Single bounding box annotation."""

    __slots__ = ("track_id", "x", "y", "w", "h", "conf", "class_id")

    def __init__(self, track_id: int, x: float, y: float,
                 w: float, h: float, conf: float = 1.0, class_id: int = -1):
        self.track_id = track_id
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.conf = conf
        self.class_id = class_id

    # ── Pixel coordinates ────────────────────────────────────────────
    @property
    def x1(self) -> int:
        return int(self.x)

    @property
    def y1(self) -> int:
        return int(self.y)

    @property
    def x2(self) -> int:
        return int(self.x + self.w)

    @property
    def y2(self) -> int:
        return int(self.y + self.h)

    @property
    def cx(self) -> int:
        return int(self.x + self.w / 2)

    @property
    def cy(self) -> int:
        return int(self.y + self.h / 2)

    def contains(self, px: int, py: int, margin: int = 8) -> bool:
        return (self.x1 - margin <= px <= self.x2 + margin and
                self.y1 - margin <= py <= self.y2 + margin)

    def corner_near(self, px: int, py: int, margin: int = 14) -> str | None:
        """Return corner name if click is near a resize handle."""
        corners = {
            "tl": (self.x1, self.y1),
            "tr": (self.x2, self.y1),
            "bl": (self.x1, self.y2),
            "br": (self.x2, self.y2),
        }
        for name, (cx, cy) in corners.items():
            if abs(px - cx) < margin and abs(py - cy) < margin:
                return name
        return None

    def to_mot_row(self, frame_id: int) -> list:
        return [
            frame_id, self.track_id,
            round(self.x, 2), round(self.y, 2),
            round(self.w, 2), round(self.h, 2),
            round(self.conf, 4), self.class_id, -1, -1
        ]

    def clone(self) -> "BBox":
        return BBox(self.track_id, self.x, self.y, self.w, self.h,
                    self.conf, self.class_id)


# =====================================================================
# Main annotator
# =====================================================================

class GTAnnotator:
    """Interactive Ground-Truth annotation GUI."""

    WINDOW = "GT Annotator - ITERA Smart Sentinel"

    def __init__(self, mot_dir: str):
        self.mot_dir = Path(mot_dir)
        self.frames_dir = self.mot_dir / "frames"
        self.pred_path = self.mot_dir / "pred.txt"
        self.gt_path = self.mot_dir / "gt.txt"

        if not self.frames_dir.exists():
            raise FileNotFoundError(f"Frames directory not found: {self.frames_dir}")

        # Sorted frame image list
        self.frame_files = sorted(self.frames_dir.glob("*.jpg"))
        if not self.frame_files:
            self.frame_files = sorted(self.frames_dir.glob("*.png"))
        if not self.frame_files:
            raise FileNotFoundError(f"No frame images in: {self.frames_dir}")

        self.total_frames = len(self.frame_files)
        self.current_frame_idx = 0

        # Annotations:  frame_id (1-based) -> list[BBox]
        self.annotations: dict[int, list[BBox]] = defaultdict(list)
        self._next_track_id = 1

        self._load_annotations()

        # ── UI / interaction state ───────────────────────────────────
        self.selected_idx: int | None = None
        self.dragging      = False
        self.resizing      = False
        self.resize_corner: str | None = None
        self.drag_offset   = (0.0, 0.0)

        self.new_box_mode  = False
        self.new_box_start: tuple[int, int] | None = None
        self.new_box_end:   tuple[int, int] | None = None

        self.show_help      = False
        self.unsaved_changes = False
        self.auto_play      = False

        # Cache to avoid reloading same frame
        self._cached_frame_idx = -1
        self._cached_frame: np.ndarray | None = None

    # ── Data I/O ─────────────────────────────────────────────────────

    def _load_annotations(self):
        """Load existing GT (preferred) or predictions as starting point."""
        source = (self.gt_path
                  if self.gt_path.exists() and self.gt_path.stat().st_size > 0
                  else self.pred_path)

        if not source.exists():
            logger.warning(f"No annotation source in {self.mot_dir}")
            return

        logger.info(f"Loading annotations from: {source}")
        count = 0
        max_tid = 0

        with open(source, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 7:
                    continue
                fid      = int(row[0])
                tid      = int(row[1])
                x, y     = float(row[2]), float(row[3])
                w, h     = float(row[4]), float(row[5])
                conf     = float(row[6])
                cls_id   = int(row[7]) if len(row) > 7 else -1

                self.annotations[fid].append(BBox(tid, x, y, w, h, conf, cls_id))
                max_tid = max(max_tid, tid)
                count += 1

        self._next_track_id = max_tid + 1
        logger.info(f"Loaded {count} boxes across {len(self.annotations)} frames")

    def save_gt(self):
        """Write annotations to gt.txt (MOT format)."""
        rows = []
        for fid in sorted(self.annotations.keys()):
            for box in self.annotations[fid]:
                rows.append(box.to_mot_row(fid))

        with open(self.gt_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        self.unsaved_changes = False
        logger.info(f"Saved {len(rows)} GT annotations -> {self.gt_path}")

    # ── Frame helpers ────────────────────────────────────────────────

    def _frame_id(self) -> int:
        """Current 1-based frame ID."""
        return self.current_frame_idx + 1

    def _load_frame(self) -> np.ndarray:
        if self._cached_frame_idx == self.current_frame_idx and self._cached_frame is not None:
            return self._cached_frame
        path = self.frame_files[self.current_frame_idx]
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Cannot read: {path}")
        self._cached_frame = frame
        self._cached_frame_idx = self.current_frame_idx
        return frame

    def _goto_frame(self, idx: int):
        """Navigate to frame index (clamped)."""
        old = self.current_frame_idx
        self.current_frame_idx = max(0, min(idx, self.total_frames - 1))
        if self.current_frame_idx != old:
            self.selected_idx = None
            self._cancel_drag()

    # ── Mouse handling ───────────────────────────────────────────────

    def _cancel_drag(self):
        self.dragging = False
        self.resizing = False
        self.resize_corner = None
        self.new_box_start = None
        self.new_box_end = None

    def _on_mouse(self, event, mx, my, flags, _param):
        fid = self._frame_id()
        boxes = self.annotations.get(fid, [])

        # ---- New-box creation mode ----
        if self.new_box_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.new_box_start = (mx, my)
                self.new_box_end = (mx, my)
            elif event == cv2.EVENT_MOUSEMOVE and self.new_box_start is not None:
                self.new_box_end = (mx, my)            # live preview
            elif event == cv2.EVENT_LBUTTONUP and self.new_box_start is not None:
                sx, sy = self.new_box_start
                bx, by = min(sx, mx), min(sy, my)
                bw, bh = abs(mx - sx), abs(my - sy)
                if bw > 10 and bh > 10:
                    nb = BBox(self._next_track_id, bx, by, bw, bh, 1.0, 1)
                    self._next_track_id += 1
                    self.annotations[fid].append(nb)
                    self.selected_idx = len(self.annotations[fid]) - 1
                    self.unsaved_changes = True
                    logger.info(f"Added box #{nb.track_id} at frame {fid}")
                self.new_box_start = None
                self.new_box_end = None
                self.new_box_mode = False              # auto-exit after creation
            return

        # ---- Normal select / drag / resize mode ----
        if event == cv2.EVENT_LBUTTONDOWN:
            idx = self._find_nearest_box(mx, my)
            self.selected_idx = idx
            if idx is not None and idx < len(boxes):
                box = boxes[idx]
                corner = box.corner_near(mx, my)
                if corner:
                    self.resizing = True
                    self.resize_corner = corner
                else:
                    self.dragging = True
                    self.drag_offset = (mx - box.x, my - box.y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging and self.selected_idx is not None and self.selected_idx < len(boxes):
                box = boxes[self.selected_idx]
                box.x = mx - self.drag_offset[0]
                box.y = my - self.drag_offset[1]
                self.unsaved_changes = True

            elif self.resizing and self.selected_idx is not None and self.selected_idx < len(boxes):
                box = boxes[self.selected_idx]
                self._apply_resize(box, mx, my)
                self.unsaved_changes = True

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.resizing = False
            self.resize_corner = None

    def _apply_resize(self, box: BBox, mx: int, my: int):
        """Resize box from the active corner."""
        c = self.resize_corner
        if c == "br":
            box.w = max(10, mx - box.x)
            box.h = max(10, my - box.y)
        elif c == "bl":
            old_x2 = box.x + box.w
            box.x = min(mx, old_x2 - 10)
            box.w = old_x2 - box.x
            box.h = max(10, my - box.y)
        elif c == "tr":
            old_y2 = box.y + box.h
            box.w = max(10, mx - box.x)
            box.y = min(my, old_y2 - 10)
            box.h = old_y2 - box.y
        elif c == "tl":
            old_x2 = box.x + box.w
            old_y2 = box.y + box.h
            box.x = min(mx, old_x2 - 10)
            box.y = min(my, old_y2 - 10)
            box.w = old_x2 - box.x
            box.h = old_y2 - box.y

    def _find_nearest_box(self, px: int, py: int) -> int | None:
        """Return index of the box whose centre is nearest to (px,py),
        among boxes that contain the click point."""
        boxes = self.annotations.get(self._frame_id(), [])
        best_idx = None
        best_dist = float("inf")
        for i, box in enumerate(boxes):
            if box.contains(px, py):
                d = (px - box.cx) ** 2 + (py - box.cy) ** 2
                if d < best_dist:
                    best_dist = d
                    best_idx = i
        return best_idx

    # ── Drawing ──────────────────────────────────────────────────────

    def _draw(self, frame: np.ndarray) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]
        fid = self._frame_id()
        boxes = self.annotations.get(fid, [])

        # ---- Draw all boxes ----
        for i, box in enumerate(boxes):
            sel = (i == self.selected_idx)
            cls_clr = COLORS[box.class_id] if 0 <= box.class_id < len(COLORS) else (150, 150, 150)
            color = CLR_SELECTED if sel else cls_clr
            thick = 3 if sel else 2

            cv2.rectangle(out, (box.x1, box.y1), (box.x2, box.y2), color, thick)

            # Label pill
            cls_name = CLASS_NAMES.get(box.class_id, f"cls{box.class_id}")
            label = f"#{box.track_id} {cls_name}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            lx, ly = box.x1, box.y1
            cv2.rectangle(out, (lx, ly - th - 8), (lx + tw + 8, ly), color, -1)
            cv2.putText(out, label, (lx + 4, ly - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_TEXT, 1, cv2.LINE_AA)

            # Resize handles when selected
            if sel:
                for cx, cy in [(box.x1, box.y1), (box.x2, box.y1),
                               (box.x1, box.y2), (box.x2, box.y2)]:
                    cv2.rectangle(out, (cx - 5, cy - 5), (cx + 5, cy + 5), CLR_SELECTED, -1)
                    cv2.rectangle(out, (cx - 5, cy - 5), (cx + 5, cy + 5), (0, 0, 0), 1)

        # ---- New-box preview (rubber-band rectangle) ----
        if self.new_box_mode and self.new_box_start and self.new_box_end:
            sx, sy = self.new_box_start
            ex, ey = self.new_box_end
            _draw_dashed_rect(out, min(sx, ex), min(sy, ey),
                              max(sx, ex), max(sy, ey), CLR_NEW_BOX, 2, 10)
            # dimension label
            bw, bh = abs(ex - sx), abs(ey - sy)
            cv2.putText(out, f"{bw}x{bh}", (min(sx, ex), min(sy, ey) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, CLR_NEW_BOX, 1, cv2.LINE_AA)

        # ---- Top status bar ----
        _draw_bar(out, 0, 0, w, 38)
        status = "MODIFIED" if self.unsaved_changes else "saved"
        mode = "DRAW NEW BOX (drag)" if self.new_box_mode else "SELECT"
        play_st = "PLAY" if self.auto_play else "PAUSE"
        info = (f"Frame {fid}/{self.total_frames}  |  "
                f"Boxes: {len(boxes)}  |  {mode}  |  {play_st}  |  [{status}]  |  [H] Help")
        cv2.putText(out, info, (8, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, CLR_TEXT, 1, cv2.LINE_AA)

        # ---- Bottom hint bar ----
        _draw_bar(out, 0, h - 28, w, h)
        hint = ("A/D: frame  |  W/S: +/-10  |  N: new box  |  "
                "X: delete  |  C: class  |  I: track ID  |  "
                "R: propagate  |  Ctrl+S: save  |  Q: quit")
        cv2.putText(out, hint, (8, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, CLR_HINT, 1, cv2.LINE_AA)

        # ---- Selected box info badge ----
        if self.selected_idx is not None and self.selected_idx < len(boxes):
            box = boxes[self.selected_idx]
            det = (f"Sel: #{box.track_id}  {CLASS_NAMES.get(box.class_id,'?')}  "
                   f"({box.x1},{box.y1})-({box.x2},{box.y2})  "
                   f"{int(box.w)}x{int(box.h)}  conf={box.conf:.2f}")
            (tw2, _), _ = cv2.getTextSize(det, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
            _draw_bar(out, 0, 38, tw2 + 16, 56)
            cv2.putText(out, det, (8, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, CLR_SELECTED, 1, cv2.LINE_AA)

        # ---- Help overlay ----
        if self.show_help:
            self._draw_help(out)

        return out

    @staticmethod
    def _draw_help(frame: np.ndarray):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        pad = 40
        cv2.rectangle(overlay, (pad, pad), (w - pad, h - pad), CLR_BG, -1)
        cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)

        lines = [
            "=== GT ANNOTATOR HELP ===",
            "",
            "NAVIGATION",
            "  D / Right     Next frame",
            "  A / Left      Previous frame",
            "  W / Up        Jump +10 frames",
            "  S / Down      Jump -10 frames",
            "  Trackbar      Drag to any frame",
            "  Space         Toggle auto-play",
            "",
            "EDITING",
            "  Click         Select box (nearest to cursor)",
            "  Drag box      Move selected box",
            "  Drag corner   Resize selected box",
            "  N             Enter new-box mode, then drag to draw",
            "  X             Delete selected box",
            "  C             Cycle class (HELMET->NO_HELMET->MOTORCYCLE)",
            "  I             Set track ID (type number in terminal)",
            "  R             Propagate box to next N frames",
            "",
            "FILE",
            "  Ctrl+S        Save ground truth (gt.txt)",
            "  Q / Esc       Quit (prompts to save)",
            "",
            "Press H again to close",
        ]
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (pad + 30, pad + 28 + i * 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.43, CLR_TEXT, 1, cv2.LINE_AA)

    # ── Trackbar callback ────────────────────────────────────────────

    def _on_trackbar(self, val: int):
        self._goto_frame(val)

    # ── Main loop ────────────────────────────────────────────────────

    def run(self):
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.WINDOW, self._on_mouse)
        cv2.createTrackbar("Frame", self.WINDOW, 0, max(1, self.total_frames - 1),
                           self._on_trackbar)

        logger.info(f"GT Annotator started - {self.total_frames} frames")
        logger.info("Press H for help")

        while True:
            # Sync trackbar position
            cv2.setTrackbarPos("Frame", self.WINDOW, self.current_frame_idx)

            frame = self._load_frame()
            display = self._draw(frame)
            cv2.imshow(self.WINDOW, display)

            # waitKeyEx returns FULL keycode (-1 = no key)
            delay = 40 if self.auto_play else 30
            key = cv2.waitKeyEx(delay)

            # Auto-play: advance when no key
            if self.auto_play and key == -1:
                self._goto_frame(self.current_frame_idx + 1)
                continue

            if key == -1:
                continue      # no key pressed -> do nothing

            self._handle_key(key)
            if key == ord("q") or key == KEY_ESC:
                break

        cv2.destroyAllWindows()
        logger.info("GT Annotator closed")

    def _handle_key(self, key: int):
        """Dispatch a single keypress."""
        fid = self._frame_id()
        boxes = self.annotations.get(fid, [])

        # ── Quit ────────────────────────────────────────────────────
        if key == ord("q") or key == KEY_ESC:
            if self.unsaved_changes:
                logger.warning("Unsaved changes!")
                try:
                    ans = input("Save before quitting? [y/n]: ").strip().lower()
                    if ans == "y":
                        self.save_gt()
                except EOFError:
                    pass
            return                                     # caller breaks the loop

        # ── Navigation ──────────────────────────────────────────────
        if key == ord("d") or key == KEY_RIGHT:
            self._goto_frame(self.current_frame_idx + 1)
        elif key == ord("a") or key == KEY_LEFT:
            self._goto_frame(self.current_frame_idx - 1)
        elif key == ord("w") or key == KEY_UP:
            self._goto_frame(self.current_frame_idx + 10)
        elif key == ord("s") or key == KEY_DOWN:
            self._goto_frame(self.current_frame_idx - 10)
        elif key == KEY_SPACE:
            self.auto_play = not self.auto_play
            logger.info(f"Auto-play: {'ON' if self.auto_play else 'OFF'}")

        # ── Mode toggles ───────────────────────────────────────────
        elif key == ord("n"):
            self.new_box_mode = not self.new_box_mode
            if not self.new_box_mode:
                self.new_box_start = self.new_box_end = None
            logger.info(f"New-box mode: {'ON - drag to draw' if self.new_box_mode else 'OFF'}")

        elif key == ord("h"):
            self.show_help = not self.show_help

        # ── Delete (only X key or Delete key, NOT the no-key code) ──
        elif key == ord("x") or key == KEY_DELETE:
            if self.selected_idx is not None and self.selected_idx < len(boxes):
                removed = boxes.pop(self.selected_idx)
                logger.info(f"Deleted box #{removed.track_id} at frame {fid}")
                self.selected_idx = None
                self.unsaved_changes = True

        # ── Cycle class ─────────────────────────────────────────────
        elif key == ord("c"):
            if self.selected_idx is not None and self.selected_idx < len(boxes):
                box = boxes[self.selected_idx]
                box.class_id = (box.class_id + 1) % 3
                logger.info(f"#{box.track_id} class -> {CLASS_NAMES.get(box.class_id,'?')}")
                self.unsaved_changes = True

        # ── Set track ID (via terminal input) ───────────────────────
        elif key == ord("i"):
            if self.selected_idx is not None and self.selected_idx < len(boxes):
                box = boxes[self.selected_idx]
                try:
                    new_id = int(input(f"  New track ID for #{box.track_id}: "))
                    old_id = box.track_id
                    box.track_id = new_id
                    self.unsaved_changes = True
                    logger.info(f"Track ID #{old_id} -> #{new_id}")
                except (ValueError, EOFError):
                    logger.warning("Cancelled / invalid ID")

        # ── Propagate box to next N frames ──────────────────────────
        elif key == ord("r"):
            if self.selected_idx is not None and self.selected_idx < len(boxes):
                box = boxes[self.selected_idx]
                try:
                    n = int(input(f"  Propagate #{box.track_id} to next N frames: "))
                    count = 0
                    for offset in range(1, n + 1):
                        target_fid = fid + offset
                        if target_fid > self.total_frames:
                            break
                        # Don't duplicate if same track_id already there
                        existing_tids = {b.track_id for b in self.annotations.get(target_fid, [])}
                        if box.track_id not in existing_tids:
                            self.annotations[target_fid].append(box.clone())
                            count += 1
                    self.unsaved_changes = True
                    logger.info(f"Propagated #{box.track_id} to {count} frames")
                except (ValueError, EOFError):
                    logger.warning("Cancelled / invalid number")

        # ── Save (Ctrl+S) ──────────────────────────────────────────
        elif key == 19:                                # Ctrl+S
            self.save_gt()


# =====================================================================
# Drawing helpers (module level)
# =====================================================================

def _draw_bar(img: np.ndarray, x1: int, y1: int, x2: int, y2: int):
    """Semi-transparent dark bar."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), CLR_BG, -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)


def _draw_dashed_rect(img, x1, y1, x2, y2, color, thickness=2, gap=10):
    """Dashed rectangle for new-box preview."""
    for (sx, sy), (ex, ey) in [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
                                ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]:
        dx, dy = ex - sx, ey - sy
        length = max(abs(dx), abs(dy))
        if length == 0:
            continue
        ndx, ndy = dx / length, dy / length
        pos, draw = 0, True
        while pos < length:
            seg = min(gap, length - pos)
            px1 = int(sx + ndx * pos)
            py1 = int(sy + ndy * pos)
            px2 = int(sx + ndx * (pos + seg))
            py2 = int(sy + ndy * (pos + seg))
            if draw:
                cv2.line(img, (px1, py1), (px2, py2), color, thickness, cv2.LINE_AA)
            pos += gap
            draw = not draw


# =====================================================================
# CLI entry point
# =====================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ground Truth Annotator for MOT evaluation"
    )
    parser.add_argument(
        "--mot-dir", type=str, required=True,
        help="MOT output directory (from mot_exporter / pipeline --export-mot)"
    )
    args = parser.parse_args()

    annotator = GTAnnotator(args.mot_dir)
    annotator.run()
