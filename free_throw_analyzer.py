from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import cv2  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
from ultralytics import YOLO  # pyright: ignore[reportMissingImports]

try:
    from filterpy.kalman import KalmanFilter  # pyright: ignore[reportMissingImports]
except Exception:  # pragma: no cover - optional dependency at runtime
    KalmanFilter = None


BBOX = Tuple[int, int, int, int]
POINT = Tuple[int, int]


@dataclass
class ShotEvent:
    result: str
    frame_idx: int


def _bbox_center(box: BBOX) -> POINT:
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def _point_in_bbox(point: POINT, box: BBOX) -> bool:
    px, py = point
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def _distance(p1: POINT, p2: POINT) -> float:
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


class BallTracker:
    """Optional Kalman-smoothed ball center tracking."""

    def __init__(self) -> None:
        self.last_center: Optional[POINT] = None
        self.kf = None
        if KalmanFilter is not None:
            self.kf = KalmanFilter(dim_x=4, dim_z=2)
            self.kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
            self.kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
            self.kf.P *= 1000.0
            self.kf.R *= 5.0
            self.kf.Q *= 0.05

    def update(self, observed_center: Optional[POINT]) -> Optional[POINT]:
        if self.kf is None:
            if observed_center is not None:
                self.last_center = observed_center
            return self.last_center

        if observed_center is not None and self.last_center is None:
            self.kf.x = np.array([[observed_center[0]], [observed_center[1]], [0.0], [0.0]])
            self.last_center = observed_center
            return observed_center

        if self.last_center is None:
            return None

        self.kf.predict()
        if observed_center is not None:
            self.kf.update(np.array([observed_center[0], observed_center[1]]))
        predicted = (int(self.kf.x[0][0]), int(self.kf.x[1][0]))
        self.last_center = predicted
        return predicted


class FreeThrowAnalyzer:
    def __init__(
        self,
        model_path: str = "best.pt",
        release_distance_threshold: float = 120.0,
        max_shot_frames: int = 120,
    ) -> None:
        self.model = YOLO(model_path)
        self.release_threshold = release_distance_threshold
        self.max_shot_frames = max_shot_frames

        self.ball_tracker = BallTracker()
        self.trajectory: Deque[POINT] = deque(maxlen=64)
        self.events: List[ShotEvent] = []

        self.score = 0
        self.miss = 0
        self.shot_active = False
        self.shot_start_frame = -1
        self.prev_ball_human_distance: Optional[float] = None
        self.ball_was_above_rim = False
        self.ball_entered_rim = False
        self.current_shot_scored = False

    def _pick_primary(self, boxes: List[Tuple[BBOX, float]]) -> Optional[BBOX]:
        if not boxes:
            return None
        boxes = sorted(boxes, key=lambda x: x[1], reverse=True)
        return boxes[0][0]

    def _extract_detections(self, frame: np.ndarray) -> Dict[str, Optional[BBOX]]:
        result = self.model.predict(frame, verbose=False)[0]

        detections: Dict[str, List[Tuple[BBOX, float]]] = {"ball": [], "human": [], "rim": []}
        names = result.names

        if result.boxes is not None:
            xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(xyxy, confs, classes):
                cls_name = names.get(int(cls_id), "")
                if cls_name in detections:
                    x1, y1, x2, y2 = box.tolist()
                    detections[cls_name].append(((x1, y1, x2, y2), float(conf)))

        return {
            "ball": self._pick_primary(detections["ball"]),
            "human": self._pick_primary(detections["human"]),
            "rim": self._pick_primary(detections["rim"]),
        }

    def _update_shot_logic(self, det: Dict[str, Optional[BBOX]], frame_idx: int) -> None:
        ball_box, human_box, rim_box = det["ball"], det["human"], det["rim"]
        ball_center: Optional[POINT] = _bbox_center(ball_box) if ball_box else None
        human_center: Optional[POINT] = _bbox_center(human_box) if human_box else None

        smoothed_ball_center = self.ball_tracker.update(ball_center)
        if smoothed_ball_center is not None:
            self.trajectory.append(smoothed_ball_center)

        if smoothed_ball_center is not None and human_center is not None:
            dist = _distance(smoothed_ball_center, human_center)
            if (
                not self.shot_active
                and self.prev_ball_human_distance is not None
                and self.prev_ball_human_distance <= self.release_threshold
                and dist > self.release_threshold
            ):
                self.shot_active = True
                self.shot_start_frame = frame_idx
                self.ball_was_above_rim = False
                self.ball_entered_rim = False
                self.current_shot_scored = False
            self.prev_ball_human_distance = dist

        if not self.shot_active:
            return

        if rim_box and smoothed_ball_center:
            rx1, ry1, rx2, ry2 = rim_box
            bx, by = smoothed_ball_center
            rim_mid_x = (rx1 + rx2) // 2
            horizontal_gate = abs(bx - rim_mid_x) <= max(25, (rx2 - rx1))

            if by < ry1 and horizontal_gate:
                self.ball_was_above_rim = True
            if _point_in_bbox(smoothed_ball_center, rim_box):
                self.ball_entered_rim = True
            if self.ball_was_above_rim and self.ball_entered_rim and by > ry2 and horizontal_gate:
                self.current_shot_scored = True

        shot_timed_out = (frame_idx - self.shot_start_frame) > self.max_shot_frames
        if shot_timed_out or self.current_shot_scored:
            if self.current_shot_scored:
                self.score += 1
                self.events.append(ShotEvent("BASKET", frame_idx))
            else:
                self.miss += 1
                self.events.append(ShotEvent("MISS", frame_idx))
            self.shot_active = False

    def _draw(self, frame: np.ndarray, det: Dict[str, Optional[BBOX]]) -> np.ndarray:
        vis = frame.copy()

        color_map = {"ball": (0, 255, 255), "human": (0, 255, 0), "rim": (0, 128, 255)}
        for label, box in det.items():
            if not box:
                continue
            x1, y1, x2, y2 = box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color_map[label], 2)
            cv2.putText(vis, label, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_map[label], 2)

        if len(self.trajectory) > 1:
            pts = np.array(self.trajectory, dtype=np.int32)
            cv2.polylines(vis, [pts], False, (255, 50, 50), 2)

        cv2.putText(vis, f"BASKET: {self.score}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 3)
        cv2.putText(vis, f"MISS: {self.miss}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 220), 3)
        status = "SHOT ACTIVE" if self.shot_active else "READY"
        cv2.putText(vis, f"STATE: {status}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        if self.events:
            cv2.putText(
                vis,
                f"LAST: {self.events[-1].result}",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )
        return vis

    def analyze_video(self, input_video: str, output_video: Optional[str] = None, display: bool = False) -> Dict[str, int]:
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise RuntimeError(f"Video could not be opened: {input_video}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            det = self._extract_detections(frame)
            self._update_shot_logic(det, frame_idx)
            vis = self._draw(frame, det)

            if writer is not None:
                writer.write(vis)
            if display:
                cv2.imshow("Free Throw Analysis", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_idx += 1

        cap.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()

        total = self.score + self.miss
        accuracy = (self.score / total) if total else 0.0
        return {
            "basket": self.score,
            "miss": self.miss,
            "total_shots": total,
            "accuracy_percent": round(accuracy * 100.0, 2),
        }
