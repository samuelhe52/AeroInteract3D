from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time

import cv2

from src.constants import DEBUG_FPS_SAMPLE_WINDOW
from src.contracts import GesturePacket, Vec3
from src.gesture.runtime import RawHandObservation


HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)


@dataclass(slots=True)
class OverlayColors:
    landmarks: tuple[int, int, int] = (80, 225, 120)
    bones: tuple[int, int, int] = (45, 145, 245)
    text: tuple[int, int, int] = (240, 240, 240)
    panel: tuple[int, int, int] = (20, 24, 32)


class GesturePreviewWindow:
    def __init__(self, *, window_name: str = "AeroInteract3D Gesture Preview") -> None:
        self.window_name = window_name
        self.is_open = True
        self._colors = OverlayColors()
        self._sample_times: deque[float] = deque(maxlen=DEBUG_FPS_SAMPLE_WINDOW)

    def render(self, frame_bgr, *, observation: RawHandObservation | None, packet: GesturePacket) -> None:
        if not self.is_open:
            return None

        now = time.perf_counter()
        self._sample_times.append(now)
        canvas = frame_bgr.copy()
        height, width = canvas.shape[:2]
        self._draw_panel(canvas)
        self._draw_status_text(canvas, packet=packet, fps=self._measured_fps())

        if observation is not None:
            self._draw_landmarks(canvas, observation.landmarks, width=width, height=height)
            self._draw_focus_points(canvas, packet=packet, width=width, height=height)

        cv2.imshow(self.window_name, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in {27, ord("q")}:
            self.close()
        return None

    def close(self) -> None:
        if not self.is_open:
            return None
        self.is_open = False
        cv2.destroyWindow(self.window_name)

    def _draw_panel(self, canvas) -> None:
        cv2.rectangle(canvas, (12, 12), (365, 165), self._colors.panel, thickness=-1)
        cv2.rectangle(canvas, (12, 12), (365, 165), (60, 68, 86), thickness=1)

    def _draw_status_text(self, canvas, *, packet: GesturePacket, fps: float) -> None:
        lines = (
            f"frame: {packet.frame_id}",
            f"tracking: {packet.tracking_state}",
            f"pinch: {packet.pinch_state}",
            f"confidence: {packet.confidence:.2f}",
            f"pinch_distance: {0.0 if packet.pinch_distance is None else packet.pinch_distance:.3f}",
            f"wrist: ({packet.wrist.x:+.2f}, {packet.wrist.y:+.2f}, {packet.wrist.z:+.2f})",
            f"fps: {fps:.1f}",
        )
        for index, line in enumerate(lines):
            y = 34 + (index * 18)
            cv2.putText(
                canvas,
                line,
                (24, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self._colors.text,
                1,
                cv2.LINE_AA,
            )

    def _draw_landmarks(self, canvas, landmarks: list[Vec3], *, width: int, height: int) -> None:
        points = [self._image_point(landmark, width=width, height=height) for landmark in landmarks]
        for start_index, end_index in HAND_CONNECTIONS:
            cv2.line(canvas, points[start_index], points[end_index], self._colors.bones, thickness=2)
        for point in points:
            cv2.circle(canvas, point, 4, self._colors.landmarks, thickness=-1)

    def _draw_focus_points(self, canvas, *, packet: GesturePacket, width: int, height: int) -> None:
        for point, color in (
            (packet.index_tip, (255, 220, 50)),
            (packet.thumb_tip, (50, 255, 220)),
            (packet.wrist, (220, 120, 255)),
        ):
            image_point = self._camera_norm_point(point, width=width, height=height)
            cv2.circle(canvas, image_point, 7, color, thickness=2)

    def _image_point(self, landmark: Vec3, *, width: int, height: int) -> tuple[int, int]:
        return (int(landmark.x * width), int(landmark.y * height))

    def _camera_norm_point(self, point: Vec3, *, width: int, height: int) -> tuple[int, int]:
        normalized_x = (point.x + 1.0) * 0.5
        normalized_y = (1.0 - point.y) * 0.5
        return (int(normalized_x * width), int(normalized_y * height))

    def _measured_fps(self) -> float:
        if len(self._sample_times) < 2:
            return 0.0
        elapsed = self._sample_times[-1] - self._sample_times[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._sample_times) - 1) / elapsed


__all__ = ["GesturePreviewWindow"]