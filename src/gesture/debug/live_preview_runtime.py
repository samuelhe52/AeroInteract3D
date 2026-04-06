from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time
from typing import Any

import cv2

from src.gesture.constants import DEBUG_FPS_SAMPLE_WINDOW
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
    secondary_landmarks: tuple[int, int, int] = (255, 180, 80)
    secondary_bones: tuple[int, int, int] = (170, 90, 255)
    text: tuple[int, int, int] = (240, 240, 240)
    panel: tuple[int, int, int] = (20, 24, 32)
    scale_neutral: tuple[int, int, int] = (220, 220, 220)
    scale_up: tuple[int, int, int] = (90, 240, 120)
    scale_down: tuple[int, int, int] = (255, 165, 80)


class GesturePreviewWindow:
    def __init__(self, *, window_name: str = "AeroInteract3D Gesture Preview") -> None:
        self.window_name = window_name
        self.is_open = True
        self._colors = OverlayColors()
        self._sample_times: deque[float] = deque(maxlen=DEBUG_FPS_SAMPLE_WINDOW)

    def render(
        self,
        frame_bgr,
        *,
        observation: RawHandObservation | None,
        packet: GesturePacket,
        secondary_observation: RawHandObservation | None = None,
    ) -> None:
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
        if secondary_observation is not None:
            self._draw_landmarks(
                canvas,
                secondary_observation.landmarks,
                width=width,
                height=height,
                bone_color=self._colors.secondary_bones,
                landmark_color=self._colors.secondary_landmarks,
            )
        self._draw_secondary_hand_overlay(canvas, packet=packet, width=width, height=height)

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
        cv2.rectangle(canvas, (12, 12), (365, 260), self._colors.panel, thickness=-1)
        cv2.rectangle(canvas, (12, 12), (365, 260), (60, 68, 86), thickness=1)

    def _draw_status_text(self, canvas, *, packet: GesturePacket, fps: float) -> None:
        primary_hand = self._hand_payload(packet, "primary_hand")
        secondary_hand = self._hand_payload(packet, "secondary_hand")
        primary_label = self._hand_label(primary_hand)
        secondary_label = self._hand_label(secondary_hand)
        primary_detected = self._is_hand_detected(primary_hand)
        secondary_detected = self._is_hand_detected(secondary_hand)
        primary_state = "已识别" if primary_detected else "未识别"
        secondary_state = "已识别" if secondary_detected else "未识别"
        both_hands_recognized = (
            primary_detected
            and secondary_detected
        )
        lines = (
            f"frame: {packet.frame_id}",
            f"tracking: {packet.tracking_state}",
            f"pinch: {packet.pinch_state}",
            f"主手识别: {primary_state}",
            f"主手详情: {primary_label}",
            f"副手识别: {secondary_state}",
            f"副手详情: {secondary_label if secondary_hand is not None else '无'}",
            f"双手同时识别: {'是' if both_hands_recognized else '否'}",
            f"双手数: {1 + int(secondary_hand is not None)}",
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

        scale_ratio = self._scale_ratio(packet)
        scale_color = self._scale_ratio_color(scale_ratio)
        cv2.putText(
            canvas,
            f"scale_ratio: {scale_ratio:.2f}x",
            (24, 34 + (len(lines) * 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            scale_color,
            2,
            cv2.LINE_AA,
        )

    def _scale_ratio(self, packet: GesturePacket) -> float:
        debug_payload = packet.debug if isinstance(packet.debug, dict) else {}
        dual_scale = debug_payload.get("dual_scale") if isinstance(debug_payload, dict) else None
        if isinstance(dual_scale, dict):
            ratio = dual_scale.get("ratio")
            if isinstance(ratio, (int, float)):
                return float(ratio)

        dual_hand = debug_payload.get("dual_hand") if isinstance(debug_payload, dict) else None
        if isinstance(dual_hand, dict):
            ratio = dual_hand.get("scale_ratio")
            if isinstance(ratio, (int, float)):
                return float(ratio)

        return 1.0

    def _scale_ratio_color(self, ratio: float) -> tuple[int, int, int]:
        if ratio > 1.02:
            return self._colors.scale_up
        if ratio < 0.98:
            return self._colors.scale_down
        return self._colors.scale_neutral

    def _draw_landmarks(
        self,
        canvas,
        landmarks: list[Vec3],
        *,
        width: int,
        height: int,
        bone_color: tuple[int, int, int] | None = None,
        landmark_color: tuple[int, int, int] | None = None,
    ) -> None:
        resolved_bone_color = self._colors.bones if bone_color is None else bone_color
        resolved_landmark_color = self._colors.landmarks if landmark_color is None else landmark_color
        points = [self._image_point(landmark, width=width, height=height) for landmark in landmarks]
        for start_index, end_index in HAND_CONNECTIONS:
            cv2.line(canvas, points[start_index], points[end_index], resolved_bone_color, thickness=2)
        for point in points:
            cv2.circle(canvas, point, 4, resolved_landmark_color, thickness=-1)

    def _draw_focus_points(self, canvas, *, packet: GesturePacket, width: int, height: int) -> None:
        for point, color in (
            (packet.index_tip, (255, 220, 50)),
            (packet.thumb_tip, (50, 255, 220)),
            (packet.wrist, (220, 120, 255)),
        ):
            image_point = self._camera_norm_point(point, width=width, height=height)
            cv2.circle(canvas, image_point, 7, color, thickness=2)

    def _draw_secondary_hand_overlay(self, canvas, *, packet: GesturePacket, width: int, height: int) -> None:
        secondary_hand = self._hand_payload(packet, "secondary_hand")
        if secondary_hand is None:
            return None

        self._draw_debug_hand_points(
            canvas,
            hand_payload=secondary_hand,
            width=width,
            height=height,
            point_colors=(
                (255, 205, 90),
                (90, 205, 255),
                (255, 120, 220),
            ),
            label="secondary",
        )
        return None

    def _draw_debug_hand_points(
        self,
        canvas,
        *,
        hand_payload: dict[str, Any],
        width: int,
        height: int,
        point_colors: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
        label: str,
    ) -> None:
        for key, color in zip(("index_tip", "thumb_tip", "wrist"), point_colors, strict=True):
            point = hand_payload.get(key)
            if not isinstance(point, dict):
                continue
            image_point = self._camera_norm_point(Vec3(point["x"], point["y"], point["z"]), width=width, height=height)
            cv2.circle(canvas, image_point, 8, color, thickness=2)

        wrist = hand_payload.get("wrist")
        if isinstance(wrist, dict):
            anchor = self._camera_norm_point(Vec3(wrist["x"], wrist["y"], wrist["z"]), width=width, height=height)
            cv2.putText(
                canvas,
                f"{label}:{self._hand_label(hand_payload)}",
                (anchor[0] + 10, anchor[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self._colors.text,
                1,
                cv2.LINE_AA,
            )

            cv2.circle(canvas, anchor, 10, (60, 220, 60) if label == "primary" else (255, 170, 60), thickness=2)

    def _hand_payload(self, packet: GesturePacket, field_name: str) -> dict[str, Any] | None:
        debug_payload = packet.debug if isinstance(packet.debug, dict) else None
        if not isinstance(debug_payload, dict):
            return None
        payload = debug_payload.get(field_name)
        if isinstance(payload, dict):
            return payload
        dual_hand = debug_payload.get("dual_hand")
        if isinstance(dual_hand, dict):
            nested = dual_hand.get(field_name)
            if isinstance(nested, dict):
                return nested
        return None

    def _hand_label(self, hand_payload: dict[str, Any] | None) -> str:
        if not isinstance(hand_payload, dict):
            return "n/a"
        handedness = hand_payload.get("handedness", "unknown")
        hand_id = hand_payload.get("hand_id", "hand")
        tracking_state = hand_payload.get("tracking_state", "unknown")
        return f"{hand_id}/{handedness}/{tracking_state}"

    def _is_hand_detected(self, hand_payload: dict[str, Any] | None) -> bool:
        if not isinstance(hand_payload, dict):
            return False
        tracking_state = hand_payload.get("tracking_state")
        if not isinstance(tracking_state, str):
            return False
        return tracking_state != "not_detected"

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