from __future__ import annotations

import math
import time
from typing import Any

from src.contracts import GesturePacket, Vec3
from src.ports import GestureInputPort


class GestureInputServiceStub(GestureInputPort):
    """手势输入服务的 MVP 模拟实现。

    当前版本不依赖真实摄像头，目标是把 gesture 模块的完整结构先跑通：
    - 生命周期完整
    - `GesturePacket` 字段完整
    - 能模拟 tracked / temporarily_lost / not_detected
    - 能模拟 open / pinch_candidate / pinched / release_candidate
    """

    def __init__(self) -> None:
        self._status = "STOPPED"
        self._started = False
        self._frame_id = 0
        self._last_timestamp_ms = 0
        self._last_error: str | None = None
        self._hand_id = "hand-0"
        self._synthetic_tick = 0

        self._tracking_state = "not_detected"
        self._pinch_state = "open"
        self._confidence = 0.0
        self._tracking_loss_streak = 0
        self._pinch_candidate_streak = 0
        self._release_candidate_streak = 0

        self._last_index_tip = Vec3(0.0, 0.0, 0.0)
        self._last_thumb_tip = Vec3(0.0, 0.0, 0.0)
        self._last_palm_center = Vec3(0.0, 0.0, 0.0)
        self._last_velocity: Vec3 | None = None

        self._backend: Any = None

    def start(self) -> None:
        if self._started:
            return

        self._status = "INITIALIZING"
        self._last_error = None
        self._reset_runtime_state()

        try:
            self._setup_backend()
            self._started = True
            self._status = "RUNNING"
        except Exception as exc:
            self._last_error = str(exc)
            self._status = "DEGRADED"
            raise

    def poll(self) -> GesturePacket | None:
        if not self._started:
            return None

        try:
            raw_frame = self._read_frame()
            hand_data = self._detect_hand(raw_frame)

            if hand_data is None:
                self._tracking_loss_streak += 1
                tracking_state = self._compute_tracking_state(None)
                pinch_state = self._compute_pinch_state(None, None)
                confidence = self._compute_confidence(None, tracking_state)
                velocity = Vec3(0.0, 0.0, 0.0)

                self._tracking_state = tracking_state
                self._pinch_state = pinch_state
                self._confidence = confidence
                self._last_velocity = velocity

                return self._build_packet(
                    tracking_state=tracking_state,
                    pinch_state=pinch_state,
                    confidence=confidence,
                    index_tip=self._last_index_tip,
                    thumb_tip=self._last_thumb_tip,
                    palm_center=self._last_palm_center,
                    velocity=velocity,
                    debug={
                        "reason": "no_hand_detected",
                        "tracking_loss_streak": self._tracking_loss_streak,
                    },
                )

            self._tracking_loss_streak = 0

            index_tip = self._normalize_vec3(hand_data["index_tip"])
            thumb_tip = self._normalize_vec3(hand_data["thumb_tip"])
            palm_center = self._normalize_vec3(hand_data["palm_center"])

            tracking_state = self._compute_tracking_state(hand_data)
            pinch_state = self._compute_pinch_state(index_tip, thumb_tip)
            confidence = self._compute_confidence(hand_data, tracking_state)

            index_tip = self._smooth_vec3(index_tip, self._last_index_tip)
            thumb_tip = self._smooth_vec3(thumb_tip, self._last_thumb_tip)
            palm_center = self._smooth_vec3(palm_center, self._last_palm_center)
            velocity = self._compute_velocity(palm_center, self._last_palm_center)

            self._last_index_tip = index_tip
            self._last_thumb_tip = thumb_tip
            self._last_palm_center = palm_center
            self._tracking_state = tracking_state
            self._pinch_state = pinch_state
            self._confidence = confidence
            self._last_velocity = velocity

            return self._build_packet(
                tracking_state=tracking_state,
                pinch_state=pinch_state,
                confidence=confidence,
                index_tip=index_tip,
                thumb_tip=thumb_tip,
                palm_center=palm_center,
                velocity=velocity,
                debug={
                    "source": "stub_backend",
                    "synthetic_tick": self._synthetic_tick,
                },
            )
        except Exception as exc:
            self._last_error = str(exc)
            self._status = "DEGRADED"
            return None

    def health(self) -> dict:
        return {
            "status": self._status,
            "started": self._started,
            "frame_id": self._frame_id,
            "hand_id": self._hand_id,
            "tracking_state": self._tracking_state,
            "pinch_state": self._pinch_state,
            "confidence": self._confidence,
            "tracking_loss_streak": self._tracking_loss_streak,
            "last_error": self._last_error,
        }

    def stop(self) -> None:
        if not self._started and self._status == "STOPPED":
            return

        self._teardown_backend()
        self._started = False
        self._status = "STOPPED"

    def _reset_runtime_state(self) -> None:
        self._frame_id = 0
        self._last_timestamp_ms = 0
        self._synthetic_tick = 0
        self._tracking_state = "not_detected"
        self._pinch_state = "open"
        self._confidence = 0.0
        self._tracking_loss_streak = 0
        self._pinch_candidate_streak = 0
        self._release_candidate_streak = 0
        self._last_index_tip = Vec3(0.0, 0.0, 0.0)
        self._last_thumb_tip = Vec3(0.0, 0.0, 0.0)
        self._last_palm_center = Vec3(0.0, 0.0, 0.0)
        self._last_velocity = None

    def _setup_backend(self) -> None:
        self._backend = {"mode": "stub"}

    def _teardown_backend(self) -> None:
        self._backend = None

    def _read_frame(self) -> dict[str, Any]:
        self._synthetic_tick += 1
        return {
            "timestamp_ms": self._next_timestamp_ms(),
            "mode": "stub_frame",
            "tick": self._synthetic_tick,
        }

    def _detect_hand(self, raw_frame: dict[str, Any]) -> dict[str, Any] | None:
        tick = int(raw_frame["tick"])

        if 90 <= tick % 180 <= 96:
            return None

        t = tick / 10.0
        palm_center = Vec3(
            x=0.2 * math.sin(t),
            y=0.1 * math.cos(t),
            z=0.1,
        )
        index_tip = Vec3(
            x=palm_center.x + 0.08,
            y=palm_center.y + 0.06,
            z=0.1,
        )

        pinch_cycle = tick % 80
        if pinch_cycle < 25:
            pinch_offset = 0.09
        elif pinch_cycle < 40:
            pinch_offset = 0.045
        elif pinch_cycle < 55:
            pinch_offset = 0.02
        elif pinch_cycle < 70:
            pinch_offset = 0.055
        else:
            pinch_offset = 0.085

        thumb_tip = Vec3(
            x=index_tip.x - pinch_offset,
            y=index_tip.y - 0.01,
            z=0.1,
        )

        return {
            "index_tip": index_tip,
            "thumb_tip": thumb_tip,
            "palm_center": palm_center,
            "raw_confidence": 0.9,
        }

    def _normalize_vec3(self, value: Vec3) -> Vec3:
        """把坐标限制在 [-1.0, 1.0]，视为 camera_norm。"""
        return Vec3(
            x=max(-1.0, min(1.0, value.x)),
            y=max(-1.0, min(1.0, value.y)),
            z=max(-1.0, min(1.0, value.z)),
        )

    def _compute_tracking_state(self, hand_data: dict[str, Any] | None) -> str:
        if hand_data is None:
            if self._tracking_loss_streak <= 2:
                return "temporarily_lost"
            return "not_detected"
        return "tracked"

    def _compute_pinch_state(self, index_tip: Vec3 | None, thumb_tip: Vec3 | None) -> str:
        if index_tip is None or thumb_tip is None:
            if self._pinch_state in {"pinched", "pinch_candidate", "release_candidate"}:
                self._release_candidate_streak += 1
                if self._release_candidate_streak <= 2:
                    return "release_candidate"
            self._pinch_candidate_streak = 0
            self._release_candidate_streak = 0
            return "open"

        pinch_distance = self._distance(index_tip, thumb_tip)
        pinch_enter_threshold = 0.05
        pinch_hold_threshold = 0.03
        release_threshold = 0.075

        if pinch_distance <= pinch_hold_threshold:
            self._pinch_candidate_streak += 1
            self._release_candidate_streak = 0
            if self._pinch_candidate_streak >= 2:
                return "pinched"
            return "pinch_candidate"

        if pinch_distance <= pinch_enter_threshold:
            self._pinch_candidate_streak += 1
            self._release_candidate_streak = 0
            if self._pinch_state == "pinched":
                return "pinched"
            return "pinch_candidate"

        self._pinch_candidate_streak = 0

        if self._pinch_state in {"pinched", "pinch_candidate", "release_candidate"} and pinch_distance >= release_threshold:
            self._release_candidate_streak += 1
            if self._release_candidate_streak <= 2:
                return "release_candidate"
            self._release_candidate_streak = 0
            return "open"

        self._release_candidate_streak = 0
        return "open"

    def _compute_confidence(self, hand_data: dict[str, Any] | None, tracking_state: str) -> float:
        if tracking_state != "tracked":
            if tracking_state == "temporarily_lost":
                return 0.25
            return 0.0

        assert hand_data is not None
        raw_confidence = float(hand_data.get("raw_confidence", 0.5))
        confidence_bonus = 0.05 if self._pinch_state == "pinched" else 0.0
        return max(0.0, min(1.0, raw_confidence + confidence_bonus))

    def _smooth_vec3(self, current: Vec3, previous: Vec3, alpha: float = 0.7) -> Vec3:
        return Vec3(
            x=current.x * alpha + previous.x * (1 - alpha),
            y=current.y * alpha + previous.y * (1 - alpha),
            z=current.z * alpha + previous.z * (1 - alpha),
        )

    def _build_packet(
        self,
        tracking_state: str,
        pinch_state: str,
        confidence: float,
        index_tip: Vec3,
        thumb_tip: Vec3,
        palm_center: Vec3,
        velocity: Vec3 | None,
        debug: dict[str, Any] | None = None,
    ) -> GesturePacket:
        self._frame_id += 1

        return GesturePacket(
            contract_version="0.1.0",
            frame_id=self._frame_id,
            timestamp_ms=self._next_timestamp_ms(),
            hand_id=self._hand_id,
            tracking_state=tracking_state,
            confidence=confidence,
            pinch_state=pinch_state,
            index_tip=index_tip,
            thumb_tip=thumb_tip,
            palm_center=palm_center,
            coordinate_space="camera_norm",
            pinch_distance=self._distance(index_tip, thumb_tip),
            velocity=velocity,
            smoothing_hint={"method": "linear", "alpha": 0.7},
            debug=debug,
        )

    def _compute_velocity(self, current: Vec3, previous: Vec3) -> Vec3:
        return Vec3(
            x=current.x - previous.x,
            y=current.y - previous.y,
            z=current.z - previous.z,
        )

    def _next_timestamp_ms(self) -> int:
        now = int(time.time() * 1000)
        if now <= self._last_timestamp_ms:
            now = self._last_timestamp_ms + 1
        self._last_timestamp_ms = now
        return now

    def _distance(self, a: Vec3, b: Vec3) -> float:
        dx = a.x - b.x
        dy = a.y - b.y
        dz = a.z - b.z
        return (dx * dx + dy * dy + dz * dz) ** 0.5
