from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from src.contracts import GesturePacket, Vec3
from src.gesture.debug.runtime import (
    DEFAULT_PINCH_ENTER_THRESHOLD,
    DEFAULT_PINCH_HOLD_THRESHOLD,
    DEFAULT_PINCH_RELEASE_THRESHOLD,
    DebugVideoConfig,
    create_hand_detector,
    distance,
    run_live_preview,
)
from src.ports import GestureInputPort
from src.utils.contracts import EXPECTED_CONTRACT_VERSION
from src.utils.runtime import (
    LIFECYCLE_DEGRADED,
    LIFECYCLE_INITIALIZING,
    LIFECYCLE_RUNNING,
    LIFECYCLE_STOPPED,
    build_health,
    error_entry,
)

# 这份文件是“学习版实现”，目标不是压缩代码量，
# 而是把 GestureInputServiceImpl 的每个设计决策讲清楚：
# 1. 输入从哪里来。
# 2. 状态怎么演化。
# 3. 为什么要在丢帧时继续产包。
# 4. 输出包里每个字段是如何得到的。
# 这份文件只用于学习和对照阅读，不应该被项目运行时路径依赖。

# 这一组阈值控制“手丢失多久算真的丢了”和“pinch 进入/保持/释放”的状态机灵敏度。
# TRACKING_TEMPORARY_LOSS_FRAMES：允许短时间检测失败，但不立刻判定整只手消失。
# PINCH_ENTER_THRESHOLD：进入 pinch 候选区的较宽阈值。
# PINCH_HOLD_THRESHOLD：确认已经 pinched 的更严格阈值。
# PINCH_RELEASE_THRESHOLD：从 pinch 状态退出时使用的释放阈值。
# PINCH_CONFIRM_FRAMES / RELEASE_CONFIRM_FRAMES：要求状态连续稳定几帧才真正切换。
TRACKING_TEMPORARY_LOSS_FRAMES = 2
PINCH_ENTER_THRESHOLD = DEFAULT_PINCH_ENTER_THRESHOLD
PINCH_HOLD_THRESHOLD = DEFAULT_PINCH_HOLD_THRESHOLD
PINCH_RELEASE_THRESHOLD = DEFAULT_PINCH_RELEASE_THRESHOLD
PINCH_CONFIRM_FRAMES = 2
RELEASE_CONFIRM_FRAMES = 2

# 平滑参数。越接近 1，越偏向当前帧；越接近 0，越偏向上一帧。
SMOOTHING_ALPHA = 0.65
HAND_MODEL_ENV_VAR = "AEROINTERACT3D_HAND_MODEL"


# 这段兼容逻辑是为了支持“直接运行这个文件”的场景。
# 如果不是作为包导入，而是 python src/gesture/service_impl_annotated.py 这种方式运行，
# 就手动把仓库根目录塞进 sys.path，保证 src.xxx 这种导入还能工作。
if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


@dataclass(slots=True)
class GestureMetrics:
    # 调了多少次 poll。
    polls_attempted: int = 0
    # 实际产出了多少个 GesturePacket。
    packets_emitted: int = 0
    # 其中有手被稳定检测到的包有多少个。
    tracked_packets: int = 0
    # 其中没有检测到手、但为了保持信号连续仍然发出的包有多少个。
    empty_packets: int = 0
    # 底层后端真正抛错的次数。
    backend_failures: int = 0


@dataclass(slots=True)
class GesturePreviewConfig:
    # 这一组字段描述“如何启动 gesture 调试预览”。
    # 它和 main.py 里的 AppConfig 属于同一种设计思路：
    # 先把命令行参数收束成一个 dataclass，再交给 build_xxx 函数去组装对象。
    log_level: str = "INFO"
    camera_index: int = 0
    target_fps: int = 30
    max_frames: int = 0
    hand_model: str | None = None
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    model_complexity: int = 1
    window_name: str = "Gesture Live Preview"
    mirror: bool = True
    draw_coordinates: bool = True


def _default_hand_model_path() -> Path:
    # 默认模型来自仓库根目录，所以在任意平台都能通过相对仓库位置找到。
    return Path(__file__).resolve().parents[2] / "hand_landmarker.task"


def _resolve_hand_model_path(hand_model: str | None) -> str | None:
    # 配置优先级：显式参数 > 环境变量 > 仓库根目录默认文件。
    if hand_model:
        return hand_model

    env_hand_model = os.getenv(HAND_MODEL_ENV_VAR)
    if env_hand_model:
        return env_hand_model

    default_model = _default_hand_model_path()
    if default_model.exists():
        return str(default_model)
    return None


def _close_detector_resource(detector_owner: Any | None, detector: Any | None = None) -> None:
    # 初始化中途失败时，需要把已经拿到的 detector 资源及时放掉。
    if detector_owner is not None and hasattr(detector_owner, "__exit__"):
        detector_owner.__exit__(None, None, None)
        return

    if detector is not None and hasattr(detector, "close"):
        detector.close()
        return

    if detector_owner is not None and hasattr(detector_owner, "close"):
        detector_owner.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    # parse_args 负责把 CLI 文本参数转成结构化 Namespace。
    # 这里的参数基本和 debug/live_preview 原来那套一致，
    # 只是现在统一收口到 service_impl，避免两个入口文件各维护一份参数表。
    parser = argparse.ArgumentParser(description="Gesture live preview debug entrypoint")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--target-fps", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means no frame limit")
    parser.add_argument("--window-name", default="Gesture Live Preview")
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--hide-coordinates", action="store_true")
    parser.add_argument("--hand-model", help="path to a MediaPipe hand_landmarker.task model file")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--model-complexity", type=int, choices=[0, 1, 2], default=1)
    return parser.parse_args(argv)


def setup_logging(level: str) -> None:
    # 入口层统一日志格式，方便 CLI 运行时看见一致的时间、级别和消息结构。
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def build_config(args: argparse.Namespace) -> GesturePreviewConfig:
    # 这一步把 argparse.Namespace 收紧成我们自己的配置数据对象。
    # 这样后续逻辑就不再依赖 argparse 的字段命名细节了。
    return GesturePreviewConfig(
        log_level=args.log_level.upper(),
        camera_index=args.camera_index,
        target_fps=args.target_fps,
        max_frames=args.max_frames,
        hand_model=args.hand_model,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        model_complexity=args.model_complexity,
        window_name=args.window_name,
        mirror=not args.no_mirror,
        draw_coordinates=not args.hide_coordinates,
    )


def build_service(config: GesturePreviewConfig) -> GestureInputServiceImpl:
    # build_service 是“配置 -> 正式服务对象”的翻译层。
    # 这样 main / 测试 / 其他入口如果要创建服务，就都走同一个构造规则。
    return GestureInputServiceImpl(
        camera_index=config.camera_index,
        hand_model=config.hand_model,
        min_detection_confidence=config.min_detection_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
        model_complexity=config.model_complexity,
    )


def build_preview_config(config: GesturePreviewConfig) -> DebugVideoConfig:
    # 调试预览底层跑的是 debug.runtime 里的 run_live_preview，
    # 它认的是 DebugVideoConfig，所以这里负责做第二次翻译。
    hand_model = _resolve_hand_model_path(config.hand_model)

    return DebugVideoConfig(
        input_video=None,
        camera_index=config.camera_index,
        hand_model=hand_model,
        output_dir=None,
        target_fps=float(config.target_fps),
        max_frames=config.max_frames,
        min_detection_confidence=config.min_detection_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
        model_complexity=config.model_complexity,
        window_name=config.window_name,
        mirror=config.mirror,
        draw_coordinates=config.draw_coordinates,
    )


class GestureInputServiceImpl(GestureInputPort):
    # 这就是 gesture 模块真正对外提供的实现类。
    # 它做的事情可以概括成一句话：
    # 持续读取摄像头帧 -> 提取关键点 -> 推断状态 -> 产出合同化的 GesturePacket。
    # 默认模型文件位置。live preview 和主程序都会依赖这个常量来给出提示。
    DEFAULT_HAND_MODEL_PATH = _default_hand_model_path()

    def __init__(
        self,
        camera_index: int = 0,
        hand_model: str | None = None,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
    ) -> None:
        # 仓库根目录主要用来拼 report/live 这种输出路径。
        self._repo_root = Path(__file__).resolve().parents[2]

        # 这些是输入配置，start() 之后会用于初始化摄像头和 detector。
        self._camera_index = camera_index
        self._hand_model = self._resolve_hand_model(hand_model)
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._model_complexity = model_complexity

        # _backend 里实际装的是 capture、detector、detector_owner 这些底层对象。
        # capture 负责取图像帧。
        # detector 负责从帧中检测手部关键点。
        # detector_owner 是可选的上下文拥有者，某些实现需要它来正确关闭资源。
        self._backend: dict[str, Any] | None = None
        self._detector_backend = "uninitialized"

        # 生命周期与 health 输出需要的公共状态。
        # _started 是“有没有启动成功”的布尔量。
        # lifecycle_state 是更细的生命周期语义，给 health 和上层 orchestration 看。
        self._started = False
        self.lifecycle_state = LIFECYCLE_STOPPED
        self._hand_id = "hand-0"
        self._errors: list[dict[str, Any]] = []
        self._metrics = GestureMetrics()

        # 下面这些是运行时缓存，主要有三类：
        # 1. 单调递增相关：frame_id / timestamp。
        # 2. 当前状态机相关：tracking / pinch / confidence。
        # 3. 上一帧几何信息：用于平滑和速度计算。
        self._frame_id = 0
        self._last_timestamp_ms = 0
        self._clock_timestamp_ms = 0
        self._capture_tick = 0
        self._tracking_state = "not_detected"
        self._pinch_state = "open"
        self._confidence = 0.0
        self._tracking_loss_streak = 0
        self._pinch_candidate_streak = 0
        self._release_candidate_streak = 0
        self._last_pinch_distance = 0.0
        self._last_index_tip = Vec3(0.0, 0.0, 0.0)
        self._last_thumb_tip = Vec3(0.0, 0.0, 0.0)
        self._last_palm_center = Vec3(0.0, 0.0, 0.0)
        self._last_velocity = Vec3(0.0, 0.0, 0.0)

    def start(self) -> None:
        # start 的职责只有一个：把“尚未运行”的对象安全切到 RUNNING。
        # 它不负责产包，也不负责业务循环。
        # start 是幂等的，避免上层重复初始化时把底层资源打乱。
        if self._started:
            return None

        # 进入初始化态，清空错误和统计，再把运行时缓存重置回起点。
        self.lifecycle_state = LIFECYCLE_INITIALIZING
        self._errors = []
        self._metrics = GestureMetrics()
        self._reset_runtime_state()

        try:
            self._setup_backend()
        except Exception as exc:
            # 初始化失败必须明确进入 DEGRADED，而不是假装还能跑。
            self.lifecycle_state = LIFECYCLE_DEGRADED
            self._record_error(
                error_entry(
                    "gesture.backend.startup_failure",
                    "Failed to initialize gesture input backend",
                    recoverable=False,
                    hint="Verify camera access and the hand_landmarker.task path before restarting the service.",
                    details={
                        "camera_index": self._camera_index,
                        "detector_backend": self._detector_backend,
                        "error": str(exc),
                    },
                )
            )
            raise

        # 只有底层后端已经准备好，started 和 lifecycle_state 才会一起进入“运行中”。
        self._started = True
        self.lifecycle_state = LIFECYCLE_RUNNING
        return None

    def poll(self) -> GesturePacket | None:
        # poll 是整个类最核心的公开方法。
        # 每调用一次，它就尝试消费“一帧新的输入事实”，然后把这帧翻译成一个 GesturePacket。
        # 上层主循环通常会以固定频率调用它。
        # 没 start 前不产包，保持接口无副作用。
        if not self._started:
            return None

        try:
            self._metrics.polls_attempted += 1

            # 先读帧，再解析时间戳，再做检测。这三步决定本轮输入事实是什么。
            raw_frame = self._read_frame()
            timestamp_ms = self._resolve_timestamp_ms(raw_frame)
            hand_data = self._detect_hand(raw_frame)

            if hand_data is None:
                # 这一支表示“摄像头和 detector 正常工作，但这帧没识别到手”。
                # 这不应该等同于程序错误，更不应该让输出流断掉。
                # 没检测到手时，不直接断流，而是输出一个降级包。
                # 这样 bridge 仍然能收到连续的 tracking / pinch 信号变化。
                self._tracking_loss_streak += 1
                tracking_state = self._compute_tracking_state(None)
                pinch_state = self._compute_pinch_state(None, None)
                confidence = self._compute_confidence(None, tracking_state, pinch_state)
                velocity = Vec3(0.0, 0.0, 0.0)

                self._tracking_state = tracking_state
                self._pinch_state = pinch_state
                self._confidence = confidence
                self._last_velocity = velocity

                # 注意：这里沿用上一帧缓存的坐标，而不是全量清零。
                # 这么做是为了让下游区分“暂时看不见手”和“坐标突然跳回原点”。
                packet = self._build_packet(
                    timestamp_ms=timestamp_ms,
                    tracking_state=tracking_state,
                    pinch_state=pinch_state,
                    confidence=confidence,
                    index_tip=self._last_index_tip,
                    thumb_tip=self._last_thumb_tip,
                    palm_center=self._last_palm_center,
                    velocity=velocity,
                    debug={
                        "backend": self._detector_backend,
                        "tick": raw_frame.get("tick"),
                        "has_hand": False,
                        "tracking_loss_streak": self._tracking_loss_streak,
                    },
                )
                self.lifecycle_state = LIFECYCLE_RUNNING
                self._metrics.packets_emitted += 1
                self._metrics.empty_packets += 1
                return packet

            # 只要这一帧成功看到手，就把丢失计数清零。
            # 这代表 tracking 状态有机会重新回到 tracked。
            self._tracking_loss_streak = 0

            # 先把 detector 输出裁剪到 [-1, 1]，统一坐标边界。
            index_tip = self._normalize_vec3(hand_data["index_tip"])
            thumb_tip = self._normalize_vec3(hand_data["thumb_tip"])
            palm_center = self._normalize_vec3(hand_data["palm_center"])

            # 这三项是“状态解释层”：tracking / pinch / confidence。
            # 它们回答的是“这一帧在语义上意味着什么”。
            tracking_state = self._compute_tracking_state(hand_data)
            pinch_state = self._compute_pinch_state(index_tip, thumb_tip)
            confidence = self._compute_confidence(hand_data, tracking_state, pinch_state)

            # 这三项是“运动学层”：平滑后的点和速度。
            # 它们回答的是“这一帧的几何位置和运动趋势是什么”。
            smoothed_index_tip = self._smooth_vec3(index_tip, self._last_index_tip)
            smoothed_thumb_tip = self._smooth_vec3(thumb_tip, self._last_thumb_tip)
            smoothed_palm_center = self._smooth_vec3(palm_center, self._last_palm_center)
            velocity = self._compute_velocity(smoothed_palm_center, self._last_palm_center)

            # 更新缓存，给下一帧继续用。
            self._last_index_tip = smoothed_index_tip
            self._last_thumb_tip = smoothed_thumb_tip
            self._last_palm_center = smoothed_palm_center
            self._last_velocity = velocity
            self._tracking_state = tracking_state
            self._pinch_state = pinch_state
            self._confidence = confidence

            # 这里产出的就是“正常跟踪帧”的标准输出包。
            packet = self._build_packet(
                timestamp_ms=timestamp_ms,
                tracking_state=tracking_state,
                pinch_state=pinch_state,
                confidence=confidence,
                index_tip=smoothed_index_tip,
                thumb_tip=smoothed_thumb_tip,
                palm_center=smoothed_palm_center,
                velocity=velocity,
                debug={
                    "backend": self._detector_backend,
                    "tick": raw_frame.get("tick"),
                    "has_hand": True,
                    "raw_confidence": hand_data.get("raw_confidence"),
                },
            )
            self.lifecycle_state = LIFECYCLE_RUNNING
            self._metrics.packets_emitted += 1
            self._metrics.tracked_packets += 1
            return packet
        except Exception as exc:
            # 这一支和 hand_data is None 不同。
            # 它表示真正的系统错误，例如摄像头读帧失败、detector 抛异常、资源状态错乱等。
            # 真正的后端失败会把服务切到 DEGRADED，并抛一个统一错误给上层。
            self.lifecycle_state = LIFECYCLE_DEGRADED
            self._metrics.backend_failures += 1
            self._record_error(
                error_entry(
                    "gesture.backend.failure",
                    "Gesture input backend failure",
                    recoverable=False,
                    hint="Inspect camera availability and detector health before resuming polling.",
                    details={
                        "camera_index": self._camera_index,
                        "detector_backend": self._detector_backend,
                        "error": str(exc),
                    },
                )
            )
            raise RuntimeError("Gesture input backend failure") from exc

    def health(self) -> dict[str, Any]:
        # health 不做 IO，只把当前快照组织成统一结构。
        # 最近一条错误消息单独抄到 stats.last_error，便于上层快速展示。
        # 你可以把它理解成“这个服务此刻的仪表盘读数”。
        last_error = self._errors[-1]["message"] if self._errors else None
        return build_health(
            component="gesture",
            lifecycle_state=self.lifecycle_state,
            errors=self._errors,
            stats={
                "started": self._started,
                "frame_id": self._frame_id,
                "hand_id": self._hand_id,
                "tracking_state": self._tracking_state,
                "pinch_state": self._pinch_state,
                "confidence": self._confidence,
                "tracking_loss_streak": self._tracking_loss_streak,
                "detector_backend": self._detector_backend,
                "last_error": last_error,
                "polls_attempted": self._metrics.polls_attempted,
                "packets_emitted": self._metrics.packets_emitted,
                "tracked_packets": self._metrics.tracked_packets,
                "empty_packets": self._metrics.empty_packets,
                "backend_failures": self._metrics.backend_failures,
            },
        )

    def stop(self) -> None:
        # stop 对应 start，负责资源释放和生命周期回收。
        # 它的原则是：尽量把资源关干净，但不要因为清理失败影响整个应用退出。
        # stop 也是幂等的。已经停了就直接返回。
        if not self._started and self.lifecycle_state == LIFECYCLE_STOPPED:
            return None

        try:
            self._teardown_backend()
        except Exception as exc:
            # 收尾错误不阻止整个应用继续退出，但要留下诊断信息。
            self._record_error(
                error_entry(
                    "gesture.backend.teardown_failure",
                    "Failed to release gesture backend resources cleanly",
                    recoverable=True,
                    hint="Release the capture device manually if the process still holds the camera.",
                    details={
                        "camera_index": self._camera_index,
                        "detector_backend": self._detector_backend,
                        "error": str(exc),
                    },
                )
            )
        finally:
            self._backend = None
            self._started = False
            self.lifecycle_state = LIFECYCLE_STOPPED
        return None

    def _reset_runtime_state(self) -> None:
        # 把一切会影响下一次启动的“运行中痕迹”全部清掉。
        # 这样可以避免上一次运行残留的数据污染下一次 start 后的前几帧结果。
        self._frame_id = 0
        self._last_timestamp_ms = 0
        self._clock_timestamp_ms = 0
        self._capture_tick = 0
        self._tracking_state = "not_detected"
        self._pinch_state = "open"
        self._confidence = 0.0
        self._tracking_loss_streak = 0
        self._pinch_candidate_streak = 0
        self._release_candidate_streak = 0
        self._last_pinch_distance = 0.0
        self._last_index_tip = Vec3(0.0, 0.0, 0.0)
        self._last_thumb_tip = Vec3(0.0, 0.0, 0.0)
        self._last_palm_center = Vec3(0.0, 0.0, 0.0)
        self._last_velocity = Vec3(0.0, 0.0, 0.0)

    def _setup_backend(self) -> None:
        # 这个方法专门做底层资源初始化。
        # 它成功返回后，poll 才有条件真正进入“读帧 -> 检测 -> 出包”的主流程。
        # 没模型就不伪造输入，直接明确失败。
        if self._hand_model is None:
            raise RuntimeError(
                "Missing hand_landmarker.task. Put the model at "
                f"{self.DEFAULT_HAND_MODEL_PATH} or pass --hand-model with a valid file path."
            )

        capture = cv2.VideoCapture(self._camera_index)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Unable to open camera source: {self._camera_index}")

        # 这里复用 debug/runtime 里的 detector 构造逻辑，
        # 保证主实现和调试工具看到的是同一类 detector 行为。
        detector_config = DebugVideoConfig(
            input_video=None,
            camera_index=self._camera_index,
            hand_model=self._hand_model,
            output_dir=self._repo_root / "report" / "live",
            target_fps=None,
            max_frames=0,
            min_detection_confidence=self._min_detection_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
            model_complexity=self._model_complexity,
        )
        detector_owner: Any | None = None
        detector: Any | None = None
        try:
            detector_owner = create_hand_detector(detector_config)
            detector = detector_owner.__enter__() if hasattr(detector_owner, "__enter__") else detector_owner
            backend_name = getattr(detector, "backend_name", type(detector).__name__)
        except Exception:
            _close_detector_resource(detector_owner, detector)
            capture.release()
            raise

        # passthrough 代表没有真正的 MediaPipe backend，继续跑没有意义，直接失败。
        if backend_name == "passthrough":
            _close_detector_resource(detector_owner, detector)
            capture.release()
            raise RuntimeError(
                "No MediaPipe hand detector backend is available. "
                f"Put the model at {self.DEFAULT_HAND_MODEL_PATH} or pass --hand-model with a valid hand_landmarker.task file."
            )

        self._backend = {
            "capture": capture,
            "detector": detector,
            "detector_owner": detector_owner,
        }
        self._detector_backend = backend_name

    def _teardown_backend(self) -> None:
        # 和 _setup_backend 成对出现，职责是把底层资源按正确顺序关掉。
        # 如果 backend 压根没建起来，就只把状态改回未初始化。
        if self._backend is None:
            self._detector_backend = "uninitialized"
            return

        detector_owner = self._backend.get("detector_owner")
        capture = self._backend.get("capture")
        if detector_owner is not None and hasattr(detector_owner, "__exit__"):
            detector_owner.__exit__(None, None, None)
        if capture is not None:
            capture.release()
        self._backend = None
        self._detector_backend = "uninitialized"

    def _read_frame(self) -> dict[str, Any]:
        # 这里统一返回一个字典，而不是只返回原始 frame，
        # 是因为后面还要顺带带出 tick、timestamp、source 这些调试信息。
        # 这些字段会一路向下传到 debug 信息里，方便回放和定位问题。
        if self._backend is None:
            raise RuntimeError("Gesture backend is not initialized")

        capture = self._backend["capture"]
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("camera frame read failed")

        self._capture_tick += 1
        return {
            "timestamp_ms": self._next_timestamp_ms(),
            "frame": frame,
            "tick": self._capture_tick,
            "source": f"camera:{self._camera_index}",
        }

    def _detect_hand(self, raw_frame: dict[str, Any]) -> dict[str, Any] | None:
        # 这里把 detector 的输出重新整理成一个更稳定的内部数据结构。
        # 后续逻辑只关心 index_tip / thumb_tip / palm_center / raw_confidence，
        # 所以这里先把底层第三方输出裁剪成服务内部真正需要的字段。
        if self._backend is None:
            raise RuntimeError("Gesture backend is not initialized")

        frame = raw_frame["frame"]
        detector = self._backend["detector"]
        timestamp_ms = int(raw_frame["timestamp_ms"])
        landmarks, raw_confidence, _raw_landmarks = detector.detect(frame, timestamp_ms)
        if landmarks is None:
            return None

        return {
            "index_tip": landmarks["index_finger_tip"],
            "thumb_tip": landmarks["thumb_tip"],
            "palm_center": landmarks["wrist"],
            "raw_confidence": raw_confidence,
            "source": self._detector_backend,
        }

    def _normalize_vec3(self, value: Vec3) -> Vec3:
        # detector 的原始值理论上应该在规范区间内，但这里再夹一层，
        # 防止上游异常值把 contract 打穿。
        # 这是一道“输出卫生检查”，能减少坏数据向下游扩散。
        return Vec3(
            x=max(-1.0, min(1.0, float(value.x))),
            y=max(-1.0, min(1.0, float(value.y))),
            z=max(-1.0, min(1.0, float(value.z))),
        )

    def _compute_tracking_state(self, hand_data: dict[str, Any] | None) -> str:
        # tracking_state 是“整只手是否还在视野中”的粗粒度判断。
        # 它比 pinch_state 更基础，因为 pinch 前提是先有手。
        # 有手就是 tracked。
        if hand_data is not None:
            return "tracked"

        # 短暂丢失先给 temporarily_lost，给 bridge 一个缓冲区。
        if self._tracking_loss_streak <= TRACKING_TEMPORARY_LOSS_FRAMES:
            return "temporarily_lost"

        # 连续更多帧丢失后才落到 not_detected。
        return "not_detected"

    def _compute_pinch_state(self, index_tip: Vec3 | None, thumb_tip: Vec3 | None) -> str:
        # pinch_state 是“食指和拇指之间关系”的细粒度判断。
        # 它的重点不是一帧瞬时值，而是连续几帧是否稳定，因此这里会维护多个 streak 计数。
        # 没有两个关键点，就没法算 pinch 距离。
        # 这里不直接从 pinched 掉回 open，而是先给 release_candidate，
        # 这样可以吸收短暂丢帧导致的状态抖动。
        if index_tip is None or thumb_tip is None:
            self._pinch_candidate_streak = 0
            if self._pinch_state in {"pinched", "pinch_candidate", "release_candidate"}:
                self._release_candidate_streak += 1
                if self._release_candidate_streak <= RELEASE_CONFIRM_FRAMES:
                    return "release_candidate"
            self._release_candidate_streak = 0
            self._last_pinch_distance = 0.0
            return "open"

        pinch_distance = distance(index_tip, thumb_tip)
        self._last_pinch_distance = pinch_distance

        # 这里的 pinch_distance 就是 index_tip 和 thumb_tip 的欧氏距离。
        # 值越小，说明两个手指越接近；值越大，说明已经张开。

        # 进入 hold 阈值说明手指已经足够近，连续几帧后确认 pinched。
        if pinch_distance <= PINCH_HOLD_THRESHOLD:
            self._pinch_candidate_streak += 1
            self._release_candidate_streak = 0
            if self._pinch_candidate_streak >= PINCH_CONFIRM_FRAMES:
                return "pinched"
            return "pinch_candidate"

        # 进入 enter 阈值说明“有 pinch 意图”，但还没紧到稳定 pinched。
        if pinch_distance <= PINCH_ENTER_THRESHOLD:
            self._pinch_candidate_streak += 1
            self._release_candidate_streak = 0
            if self._pinch_state == "pinched":
                return "pinched"
            return "pinch_candidate"

        # 距离重新拉开后，看是不是满足 release 阈值。
        self._pinch_candidate_streak = 0
        if self._pinch_state in {"pinched", "pinch_candidate", "release_candidate"} and pinch_distance >= PINCH_RELEASE_THRESHOLD:
            self._release_candidate_streak += 1
            if self._release_candidate_streak <= RELEASE_CONFIRM_FRAMES:
                return "release_candidate"
            self._release_candidate_streak = 0
            return "open"

        self._release_candidate_streak = 0
        return "open"

    def _compute_confidence(
        self,
        hand_data: dict[str, Any] | None,
        tracking_state: str,
        pinch_state: str,
    ) -> float:
        # confidence 是给下游的“信号可信度”估计，不是 detector 原始分数的原样转发。
        # 这里做了一个很轻量的二次加工，让暂失、候选态、稳定态在数值上有可区分性。
        # 暂失时给低但非零置信度，表达“我可能还在跟，只是这一帧没看清”。
        if tracking_state == "temporarily_lost":
            return max(0.05, 0.3 - 0.05 * max(self._tracking_loss_streak - 1, 0))

        # 真没检测到手就给 0。
        if tracking_state == "not_detected":
            return 0.0

        # 正常检测到手时，基于 detector 原始分数做一个轻微状态修正。
        assert hand_data is not None
        raw_confidence = float(hand_data.get("raw_confidence", 0.8))
        # 已确认 pinched 时略微上调，表达“这个交互状态更稳定”。
        if pinch_state == "pinched":
            raw_confidence += 0.05
        # 候选态和释放态略微下调，表达“状态尚未完全稳定”。
        elif pinch_state in {"pinch_candidate", "release_candidate"}:
            raw_confidence -= 0.05

        # 最后统一裁剪到 [0, 1]，保证 contract 合法。
        return max(0.0, min(1.0, raw_confidence))

    def _smooth_vec3(self, current: Vec3, previous: Vec3) -> Vec3:
        # 这是最简单的指数平滑写法。
        # 可以把它理解成：当前位置 = 当前观测 * alpha + 上一帧结果 * (1 - alpha)。
        # 优点是实现简单、计算便宜；缺点是有轻微滞后。
        # 第一帧没有历史值，直接返回当前值。
        if self._frame_id == 0:
            return current
        alpha = SMOOTHING_ALPHA
        return Vec3(
            x=current.x * alpha + previous.x * (1.0 - alpha),
            y=current.y * alpha + previous.y * (1.0 - alpha),
            z=current.z * alpha + previous.z * (1.0 - alpha),
        )

    def _compute_velocity(self, current: Vec3, previous: Vec3) -> Vec3:
        # 这里速度是“相邻两帧位移差”，不是物理意义上的 m/s。
        # 对 bridge 来说，这个量主要用于感知变化趋势，而不是做精确动力学。
        # 再做一次 normalize，是为了把异常突刺限制在 contract 允许范围内。
        return self._normalize_vec3(
            Vec3(
                x=current.x - previous.x,
                y=current.y - previous.y,
                z=current.z - previous.z,
            )
        )

    def _build_packet(
        self,
        timestamp_ms: int,
        tracking_state: str,
        pinch_state: str,
        confidence: float,
        index_tip: Vec3,
        thumb_tip: Vec3,
        palm_center: Vec3,
        velocity: Vec3,
        debug: dict[str, Any] | None,
    ) -> GesturePacket:
        # 真正出包的地方。所有上游计算最后都收束到这里。
        # 如果你要追“为什么某个字段是这个值”，最终一定会回到这里看传进来的参数。
        self._frame_id += 1
        self._last_timestamp_ms = timestamp_ms
        return GesturePacket(
            contract_version=EXPECTED_CONTRACT_VERSION,
            frame_id=self._frame_id,
            timestamp_ms=timestamp_ms,
            hand_id=self._hand_id,
            tracking_state=tracking_state,
            confidence=max(0.0, min(1.0, confidence)),
            pinch_state=pinch_state,
            index_tip=index_tip,
            thumb_tip=thumb_tip,
            palm_center=palm_center,
            coordinate_space="camera_norm",
            pinch_distance=self._last_pinch_distance,
            velocity=velocity,
            smoothing_hint={"method": "linear", "alpha": SMOOTHING_ALPHA},
            debug=debug,
        )

    def _resolve_timestamp_ms(self, raw_frame: dict[str, Any]) -> int:
        # 这个方法负责把“输入时间”修正成“对外可发布时间”。
        # 关键要求只有一个：不能倒退。
        # 优先用读帧阶段已经带上的 timestamp。
        raw_timestamp = raw_frame.get("timestamp_ms")
        timestamp_ms = int(raw_timestamp) if raw_timestamp is not None else self._next_timestamp_ms()

        # 如果上游给了倒退时间，就强行修正为单调递增。
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        if timestamp_ms > self._clock_timestamp_ms:
            self._clock_timestamp_ms = timestamp_ms
        return timestamp_ms

    def _next_timestamp_ms(self) -> int:
        # 本地兜底时间源，同样保证严格递增。
        # 即使操作系统时间在极短间隔内取到相同毫秒值，这里也会手动 +1。
        now = int(time.time() * 1000)
        if now <= self._clock_timestamp_ms:
            now = self._clock_timestamp_ms + 1
        self._clock_timestamp_ms = now
        return now

    def _resolve_hand_model(self, hand_model: str | None) -> str | None:
        # 显式传入优先；否则看默认路径里有没有模型文件。
        # 这个方法本质上是在做“配置优先级决策”。
        return _resolve_hand_model_path(hand_model)

    def _record_error(self, error: dict[str, Any]) -> None:
        # 错误列表只保留最近 10 条，避免 health 无限膨胀。
        # 这是一种常见的环形窗口思路：保留最近诊断信息，但不让状态对象无限长大。
        self._errors.append(error)
        self._errors = self._errors[-10:]


def main(argv: list[str] | None = None) -> int:
    # 这里现在不再简单转发到别的文件，而是自己走完整入口链：
    # parse_args -> setup_logging -> build_config -> build_preview_config -> run_live_preview。
    # 这样 service_impl 就和根目录 main.py 一样，具备完整的“解析参数并启动”的入口形态。
    args = parse_args(argv)
    setup_logging(args.log_level)
    config = build_config(args)
    try:
        run_live_preview(build_preview_config(config))
        return 0
    except Exception as exc:
        # 入口层统一兜底，把默认模型路径一起打出来，方便用户定位环境问题。
        logging.exception(
            "Live gesture preview failed. Expected model path: %s. Error: %s",
            GestureInputServiceImpl.DEFAULT_HAND_MODEL_PATH,
            exc,
        )
        return 1