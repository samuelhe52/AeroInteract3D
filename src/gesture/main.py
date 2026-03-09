from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path


if __package__ in {None, ""}:
	repo_root = Path(__file__).resolve().parents[2]
	if str(repo_root) not in sys.path:
		sys.path.insert(0, str(repo_root))

from src.gesture.service_stub import GestureInputServiceStub
from src.ports import GestureInputPort


LIFECYCLE_INITIALIZING = "INITIALIZING"
LIFECYCLE_RUNNING = "RUNNING"
LIFECYCLE_DEGRADED = "DEGRADED"
LIFECYCLE_STOPPED = "STOPPED"


@dataclass(slots=True)
class GestureRunnerConfig:
	"""手势模块独立运行时的配置。"""

	log_level: str = "INFO"
	target_fps: int = 30
	max_frames: int = 300
	print_every: int = 30


class GestureApp:
	"""手势模块的独立运行骨架。

	作用：
	- 启动 `GestureInputPort`
	- 按固定频率调用 `poll()`
	- 输出健康信息和包摘要
	- 处理优雅停止
	"""

	def __init__(self, config: GestureRunnerConfig, gesture_input: GestureInputPort) -> None:
		self.config = config
		self.gesture_input = gesture_input
		self.lifecycle_state = LIFECYCLE_INITIALIZING
		self._running = False
		self._frames_seen = 0
		self._packets_seen = 0

	def initialize(self) -> None:
		logging.info("Initializing gesture module")
		self.gesture_input.start()
		self.lifecycle_state = LIFECYCLE_RUNNING

	def run(self) -> None:
		if self.lifecycle_state != LIFECYCLE_RUNNING:
			raise RuntimeError("Gesture module is not ready to run")

		frame_interval = 1.0 / max(self.config.target_fps, 1)
		self._running = True
		logging.info("Gesture loop started at target_fps=%s", self.config.target_fps)

		while self._running:
			loop_start = time.perf_counter()
			self._frames_seen += 1

			packet = self.gesture_input.poll()
			if packet is not None:
				self._packets_seen += 1
				self._handle_packet(packet)

			if self.config.max_frames > 0 and self._frames_seen >= self.config.max_frames:
				logging.info("Reached max_frames=%s, stopping loop", self.config.max_frames)
				self.request_stop()

			elapsed = time.perf_counter() - loop_start
			sleep_for = frame_interval - elapsed
			if sleep_for > 0:
				time.sleep(sleep_for)

	def _handle_packet(self, packet: object) -> None:
		if self._packets_seen % max(self.config.print_every, 1) != 0:
			return

		logging.info(
			"packet #%s frame_id=%s tracking=%s pinch=%s confidence=%.2f",
			self._packets_seen,
			getattr(packet, "frame_id", "?"),
			getattr(packet, "tracking_state", "?"),
			getattr(packet, "pinch_state", "?"),
			getattr(packet, "confidence", 0.0),
		)

	def request_stop(self) -> None:
		self._running = False

	def shutdown(self) -> None:
		logging.info("Shutting down gesture module")
		try:
			self.gesture_input.stop()
		except Exception:
			logging.exception("Gesture shutdown error")
			self.lifecycle_state = LIFECYCLE_DEGRADED
			return

		self.lifecycle_state = LIFECYCLE_STOPPED

	def health_snapshot(self) -> dict:
		return {
			"lifecycle_state": self.lifecycle_state,
			"frames_seen": self._frames_seen,
			"packets_seen": self._packets_seen,
			"gesture": self.gesture_input.health(),
		}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Gesture module standalone entrypoint")
	parser.add_argument("--log-level", default="INFO")
	parser.add_argument("--target-fps", type=int, default=30)
	parser.add_argument(
		"--max-frames",
		type=int,
		default=300,
		help="default is 300 frames; use 0 to keep running until stopped",
	)
	parser.add_argument(
		"--print-every",
		type=int,
		default=30,
		help="log one packet summary every N packets",
	)
	return parser.parse_args(argv)


def setup_logging(level: str) -> None:
	numeric_level = getattr(logging, level.upper(), logging.INFO)
	logging.basicConfig(
		level=numeric_level,
		format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
	)


def build_config(args: argparse.Namespace) -> GestureRunnerConfig:
	return GestureRunnerConfig(
		log_level=args.log_level.upper(),
		target_fps=args.target_fps,
		max_frames=args.max_frames,
		print_every=args.print_every,
	)


def build_app(config: GestureRunnerConfig) -> GestureApp:
	gesture_input = GestureInputServiceStub()
	return GestureApp(config, gesture_input)


def main(argv: list[str] | None = None) -> int:
	args = parse_args(argv)
	setup_logging(args.log_level)
	config = build_config(args)
	app = build_app(config)

	def _handle_signal(signum: int, _frame: object) -> None:
		logging.info("Received signal %s, requesting shutdown", signum)
		app.request_stop()

	signal.signal(signal.SIGINT, _handle_signal)
	signal.signal(signal.SIGTERM, _handle_signal)

	try:
		app.initialize()
		logging.info("Health snapshot: %s", app.health_snapshot())
		app.run()
		return 0
	except KeyboardInterrupt:
		logging.info("Interrupted by user")
		return 130
	except Exception:
		logging.exception("Fatal gesture module error")
		return 1
	finally:
		app.shutdown()


if __name__ == "__main__":
	raise SystemExit(main(sys.argv[1:]))
