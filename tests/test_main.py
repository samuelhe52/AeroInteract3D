from __future__ import annotations

from main import App, AppConfig, LIFECYCLE_RUNNING


class FakeGestureInput:
    def __init__(self) -> None:
        self.app: App | None = None

    def start(self) -> None:
        return None

    def poll(self):
        assert self.app is not None
        self.app.request_stop()
        return None

    def health(self) -> dict:
        return {"lifecycle_state": LIFECYCLE_RUNNING}

    def stop(self) -> None:
        return None


class FakeBridge:
    def start(self) -> None:
        return None

    def process(self, packet):
        return []

    def health(self) -> dict:
        return {"lifecycle_state": LIFECYCLE_RUNNING}

    def stop(self) -> None:
        return None


class FakeRenderOutput:
    def __init__(self) -> None:
        self.step_calls = 0

    def start(self) -> None:
        return None

    def push(self, command) -> None:
        return None

    def step(self) -> None:
        self.step_calls += 1

    def health(self) -> dict:
        return {"lifecycle_state": LIFECYCLE_RUNNING}

    def stop(self) -> None:
        return None


def test_app_run_steps_render_output_every_loop_iteration() -> None:
    config = AppConfig(target_fps=60)
    gesture_input = FakeGestureInput()
    bridge = FakeBridge()
    render_output = FakeRenderOutput()
    app = App(config, gesture_input, bridge, render_output)
    gesture_input.app = app

    app.lifecycle_state = LIFECYCLE_RUNNING

    app.run()

    assert render_output.step_calls == 1