"""Bridge-driven stylized two-finger hand rendering in world/scene space."""

from __future__ import annotations

from typing import Optional

from panda3d.core import NodePath, TransparencyAttrib, Vec3


class VirtualHand:
    WRIST_MARKER_SCALE = (0.11, 0.06, 0.035)
    JOINT_MARKER_SCALE = (0.06, 0.06, 0.06)
    ANCHOR_MARKER_SCALE = (0.085, 0.085, 0.05)
    PALM_SEGMENT_THICKNESS = 0.045
    FINGER_SEGMENT_THICKNESS = 0.035
    PINCH_BAR_THICKNESS_OPEN = 0.012
    PINCH_BAR_THICKNESS_CANDIDATE = 0.02
    PINCH_BAR_THICKNESS_PINCHED = 0.03
    PINCH_CENTER_SCALE_OPEN = 0.035
    PINCH_CENTER_SCALE_CANDIDATE = 0.055
    PINCH_CENTER_SCALE_PINCHED = 0.08
    PINCH_CANDIDATE_DISTANCE = 0.16
    PINCH_LOCK_DISTANCE = 0.09
    _CENTER_OFFSET = (-0.5, -0.5, -0.5)

    def __init__(self, base, root_np: NodePath, config: dict | None = None):
        self.base = base
        self.root = root_np.attachNewNode("virtual_hand")
        self.config = config or {}
        self._box_template = self._load_box_template()
        self._marker_colors = {
            "wrist": self.config.get("wrist_color", [0.36, 0.44, 0.52]),
            "thumb_tip": self.config.get("thumb_color", [0.98, 0.60, 0.22]),
            "index_tip": self.config.get("index_color", [0.96, 0.74, 0.30]),
            "anchor": self.config.get("anchor_color", [0.24, 0.88, 0.86]),
        }
        self._segment_colors = {
            "palm": self.config.get("palm_color", [0.30, 0.36, 0.44]),
            "thumb": self.config.get("thumb_segment_color", [0.96, 0.56, 0.24]),
            "index": self.config.get("index_segment_color", [0.95, 0.70, 0.26]),
            "pinch_open": self.config.get("pinch_open_color", [0.99, 0.82, 0.38]),
            "pinch_candidate": self.config.get("pinch_candidate_color", [1.0, 0.72, 0.26]),
            "pinch_locked": self.config.get("pinch_locked_color", [1.0, 0.42, 0.18]),
        }
        self._points: dict[str, NodePath] = {}
        self._segments: dict[str, NodePath] = {}
        self._pinch_center = self._create_marker("pinch_center", [1.0, 0.72, 0.26], (0.05, 0.05, 0.05), alpha=0.78)
        self._init_points()
        self._init_segments()
        self.root.hide()

    def _load_box_template(self) -> NodePath:
        box_model = self.base.loader.loadModel("box")
        if box_model.isEmpty():
            raise RuntimeError("Failed to load box model for virtual hand")
        box_model.setTextureOff(1)
        box_model.setTransparency(TransparencyAttrib.MAlpha)
        box_model.setPos(*self._CENTER_OFFSET)
        return box_model

    def _init_points(self) -> None:
        self._points["wrist"] = self._create_marker(
            "wrist_marker",
            self._marker_colors["wrist"],
            self.WRIST_MARKER_SCALE,
            alpha=0.95,
        )
        self._points["thumb_tip"] = self._create_marker(
            "thumb_marker",
            self._marker_colors["thumb_tip"],
            self.JOINT_MARKER_SCALE,
            alpha=0.96,
        )
        self._points["index_tip"] = self._create_marker(
            "index_marker",
            self._marker_colors["index_tip"],
            self.JOINT_MARKER_SCALE,
            alpha=0.96,
        )
        self._points["anchor"] = self._create_marker(
            "anchor_marker",
            self._marker_colors["anchor"],
            self.ANCHOR_MARKER_SCALE,
            alpha=0.92,
        )

    def _init_segments(self) -> None:
        self._segments["wrist_anchor"] = self._create_segment("wrist_anchor", self._segment_colors["palm"], alpha=0.82)
        self._segments["anchor_thumb"] = self._create_segment("anchor_thumb", self._segment_colors["thumb"], alpha=0.92)
        self._segments["anchor_index"] = self._create_segment("anchor_index", self._segment_colors["index"], alpha=0.92)
        self._segments["pinch_bar"] = self._create_segment("pinch_bar", self._segment_colors["pinch_open"], alpha=0.72)

    def update_points(self, points: Optional[dict[str, Vec3]]) -> None:
        if not points:
            self._hide_all()
            return

        required = {"wrist", "thumb_tip", "index_tip", "anchor"}
        if any(name not in points for name in required):
            self._hide_all()
            return

        self.root.show()
        for name, node in self._points.items():
            node.show()
            node.setPos(points[name])

        wrist = points["wrist"]
        anchor = points["anchor"]
        thumb_tip = points["thumb_tip"]
        index_tip = points["index_tip"]
        pinch_distance = (thumb_tip - index_tip).length()

        self._update_segment(
            self._segments["wrist_anchor"],
            wrist,
            anchor,
            thickness=self.PALM_SEGMENT_THICKNESS,
        )
        self._update_segment(
            self._segments["anchor_thumb"],
            anchor,
            thumb_tip,
            thickness=self.FINGER_SEGMENT_THICKNESS,
        )
        self._update_segment(
            self._segments["anchor_index"],
            anchor,
            index_tip,
            thickness=self.FINGER_SEGMENT_THICKNESS,
        )
        self._update_pinch_visual(thumb_tip, index_tip, pinch_distance)

    def _update_pinch_visual(self, thumb_tip: Vec3, index_tip: Vec3, pinch_distance: float) -> None:
        if pinch_distance <= self.PINCH_LOCK_DISTANCE:
            color = self._segment_colors["pinch_locked"]
            bar_thickness = self.PINCH_BAR_THICKNESS_PINCHED
            center_scale = self.PINCH_CENTER_SCALE_PINCHED
            alpha = 0.95
        elif pinch_distance <= self.PINCH_CANDIDATE_DISTANCE:
            color = self._segment_colors["pinch_candidate"]
            bar_thickness = self.PINCH_BAR_THICKNESS_CANDIDATE
            center_scale = self.PINCH_CENTER_SCALE_CANDIDATE
            alpha = 0.88
        else:
            color = self._segment_colors["pinch_open"]
            bar_thickness = self.PINCH_BAR_THICKNESS_OPEN
            center_scale = self.PINCH_CENTER_SCALE_OPEN
            alpha = 0.62

        pinch_bar = self._segments["pinch_bar"]
        pinch_bar.show()
        self._set_segment_color(pinch_bar, color, alpha=alpha)
        self._update_segment(
            pinch_bar,
            thumb_tip,
            index_tip,
            thickness=bar_thickness,
        )

        pinch_center_pos = (thumb_tip + index_tip) * 0.5
        self._pinch_center.show()
        self._pinch_center.setPos(pinch_center_pos)
        self._pinch_center.setScale(center_scale, center_scale, center_scale)
        self._set_node_color(self._pinch_center, color, alpha=alpha)

    def _hide_all(self) -> None:
        self.root.hide()
        for node in self._points.values():
            node.hide()
        for node in self._segments.values():
            node.hide()
        self._pinch_center.hide()

    def _create_marker(
        self,
        name: str,
        color: list[float],
        scale: tuple[float, float, float],
        *,
        alpha: float,
    ) -> NodePath:
        marker_np = self.root.attachNewNode(name)
        model_np = self._box_template.copyTo(marker_np)
        marker_np.setScale(*scale)
        self._set_node_color(marker_np, color, alpha=alpha)
        return marker_np

    def _create_segment(self, name: str, color: list[float], *, alpha: float) -> NodePath:
        segment_np = self.root.attachNewNode(name)
        self._box_template.copyTo(segment_np)
        self._set_segment_color(segment_np, color, alpha=alpha)
        return segment_np

    def _update_segment(self, segment_np: NodePath, start: Vec3, end: Vec3, *, thickness: float) -> None:
        direction = end - start
        length = direction.length()
        if length <= 1e-4:
            segment_np.hide()
            return

        midpoint = start + (direction * 0.5)
        segment_np.show()
        segment_np.setPos(midpoint)
        segment_np.lookAt(end)
        segment_np.setScale(thickness, length, thickness)

    @staticmethod
    def _set_node_color(node: NodePath, color: list[float], *, alpha: float) -> None:
        node.setTransparency(TransparencyAttrib.MAlpha)
        node.setColor(float(color[0]), float(color[1]), float(color[2]), float(alpha))

    def _set_segment_color(self, node: NodePath, color: list[float], *, alpha: float) -> None:
        self._set_node_color(node, color, alpha=alpha)

