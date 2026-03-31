"""Bridge-driven stylized two-finger hand rendering in world/scene space."""

from __future__ import annotations

from typing import Optional

from panda3d.core import NodePath, TransparencyAttrib, Vec3


class VirtualHand:
    WRIST_SCALE = (0.10, 0.06, 0.03)
    ANCHOR_SCALE = (0.055, 0.055, 0.028)
    ROOT_JOINT_SCALE = (0.065, 0.065, 0.04)
    TIP_JOINT_SCALE = (0.05, 0.05, 0.04)
    PALM_BAR_THICKNESS = 0.05
    FINGER_BAR_THICKNESS = 0.032
    PINCH_BAR_THICKNESS_OPEN = 0.01
    PINCH_BAR_THICKNESS_CANDIDATE = 0.016
    PINCH_BAR_THICKNESS_PINCHED = 0.024
    PINCH_CENTER_SCALE_OPEN = (0.024, 0.024, 0.024)
    PINCH_CENTER_SCALE_CANDIDATE = (0.042, 0.042, 0.042)
    PINCH_CENTER_SCALE_PINCHED = (0.06, 0.06, 0.06)
    PINCH_CANDIDATE_DISTANCE = 0.12
    PINCH_LOCK_DISTANCE = 0.055
    _CENTER_OFFSET = (-0.5, -0.5, -0.5)

    def __init__(self, base, root_np: NodePath, config: dict | None = None):
        self.base = base
        self.root = root_np.attachNewNode("virtual_hand")
        self.config = config or {}
        self._box_template = self._load_box_template()
        self._colors = {
            "wrist": self.config.get("wrist_color", [0.32, 0.38, 0.46]),
            "anchor": self.config.get("anchor_color", [0.22, 0.84, 0.84]),
            "palm": self.config.get("palm_color", [0.44, 0.48, 0.54]),
            "thumb": self.config.get("thumb_color", [0.95, 0.58, 0.22]),
            "index": self.config.get("index_color", [0.93, 0.73, 0.30]),
            "pinch_open": self.config.get("pinch_open_color", [1.0, 0.82, 0.38]),
            "pinch_candidate": self.config.get("pinch_candidate_color", [1.0, 0.66, 0.22]),
            "pinch_locked": self.config.get("pinch_locked_color", [1.0, 0.36, 0.16]),
        }
        self._markers: dict[str, NodePath] = {}
        self._segments: dict[str, NodePath] = {}
        self._init_markers()
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

    def _init_markers(self) -> None:
        self._markers["wrist"] = self._create_box_node(
            "wrist_marker",
            self._colors["wrist"],
            self.WRIST_SCALE,
            alpha=0.92,
        )
        self._markers["anchor"] = self._create_box_node(
            "anchor_marker",
            self._colors["anchor"],
            self.ANCHOR_SCALE,
            alpha=0.88,
        )
        self._markers["thumb_base"] = self._create_box_node(
            "thumb_root_marker",
            self._colors["thumb"],
            self.ROOT_JOINT_SCALE,
            alpha=0.94,
        )
        self._markers["index_base"] = self._create_box_node(
            "index_root_marker",
            self._colors["index"],
            self.ROOT_JOINT_SCALE,
            alpha=0.94,
        )
        self._markers["thumb_tip"] = self._create_box_node(
            "thumb_tip_marker",
            self._colors["thumb"],
            self.TIP_JOINT_SCALE,
            alpha=0.96,
        )
        self._markers["index_tip"] = self._create_box_node(
            "index_tip_marker",
            self._colors["index"],
            self.TIP_JOINT_SCALE,
            alpha=0.96,
        )
        self._markers["pinch_center"] = self._create_box_node(
            "pinch_center",
            self._colors["pinch_candidate"],
            self.PINCH_CENTER_SCALE_CANDIDATE,
            alpha=0.84,
        )

    def _init_segments(self) -> None:
        self._segments["palm_bridge"] = self._create_segment("palm_bridge", self._colors["palm"], alpha=0.78)
        self._segments["thumb_root"] = self._create_segment("thumb_root", self._colors["thumb"], alpha=0.82)
        self._segments["index_root"] = self._create_segment("index_root", self._colors["index"], alpha=0.82)
        self._segments["thumb_finger"] = self._create_segment("thumb_finger", self._colors["thumb"], alpha=0.94)
        self._segments["index_finger"] = self._create_segment("index_finger", self._colors["index"], alpha=0.94)
        self._segments["pinch_bar"] = self._create_segment("pinch_bar", self._colors["pinch_open"], alpha=0.66)

    def update_points(self, points: Optional[dict[str, Vec3]]) -> None:
        if not points:
            self._hide_all()
            return

        required = {"wrist", "thumb_tip", "index_tip", "anchor", "thumb_base", "index_base"}
        if any(name not in points for name in required):
            self._hide_all()
            return

        self.root.show()
        wrist = points["wrist"]
        anchor = points["anchor"]
        thumb_base = points["thumb_base"]
        index_base = points["index_base"]
        thumb_tip = points["thumb_tip"]
        index_tip = points["index_tip"]

        marker_positions = {
            "wrist": wrist,
            "anchor": anchor,
            "thumb_base": thumb_base,
            "index_base": index_base,
            "thumb_tip": thumb_tip,
            "index_tip": index_tip,
        }
        for name, pos in marker_positions.items():
            node = self._markers[name]
            node.show()
            node.setPos(pos)

        self._update_segment(self._segments["palm_bridge"], thumb_base, index_base, thickness=self.PALM_BAR_THICKNESS)
        self._update_segment(self._segments["thumb_root"], wrist, thumb_base, thickness=self.PALM_BAR_THICKNESS * 0.72)
        self._update_segment(self._segments["index_root"], wrist, index_base, thickness=self.PALM_BAR_THICKNESS * 0.72)
        self._update_segment(self._segments["thumb_finger"], thumb_base, thumb_tip, thickness=self.FINGER_BAR_THICKNESS)
        self._update_segment(self._segments["index_finger"], index_base, index_tip, thickness=self.FINGER_BAR_THICKNESS)
        self._update_pinch_visual(thumb_tip, index_tip)

    def _update_pinch_visual(self, thumb_tip: Vec3, index_tip: Vec3) -> None:
        pinch_distance = (thumb_tip - index_tip).length()
        if pinch_distance <= self.PINCH_LOCK_DISTANCE:
            color = self._colors["pinch_locked"]
            bar_thickness = self.PINCH_BAR_THICKNESS_PINCHED
            center_scale = self.PINCH_CENTER_SCALE_PINCHED
            alpha = 0.95
        elif pinch_distance <= self.PINCH_CANDIDATE_DISTANCE:
            color = self._colors["pinch_candidate"]
            bar_thickness = self.PINCH_BAR_THICKNESS_CANDIDATE
            center_scale = self.PINCH_CENTER_SCALE_CANDIDATE
            alpha = 0.86
        else:
            color = self._colors["pinch_open"]
            bar_thickness = self.PINCH_BAR_THICKNESS_OPEN
            center_scale = self.PINCH_CENTER_SCALE_OPEN
            alpha = 0.54

        pinch_bar = self._segments["pinch_bar"]
        self._set_node_color(pinch_bar, color, alpha=alpha)
        self._update_segment(pinch_bar, thumb_tip, index_tip, thickness=bar_thickness)

        pinch_center = self._markers["pinch_center"]
        pinch_center.show()
        pinch_center.setPos((thumb_tip + index_tip) * 0.5)
        pinch_center.setScale(*center_scale)
        self._set_node_color(pinch_center, color, alpha=alpha)

    def _create_box_node(
        self,
        name: str,
        color: list[float],
        scale: tuple[float, float, float],
        *,
        alpha: float,
    ) -> NodePath:
        node = self.root.attachNewNode(name)
        self._box_template.copyTo(node)
        node.setScale(*scale)
        self._set_node_color(node, color, alpha=alpha)
        return node

    def _create_segment(self, name: str, color: list[float], *, alpha: float) -> NodePath:
        node = self.root.attachNewNode(name)
        self._box_template.copyTo(node)
        self._set_node_color(node, color, alpha=alpha)
        return node

    def _update_segment(self, segment: NodePath, start: Vec3, end: Vec3, *, thickness: float) -> None:
        direction = end - start
        length = direction.length()
        if length <= 1e-4:
            segment.hide()
            return
        segment.show()
        segment.setPos(start + (direction * 0.5))
        segment.lookAt(end)
        segment.setScale(thickness, length, thickness)

    def _hide_all(self) -> None:
        self.root.hide()
        for node in self._markers.values():
            node.hide()
        for node in self._segments.values():
            node.hide()

    @staticmethod
    def _set_node_color(node: NodePath, color: list[float], *, alpha: float) -> None:
        node.setTransparency(TransparencyAttrib.MAlpha)
        node.setColor(float(color[0]), float(color[1]), float(color[2]), float(alpha))
