"""Bridge-driven hand overlay rendering in world/scene space."""

from __future__ import annotations

from typing import Optional

from panda3d.core import LineSegs, NodePath, TransparencyAttrib, Vec3, Vec4


class VirtualHand:
    POINT_MARKER_SCALE = 0.06
    ANCHOR_MARKER_SCALE = 0.08
    LINE_THICKNESS = 3

    def __init__(self, base, root_np: NodePath, config: dict | None = None):
        self.base = base
        self.root = root_np.attachNewNode("virtual_hand")
        self.config = config or {}
        self._point_color = self.config.get("point_color", [0.98, 0.58, 0.12])
        self._anchor_color = self.config.get("anchor_color", [0.2, 0.9, 0.9])
        self._line_color = self.config.get("line_color", [1.0, 0.72, 0.28])
        self._points: dict[str, NodePath] = {}
        self._line_root = self.root.attachNewNode("hand_lines")
        self._init_points()
        self.root.hide()

    def _init_points(self) -> None:
        for name in ("wrist", "thumb_tip", "index_tip"):
            node = self._create_cross_marker(color=self._point_color, alpha=0.95, thickness=2)
            node.setScale(self.POINT_MARKER_SCALE)
            node.reparentTo(self.root)
            self._points[name] = node

        anchor = self._create_cross_marker(color=self._anchor_color, alpha=1.0, thickness=3)
        anchor.setScale(self.ANCHOR_MARKER_SCALE)
        anchor.reparentTo(self.root)
        self._points["anchor"] = anchor

    def update_points(self, points: Optional[dict[str, Vec3]]) -> None:
        if not points:
            self.root.hide()
            self._clear_lines()
            return

        required = {"wrist", "thumb_tip", "index_tip", "anchor"}
        if any(name not in points for name in required):
            self.root.hide()
            self._clear_lines()
            return

        self.root.show()
        self._clear_lines()
        for name, node in self._points.items():
            node.setPos(points[name])

        self._draw_line(points["wrist"], points["thumb_tip"])
        self._draw_line(points["wrist"], points["index_tip"])
        self._draw_line(points["thumb_tip"], points["index_tip"])
        self._draw_line(points["anchor"], points["thumb_tip"])
        self._draw_line(points["anchor"], points["index_tip"])

    def _draw_line(self, start: Vec3, end: Vec3) -> None:
        segs = LineSegs()
        segs.setColor(Vec4(*self._line_color, 0.9))
        segs.setThickness(self.LINE_THICKNESS)
        segs.moveTo(start)
        segs.drawTo(end)
        self._line_root.attachNewNode(segs.create())

    def _clear_lines(self) -> None:
        for child in self._line_root.get_children():
            child.detach_node()

    @staticmethod
    def _create_cross_marker(*, color: list[float], alpha: float, thickness: float) -> NodePath:
        segs = LineSegs()
        segs.setColor(float(color[0]), float(color[1]), float(color[2]), float(alpha))
        segs.setThickness(float(thickness))

        half = 0.5
        segs.moveTo(-half, 0.0, 0.0)
        segs.drawTo(half, 0.0, 0.0)
        segs.moveTo(0.0, -half, 0.0)
        segs.drawTo(0.0, half, 0.0)
        segs.moveTo(0.0, 0.0, -half)
        segs.drawTo(0.0, 0.0, half)

        node = NodePath(segs.create())
        node.setTransparency(TransparencyAttrib.MAlpha)
        return node
