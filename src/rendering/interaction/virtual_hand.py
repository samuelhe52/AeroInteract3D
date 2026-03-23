"""Virtual hand rendering with collision detection"""

from typing import List, Optional
from panda3d.core import (
    CollisionNode,
    CollisionSphere,
    LineSegs,
    NodePath,
    TransparencyAttrib,
    Vec3,
)
from src.gesture.debug.live_preview_runtime import HAND_CONNECTIONS


class VirtualHand:
    LANDMARK_MARKER_SCALE = 0.05
    COLLIDER_VISUAL_SCALE = 0.2
    LANDMARK_MARKER_THICKNESS = 2
    COLLIDER_MARKER_THICKNESS = 3
    MIN_VISUAL_SCALE = 0.7
    MAX_VISUAL_SCALE = 1.35

    def __init__(self, base, root_np: NodePath, config: dict | None = None):
        self.base = base
        self.root = root_np.attachNewNode("virtual_hand")
        
        # 配置参数
        self.config = config or {}
        self.SCALE = self.config.get("scale", 4.0)
        self.DEPTH_SCALE = self.config.get("depth_scale", 4.0)
        self.PERSPECTIVE_SCALE = self.config.get("perspective_scale", 0.4)
        self.BONE_COLOR = self.config.get("bone_color", [1.0, 0.5, 0.0])
        self.BONE_WIDTH = self.config.get("bone_width", 2.0)
        self.LANDMARK_COLOR = self.config.get("landmark_color", [1.0, 0.72, 0.28])
        self.COLLIDER_COLOR = self.config.get("collider_color", [1.0, 0.3, 0.3])
        
        # 21个关键点小球
        self.landmark_spheres: List[NodePath] = []
        self._init_landmark_spheres()
        
        # 骨骼连线
        self.bone_lines: List[NodePath] = []
        
        # 指尖碰撞球（食指8，拇指4）
        self.index_collider: Optional[NodePath] = None
        self.thumb_collider: Optional[NodePath] = None
        self._index_visual: Optional[NodePath] = None
        self._thumb_visual: Optional[NodePath] = None
        self.COLLIDER_RADIUS = 0.1
        self._init_colliders()
        
        # 初始隐藏
        self.root.hide()
        
        # 初始化骨骼连线
        self._init_bone_lines()

    def _init_landmark_spheres(self):
        """初始化21个关键点小球"""
        for i in range(21):
            marker = self._create_cross_marker(
                color=self.LANDMARK_COLOR,
                alpha=1.0,
                thickness=self.LANDMARK_MARKER_THICKNESS,
            )
            marker.setScale(self.LANDMARK_MARKER_SCALE)
            marker.reparentTo(self.root)
            self.landmark_spheres.append(marker)

    def _init_colliders(self):
        """初始化指尖碰撞球（带红色可视化）"""
        # 食指尖
        idx_sphere = CollisionSphere(0, 0, 0, self.COLLIDER_RADIUS)
        idx_node = CollisionNode("index_tip")
        idx_node.addSolid(idx_sphere)
        self.index_collider = self.root.attachNewNode(idx_node)
        # 可视化
        vis = self._create_cross_marker(
            color=self.COLLIDER_COLOR,
            alpha=0.45,
            thickness=self.COLLIDER_MARKER_THICKNESS,
        )
        vis.setScale(self.COLLIDER_VISUAL_SCALE)
        vis.reparentTo(self.index_collider)
        self._index_visual = vis
        
        # 拇指尖
        thumb_sphere = CollisionSphere(0, 0, 0, self.COLLIDER_RADIUS)
        thumb_node = CollisionNode("thumb_tip")
        thumb_node.addSolid(thumb_sphere)
        self.thumb_collider = self.root.attachNewNode(thumb_node)
        # 可视化
        vis = self._create_cross_marker(
            color=self.COLLIDER_COLOR,
            alpha=0.45,
            thickness=self.COLLIDER_MARKER_THICKNESS,
        )
        vis.setScale(self.COLLIDER_VISUAL_SCALE)
        vis.reparentTo(self.thumb_collider)
        self._thumb_visual = vis
    
    def _init_bone_lines(self):
        """初始化骨骼连线"""
        for start_idx, end_idx in HAND_CONNECTIONS:
            # 创建线段
            segs = LineSegs()
            segs.setColor(*self.BONE_COLOR, 1)  # 从配置读取颜色
            segs.setThickness(self.BONE_WIDTH)  # 从配置读取线宽
            segs.moveTo(0, 0, 0)  # 临时位置
            segs.drawTo(0, 0, 0)  # 临时位置
            # 创建节点并添加到root
            line_node = segs.create()
            line_np = self.root.attachNewNode(line_node)
            self.bone_lines.append(line_np)

    def update(self, landmarks: Optional[List[Vec3]]):
        """更新虚拟手（核心）"""
        if not landmarks or len(landmarks) != 21:
            self.root.hide()
            return
        
        # 【关键】有数据时强制显示
        self.root.show()
        
        # The virtual hand receives camera_norm-style landmarks:
        # x/right, y/up, z/toward camera.
        converted = []
        for lm in landmarks:
            x = lm.x * (self.SCALE * 0.5)
            y = -lm.z * self.DEPTH_SCALE
            z = lm.y * (self.SCALE * 0.5)
            converted.append(Vec3(x, y, z))

        center = self._centroid(converted)
        average_depth = sum(lm.z for lm in landmarks) / len(landmarks)
        visual_scale = self._visual_scale_from_depth(average_depth)
        scaled_positions = [
            center + ((pos - center) * visual_scale)
            for pos in converted
        ]
        
        # 更新关键点
        for i, pos in enumerate(scaled_positions):
            if i < len(self.landmark_spheres):
                self.landmark_spheres[i].setPos(pos)
                self.landmark_spheres[i].setScale(self.LANDMARK_MARKER_SCALE * visual_scale)
        
        # 更新碰撞球位置
        if len(scaled_positions) >= 9:
            self.index_collider.setPos(scaled_positions[8])
            if self._index_visual is not None:
                self._index_visual.setScale(self.COLLIDER_VISUAL_SCALE * visual_scale)
        if len(scaled_positions) >= 5:
            self.thumb_collider.setPos(scaled_positions[4])
            if self._thumb_visual is not None:
                self._thumb_visual.setScale(self.COLLIDER_VISUAL_SCALE * visual_scale)
        
        # 更新骨骼连线（修复版）
        for i, (start_idx, end_idx) in enumerate(HAND_CONNECTIONS):
            if i < len(self.bone_lines) and start_idx < len(scaled_positions) and end_idx < len(scaled_positions):
                start_pos = scaled_positions[start_idx]
                end_pos = scaled_positions[end_idx]
                
                # 1. 先移除旧的线段节点
                self.bone_lines[i].removeNode()
                
                # 2. 创建新的线段
                segs = LineSegs()
                segs.setColor(*self.BONE_COLOR, 1.0)
                segs.setThickness(self.BONE_WIDTH)
                segs.moveTo(start_pos)
                segs.drawTo(end_pos)
                
                # 3. 生成新节点并挂载
                new_line_np = self.root.attachNewNode(segs.create())
                
                # 4. 更新列表里的引用
                self.bone_lines[i] = new_line_np

    @staticmethod
    def _centroid(points: List[Vec3]) -> Vec3:
        count = max(len(points), 1)
        return Vec3(
            sum(point.x for point in points) / count,
            sum(point.y for point in points) / count,
            sum(point.z for point in points) / count,
        )

    def _visual_scale_from_depth(self, average_depth: float) -> float:
        scale = 1.0 - (float(average_depth) * float(self.PERSPECTIVE_SCALE))
        return max(self.MIN_VISUAL_SCALE, min(self.MAX_VISUAL_SCALE, scale))

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
