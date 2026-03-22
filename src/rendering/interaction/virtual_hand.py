"""Virtual hand rendering with collision detection"""

from panda3d.core import NodePath, CollisionSphere, CollisionNode, Vec3, LineSegs
from src.gesture.debug.live_preview_runtime import HAND_CONNECTIONS
from typing import List, Optional


class VirtualHand:
    def __init__(self, base, root_np: NodePath):
        self.base = base
        self.root = root_np.attachNewNode("virtual_hand")
        
        # 21个关键点小球
        self.landmark_spheres: List[NodePath] = []
        self._init_landmark_spheres()
        
        # 骨骼连线
        self.bone_lines: List[NodePath] = []
        
        # 指尖碰撞球（食指8，拇指4）
        self.index_collider: Optional[NodePath] = None
        self.thumb_collider: Optional[NodePath] = None
        self.COLLIDER_RADIUS = 0.1
        self._init_colliders()
        
        # 初始隐藏
        self.root.hide()
        
        # 初始化骨骼连线
        self._init_bone_lines()

    def _init_landmark_spheres(self):
        """初始化21个关键点小球"""
        for i in range(21):
            sphere = self.base.loader.loadModel("box")
            sphere.setScale(0.05)
            sphere.reparentTo(self.root)
            self.landmark_spheres.append(sphere)

    def _init_colliders(self):
        """初始化指尖碰撞球（带红色可视化）"""
        # 食指尖
        idx_sphere = CollisionSphere(0, 0, 0, self.COLLIDER_RADIUS)
        idx_node = CollisionNode("index_tip")
        idx_node.addSolid(idx_sphere)
        self.index_collider = self.root.attachNewNode(idx_node)
        # 可视化
        vis = self.base.loader.loadModel("box")
        vis.setColor(1, 0, 0, 0.5)
        vis.setScale(self.COLLIDER_RADIUS * 2)
        vis.reparentTo(self.index_collider)
        
        # 拇指尖
        thumb_sphere = CollisionSphere(0, 0, 0, self.COLLIDER_RADIUS)
        thumb_node = CollisionNode("thumb_tip")
        thumb_node.addSolid(thumb_sphere)
        self.thumb_collider = self.root.attachNewNode(thumb_node)
        # 可视化
        vis = self.base.loader.loadModel("box")
        vis.setColor(1, 0, 0, 0.5)
        vis.setScale(self.COLLIDER_RADIUS * 2)
        vis.reparentTo(self.thumb_collider)
    
    def _init_bone_lines(self):
        """初始化骨骼连线"""
        for start_idx, end_idx in HAND_CONNECTIONS:
            # 创建线段
            segs = LineSegs()
            segs.setColor(1, 0.5, 0, 1)  # 橙色，和左上角预览一致
            segs.setThickness(2)  # 线宽
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
        
        # 【核心】正确的坐标转换，禁止乱改
        SCALE = 3.0  # 调整缩放系数
        DEPTH_SCALE = 4.0  # 调整深度缩放
        converted = []
        for lm in landmarks:
            x = -(lm.x - 0.5) * SCALE  # 左右
            y = -lm.z * DEPTH_SCALE     # 前后深度（负号确保在相机前方）
            z = (0.5 - lm.y) * SCALE    # 上下
            converted.append(Vec3(x, y, z))
        
        # 更新关键点
        for i, pos in enumerate(converted):
            if i < len(self.landmark_spheres):
                self.landmark_spheres[i].setPos(pos)
        
        # 更新碰撞球位置
        if len(converted) >= 9:
            self.index_collider.setPos(converted[8])
        if len(converted) >= 5:
            self.thumb_collider.setPos(converted[4])
        
        # 更新骨骼连线
        for i, (start_idx, end_idx) in enumerate(HAND_CONNECTIONS):
            if i < len(self.bone_lines) and start_idx < len(converted) and end_idx < len(converted):
                start_pos = converted[start_idx]
                end_pos = converted[end_idx]
                # 重新创建线段
                segs = LineSegs()
                segs.setColor(1, 0.5, 0, 1)  # 橙色
                segs.setThickness(2)  # 线宽
                segs.moveTo(start_pos)
                segs.drawTo(end_pos)
                # 更新线段
                line_node = segs.create()
                self.bone_lines[i].removeNode()
                self.bone_lines[i] = self.root.attachNewNode(line_node)