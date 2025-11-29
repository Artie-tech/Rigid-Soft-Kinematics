"""
文件功能：机器人运动学模型定义

本文件定义了 `HydraulicSoftArmKinematics` 类，它是整个项目的核心，负责处理机械臂的正向运动学（Forward Kinematics, FK）。

主要功能：
1.  **定义机器人结构**:
    - 存储刚性部分的 D-H (Denavit-Hartenberg) 参数。
    - 定义软体部分的几何参数（如分段数）。
    - 设定所有关节（刚性和软体）的运动限位。

2.  **实现正向运动学**:
    - `forward_kinematics` 方法是核心，它接收一组关节角度作为输入。
    - 首先，根据 D-H 参数计算刚性臂各关节的位姿。
    - 然后，基于刚性臂末端的位姿，使用 PCC (Piecewise Constant Curvature) 模型计算软体部分的形状和末端位姿。
    -最终，输出机器人所有关键点的三维坐标以及末端执行器的位姿矩阵。
"""

import numpy as np

class HydraulicSoftArmKinematics:
    def __init__(self):
        self.base_z_offset = 500.0 # mm
        
        # Modified D-H 参数 (3个可动关节 + 1个固定延长段)
        self.rigid_dh_params = [
            {'alpha': 0, 'a': 190, 'd': 0},             # Link 1
            {'alpha': np.radians(90), 'a': 90, 'd': 0}, # Link 2
            {'alpha': 0, 'a': 605, 'd': 0},             # Link 3
            {'alpha': 0, 'a': 290, 'd': 0}              # Link 4 (固定延长段)
        ]
        
        self.soft_segments = 20

    def mdh_matrix(self, alpha, a, theta, d):
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        return np.array([
            [ct,    -st,    0,    a],
            [st*ca, ct*ca, -sa, -d*sa],
            [st*sa, ct*sa,  ca,  d*ca],
            [0,     0,      0,    1]
        ])

    def forward_kinematics(self, q_degrees):
        """
        修正后的输入解析：
        q_degrees 必须是 7 维向量: [q1, q2, q3, q4_fixed, bend, phi, len]
        """
        # 1. 提取刚性关节角度 (前3个)-
        # q4_fixed (index 3) 被忽略，但在循环中会自动处理为 0
        q_rigid_rad = [np.deg2rad(q_degrees[i]) for i in range(3)]
        
        # 2. 提取软体参数 (注意索引偏移！)
        theta_bend = np.deg2rad(q_degrees[4]) # Index 4: Bend
        phi_dir = np.deg2rad(q_degrees[5])    # Index 5: Phi
        length_mm = q_degrees[6]              # Index 6: Length
        
        # --- 计算刚性部分 ---
        T_cum = np.eye(4)
        T_cum[2, 3] = self.base_z_offset
        rigid_points = [T_cum[:3, 3]]
        
        for i in range(4):
            # 前3个用变量，第4个(index 3)固定为 0
            theta = q_rigid_rad[i] if i < 3 else 0 
            
            T_i = self.mdh_matrix(self.rigid_dh_params[i]['alpha'], 
                                  self.rigid_dh_params[i]['a'], 
                                  theta, 
                                  self.rigid_dh_params[i]['d'])
            T_cum = T_cum @ T_i
            rigid_points.append(T_cum[:3, 3])
            
        T_base_soft = T_cum
        
        # --- 计算软体部分 (PCC) ---
        soft_points_local = []
        if abs(theta_bend) < 1e-4:
            for i in range(self.soft_segments + 1):
                s = (i / self.soft_segments) * length_mm
                soft_points_local.append([s, 0, 0, 1])
            # 末端局部变换 (纯平移)
            T_tip_local = np.eye(4)
            T_tip_local[0, 3] = length_mm
        else:
            R = length_mm / theta_bend
            for i in range(self.soft_segments + 1):
                sigma = (i / self.soft_segments) * theta_bend
                # 常曲率圆弧参数方程
                x = R * np.sin(sigma)
                y_raw = R * (1 - np.cos(sigma))
                
                # 空间旋转 Phi
                y = y_raw * np.cos(phi_dir)
                z = y_raw * np.sin(phi_dir)
                soft_points_local.append([x, y, z, 1])
            
            # 末端局部变换
            tip_x = R * np.sin(theta_bend)
            tip_y_raw = R * (1 - np.cos(theta_bend))
            
            # --- 精确计算末端姿态 (Frenet-Serret frame) ---
            # 1. 计算切线方向 (Tangent vector) -> X轴
            # 在弯曲平面内，切线方向是圆弧的切向
            x_axis = np.array([
                np.cos(theta_bend),
                np.sin(theta_bend) * np.cos(phi_dir),
                np.sin(theta_bend) * np.sin(phi_dir)
            ])
            x_axis /= np.linalg.norm(x_axis)

            # 2. 计算法线方向 (Normal vector) -> Z轴
            # 法线指向圆心（向内）
            z_axis = np.array([
                -np.sin(theta_bend),
                np.cos(theta_bend) * np.cos(phi_dir),
                np.cos(theta_bend) * np.sin(phi_dir)
            ])
            z_axis /= np.linalg.norm(z_axis)

            # 3. 计算副法线 (Binormal vector) -> Y轴
            # Y = Z × X (右手坐标系)
            y_axis = np.cross(z_axis, x_axis)
            y_axis /= np.linalg.norm(y_axis)
            
            # 4. 构建末端局部变换矩阵
            T_tip_local = np.eye(4)
            T_tip_local[:3, 0] = x_axis  # X轴方向
            T_tip_local[:3, 1] = y_axis  # Y轴方向
            T_tip_local[:3, 2] = z_axis  # Z轴方向（法线）
            T_tip_local[:3, 3] = [tip_x, tip_y_raw*np.cos(phi_dir), tip_y_raw*np.sin(phi_dir)]
            
        # 坐标变换：局部 -> 全局
        soft_points_global = (T_base_soft @ np.array(soft_points_local).T).T
        T_tip_global = T_base_soft @ T_tip_local
        
        return np.array(rigid_points), soft_points_global[:, :3], T_base_soft, T_tip_global