import numpy as np

class HydraulicSoftArmKinematics:
    def __init__(self):
        self.base_z_offset = 500.0 # mm
        
        # Modified D-H 参数 (3个刚性关节 + 1个固定延长段)
        self.rigid_dh_params = [
            {'alpha': 0, 'a': 190, 'd': 0},             # Link 1
            {'alpha': np.radians(90), 'a': 90, 'd': 0}, # Link 2
            {'alpha': 0, 'a': 605, 'd': 0},             # Link 3
            {'alpha': 0, 'a': 290, 'd': 0}              # Link 4 (固定延长段)
        ]
        
        self.soft_segments = 20 # 软体分段数

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
        核心修正：确保输入是 7 维向量
        q = [q1, q2, q3, q4(Fixed), bend, phi, len]
        """
        # 1. 解析参数
        q_rigid_rad = [np.deg2rad(q_degrees[i]) for i in range(3)]
        
        # 软体参数 (注意索引: 跳过 index 3 的 q4)
        theta_bend = np.deg2rad(q_degrees[4]) 
        phi_dir = np.deg2rad(q_degrees[5])    
        length_mm = q_degrees[6]              
        
        # 2. 计算刚性部分
        T_cum = np.eye(4)
        T_cum[2, 3] = self.base_z_offset
        rigid_points = [T_cum[:3, 3]]
        
        for i in range(4):
            # 前3个用变量，第4个固定为0
            theta = q_rigid_rad[i] if i < 3 else 0
            T_i = self.mdh_matrix(self.rigid_dh_params[i]['alpha'], 
                                  self.rigid_dh_params[i]['a'], 
                                  theta, 
                                  self.rigid_dh_params[i]['d'])
            T_cum = T_cum @ T_i
            rigid_points.append(T_cum[:3, 3])
            
        T_base_soft = T_cum
        
        # 3. 计算软体部分 (几何生成)
        soft_points_local = []
        # 避免除以零
        if abs(theta_bend) < 1e-4:
            for i in range(self.soft_segments + 1):
                s = (i / self.soft_segments) * length_mm
                soft_points_local.append([s, 0, 0, 1])
            T_tip_local = np.eye(4)
            T_tip_local[0, 3] = length_mm
        else:
            R = length_mm / theta_bend
            for i in range(self.soft_segments + 1):
                sigma = (i / self.soft_segments) * theta_bend
                x = R * np.sin(sigma)
                y_raw = R * (1 - np.cos(sigma))
                
                # 空间旋转
                y = y_raw * np.cos(phi_dir)
                z = y_raw * np.sin(phi_dir)
                soft_points_local.append([x, y, z, 1])
            
            # 末端局部变换
            tip_x = R * np.sin(theta_bend)
            tip_y_raw = R * (1 - np.cos(theta_bend))
            T_tip_local = np.eye(4)
            T_tip_local[:3, 3] = [tip_x, tip_y_raw*np.cos(phi_dir), tip_y_raw*np.sin(phi_dir)]
            
        # 坐标变换
        soft_points_global = (T_base_soft @ np.array(soft_points_local).T).T
        T_tip_global = T_base_soft @ T_tip_local
        
        return np.array(rigid_points), soft_points_global[:, :3], T_base_soft, T_tip_global