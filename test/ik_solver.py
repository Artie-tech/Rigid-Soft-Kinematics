import numpy as np
from robot_model import HydraulicSoftArmKinematics

class IKSolver:
    def __init__(self):
        self.model = HydraulicSoftArmKinematics()
        self.learning_rate = 0.8 
        
        # 刚性关节限位
        self.limits_rad = [
            np.deg2rad([-60, 60]),   # J1
            np.deg2rad([-28, 90]),   # J2
            np.deg2rad([-152, -42])  # J3
        ]
        # 软体臂有效工作长度范围
        self.soft_range = [140.0, 250.0]

    def solve(self, target_pos, current_q_deg):
        target = np.array(target_pos)
        
        # === 1. 刚性臂 IK (DLS + 范围约束) ===
        # 提取当前刚性角度
        q = np.array([np.deg2rad(v) for v in current_q_deg[:3]])
        
        for _ in range(5): # 迭代求解
            # 计算当前刚性末端位置
            # 构造临时 7维向量 (后4位补0即可，只算刚性)
            q_calc = np.degrees(q).tolist() + [0, 0, 0, 200]
            r_pts, _, _, _ = self.model.forward_kinematics(q_calc)
            current_tip = r_pts[-1]
            
            # 计算距离误差
            diff_vec = target - current_tip
            dist = np.linalg.norm(diff_vec)
            
            # 范围约束逻辑：
            # 如果距离在 [140, 250] 之间，说明刚性臂位置完美，不需要动
            if dist < self.soft_range[0]:
                target_dist = self.soft_range[0] # 太近，退一点
            elif dist > self.soft_range[1]:
                target_dist = self.soft_range[1] # 太远，追一点
            else:
                target_dist = dist # 保持
            
            if dist > 1e-4:
                direction = diff_vec / dist
                desired_pos = target - direction * target_dist
            else:
                desired_pos = current_tip
                
            error = desired_pos - current_tip
            if np.linalg.norm(error) < 2.0: break # 精度满足

            # 计算雅可比
            J = np.zeros((3, 3))
            delta = 0.001
            for i in range(3):
                q_tmp = q.copy(); q_tmp[i] += delta
                q_full_tmp = np.degrees(q_tmp).tolist() + [0, 0, 0, 200]
                r_tmp, _, _, _ = self.model.forward_kinematics(q_full_tmp)
                J[:, i] = (r_tmp[-1] - current_tip) / delta
            
            # DLS 更新
            lambda_sq = 0.04
            dq = np.linalg.inv(J.T @ J + lambda_sq * np.eye(3)) @ J.T @ error
            q += dq
            
            # 限位
            for i in range(3):
                q[i] = np.clip(q[i], self.limits_rad[i][0], self.limits_rad[i][1])

        new_q_deg = np.degrees(q).tolist()

        # === 2. 软体臂参数计算 (几何解析) ===
        # 使用计算出的新刚性角度，再次正解得到基座位置
        q_full_new = new_q_deg + [0, 0, 0, 200]
        r_pts, _, T_base_soft, _ = self.model.forward_kinematics(q_full_new)
        
        soft_base = r_pts[-1]
        R_base = T_base_soft[:3, :3]
        
        vec_to_target = target - soft_base
        dist_final = np.linalg.norm(vec_to_target)
        vec_local = R_base.T @ vec_to_target
        
        # 计算 Phi (对准目标)
        new_phi = np.degrees(np.arctan2(vec_local[2], vec_local[1]))
        
        # 计算 Length (伸缩)
        new_len = np.clip(dist_final, self.soft_range[0], self.soft_range[1])
        
        # 计算 Bend (抓取动作)
        # 只有当基座距离合适时才弯曲
        if dist_final < 50: 
            new_bend = 120 # 贴脸了，完全卷起
        elif dist_final < 260:
            # 距离越近弯得越多
            new_bend = (1.0 - (dist_final - 50) / 210) * 120
        else:
            new_bend = 0

        # 返回完整 7 维向量
        return new_q_deg + [0, new_bend, new_phi, new_len]