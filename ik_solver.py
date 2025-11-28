import numpy as np
from robot_model import HydraulicSoftArmKinematics

class IKSolver:
    def __init__(self):
        self.model = HydraulicSoftArmKinematics()
        self.learning_rate = 0.6 
        self.limits_rad = [
            np.deg2rad([-60, 60]),   # J1
            np.deg2rad([-28, 90]),   # J2
            np.deg2rad([-152, -42])  # J3
        ]

    def solve(self, target_pos, current_q_deg):
        target = np.array(target_pos)
        
        # 1. 软体参数计算
        # 构造 7 维向量获取当前基座位置
        q_full_current = current_q_deg[:3] + [0, 0, 0, 200] 
        r_pts, _, T_base_soft, _ = self.model.forward_kinematics(q_full_current)
        soft_base_pos = r_pts[-1]
        R_base = T_base_soft[:3, :3]
        
        vec_to_target = target - soft_base_pos
        dist_to_target = np.linalg.norm(vec_to_target)
        vec_local = R_base.T @ vec_to_target
        
        new_phi_deg = np.degrees(np.arctan2(vec_local[2], vec_local[1]))
        new_len = np.clip(dist_to_target, 140, 250)
        
        if dist_to_target < 50: new_bend = 120
        elif dist_to_target < 250: new_bend = (250 - dist_to_target) / 200 * 100
        else: new_bend = 0

        # 2. 刚性臂 IK (DLS)
        q = np.array([np.deg2rad(v) for v in current_q_deg[:3]])
        
        for _ in range(3):
            # 构造 7 维向量用于计算
            q_calc = np.degrees(q).tolist() + [0, 0, 0, 200]
            r_pts, _, _, _ = self.model.forward_kinematics(q_calc)
            current_tip = r_pts[-1]
            
            diff = target - current_tip
            dist = np.linalg.norm(diff)
            
            if dist > 1e-3:
                desired_pos = target - (diff / dist) * 160.0
            else:
                desired_pos = current_tip
            
            error = desired_pos - current_tip
            if np.linalg.norm(error) < 1.0: break

            J = np.zeros((3, 3))
            delta = 0.001
            for i in range(3):
                q_temp = q.copy()
                q_temp[i] += delta
                # 构造 7 维向量用于微分
                q_full_temp = np.degrees(q_temp).tolist() + [0, 0, 0, 200]
                r_pts_temp, _, _, _ = self.model.forward_kinematics(q_full_temp)
                J[:, i] = (r_pts_temp[-1] - current_tip) / delta
            
            eye = np.eye(3) * 0.01 # Damping
            dq = np.linalg.inv(J.T @ J + eye) @ J.T @ error
            
            # 限幅
            dq_deg = np.degrees(dq)
            max_dq = np.max(np.abs(dq_deg))
            if max_dq > 2.0: dq *= (2.0 / max_dq)
                
            q += dq
            for i in range(3): q[i] = np.clip(q[i], self.limits_rad[i][0], self.limits_rad[i][1])

        new_q_deg = np.degrees(q).tolist()
        
        # 返回 7 维向量 [q1, q2, q3, q4=0, bend, phi, len]
        return new_q_deg + [0, new_bend, new_phi_deg, new_len]