import numpy as np
from robot_model import HydraulicSoftArmKinematics

class IKSolver:
    def __init__(self):
        self.model = HydraulicSoftArmKinematics()
        
        # === 抗抖动参数配置 ===
        self.damping = 0.1        # 阻尼系数 (DLS核心)：越大越稳，越小越准
        self.step_limit = 2.0     # 单帧最大转动角度 (度)：限制瞬时剧烈运动
        self.smooth_factor = 0.3  # 平滑因子 (0~1)：越小越平滑，但会有延迟感
        
        # 上一帧的关节角度 (用于平滑滤波)
        self.prev_q = None
        
        # 刚性关节限位 (Rad)
        self.limits_rad = [
            np.deg2rad([-60, 60]),   # J1
            np.deg2rad([-28, 90]),   # J2
            np.deg2rad([-152, -42])  # J3
        ]

    def solve(self, target_pos, current_q_deg):
        """
        升级版 IK 求解器：使用阻尼最小二乘法 (DLS) + 动态平滑
        """
        target = np.array(target_pos)
        
        # 初始化上一帧角度
        if self.prev_q is None:
            self.prev_q = np.array(current_q_deg[:3])

        # === 第一步：计算软体臂参数 (几何法，保持不变) ===
        # 获取刚性臂当前状态
        q_full_current = current_q_deg[:3] + [0, 0, 0, 200]
        r_pts, _, T_base_soft, _ = self.model.forward_kinematics(q_full_current)
        soft_base_pos = r_pts[-1]
        R_base = T_base_soft[:3, :3]
        
        # 计算软体参数
        vec_to_target = target - soft_base_pos
        dist_to_target = np.linalg.norm(vec_to_target)
        vec_local = R_base.T @ vec_to_target # 转到局部坐标
        
        # 1. Phi (旋转)
        new_phi_deg = np.degrees(np.arctan2(vec_local[2], vec_local[1]))
        
        # 2. Length (伸缩)
        new_len = np.clip(dist_to_target, 140, 250)
        
        # 3. Bend (弯曲)
        if dist_to_target < 50: new_bend = 120
        elif dist_to_target < 250: new_bend = (250 - dist_to_target) / 200 * 100
        else: new_bend = 0

        # === 第二步：刚性臂 IK (DLS 算法) ===
        # 目标：让刚性末端停在距离物体 160mm 处
        q = np.array([np.deg2rad(v) for v in current_q_deg[:3]])
        
        # 迭代几次 (DLS 收敛很快，3次通常够了)
        for _ in range(3):
            # 正向运动学
            q_calc = np.degrees(q).tolist() + [0, 0, 0, 200]
            r_pts, _, _, _ = self.model.forward_kinematics(q_calc)
            current_tip = r_pts[-1]
            
            # 计算这一步的目标位置
            diff = target - current_tip
            dist = np.linalg.norm(diff)
            if dist > 1e-3:
                direction = diff / dist
                desired_pos = target - direction * 160.0 # 保持安全距离
            else:
                desired_pos = current_tip
            
            error = desired_pos - current_tip
            error_norm = np.linalg.norm(error)
            
            # 如果误差很小，就不用算了，防止微小抖动
            if error_norm < 1.0:
                break

            # 计算雅可比 (Jacobian)
            J = np.zeros((3, 3))
            delta = 0.001
            for i in range(3):
                q_temp = q.copy()
                q_temp[i] += delta
                q_full_temp = np.degrees(q_temp).tolist() + [0, 0, 0, 200]
                r_pts_temp, _, _, _ = self.model.forward_kinematics(q_full_temp)
                J[:, i] = (r_pts_temp[-1] - current_tip) / delta
            
            # --- DLS 核心公式 ---
            # dq = (J^T * J + lambda^2 * I)^-1 * J^T * error
            # 这种方法在奇异点附近非常稳定
            eye = np.eye(3) * (self.damping ** 2)
            dq = np.linalg.inv(J.T @ J + eye) @ J.T @ error
            
            # --- 速度限幅 (Clamping) ---
            # 限制每一此迭代关节最多转动 step_limit 度
            dq_deg = np.degrees(dq)
            max_dq = np.max(np.abs(dq_deg))
            if max_dq > self.step_limit:
                scale = self.step_limit / max_dq
                dq *= scale
                
            q += dq
            
            # 关节限位
            for i in range(3):
                q[i] = np.clip(q[i], self.limits_rad[i][0], self.limits_rad[i][1])

        # === 第三步：输出滤波 ===
        # 将计算出的新角度与上一帧角度做平滑插值
        new_q_raw = np.degrees(q)
        
        # Low-pass filter: Q_out = alpha * Q_new + (1-alpha) * Q_old
        filtered_q = self.smooth_factor * new_q_raw + (1 - self.smooth_factor) * self.prev_q
        self.prev_q = filtered_q # 更新历史值
        
        return filtered_q.tolist() + [0, new_bend, new_phi_deg, new_len]