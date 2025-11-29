import numpy as np
from robot_model import HydraulicSoftArmKinematics

class IKSolver:
    def __init__(self):
        self.model = HydraulicSoftArmKinematics()
        self.learning_rate = 0.8 
        # 刚性关节限位 (Rad)
        self.limits_rad = [
            np.deg2rad([-60, 60]),   # J1: 基座
            np.deg2rad([-28, 90]),   # J2: 大臂
            np.deg2rad([-152, -42])  # J3: 小臂
        ]
        # 软体臂的伸缩范围 (mm)
        self.soft_range = [140.0, 250.0]

    def solve(self, target_pos, current_q_deg):
        target = np.array(target_pos)
        
        # 尝试求解。如果第一次失败（陷入局部极小值），尝试随机姿态重启
        # 我们最多尝试 3 次重启，保证实时性
        best_q = None
        min_error = float('inf')
        
        # 第一次尝试：从当前角度开始 (利用时间连续性)
        seeds = [current_q_deg[:3]]
        # 后续尝试：加入随机种子 (在限位范围内随机采样)
        for _ in range(2):
            random_q = [np.random.uniform(l[0], l[1]) for l in self.limits_rad]
            seeds.append(np.degrees(random_q))

        for start_q in seeds:
            q_res, error = self._gradient_descent_range(target, start_q)
            
            # 如果成功进入范围，直接返回
            if error < 5.0:
                return self._compose_result(target, q_res)
            
            # 否则记录最优解，以此保底
            if error < min_error:
                min_error = error
                best_q = q_res

        # 如果都试过了还是不行，就用那个“离得最近”的解
        return self._compose_result(target, best_q)

    def _gradient_descent_range(self, target, start_q_deg):
        """
        核心算法：范围约束 IK
        目标不是重合，而是让 distance 在 [min_len, max_len] 之间
        """
        q = np.array([np.deg2rad(v) for v in start_q_deg])
        final_error = float('inf')

        for _ in range(15): # 增加迭代次数以换取精度
            # 1. 正向运动学获取刚性末端
            q_calc = np.degrees(q).tolist() + [0, 0, 0, 200]
            r_pts, _, _, _ = self.model.forward_kinematics(q_calc)
            current_tip = r_pts[-1]
            
            # 2. 计算距离向量
            diff_vec = target - current_tip
            dist = np.linalg.norm(diff_vec)
            
            # 3. 【核心改进】范围误差计算
            # 我们不要求 dist = 160，只要求 140 < dist < 250
            if dist < self.soft_range[0]:
                # 太近了，需要退后。目标距离设为下限
                target_dist = self.soft_range[0]
            elif dist > self.soft_range[1]:
                # 太远了，需要前进。目标距离设为上限
                target_dist = self.soft_range[1]
            else:
                # 在射程范围内！不需要刚性臂动了，误差为0
                target_dist = dist
            
            # 构造一个虚拟目标点：沿着连线方向，距离刚性末端 target_dist 的位置
            if dist > 1e-4:
                direction = diff_vec / dist
                # 刚性臂希望到达的位置 = 目标物体位置 - 理想的伸长向量
                desired_tip_pos = target - direction * target_dist
            else:
                desired_tip_pos = current_tip

            # 计算位移误差
            error_vec = desired_tip_pos - current_tip
            error_val = np.linalg.norm(error_vec)
            final_error = error_val
            
            # 如果误差足够小（刚性臂已经把软体臂送到了射程内），停止迭代
            if error_val < 5.0:
                break

            # 4. 雅可比迭代 (DLS)
            J = np.zeros((3, 3))
            delta = 0.001
            for i in range(3):
                q_temp = q.copy()
                q_temp[i] += delta
                q_full_temp = np.degrees(q_temp).tolist() + [0, 0, 0, 200]
                r_pts_temp, _, _, _ = self.model.forward_kinematics(q_full_temp)
                J[:, i] = (r_pts_temp[-1] - current_tip) / delta
            
            # DLS 求解
            lambda_sq = 0.04 # 阻尼系数
            dq = np.linalg.inv(J.T @ J + lambda_sq * np.eye(3)) @ J.T @ error_vec
            
            # 5. 更新并限位
            q += dq
            for i in range(3):
                q[i] = np.clip(q[i], self.limits_rad[i][0], self.limits_rad[i][1])
        
        return np.degrees(q), final_error

    def _compose_result(self, target, rigid_q_deg):
        """
        根据计算出的刚性臂角度，自动适配软体臂的参数(弯曲、旋转、伸长)
        """
        # 1. 获取刚性末端位置和姿态
        q_full = rigid_q_deg.tolist() + [0, 0, 0, 200]
        r_pts, _, T_base_soft, _ = self.model.forward_kinematics(q_full)
        soft_base_pos = r_pts[-1]
        R_base = T_base_soft[:3, :3]
        
        # 2. 计算软体参数
        vec_to_target = target - soft_base_pos
        dist = np.linalg.norm(vec_to_target)
        vec_local = R_base.T @ vec_to_target # 转到局部坐标
        
        # Phi (旋转): 对准目标
        new_phi = np.degrees(np.arctan2(vec_local[2], vec_local[1]))
        
        # Length (伸缩): 刚好够到目标，限制在物理范围内
        new_len = np.clip(dist, self.soft_range[0], self.soft_range[1])
        
        # Bend (弯曲): 
        # 逻辑优化：只有当刚性臂确实把基座送到了距离目标 < 260 的地方，才开始弯曲
        # 如果距离太远(>260)，说明根本够不着，保持直的(0)
        if dist > self.soft_range[1] + 20: 
            new_bend = 0
        elif dist < 50: 
            new_bend = 120
        else:
            # 距离越近，弯得越多，模拟抓取
            # 归一化距离: 250 -> 0, 50 -> 1
            ratio = 1.0 - (dist - 50) / (250 - 50)
            new_bend = np.clip(ratio * 120, 0, 120)

        return rigid_q_deg.tolist() + [0, new_bend, new_phi, new_len]