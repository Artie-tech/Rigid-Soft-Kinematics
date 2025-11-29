import numpy as np
from robot_model import HydraulicSoftArmKinematics

class IKSolver:
    def __init__(self):
        self.model = HydraulicSoftArmKinematics()
        # 刚性关节限位 (Rad)
        self.limits_rad = [
            np.deg2rad([-60, 60]),   # J1: 基座
            np.deg2rad([-28, 90]),   # J2: 大臂
            np.deg2rad([-152, -42])  # J3: 小臂
        ]
        # 软体臂的物理限制
        self.soft_len_range = [140.0, 250.0] # 长度范围 (mm)
        self.soft_bend_max = 130.0           # 最大弯曲角度 (deg)

    def solve(self, target_pos, current_q_deg):
        """
        主求解函数
        :param target_pos: 全局目标坐标 [x, y, z]
        :param current_q_deg: 当前所有关节角度 (7维)
        """
        target = np.array(target_pos)
        
        # 策略：如果当前姿态微调就能到，就直接算；否则尝试随机重启避免局部极小值
        seeds = [current_q_deg[:3]]
        
        # 增加随机种子，帮助跳出死锁
        for _ in range(2):
            random_q = [np.random.uniform(l[0], l[1]) for l in self.limits_rad]
            seeds.append(np.degrees(random_q))

        best_q_full = None
        min_total_error = float('inf')

        for start_q in seeds:
            # 1. 刚性臂粗调：将软体臂基座送到合适位置
            rigid_q_res, range_error = self._optimize_rigid_base(target, start_q)
            
            # 2. 软体臂精调：基于几何解析法计算软体参数
            final_q_full, tip_error = self._solve_soft_geometric(target, rigid_q_res)
            
            # 3. 评估总误差
            if tip_error < 2.0: # 精度满足要求 (2mm)
                return final_q_full
            
            if tip_error < min_total_error:
                min_total_error = tip_error
                best_q_full = final_q_full

        return best_q_full

    def _optimize_rigid_base(self, target, start_q_deg):
        """
        刚性臂优化：目标不是重合，而是让 Target 落入软体臂的“可达甜区”
        甜区定义：以刚性末端为球心，半径为软体臂长度中值 (约 200mm) 的球壳
        """
        q = np.array([np.deg2rad(v) for v in start_q_deg])
        
        # 理想的软体臂长度 (取中间值，留出伸缩余量)
        ideal_reach = (self.soft_len_range[0] + self.soft_len_range[1]) / 2.0
        
        for _ in range(20): # 迭代次数
            # FK 计算刚性末端 (软体部分设为 0)
            q_calc = np.degrees(q).tolist() + [0, 0, 0, 0] 
            r_pts, _, _, _ = self.model.forward_kinematics(q_calc)
            current_base = r_pts[-1] # 刚性末端 = 软体基座
            
            # 向量：从软体基座指向目标
            vec_to_target = target - current_base
            dist = np.linalg.norm(vec_to_target)
            
            # 误差逻辑：我们需要 dist 接近 ideal_reach
            # 这里的误差是标量误差，我们希望 dist = ideal_reach
            dist_error = dist - ideal_reach
            
            if abs(dist_error) < 5.0: # 已经进入理想区间
                break
                
            # 构造虚拟力：
            # 如果太远 (dist > ideal)，刚性臂需要向目标移动
            # 如果太近 (dist < ideal)，刚性臂需要远离目标
            # 移动方向沿着 vec_to_target
            direction = vec_to_target / (dist + 1e-6)
            
            # 我们希望刚性末端移动到的位置
            desired_base_pos = target - direction * ideal_reach
            
            # 计算 Cartesian 误差向量
            cartesian_error = desired_base_pos - current_base
            
            # --- 雅可比迭代 (DLS) ---
            J = np.zeros((3, 3))
            delta = 0.001
            for i in range(3):
                q_temp = q.copy()
                q_temp[i] += delta
                q_full_temp = np.degrees(q_temp).tolist() + [0, 0, 0, 0]
                r_pts_temp, _, _, _ = self.model.forward_kinematics(q_full_temp)
                J[:, i] = (r_pts_temp[-1] - current_base) / delta
            
            # 阻尼最小二乘求解
            lambda_sq = 0.1
            dq = np.linalg.inv(J.T @ J + lambda_sq * np.eye(3)) @ J.T @ cartesian_error
            
            q += dq
            # 关节限位
            for i in range(3):
                q[i] = np.clip(q[i], self.limits_rad[i][0], self.limits_rad[i][1])
                
        return np.degrees(q), abs(dist - ideal_reach)

    def _solve_soft_geometric(self, target, rigid_q_deg):
        """
        软体臂解析解：根据刚性末端位置，计算软体臂所需的弯曲、旋转和长度
        这是一个纯几何过程，不需要迭代。
        """
        # 1. 获取刚性末端的变换矩阵 (姿态很重要)
        q_dummy = rigid_q_deg.tolist() + [0, 0, 0, 0]
        _, _, T_base_soft, _ = self.model.forward_kinematics(q_dummy)
        
        soft_base_pos = T_base_soft[:3, 3]
        R_base = T_base_soft[:3, :3] # 软体基座的旋转矩阵
        
        # 2. 将目标点转换到软体臂的【局部坐标系】
        # 这一点至关重要！所有的 PCC 计算都在局部系完成
        vec_global = target - soft_base_pos
        vec_local = R_base.T @ vec_global  # [x, y, z]
        
        x, y, z = vec_local
        
        # 注意：根据 robot_model.py，软体臂沿 X 轴生长 (s 对应 x)
        # y, z 是横截面偏移
        
        # --- A. 计算 Phi (旋转角) ---
        # 决定弯曲平面的方向
        phi_rad = np.arctan2(z, y)
        phi_deg = np.degrees(phi_rad)
        
        # --- B. 计算几何参数 ---
        # 投影到弯曲平面后的“高度” h (偏离 X 轴的距离)
        h = np.sqrt(y**2 + z**2)
        # 沿主轴的距离
        d_x = x
        
        # --- C. 计算 Bend (弯曲角 Theta) 和 Length (弧长 S) ---
        # 几何关系：恒定曲率圆弧经过 (0,0) 和 (d_x, h)
        # 圆弧方程推导：
        # 半径 R, 弯曲角 theta. 
        # 弦长 chord = sqrt(d_x^2 + h^2)
        # theta = 2 * atan2(h, d_x)  <-- PCC 几何核心公式
        
        if h < 1e-4: 
            # 直线情况
            theta_rad = 0
            arc_length = d_x
        else:
            # 曲线情况
            theta_rad = 2 * np.arctan2(h, d_x)
            
            # 避免除以零或数值不稳定
            if abs(theta_rad) < 1e-4:
                arc_length = d_x
            else:
                # 半径 R = h / (1 - cos(theta)) ??? 不对，那是另一种参数化
                # 使用更稳健的公式：R = L / theta -> L = R * theta
                # 几何推导：R = (d_x^2 + h^2) / (2*h)
                R = (d_x**2 + h**2) / (2 * h)
                arc_length = R * theta_rad

        # --- D. 约束检查与修正 ---
        theta_deg = np.degrees(theta_rad)
        
        # 1. 长度限制
        final_len = np.clip(arc_length, self.soft_len_range[0], self.soft_len_range[1])
        
        # 2. 角度限制
        final_bend = np.clip(theta_deg, -self.soft_bend_max, self.soft_bend_max)
        
        # 构造最终指令
        final_q = rigid_q_deg.tolist() + [0, final_bend, phi_deg, final_len]
        
        # --- E. 验证误差 ---
        # 使用正向运动学验证一下最终位置
        _, _, _, T_tip_real = self.model.forward_kinematics(final_q)
        tip_real = T_tip_real[:3, 3]
        error = np.linalg.norm(target - tip_real)
        
        return final_q, error