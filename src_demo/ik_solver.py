import numpy as np
from robot_model import HydraulicSoftArmKinematics

class IKSolver:
    def __init__(self):
        self.model = HydraulicSoftArmKinematics()
        
        # 1. 关节限位 (Rad)
        self.limits_rad = [
            np.deg2rad([-60, 60]),   # J1
            np.deg2rad([-28, 90]),   # J2
            np.deg2rad([-152, -42])  # J3
        ]
        
        # 2. 软体臂参数限制
        self.soft_len_range = [140.0, 200.0] # 长度 mm
        self.soft_bend_max = 100.0           # 弯曲 deg
        
        # 3. 求解配置
        self.tolerance = 2.0  # 目标精度 mm

    def solve(self, target_pos, current_q_deg):
        """
        输入: 全局目标 [x,y,z], 当前角度(7维列表)
        输出: 目标角度(7维列表) 或 None
        """
        target = np.array(target_pos)
        
        # 种子策略：尝试当前位置 + 2个随机位置
        seeds = [current_q_deg[:3]]
        for _ in range(2):
            random_q = [np.random.uniform(l[0], l[1]) for l in self.limits_rad]
            seeds.append(np.degrees(random_q))

        best_q_full = None
        min_error = float('inf')

        for start_q in seeds:
            # --- 第一步：刚性臂粗定位 ---
            # 目标：让软体臂基座到达 target 附近的“甜区”
            rigid_q_res, _ = self._optimize_rigid_base(target, start_q)
            
            # --- 第二步：软体臂几何精解 ---
            # 目标：解析法计算 Bend/Phi/Len
            q_geometric, geo_error = self._solve_soft_geometric(target, rigid_q_res, current_q_deg)
            
            final_q = q_geometric
            final_error = geo_error
            
            # --- 第三步：全局微调 (新增) ---
            # 如果几何解算误差在 "有点大但不是特别大" (2mm ~ 20mm) 之间
            # 说明可能卡在限位边界，尝试全关节数值松弛
            if 2.0 < geo_error < 50.0:
                q_refined, refined_error = self._global_refine(target, q_geometric)
                if refined_error < final_error:
                    final_q = q_refined
                    final_error = refined_error
            
            # 检查是否满足要求
            if final_error < self.tolerance:
                return final_q.tolist()
            
            # 记录当前最优
            if final_error < min_error:
                min_error = final_error
                best_q_full = final_q

        # 如果所有尝试都未达到 tolerance，返回误差最小的解
        return best_q_full.tolist() if best_q_full is not None else None

    def _optimize_rigid_base(self, target, start_q_deg):
        """
        让刚性臂将软体基座送到距离目标 ideal_reach (约195mm) 的位置
        """
        q = np.array([np.deg2rad(v) for v in start_q_deg])
        ideal_reach = (self.soft_len_range[0] + self.soft_len_range[1]) / 2.0
        
        for _ in range(15):
            # 构造完整关节向量进行 FK (软体部分设为0)
            q_full = np.degrees(q).tolist() + [0, 0, 0, 0]
            r_pts, _, _, _ = self.model.forward_kinematics(q_full)
            base_pos = r_pts[-1]
            
            # 计算当前距离向量
            vec = target - base_pos
            dist = np.linalg.norm(vec)
            
            # 距离误差 (标量)
            dist_err = dist - ideal_reach
            if abs(dist_err) < 5.0:
                break
            
            # 期望位置：在连线上，距离目标 ideal_reach 处
            direction = vec / (dist + 1e-6)
            desired_pos = target - direction * ideal_reach
            
            # 笛卡尔误差向量
            err_vec = desired_pos - base_pos
            
            # 数值雅可比
            J = np.zeros((3, 3))
            delta = 0.001
            for i in range(3):
                q_temp = q.copy()
                q_temp[i] += delta
                q_calc = np.degrees(q_temp).tolist() + [0, 0, 0, 0]
                pts_new, _, _, _ = self.model.forward_kinematics(q_calc)
                J[:, i] = (pts_new[-1] - base_pos) / delta
            
            # DLS 更新
            dq = np.linalg.pinv(J.T @ J + 0.1 * np.eye(3)) @ J.T @ err_vec
            q += dq
            
            # 限位
            for i in range(3):
                q[i] = np.clip(q[i], self.limits_rad[i][0], self.limits_rad[i][1])
                
        return np.degrees(q), abs(dist - ideal_reach)

    def _solve_soft_geometric(self, target, rigid_q_deg, old_q_full):
        """
        几何解析软体参数
        old_q_full: 用于在奇异点(直线)时保持 Phi 不变
        """
        # FK 获取软体基座坐标系
        q_dummy = rigid_q_deg.tolist() + [0, 0, 0, 0]
        _, _, T_base, _ = self.model.forward_kinematics(q_dummy)
        
        base_pos = T_base[:3, 3]
        R_base = T_base[:3, :3]
        
        # 转到局部坐标系
        vec_local = R_base.T @ (target - base_pos)
        x, y, z = vec_local # x: 前进方向, y,z: 弯曲平面
        
        # --- 1. 计算 Phi (旋转角) ---
        h = np.sqrt(y**2 + z**2)
        
        # 【优化】奇异点保护
        if h < 1e-3: 
            # 几乎是直线，Phi 失去定义。保持上一时刻的 Phi 或设为 0
            phi_deg = old_q_full[5] 
        else:
            phi_deg = np.degrees(np.arctan2(z, y))
            
        # --- 2. 计算 Bend (弯曲角) & Length (弧长) ---
        if h < 1e-3:
            theta_rad = 0.0
            arc_len = x
        else:
            # PCC 几何公式
            # theta = 2 * atan(h / x)
            theta_rad = 2 * np.arctan2(h, x)
            if abs(theta_rad) < 1e-4:
                arc_len = x
            else:
                R = (x**2 + h**2) / (2*h)
                arc_len = R * theta_rad
                
        # --- 3. 约束限位 ---
        final_bend = np.clip(np.degrees(theta_rad), -self.soft_bend_max, self.soft_bend_max)
        final_len = np.clip(arc_len, self.soft_len_range[0], self.soft_len_range[1])
        
        # 组合结果
        res_q = np.array(rigid_q_deg.tolist() + [0, final_bend, phi_deg, final_len])
        
        # 验证误差
        _, _, _, T_tip = self.model.forward_kinematics(res_q)
        error = np.linalg.norm(target - T_tip[:3, 3])
        
        return res_q, error

    def _global_refine(self, target, q_start):
        """
        【全局微调】同时调整刚性关节和软体参数
        解决 "几何解算被限位截断导致差一点点" 的问题
        """
        q = np.array(q_start, dtype=float)
        current_err = float('inf')
        
        # 参与优化的关节索引: 0,1,2 (Rigid), 4(Bend), 5(Phi), 6(Len)
        # Index 3 是固定的，不优化
        active_indices = [0, 1, 2, 4, 5, 6]
        
        for _ in range(5): # 微调次数不宜多，保证速度
            # FK
            _, _, _, T_tip = self.model.forward_kinematics(q)
            curr_pos = T_tip[:3, 3]
            err_vec = target - curr_pos
            current_err = np.linalg.norm(err_vec)
            
            if current_err < self.tolerance:
                break
                
            # 计算 3x6 雅可比
            J = np.zeros((3, 6))
            delta_arr = [0.1, 0.1, 0.1, 0, 0.1, 0.1, 1.0] # 步长: 角度用0.1度, 长度用1mm
            
            for k, idx in enumerate(active_indices):
                q_temp = q.copy()
                delta = delta_arr[idx]
                q_temp[idx] += delta
                
                _, _, _, T_test = self.model.forward_kinematics(q_temp)
                J[:, k] = (T_test[:3, 3] - curr_pos) / delta
            
            # DLS 求解
            # 这里的 lambda 需要大一点，保证稳定性
            dq_active = np.linalg.pinv(J.T @ J + 0.5 * np.eye(6)) @ J.T @ err_vec
            
            # 更新
            for k, idx in enumerate(active_indices):
                q[idx] += dq_active[k]
                
            # 限位保护
            # 刚性
            for i in range(3):
                q[i] = np.clip(q[i], np.degrees(self.limits_rad[i][0]), np.degrees(self.limits_rad[i][1]))
            # 软体
            q[4] = np.clip(q[4], -self.soft_bend_max, self.soft_bend_max)
            q[6] = np.clip(q[6], self.soft_len_range[0], self.soft_len_range[1])
            
        return q, current_err