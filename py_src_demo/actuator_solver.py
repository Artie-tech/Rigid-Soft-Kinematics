import numpy as np

def solve_actuator_velocity(l_dot, phi_dot, kappa_dot, l, phi, kappa, d):
    """
    软体臂驱动空间微分逆运动学映射
    将配置空间速度 (l_dot, phi_dot, kappa_dot) 映射到 驱动线缆速度 (l1_dot, l2_dot, l3_dot)
    
    参数:
        l_dot, phi_dot, kappa_dot : 配置参数变化率 (m/s, rad/s, 1/m*s)
        l, phi, kappa             : 当前配置参数 (m, rad, 1/m)
        d                         : 线缆分布半径 (节圆半径, m)
                                    注意：d 必须与 l 单位一致(mm或m)
    
    返回:
        l1_dot, l2_dot, l3_dot    : 三根驱动线缆的速度
    """
    
    # 初始化雅可比矩阵
    J2 = np.zeros((3, 3))
    
    # 计算线缆的相位角 (根据 Matlab 代码逻辑: 90, 210, 330 度分布)
    # 注意：输入 phi 必须是弧度制
    phi1 = np.pi/2 - phi
    phi2 = 7*np.pi/6 - phi
    phi3 = 11*np.pi/6 - phi
    
    # === 构建雅可比矩阵 ===
    # J = d[l1, l2, l3]^T / d[l, phi, kappa]
    
    # 第一行 (Cable 1)
    J2[0, 0] = 1 - kappa * d * np.cos(phi1)       # ∂l1/∂l
    J2[0, 1] = -l * kappa * d * np.sin(phi1)      # ∂l1/∂phi
    J2[0, 2] = -l * d * np.cos(phi1)              # ∂l1/∂kappa
    
    # 第二行 (Cable 2)
    J2[1, 0] = 1 - kappa * d * np.cos(phi2)
    J2[1, 1] = -l * kappa * d * np.sin(phi2)
    J2[1, 2] = -l * d * np.cos(phi2)
    
    # 第三行 (Cable 3)
    J2[2, 0] = 1 - kappa * d * np.cos(phi3)
    J2[2, 1] = -l * kappa * d * np.sin(phi3)
    J2[2, 2] = -l * d * np.cos(phi3)
    
    # === 映射速度 ===
    input_vec = np.array([l_dot, phi_dot, kappa_dot])
    result_vec = J2 @ input_vec  # 矩阵乘法
    
    # 提取结果 (并取实部，防止数值计算产生微小虚部)
    l1_dot = np.real(result_vec[0])
    l2_dot = np.real(result_vec[1])
    l3_dot = np.real(result_vec[2])
    
    return l1_dot, l2_dot, l3_dot

# ==========================================
# 测试代码
# ==========================================
if __name__ == "__main__":
    # 假设参数 (单位需统一，这里假设是 mm)
    d = 12.5       # 线缆分布半径
    L = 200.0      # 当前长度
    Phi = np.radians(30) # 当前弯曲方向 30度
    Kappa = 0.01   # 当前曲率 (对应弯曲半径 100mm)
    
    # 假设配置空间速度 (希望怎么动)
    v_L = 10.0     # 伸长速度 10mm/s
    v_Phi = 0.5    # 旋转速度 0.5 rad/s
    v_Kappa = 0.1  # 曲率变化速度 0.1 1/m*s
    
    v1, v2, v3 = solve_actuator_velocity(v_L, v_Phi, v_Kappa, L, Phi, Kappa, d)
    
    print(f"Cable 1 Velocity: {v1:.2f} mm/s")
    print(f"Cable 2 Velocity: {v2:.2f} mm/s")
    print(f"Cable 3 Velocity: {v3:.2f} mm/s")