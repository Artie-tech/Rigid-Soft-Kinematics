"""
文件功能：逆运动学测试脚本

本程序用于测试和验证刚柔耦合机械臂的逆运动学（IK）算法。
主要流程：
1. 随机生成一个目标点（末端期望位置）。
2. 以当前关节状态为初始猜测，调用 IKSolver 进行逆运动学求解，得到一组关节角度。
3. 输出求解结果，并用正向运动学（FK）验证末端位置的精度。
4. 可用于调试 IK 算法的收敛性、精度和鲁棒性。
适合单步运行和批量测试。
"""

import numpy as np
from robot_model import HydraulicSoftArmKinematics
from ik_solver import IKSolver

def test_ik():
    # 1. 初始化求解器
    solver = IKSolver()
    
    # 2. 定义一个测试目标点 [x, y, z] (单位: mm)
    # 建议先找一个大概率能到的点，比如正前方上方
    target_pos = [np.random.uniform(500, 700), np.random.uniform(50, 150), np.random.uniform(50, 150)] 
    
    # 3. 定义当前状态 (作为初始猜测值)
    # [q1, q2, q3, fixed, bend, phi, len]
    current_q = [0, 0, -90, 0, 0, 0, 200]
    
    print(f"目标坐标: {target_pos}")
    print("-" * 30)

    # 4. 运行逆运动学求解
    result_q = solver.solve(target_pos, current_q)

    if result_q is None:
        print("求解失败！未能找到解。")
        return

    print("计算出的关节角度:")
    print(f"  刚性关节 (J1-J3): {np.round(result_q[:3], 2)}")
    print(f"  软体参数 (Bend, Phi, Len): {np.round(result_q[4:], 2)}")
    
    # ==========================================
    # 核心：验证部分 (你问的那一段)
    # ==========================================
    print("-" * 30)
    print("正在验证精度...")
    
    # A. 创建一个模型对象用于正向计算
    model = HydraulicSoftArmKinematics()
    
    # B. 将求解算出的 result_q 喂给正向运动学
    # forward_kinematics 会返回 [刚性点, 软体点, 软体基座矩阵, 末端矩阵]
    _, _, _, T_tip_real = model.forward_kinematics(result_q)
    
    # C. 提取实际末端坐标 (矩阵的前3行第4列)
    actual_pos = T_tip_real[:3, 3]
    
    # D. 计算误差 (目标点 - 实际点 的欧几里得距离)
    error_vec = np.array(target_pos) - actual_pos
    error_dist = np.linalg.norm(error_vec)

    print(f"目标位置: {target_pos}")
    print(f"实际到达: {np.round(actual_pos, 2)}")
    print(f"位置误差: {error_dist:.4f} mm")
    
    if error_dist < 5.0:
        print(">> 测试通过：精度符合要求 (SUCCESS)")
    else:
        print(">> 测试警告：误差较大，可能目标点不可达或陷入局部极值 (FAIL)")

if __name__ == "__main__":
    test_ik()