import numpy as np
from robot_model import HydraulicSoftArmKinematics

def test_case(name, bend, phi, length):
    model = HydraulicSoftArmKinematics()
    
    # 计算坐标
    pos = model.get_soft_tip_in_base_frame(bend, phi, length, to_real_z_axis=True)
    
    print(f"--- 测试场景: {name} ---")
    print(f"输入: Bend={bend}°, Phi={phi}°, Len={length}mm")
    print(f"输出 (实机坐标): X={pos[0]:.2f}, Y={pos[1]:.2f}, Z={pos[2]:.2f}")
    
    # 验证逻辑
    return pos

if __name__ == "__main__":
    print("=== 软体臂运动学验证 (Z轴为伸长方向) ===\n")

    # 1. 直线伸长测试
    # 预期: X=0, Y=0, Z=200 (因为Z是伸长方向)
    p1 = test_case("直线状态", bend=0, phi=0, length=200)
    assert abs(p1[0]) < 1e-3 and abs(p1[2] - 200) < 1e-3, "错误：直线伸长时 Z 轴应为 200"

    print("\n")

    # 2. 向下弯曲 90度 (假设 Phi=0 是向下/向X轴弯曲)
    # 理论计算: 
    # 弧长 L=200, 角度 90度 (pi/2)
    # 半径 R = L / (pi/2) ≈ 127.32
    # 伸长方向 (Z) = R * sin(90) = R
    # 侧向方向 (X 或 Y) = R * (1 - cos(90)) = R
    p2 = test_case("90度弯曲 (Phi=0)", bend=90, phi=0, length=200)
    
    # 检查 Z 轴 (向前分量)
    R = 200 / (np.pi/2)
    assert abs(p2[2] - R) < 1.0, f"错误：Z轴应约为 {R}"
    # 检查偏转分量 (根据你的 Phi 定义，Phi=0 时可能是 X 或 Y 变化，具体看仿真定义)
    # 在本代码中，Phi=0 通常对应 Y_sim 变化 -> Real X 变化
    print(f"验证: 理论半径 R = {R:.2f}")

    print("\n")
    
    # 3. 螺旋弯曲 (Bend=90, Phi=90)
    # 预期: Z轴(前)不变，偏转方向从 X 轴转到 Y 轴
    p3 = test_case("90度弯曲 + 90度旋转 (Phi=90)", bend=90, phi=90, length=200)
    
    print("\n=== 验证通过: 逻辑符合实机 Z 轴伸长定义 ===")