"""
文件功能：通用测试脚本

本文件是一个通用的、临时的测试脚本。
其主要目的是为了快速验证、调试或演示项目中的某个特定功能、算法或模块，而不需要运行完整的主程序。

例如，您可以在这里：
- 单独测试 `robot_model.py` 中的正向运动学。
- 验证 `ik_solver.py` 在特定目标下的求解效果。
- 检查 `task_planner.py` 生成的轨迹点是否符合预期。
- 运行任何与项目相关的小实验。

这是一个“草稿纸”性质的文件，其内容可以根据当前的测试需求随时更改。
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def modified_dh_matrix(a, alpha, d, theta):
    """
    根据改进型D-H参数计算变换矩阵 T_{i-1}^{i}
    Rot_x(alpha) * Trans_x(a) * Rot_z(theta) * Trans_z(d)
    """
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    # 这里的矩阵乘法顺序对应改进型D-H:
    # T = Rx(alpha) * Dx(a) * Rz(theta) * Dz(d)
    return np.array([
        [ct,    -st,    0,      a],
        [st*ca, ct*ca, -sa,    -d*sa],
        [st*sa, ct*sa,  ca,     d*ca],
        [0,     0,      0,      1]
    ])

def plot_robot_arm(q_values):
    # D-H 参数表 [a_{i-1}, alpha_{i-1}, d_i, theta_i]
    # 注意：角度用弧度，长度用米 (190mm = 0.19m)
    dh_params = [
        # Link 1
        {'a': 0.190, 'alpha': 0,          'd': 0, 'theta': q_values[0]},
        # Link 2 (alpha = 90 deg)
        {'a': 0.090, 'alpha': np.pi/2,    'd': 0, 'theta': q_values[1]},
        # Link 3
        {'a': 0.605, 'alpha': 0,          'd': 0, 'theta': q_values[2]},
        # Link 4
        {'a': 0.290, 'alpha': 0,          'd': 0, 'theta': q_values[3]},
        # Link 5 (固定的末端, theta=0, alpha=-90 deg)
        {'a': 0.285, 'alpha': -np.pi/2,   'd': 0, 'theta': 0},
    ]

    # 初始化变换矩阵为单位矩阵
    T_cum = np.eye(4)
    
    # 保存关节坐标用于绘图
    positions = [T_cum[:3, 3]]
    
    # 坐标系方向用于绘制坐标轴 (x:红, y:绿, z:蓝)
    axes_dirs = []

    print("关节坐标 (x, y, z):")
    print(f"Base: {positions[0]}")

    for i, p in enumerate(dh_params):
        # 计算当前关节的变换矩阵
        T_i = modified_dh_matrix(p['a'], p['alpha'], p['d'], p['theta'])
        
        # 累乘得到相对于基座的变换
        T_cum = T_cum @ T_i
        
        # 提取位置
        pos = T_cum[:3, 3]
        positions.append(pos)
        
        # 提取旋转矩阵用于画坐标轴
        axes_dirs.append((T_cum[:3, 0], T_cum[:3, 1], T_cum[:3, 2], pos))
        
        print(f"Joint {i+1}: {np.round(pos, 3)}")

    # --- 绘图部分 ---
    positions = np.array(positions)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 1. 画连杆 (骨架)
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
            '-o', linewidth=3, color='black', label='Link')

    # 2. 画每个关节的坐标系
    scale = 0.15 # 坐标轴长度
    for x_dir, y_dir, z_dir, origin in axes_dirs:
        ax.quiver(origin[0], origin[1], origin[2], x_dir[0], x_dir[1], x_dir[2], color='r', length=scale)
        ax.quiver(origin[0], origin[1], origin[2], y_dir[0], y_dir[1], y_dir[2], color='g', length=scale)
        ax.quiver(origin[0], origin[1], origin[2], z_dir[0], z_dir[1], z_dir[2], color='b', length=scale)

    # 设置图形属性
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Hydraulic Manipulator (Modified D-H)')
    
    # 设置显示范围，保证比例一致
    limit = 1.5
    ax.set_xlim([-limit/2, limit])
    ax.set_ylim([-limit/2, limit/2])
    ax.set_zlim([0, limit])
    
    # 强制比例一致
    ax.set_box_aspect([1,1,1]) 
    
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # 输入关节角度 q1, q2, q3, q4 (弧度)
    # 这里设置一个随意的姿态来展示机械臂结构
    q_input = [0, np.pi/6, -np.pi/4, np.pi/6] 
    
    plot_robot_arm(q_input)