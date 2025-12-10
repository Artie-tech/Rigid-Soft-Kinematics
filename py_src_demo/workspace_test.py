"""
文件功能：工作空间分析与可视化

本文件用于计算和可视化机械臂的工作空间（Workspace）。
工作空间是指机器人末端执行器能够到达的所有点的集合。

主要功能：
- 通过在关节空间内进行大量随机采样，生成大量的关节角度组合。
- 对每个关节角度组合，调用正向运动学 (`robot_model.py`) 计算出末端执行器的位置。
- 将所有计算出的末端位置点绘制在三维空间中，从而直观地展示出机械臂的可达范围。

此脚本对于评估机械臂的设计（如连杆长度、关节限位）是否满足任务需求非常有用。
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import time

# 导入你的机械臂模型
try:
    from robot_model import HydraulicSoftArmKinematics
except ImportError:
    print("错误: 找不到 robot_model.py，请确保它在当前目录下。")
    exit()

def run_workspace_analysis():
    # 1. 初始化模型
    arm = HydraulicSoftArmKinematics()
    
    # 2. 设定采样数量 (越多越精确，但计算越慢)
    N_SAMPLES = 5000
    print(f"正在生成工作空间点云 (采样数: {N_SAMPLES})...")
    start_time = time.time()
    
    # 3. 定义关节限位 (Min, Max)
    # q1, q2, q3 (刚性)
    limits_rigid = [
        (-60, 60),    # q1 Base
        (-28, 90),    # q2 Shoulder
        (-152, -42)   # q3 Elbow
    ]
    # bend, phi, len (软体)
    limits_soft = [
        (0, 120),     # Bend
        (-180, 180),  # Phi
        (140, 250)    # Length
    ]
    
    points = []
    
    # 4. 蒙特卡洛采样循环
    for _ in range(N_SAMPLES):
        # 随机生成刚性角度
        q_rigid = [random.uniform(l[0], l[1]) for l in limits_rigid]
        
        # 随机生成软体参数
        q_soft = [random.uniform(l[0], l[1]) for l in limits_soft]
        
        # 组合成 7 维向量: [q1, q2, q3, q4(Fixed=0), bend, phi, len]
        q_full = q_rigid + [0] + q_soft
        
        # 正向运动学计算
        # forward_kinematics 返回: (rigid_pts, soft_pts, T_base, T_tip)
        _, _, _, T_tip = arm.forward_kinematics(q_full)
        
        # 提取末端坐标 (x, y, z)
        points.append(T_tip[:3, 3])
        
    points = np.array(points)
    duration = time.time() - start_time
    print(f"计算完成！耗时: {duration:.2f}秒")
    
    # 5. 3D 可视化
    plot_workspace(points, arm.base_z_offset)

def plot_workspace(points, base_height):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置坐标轴范围
    limit = 1200
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(0, 1800)
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'Hydraulic Arm Workspace ({len(points)} samples)', fontsize=14)
    
    # 绘制基座参考
    ax.plot([0,0], [0,0], [0, base_height], 'k-', lw=5, alpha=0.5, label='Base Stand')
    ax.scatter([0], [0], [0], s=200, c='black', marker='s')
    
    # 绘制点云
    # s=2: 点的大小
    # alpha=0.3: 透明度，这样重叠的地方颜色会深，显示出密度
    # cmap='viridis': 根据 Z 轴高度着色，好看一点
    scatter = ax.scatter(
        points[:,0], 
        points[:,1], 
        points[:,2], 
        s=2, 
        c=points[:,2], 
        cmap='viridis',
        alpha=0.3, 
        label='Reachable Points'
    )
    
    # 添加颜色条
    plt.colorbar(scatter, ax=ax, label='Height Z (mm)', shrink=0.7)
    
    # 添加图例
    ax.legend(loc='upper right')
    
    print("窗口已打开。你可以拖动鼠标旋转视角。")
    plt.show()

if __name__ == "__main__":
    run_workspace_analysis()