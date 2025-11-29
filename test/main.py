import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D
import random

# 导入模块
from robot_model import HydraulicSoftArmKinematics
from ik_solver import IKSolver

class RobotGraspDemo:
    def __init__(self):
        self.arm = HydraulicSoftArmKinematics()
        self.ik = IKSolver()
        
        # 模式：INTERACT(交互), AUTO_TEST(自动测试), DANCE(跳舞)
        self.modes = ['INTERACT', 'AUTO_TEST', 'DANCE']
        self.mode_idx = 0
        
        # 状态数据
        self.current_q = [0, 0, -90, 0, 0, 0, 200] 
        self.target_pos = np.array([700.0, 0.0, 300.0])
        self.smooth_target = np.array(self.target_pos) # 用于平滑动画
        
        # 自动测试用的计时器
        self.auto_timer = 0
        
        # 初始化绘图
        self.fig = plt.figure(figsize=(15, 9))
        self.ax3d = plt.axes([0.0, 0.0, 1.0, 1.0], projection='3d')
        self.setup_scene()
        self.setup_ui()
        
        # 绑定键盘
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # 启动
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=30, blit=False)
        print("Demo启动成功！")
        plt.show()

    def setup_scene(self):
        limit = 1200
        self.ax3d.set_xlim(-limit, limit)
        self.ax3d.set_ylim(-limit, limit)
        self.ax3d.set_zlim(0, 1800)
        self.ax3d.set_xlabel('X'); self.ax3d.set_ylabel('Y'); self.ax3d.set_zlabel('Z')
        self.ax3d.set_title("Hydraulic Soft Arm - Grasping Demo", fontsize=16)
        
        # 绘制环境
        self.ax3d.plot([0,0], [0,0], [0, 500], 'k-', lw=6, alpha=0.3) # 支架
        self.ax3d.scatter([0], [0], [0], s=400, c='black', marker='s') # 底座
        
        # 绘制机械臂 (占位符)
        self.viz_links, = self.ax3d.plot([], [], [], '-', lw=8, c='#4682B4', alpha=0.9, label='Rigid')
        self.viz_fixed, = self.ax3d.plot([], [], [], '-', lw=8, c='#505050', alpha=0.9, label='Extension')
        self.viz_joints, = self.ax3d.plot([], [], [], 'o', ms=12, mfc='white', mec='black')
        self.viz_soft, = self.ax3d.plot([], [], [], '-', lw=10, c='#FF8C00', alpha=0.8, solid_capstyle='round', label='Soft')
        
        # 绘制目标球
        self.viz_target, = self.ax3d.plot([], [], [], 'o', ms=18, c='#32CD32', label='Target')
        self.viz_shadow, = self.ax3d.plot([], [], [], 'o', ms=10, c='gray', alpha=0.3) # 投影
        
        self.ax3d.legend(loc='upper right')
        
        # 文本信息
        self.txt_mode = self.ax3d.text2D(0.02, 0.95, "", transform=self.ax3d.transAxes, fontsize=16, weight='bold')
        self.txt_info = self.ax3d.text2D(0.02, 0.90, "", transform=self.ax3d.transAxes, fontsize=12, color='blue')

    def setup_ui(self):
        # 简单的切换按钮
        ax_btn = plt.axes([0.8, 0.05, 0.15, 0.05])
        self.btn = Button(ax_btn, 'Switch Mode', color='lightblue', hovercolor='0.9')
        self.btn.on_clicked(self.toggle_mode)

    def toggle_mode(self, event):
        self.mode_idx = (self.mode_idx + 1) % len(self.modes)
        self.auto_timer = 0 # 重置计时器

    def on_key(self, event):
        # 仅在交互模式下允许键盘控制
        if self.modes[self.mode_idx] != 'INTERACT': return
        
        step = 30.0
        k = event.key.lower()
        if k == 'w': self.target_pos[0] += step
        elif k == 's': self.target_pos[0] -= step
        elif k == 'a': self.target_pos[1] -= step
        elif k == 'd': self.target_pos[1] += step
        elif k == 'q': self.target_pos[2] += step
        elif k == 'e': self.target_pos[2] -= step
        self.target_pos[2] = max(0, self.target_pos[2])

    def generate_random_target(self):
        # 生成一个机械臂大概率能抓到的随机位置
        # X: 500~900, Y: -400~400, Z: 100~800
        x = random.uniform(500, 900)
        y = random.uniform(-400, 400)
        z = random.uniform(100, 800)
        return np.array([x, y, z])

    def update(self, frame):
        mode = self.modes[self.mode_idx]
        
        # === 模式逻辑 ===
        if mode == 'INTERACT':
            # 键盘控制目标
            self.txt_mode.set_text("Mode: INTERACT (Keyboard)")
            self.txt_mode.set_color('green')
            self.txt_info.set_text("Use W/A/S/D/Q/E to move ball")
            
            # 目标平滑移动
            self.smooth_target += (self.target_pos - self.smooth_target) * 0.15
            self.current_q = self.ik.solve(self.smooth_target, self.current_q)
            
        elif mode == 'AUTO_TEST':
            # 自动随机抓取
            self.txt_mode.set_text("Mode: AUTO TEST (Random Grasp)")
            self.txt_mode.set_color('red')
            self.txt_info.set_text("Generating random targets...")
            
            self.auto_timer += 1
            if self.auto_timer > 100: # 每 100 帧换一个位置
                self.auto_timer = 0
                self.target_pos = self.generate_random_target()
                
            self.smooth_target += (self.target_pos - self.smooth_target) * 0.1
            self.current_q = self.ik.solve(self.smooth_target, self.current_q)
            
        elif mode == 'DANCE':
            # 简单的展示动作
            self.txt_mode.set_text("Mode: DANCE")
            self.txt_mode.set_color('purple')
            self.txt_info.set_text("Rigid & Soft arm coordination")
            
            t = frame * 0.05
            q1 = np.sin(t*0.5) * 45
            q2 = np.sin(t*0.5 + 1) * 30 + 10
            q3 = np.sin(t*0.6 + 2) * 40 - 90
            bend = (np.sin(t) + 1) * 60
            phi = (t * 50) % 360 - 180
            length = 180 + np.sin(t*2) * 40
            self.current_q = [q1, q2, q3, 0, bend, phi, length]
            
            # 在跳舞模式下，让球跟着机械臂跑(为了好玩)
            r, s, _, _ = self.arm.forward_kinematics(self.current_q)
            self.viz_target.set_data([s[-1,0]], [s[-1,1]])
            self.viz_target.set_3d_properties([s[-1,2]])

        # === 绘图更新 ===
        # 计算所有点
        r_pts, s_pts, _, _ = self.arm.forward_kinematics(self.current_q)
        
        # 1. 刚性臂
        self.viz_links.set_data(r_pts[0:4, 0], r_pts[0:4, 1])
        self.viz_links.set_3d_properties(r_pts[0:4, 2])
        
        # 2. 固定延长段
        self.viz_fixed.set_data(r_pts[3:5, 0], r_pts[3:5, 1])
        self.viz_fixed.set_3d_properties(r_pts[3:5, 2])
        
        # 3. 关节球
        self.viz_joints.set_data(r_pts[1:4, 0], r_pts[1:4, 1])
        self.viz_joints.set_3d_properties(r_pts[1:4, 2])
        
        # 4. 软体臂
        sx = np.concatenate(([r_pts[-1,0]], s_pts[:,0]))
        sy = np.concatenate(([r_pts[-1,1]], s_pts[:,1]))
        sz = np.concatenate(([r_pts[-1,2]], s_pts[:,2]))
        self.viz_soft.set_data(sx, sy)
        self.viz_soft.set_3d_properties(sz)
        
        # 5. 目标球 (非跳舞模式下显示真实目标)
        if mode != 'DANCE':
            self.viz_target.set_data([self.smooth_target[0]], [self.smooth_target[1]])
            self.viz_target.set_3d_properties([self.smooth_target[2]])
            
            # 地面投影 (增加立体感)
            self.viz_shadow.set_data([self.smooth_target[0]], [self.smooth_target[1]])
            self.viz_shadow.set_3d_properties([0])
        
        return self.viz_links, self.viz_soft, self.viz_target

if __name__ == "__main__":
    app = RobotGraspDemo()