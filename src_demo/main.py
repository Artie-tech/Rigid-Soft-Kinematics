"""
文件功能：主程序入口与可视化界面

本文件是整个仿真项目的启动入口。它负责：
1.  创建并集成各个模块，如机器人模型 (`robot_model`)、逆运动学求解器 (`ik_solver`) 和任务规划器 (`task_planner`)。
2.  使用 `matplotlib` 构建一个交互式的 3D 可视化界面，用于实时显示机械臂的运动。
3.  实现用户交互逻辑，包括：
    - 通过滑块手动控制每个关节。
    - 切换不同的操作模式（如手动、自动轨迹、IK 抓取）。
    - 响应键盘事件，用于在抓取模式下控制目标点的位置。
4.  管理主更新循环 (`update_loop`)，根据当前模式驱动机械臂的运动和仿真。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D

# 导入你的自定义模块 (确保这两个文件在同目录下)
from robot_model import HydraulicSoftArmKinematics
from ik_solver import IKSolver

class UnifiedRobotController:
    def __init__(self):
        # 1. 初始化核心模块
        self.arm_model = HydraulicSoftArmKinematics()
        self.ik_solver = IKSolver()
        
        # 2. 状态定义
        self.MODES = ['MANUAL', 'AUTO', 'GRASP']
        self.current_mode = 'MANUAL' # 当前模式名称
        self.time_step = 0.0
        
        # 3. 数据存储
        # [q1, q2, q3, q4(固定), bend, phi, len]
        self.current_q = [0, 0, -90, 0, 0, 0, 180] 
        self.target_pos = np.array([600.0, 0.0, 400.0]) 
        
        # 4. 初始化绘图
        self.fig = plt.figure(figsize=(15, 9))
        self.ax3d = plt.axes([0.35, 0.05, 0.60, 0.90], projection='3d')
        
        self.setup_visuals()
        self.setup_ui()
        
        # 5. 绑定键盘事件
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # 6. 启动循环
        self.ani = animation.FuncAnimation(self.fig, self.update_loop, interval=30, blit=False)
        
        # 初始高亮按钮
        self.update_button_colors()
        
        print("=== 系统启动 ===")
        print("点击左上角按钮切换模式")
        plt.show()

    def setup_visuals(self):
        xlimit_max = 1000
        xlimit_min = -400

        ylimit_max = 700
        ylimit_min = -400
        
        zlimit_max = 1200
        zlimit_min = 0

        self.ax3d.set_xlim(xlimit_min, xlimit_max)
        self.ax3d.set_ylim(ylimit_min, ylimit_max)
        self.ax3d.set_zlim(zlimit_min, zlimit_max)
        self.ax3d.set_xlabel('X')
        self.ax3d.set_ylabel('Y')
        self.ax3d.set_zlabel('Z')
        
        # 静态环境
        self.ax3d.plot([0,0], [0,0], [0, self.arm_model.base_z_offset], 'k-', lw=5, alpha=0.3)
        self.ax3d.scatter([0], [0], [0], s=300, c='black', marker='s')
        
        # 动态组件
        self.viz_links, = self.ax3d.plot([], [], [], '-', lw=8, c='#4682B4', alpha=0.9, label='Rigid Arm')
        self.viz_fixed, = self.ax3d.plot([], [], [], '-', lw=8, c='#4682B4', alpha=0.9, label='Fixed Ext')
        self.viz_joints, = self.ax3d.plot([], [], [], 'o', ms=10, mfc='white', mec='black')
        self.viz_soft,   = self.ax3d.plot([], [], [], '-', lw=8, c='#FF8C00', alpha=0.9, solid_capstyle='round', label='Soft Arm')
        
        # 目标小球
        self.viz_target, = self.ax3d.plot([], [], [], 'o', ms=6, c='gray', alpha=0.5, label='Target')
        
        # 地面
        xx, yy = np.meshgrid(range(xlimit_min-50, xlimit_max+50, 600), range(ylimit_min, ylimit_max+100, 600))
        zz = np.zeros_like(xx)
        self.ax3d.plot_surface(xx, yy, zz, color='gray', alpha=0.1)
        
        # 标题文本
        self.title_text = self.ax3d.text2D(0.05, 0.95, "Mode: MANUAL", transform=self.ax3d.transAxes, fontsize=16, weight='bold', color='blue')

        # 末端坐标轴
        self.viz_tip_axes = []
        colors = ['r', 'g', 'b']
        for c in colors:
            line, = self.ax3d.plot([], [], [], '-', lw=2, c=c)
            self.viz_tip_axes.append(line)

    def setup_ui(self):
        axcolor = 'lightgoldenrodyellow'
        
        # === 【修改点】创建三个独立的按钮 ===
        # 位置格式: [left, bottom, width, height]
        
        # 1. MANUAL 按钮
        ax_btn1 = plt.axes([0.05, 0.92, 0.08, 0.05])
        self.btn_manual = Button(ax_btn1, 'Manual', color='white', hovercolor='0.9')
        self.btn_manual.on_clicked(self.set_mode_manual)
        
        # 2. AUTO 按钮
        ax_btn2 = plt.axes([0.14, 0.92, 0.08, 0.05])
        self.btn_auto = Button(ax_btn2, 'Auto', color='white', hovercolor='0.9')
        self.btn_auto.on_clicked(self.set_mode_auto)
        
        # 3. GRASP 按钮
        ax_btn3 = plt.axes([0.23, 0.92, 0.08, 0.05])
        self.btn_grasp = Button(ax_btn3, 'Grasp', color='white', hovercolor='0.9')
        self.btn_grasp.on_clicked(self.set_mode_grasp)
        
        # 滑块配置
        self.sliders = []
        slider_params = [
            ('J1 Base', -60, 60, 0),
            ('J2 Shoulder', -28, 90, 0),
            ('J3 Elbow', -152, -42, -90),
            # J4 Fixed
            ('Soft Bend', -120, 120, 0),
            ('Soft Phi', -180, 180, 0),
            ('Soft Len', 140, 250, 180)
        ]
        
        start_y = 0.85
        for i, (label, vmin, vmax, vinit) in enumerate(slider_params):
            ax = plt.axes([0.05, start_y - i*0.06, 0.20, 0.03], facecolor=axcolor)
            s = Slider(ax, label, vmin, vmax, valinit=vinit)
            s.on_changed(self.on_slider_manual)
            self.sliders.append(s)
            
        plt.figtext(0.05, 0.35, "Controls:\n[Grasp Mode]: Use ↑ ↓ ← → + - to move ball", fontsize=10)

    # ================= 模式切换逻辑 =================
    
    def set_mode_manual(self, event):
        self.change_mode('MANUAL')

    def set_mode_auto(self, event):
        self.change_mode('AUTO')

    def set_mode_grasp(self, event):
        self.change_mode('GRASP')

    def change_mode(self, new_mode):
        self.current_mode = new_mode
        self.update_button_colors() # 更新按钮颜色
        
        # 更新标题颜色和文字
        self.title_text.set_text(f"Mode: {new_mode}")
        if new_mode == 'MANUAL':
            self.title_text.set_color('blue')
            self.viz_target.set_color('gray')
            self.viz_target.set_alpha(0.3)
        elif new_mode == 'AUTO':
            self.title_text.set_color('purple')
            self.viz_target.set_color('gray')
            self.viz_target.set_alpha(0.3)
        elif new_mode == 'GRASP':
            self.title_text.set_color('green')
            self.viz_target.set_color('#32CD32')
            self.viz_target.set_alpha(1.0)

    def update_button_colors(self):
        # 辅助函数：根据当前模式高亮对应按钮
        active_color = 'lightblue'
        inactive_color = 'white'
        
        self.btn_manual.color = active_color if self.current_mode == 'MANUAL' else inactive_color
        self.btn_auto.color = active_color if self.current_mode == 'AUTO' else inactive_color
        self.btn_grasp.color = active_color if self.current_mode == 'GRASP' else inactive_color
        
        # 强制重绘按钮
        self.btn_manual.hovercolor = self.btn_manual.color
        self.btn_auto.hovercolor = self.btn_auto.color
        self.btn_grasp.hovercolor = self.btn_grasp.color

    # ================= 其他逻辑 (保持不变) =================

    def on_key_press(self, event):
        if self.current_mode != 'GRASP': return
        step = 5.0
        key = (event.key or '').lower()
        # XY：方向键 或 小键盘 8/2/4/6
        if key in ('up', '8'):
            self.target_pos[0] += step
        elif key in ('down', '2'):
            self.target_pos[0] -= step
        elif key in ('left', '4'):
            self.target_pos[1] -= step
        elif key in ('right', '6'):
            self.target_pos[1] += step
        # Z：小键盘 + / - （兼容多种键名）或主键盘 + / -
        elif key in ('+', 'add', 'kp_add'):
            self.target_pos[2] += step
        elif key in ('-', 'subtract', 'kp_subtract'):
            self.target_pos[2] -= step
        # 保证高度不为负
        self.target_pos[2] = max(0, self.target_pos[2])

    def on_slider_manual(self, val):
        if self.current_mode == 'MANUAL':
            vals = [s.val for s in self.sliders]
            self.current_q = vals[:3] + [0] + vals[3:]

    def update_sliders_visual(self, q_vector):
        indices = [0, 1, 2, 4, 5, 6]
        for i, slider_idx in enumerate(indices):
            self.sliders[i].eventson = False
            self.sliders[i].set_val(q_vector[slider_idx])
            self.sliders[i].eventson = True

    def update_loop(self, frame):
        self.time_step += 0.05
        
        if self.current_mode == 'AUTO':
            t = self.time_step
            q1 = np.sin(t*0.1) * 45
            q2 = np.sin(t*0.1 + 1) * 30 + 10
            q3 = np.sin(t*0.1 + 2) * 40 - 90
            bend = (np.sin(0.1*t)) * 80
            bend = np.clip(bend, -120, 120)  # 限制弯曲度
            phi = (t * 50) % 360 - 180
            length = 180 + np.sin(t*2) * 40
            self.current_q = [q1, q2, q3, 0, bend, phi, length]
            self.update_sliders_visual(self.current_q)
            
        elif self.current_mode == 'GRASP':
            self.current_q = self.ik_solver.solve(self.target_pos, self.current_q)
            self.update_sliders_visual(self.current_q)
            
        # 绘图更新
        r_pts, s_pts, _, T_tip = self.arm_model.forward_kinematics(self.current_q)
        
        self.viz_links.set_data(r_pts[0:4, 0], r_pts[0:4, 1])
        self.viz_links.set_3d_properties(r_pts[0:4, 2])
        self.viz_fixed.set_data(r_pts[3:5, 0], r_pts[3:5, 1])
        self.viz_fixed.set_3d_properties(r_pts[3:5, 2])
        self.viz_joints.set_data(r_pts[1:4, 0], r_pts[1:4, 1])
        self.viz_joints.set_3d_properties(r_pts[1:4, 2])
        
        sx = np.concatenate(([r_pts[-1,0]], s_pts[:,0]))
        sy = np.concatenate(([r_pts[-1,1]], s_pts[:,1]))
        sz = np.concatenate(([r_pts[-1,2]], s_pts[:,2]))
        self.viz_soft.set_data(sx, sy)
        self.viz_soft.set_3d_properties(sz)
        
        # 更新末端坐标轴
        axis_len = 50 # 坐标轴长度
        origin = T_tip[:3, 3]
        for i in range(3):
            axis_vec = T_tip[:3, i]
            end_point = origin + axis_len * axis_vec
            line = self.viz_tip_axes[i]
            line.set_data([origin[0], end_point[0]], [origin[1], end_point[1]])
            line.set_3d_properties([origin[2], end_point[2]])

        if self.current_mode == 'GRASP':
            self.viz_target.set_data([self.target_pos[0]], [self.target_pos[1]])
            self.viz_target.set_3d_properties([self.target_pos[2]])
        
        return self.viz_links, self.viz_soft

if __name__ == "__main__":
    app = UnifiedRobotController()