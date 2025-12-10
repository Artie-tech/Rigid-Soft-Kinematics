"""
文件功能：主程序入口与可视化界面 (UI优化版)

更新日志：
1. 恢复了末端坐标轴 (RGB Axes) 的显示。
2. 增大了模式选择按钮的尺寸和字体，便于点击。
3. 包含了 TrajectoryPlanner 轨迹规划功能。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.mplot3d import Axes3D

# 导入你的自定义模块
from robot_model import HydraulicSoftArmKinematics
from ik_solver import IKSolver

# ==========================================
#  轨迹规划器类
# ==========================================
class TrajectoryPlanner:
    def __init__(self):
        self.center = np.array([1000.0, 100.0, 300.0]) # 默认轨迹中心
        self.radius = 150.0
        self.period = 10.0 # 周期(秒)
        self.type = 'NONE' # 当前轨迹类型
        
    def get_path_points(self, traj_type, steps=100):
        """生成用于绘制 3D 红线的静态路径点"""
        t = np.linspace(0, 1, steps) * self.period
        points = []
        for ti in t:
            points.append(self.calculate_pos(traj_type, ti))
        return np.array(points)

    def calculate_pos(self, traj_type, current_time):
        """根据当前时间计算目标点坐标 (x, y, z)"""
        w = 2 * np.pi * (current_time % self.period) / self.period
        x, y, z = self.center
        r = self.radius

        if traj_type == 'CIRCLE':
            x = self.center[0] + r * np.cos(w)
            y = self.center[1] + r * np.sin(w)
            z = self.center[2]
            
        elif traj_type == 'FIG8':
            scale = r * 1.2
            x = self.center[0] 
            y = self.center[1] + scale * np.sin(w)
            z = self.center[2] + scale * np.sin(2 * w) / 2.0
            
            
        elif traj_type == 'SPIRAL':
            r_spiral = r * 0.8
            x = self.center[0] + 100 * np.sin(2 * w)
            y = self.center[1] + r_spiral * np.cos(w)
            z = self.center[2] + r_spiral * np.sin(w)
            
        return np.array([x, y, z])

# ==========================================
#  主控制器类
# ==========================================
class UnifiedRobotController:
    def __init__(self):
        # 1. 初始化核心模块
        self.arm_model = HydraulicSoftArmKinematics()
        self.ik_solver = IKSolver()
        self.planner = TrajectoryPlanner()
        
        # 2. 状态定义
        self.current_mode = 'MANUAL' 
        self.time_step = 0.0
        self.is_traj_running = False
        
        # 3. 数据存储
        self.current_q = [0, 60, -90, 0, 0, 0, 180] 
        self.target_pos = self.planner.center.copy()
        
        # 4. 初始化绘图
        self.fig = plt.figure(figsize=(16, 9))
        # 调整布局: [left, bottom, width, height]
        # 给右侧留出更多空间 (0.70 -> 0.65 width)
        self.ax3d = plt.axes([0.02, 0.05, 0.65, 0.90], projection='3d')
        
        self.setup_visuals()
        self.setup_ui()
        
        # 5. 绑定键盘事件
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # 6. 启动循环
        self.ani = animation.FuncAnimation(self.fig, self.update_loop, interval=30, blit=False)
        
        print("=== 系统启动 ===")
        print("模式说明:")
        print("  MANUAL: 滑块控制关节")
        print("  IK-POINT: 键盘控制目标点，IK 求解")
        print("  TRAJ-*: 自动沿预设轨迹运行 (可用键盘调整中心)")
        plt.show()

    def setup_visuals(self):
        limit = 1000
        self.ax3d.set_xlim(-200, 1200)
        self.ax3d.set_ylim(-800, 800)
        self.ax3d.set_zlim(0, 1200)
        self.ax3d.set_xlabel('X'); self.ax3d.set_ylabel('Y'); self.ax3d.set_zlabel('Z')
        
        # 静态组件
        self.ax3d.plot([0,0], [0,0], [0, self.arm_model.base_z_offset], 'k-', lw=5, alpha=0.3)
        self.ax3d.scatter([0], [0], [0], s=300, c='black', marker='s')
        
        # 机械臂组件
        self.viz_links, = self.ax3d.plot([], [], [], '-', lw=6, c='#4682B4', alpha=0.9, label='Rigid')
        self.viz_fixed, = self.ax3d.plot([], [], [], '-', lw=6, c='#5F9EA0', alpha=0.9)
        self.viz_joints, = self.ax3d.plot([], [], [], 'o', ms=8, mfc='white', mec='black')
        self.viz_soft,   = self.ax3d.plot([], [], [], '-', lw=8, c='#FF8C00', solid_capstyle='round', label='Soft')
        
        # 目标与轨迹
        self.viz_target, = self.ax3d.plot([], [], [], 'o', ms=10, c='red', alpha=0.8, label='Target')
        self.viz_path,   = self.ax3d.plot([], [], [], '--', lw=1.5, c='red', alpha=0.5)
        
        # --- 【恢复】末端坐标轴 (RGB) ---
        self.viz_tip_axes = []
        colors = ['r', 'g', 'b']
        for c in colors:
            # 初始化三条线，分别代表 x, y, z 轴
            line, = self.ax3d.plot([], [], [], '-', lw=3, c=c)
            self.viz_tip_axes.append(line)
        
        # 地面
        xx, yy = np.meshgrid(range(-500, 1500, 1000), range(-600, 1200, 1000))
        self.ax3d.plot_surface(xx, yy, np.zeros_like(xx), color='gray', alpha=0.1)
        
        # 信息文本
        self.text_info = self.ax3d.text2D(0.02, 0.95, "Mode: MANUAL", transform=self.ax3d.transAxes, fontsize=14, color='blue')

        self.text_soft_tip = self.ax3d.text2D(
            0.02, 0.85, 
            "Soft Tip: N/A", 
            transform=self.ax3d.transAxes, 
            fontsize=11, 
            color='purple',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )


    def setup_ui(self):
        # 右侧控制面板区域设置
        panel_left = 0.72  # 向左移一点，变宽
        panel_width = 0.20 # 增加宽度
        
        # 1. 模式选择 (Radio Button)
        # 【修改】增大高度 (0.15 -> 0.25)，防止按钮挤在一起
        ax_mode = plt.axes([panel_left, 0.65, panel_width, 0.25], facecolor='#f0f0f0')
        self.radio_mode = RadioButtons(ax_mode, ('MANUAL', 'IK-POINT', 'TRAJ-CIRCLE', 'TRAJ-FIG8', 'TRAJ-SPIRAL'))
        self.radio_mode.on_clicked(self.on_mode_change)
        
        # 设置标题和字体大小
        ax_mode.set_title("Operation Mode", fontsize=12, pad=10)
        # 遍历标签，调大字体
        for label in self.radio_mode.labels:
            label.set_fontsize(11) 
        
        # 2. 滑块 (仅在 Manual 模式有效，但一直显示)
        self.sliders = []
        slider_configs = [
            ('J1 Base', -60, 60, 0),
            ('J2 Shoulder', -28, 90, 60),
            ('J3 Elbow', -152, -42, -90),
            ('Bend', -120, 120, 0),
            ('Phi', -180, 180, 0),
            ('Len', 140, 250, 180)
        ]
        
        start_y = 0.55
        for i, (lbl, vmin, vmax, vinit) in enumerate(slider_configs):
            # 调整滑块位置和高度
            ax = plt.axes([panel_left, start_y - i*0.06, panel_width, 0.03])
            s = Slider(ax, lbl, vmin, vmax, valinit=vinit, valfmt='%0.0f')
            s.label.set_fontsize(10)
            s.on_changed(self.on_slider_manual)
            self.sliders.append(s)
            
        # 说明文字
        plt.figtext(panel_left, 0.15, "Controls:\n[IK/Traj Mode]:\nArrows: Move Center XY\n+/-: Move Center Z", fontsize=10, color='#333333')

    # ================= 交互逻辑 =================

    def on_mode_change(self, label):
        self.current_mode = label
        self.time_step = 0.0 # 重置时间
        self.text_info.set_text(f"Mode: {label}")
        
        if label.startswith('TRAJ'):
            self.is_traj_running = True
            traj_type = label.split('-')[1]
            self.planner.type = traj_type
            
            # 绘制红线轨迹
            path_pts = self.planner.get_path_points(traj_type)
            self.viz_path.set_data(path_pts[:,0], path_pts[:,1])
            self.viz_path.set_3d_properties(path_pts[:,2])
            
            self.viz_target.set_visible(True)
            self.viz_path.set_visible(True)
            
        elif label == 'IK-POINT':
            self.is_traj_running = False
            self.viz_path.set_visible(False)
            self.viz_target.set_visible(True)
            self.target_pos = self.planner.center.copy()
            
        else: # MANUAL
            self.is_traj_running = False
            self.viz_path.set_visible(False)
            self.viz_target.set_visible(False)

    def on_key_press(self, event):
        if self.current_mode == 'MANUAL': return
        
        step = 10.0
        key = (event.key or '').lower()
        
        # 键盘控制中心点
        if key in ('up', '8'):    self.planner.center[0] += step
        elif key in ('down', '2'): self.planner.center[0] -= step
        elif key in ('left', '4'): self.planner.center[1] -= step
        elif key in ('right', '6'): self.planner.center[1] += step
        elif key in ('+', 'equals'): self.planner.center[2] += step
        elif key in ('-', 'minus'):  self.planner.center[2] -= step
        
        if not self.is_traj_running:
            self.target_pos = self.planner.center.copy()
            
        if self.is_traj_running:
            path_pts = self.planner.get_path_points(self.planner.type)
            self.viz_path.set_data(path_pts[:,0], path_pts[:,1])
            self.viz_path.set_3d_properties(path_pts[:,2])

    def on_slider_manual(self, val):
        if self.current_mode == 'MANUAL':
            vals = [s.val for s in self.sliders]
            self.current_q = vals[:3] + [0] + vals[3:]

    def update_sliders_visual(self, q_vector):
        indices = [0, 1, 2, 4, 5, 6]
        for i, idx in enumerate(indices):
            self.sliders[i].eventson = False
            self.sliders[i].set_val(q_vector[idx])
            self.sliders[i].eventson = True

    # ================= 主循环 =================
    
    def update_loop(self, frame):
        self.time_step += 0.05
        
        # 1. 计算目标与IK
        if self.is_traj_running:
            traj_target = self.planner.calculate_pos(self.planner.type, self.time_step)
            self.target_pos = traj_target
            
            sol = self.ik_solver.solve(self.target_pos, self.current_q)
            if sol is not None:
                self.current_q = sol
                self.update_sliders_visual(self.current_q)
                self.viz_target.set_color('lime')
            else:
                self.viz_target.set_color('red')
                
        elif self.current_mode == 'IK-POINT':
            sol = self.ik_solver.solve(self.target_pos, self.current_q)
            if sol is not None:
                self.current_q = sol
                self.update_sliders_visual(self.current_q)
                self.viz_target.set_color('lime')
            else:
                self.viz_target.set_color('red')

        # 2. 正向运动学绘图
        r_pts, s_pts, _, T_tip = self.arm_model.forward_kinematics(self.current_q)
        
        # 更新组件位置
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
        
# 1. 提取当前软体参数 (Index 4=Bend, 5=Phi, 6=Len)
        curr_bend = self.current_q[4]
        curr_phi  = self.current_q[5]
        curr_len  = self.current_q[6]
        
        # 2. 计算相对坐标 (开启 to_real_z_axis=True，让 Z 轴代表伸长方向)
        # 这里的 self.arm_model 就是 robot_model.py 中的类实例
        tip_local = self.arm_model.get_soft_tip_in_base_frame(
            curr_bend, curr_phi, curr_len, to_real_z_axis=True
        )
        
        # 3. 格式化显示的文本
        # Real X: 侧向弯曲
        # Real Y: 向上弯曲
        # Real Z: 伸长方向
        info_str = (
            f"Soft Tip (Local):\n"
            f"X: {tip_local[0]:6.1f} mm\n"
            f"Y: {tip_local[1]:6.1f} mm\n"
            f"Z: {tip_local[2]:6.1f} mm"
        )
        self.text_soft_tip.set_text(info_str)

        # 更新目标点
        if self.current_mode != 'MANUAL':
            self.viz_target.set_data([self.target_pos[0]], [self.target_pos[1]])
            self.viz_target.set_3d_properties([self.target_pos[2]])

        # --- 【恢复】更新末端坐标轴 ---
        origin = T_tip[:3, 3]
        rot_mat = T_tip[:3, :3]
        axis_len = 80 # 坐标轴显示长度 (mm)
        
        # 绘制 X, Y, Z 轴
        # viz_tip_axes[0] -> X (Red), [1] -> Y (Green), [2] -> Z (Blue)
        for i in range(3):
            # 计算轴的终点：起点 + 旋转矩阵的第i列 * 长度
            axis_vec = rot_mat[:, i] 
            end_p = origin + axis_vec * axis_len
            
            line = self.viz_tip_axes[i]
            line.set_data([origin[0], end_p[0]], [origin[1], end_p[1]])
            line.set_3d_properties([origin[2], end_p[2]])
            
        return self.viz_links, self.viz_soft

if __name__ == "__main__":
    app = UnifiedRobotController()