"""
文件功能：早期版本或特定功能的快照

本文件 (`version1.py`) 通常代表项目在某个开发阶段的稳定版本或一个特定功能的完整实现。
与 `main.py` 相比，它可能：
- 是一个更简单、更纯粹的实现，只包含核心的运动学和可视化，没有复杂的模式切换。
- 是为了测试某个特定想法（例如，不同的UI布局、运动学算法或可视化效果）而创建的独立原型。
- 作为项目重构前的备份。

简而言之，这是一个功能相对完整的、独立的演示程序，记录了项目演进过程中的一个版本。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, CheckButtons
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 1. 核心运动学类 (逻辑层保持不变)
# ==========================================
class HydraulicSoftArmKinematics:
    def __init__(self):
        self.base_z_offset = 500.0 # mm
        
        # Modified D-H 参数 (4个刚性关节)
        self.rigid_dh_params = [
            {'alpha': 0, 'a': 190, 'd': 0},             # Link 1
            {'alpha': np.radians(90), 'a': 90, 'd': 0}, # Link 2
            {'alpha': 0, 'a': 605, 'd': 0},             # Link 3
            {'alpha': 0, 'a': 290, 'd': 0}              # Link 4 (固定延长段)
        ]
        
        self.soft_segments = 25 # 增加段数让软体更平滑
        
        # 关节限位
        self.limits = {
            'q1': [-60, 60],
            'q2': [-28, 90],
            'q3': [-152, -42],
            'q4': [0, 0],       # 固定
            'bend': [0, 120],
            'phi': [-180, 180],
            'len': [140, 250]
        }

    def mdh_matrix(self, alpha, a, theta, d):
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        return np.array([
            [ct,    -st,    0,    a],
            [st*ca, ct*ca, -sa, -d*sa],
            [st*sa, ct*sa,  ca,  d*ca],
            [0,     0,      0,    1]
        ])

    def forward_kinematics(self, q_degrees):
        q_rad = [np.deg2rad(v) for v in q_degrees[:6]]
        length_mm = q_degrees[6]
        
        # 1. 计算刚性前三节
        T_cum = np.eye(4)
        T_cum[2, 3] = self.base_z_offset
        rigid_points = [T_cum[:3, 3]] # Point 0: Base
        
        for i in range(4):
            T_i = self.mdh_matrix(self.rigid_dh_params[i]['alpha'], 
                                  self.rigid_dh_params[i]['a'], 
                                  q_rad[i], 
                                  self.rigid_dh_params[i]['d'])
            T_cum = T_cum @ T_i
            rigid_points.append(T_cum[:3, 3])
            
        T_base_soft = T_cum
        
        # 2. 计算软体部分 (PCC)
        theta_bend = q_rad[4]
        phi_dir = q_rad[5]
        
        soft_points_local = []
        if abs(theta_bend) < 1e-4:
            for i in range(self.soft_segments + 1):
                s = (i / self.soft_segments) * length_mm
                soft_points_local.append([s, 0, 0, 1])
            T_tip_local = np.eye(4)
            T_tip_local[0, 3] = length_mm
        else:
            R = length_mm / theta_bend
            for i in range(self.soft_segments + 1):
                sigma = (i / self.soft_segments) * theta_bend
                x_plane = R * np.sin(sigma)
                y_plane = R * (1 - np.cos(sigma))
                y = y_plane * np.cos(phi_dir)
                z = y_plane * np.sin(phi_dir)
                soft_points_local.append([x_plane, y, z, 1])
            
            tip_x = R * np.sin(theta_bend)
            tip_y_plane = R * (1 - np.cos(theta_bend))
            ct = np.cos(theta_bend); st = np.sin(theta_bend)
            cp = np.cos(phi_dir); sp = np.sin(phi_dir)
            T_tip_local = np.eye(4)
            T_tip_local[:3, 3] = [tip_x, tip_y_plane*cp, tip_y_plane*sp]
            
        soft_points_global = (T_base_soft @ np.array(soft_points_local).T).T
        
        # rigid_points: [Base, J1, J2, J3, End_Fixed]
        return np.array(rigid_points), soft_points_global[:, :3], T_base_soft

# ==========================================
# 2. 混合控制可视化类 (视觉增强版)
# ==========================================
class HybridSimulation:
    def __init__(self):
        self.arm = HydraulicSoftArmKinematics()
        self.is_auto = False
        self.show_axis = True
        self.time_step = 0.0
        
        self.fig = plt.figure(figsize=(14, 9))
        self.ax3d = plt.axes([0.35, 0.05, 0.60, 0.9], projection='3d')
        
        self.setup_3d_plot()
        self.sliders = []
        self.setup_ui_controls()
        
        self.ani = animation.FuncAnimation(self.fig, self.update_loop, interval=40, blit=False)
        init_vals = [s.val for s in self.sliders]
        self.update_robot_viz(init_vals)
        
        plt.show()

    def setup_3d_plot(self):
        limit = 1200
        self.ax3d.set_xlim(-limit, limit)
        self.ax3d.set_ylim(-limit, limit)
        self.ax3d.set_zlim(0, 1800)
        self.ax3d.set_xlabel('X')
        self.ax3d.set_ylabel('Y')
        self.ax3d.set_zlabel('Z')
        self.ax3d.set_title("Visually Enhanced Robot Simulation", fontsize=14)
        
        # === 视觉层 1: 地面与基座 ===
        # 画一个简单的方块底座
        self.ax3d.plot([0,0], [0,0], [0, self.arm.base_z_offset], 'k-', lw=3, alpha=0.5)
        # 用散点画一个大的基座点
        self.ax3d.scatter([0], [0], [0], s=200, c='black', marker='s')
        
        # === 视觉层 2: 连杆 (Links) - 只画线，不画点 ===
        # 刚性活动部分 (J1-J3) -> 蓝色粗线
        self.viz_rigid_links, = self.ax3d.plot([], [], [], '-', lw=8, c='#4682B4', alpha=0.9, label='Hydraulic Arms')
        # 固定延长部分 (Link 4) -> 深灰色粗线
        self.viz_fixed_link, = self.ax3d.plot([], [], [], '-', lw=8, c='#4682B4', alpha=0.9, label='Fixed Extension')
        
        # === 视觉层 3: 关节 (Joints) - 只画点，不画线 ===
        # 用 scatter 画大圆球，显眼
        # 注意：这里我们使用 plot(..., 'o') 因为 scatter 在动画中更新比较麻烦
        self.viz_joints, = self.ax3d.plot([], [], [], 'o', ms=10, mfc='white', mec='black', mew=2, label='Joints')
        
        # === 视觉层 4: 软体臂 ===
        # 橙色，半透明，很粗的线，模拟硅胶
        self.viz_soft, = self.ax3d.plot([], [], [], '-', lw=8, c='#FF8C00', alpha=0.9, solid_capstyle='round', label='Soft Arm')
        
        # === 视觉层 5: 法兰连接盘 ===
        self.viz_flange, = self.ax3d.plot([], [], [], 'o', ms=10, c='black')

        # 坐标轴
        self.axis_lines = []
        for c in ['r', 'g', 'b']:
            line, = self.ax3d.plot([], [], [], '-', lw=2, color=c)
            self.axis_lines.append(line)
            
        self.ax3d.legend(loc='upper right')
        
        # 地面网格
        xx, yy = np.meshgrid(range(-limit, limit+1, 500), range(-limit, limit+1, 500))
        zz = np.zeros_like(xx)
        self.ax3d.plot_surface(xx, yy, zz, color='gray', alpha=0.1)

    def setup_ui_controls(self):
        axcolor = 'lightgoldenrodyellow'
        slider_configs = [
            ('J1 Base', -60, 60, 0),
            ('J2 Shoulder', -28, 90, 0),
            ('J3 Elbow', -152, -42, -90),
            # J4 已移除
            ('Soft Bend', 0, 120, 0),
            ('Soft Phi', -180, 180, 0),
            ('Soft Len', 140, 250, 200)
        ]
        
        start_y = 0.85
        for i, (lbl, vmin, vmax, vinit) in enumerate(slider_configs):
            ax = plt.axes([0.05, start_y - i*0.06, 0.20, 0.03], facecolor=axcolor)
            s = Slider(ax, lbl, vmin, vmax, valinit=vinit)
            s.on_changed(self.on_slider_change)
            self.sliders.append(s)

        btn_ax = plt.axes([0.05, 0.15, 0.20, 0.06])
        self.btn_mode = Button(btn_ax, 'Switch AUTO/MANUAL', color='lightgray')
        self.btn_mode.on_clicked(self.toggle_mode)
        
        chk_ax = plt.axes([0.05, 0.10, 0.20, 0.04])
        self.chk_axis = CheckButtons(chk_ax, ['Show Flange Axis'], [True])
        self.chk_axis.on_clicked(self.toggle_axis)

    def toggle_mode(self, event):
        self.is_auto = not self.is_auto
        
    def toggle_axis(self, label):
        self.show_axis = not self.show_axis
        if not self.show_axis:
            for line in self.axis_lines:
                line.set_data([], [])
                line.set_3d_properties([])
        self.fig.canvas.draw_idle()

    def get_auto_wave(self, t, limits, freq=1.0, phase=0.0):
        mid = (limits[0] + limits[1]) / 2
        amp = (limits[1] - limits[0]) / 2 * 0.9
        return mid + amp * np.sin(t*freq + phase)

    def update_loop(self, frame):
        if not self.is_auto: return
        
        self.time_step += 0.05
        t = self.time_step
        l = self.arm.limits
        
        vals = [
            self.get_auto_wave(t, l['q1'], 0.5),
            self.get_auto_wave(t, l['q2'], 0.5, 1),
            self.get_auto_wave(t, l['q3'], 0.6, 2),
            (np.sin(t)+1)/2 * 100, # Bend
            t * 50 % 360 - 180,    # Phi
            200 + 50*np.sin(t)     # Len
        ]
        
        for i, s in enumerate(self.sliders):
            s.eventson = False 
            s.set_val(vals[i])
            s.eventson = True
            
        self.update_robot_viz(vals)

    def on_slider_change(self, val):
        current_vals = [s.val for s in self.sliders]
        self.update_robot_viz(current_vals)

    def update_robot_viz(self, slider_vals):
        # 补全 q4=0
        full_q = slider_vals[:3] + [0.0] + slider_vals[3:]
        
        r_pts, s_pts, T_base = self.arm.forward_kinematics(full_q)
        
        # --- 1. 更新连杆 (Lines) ---
        # 刚性连杆: Base -> J1 -> J2 -> J3 (Array indices 0, 1, 2, 3)
        self.viz_rigid_links.set_data(r_pts[0:4, 0], r_pts[0:4, 1])
        self.viz_rigid_links.set_3d_properties(r_pts[0:4, 2])
        
        # 固定延长段: J3 -> End (Array indices 3, 4)
        self.viz_fixed_link.set_data(r_pts[3:5, 0], r_pts[3:5, 1])
        self.viz_fixed_link.set_3d_properties(r_pts[3:5, 2])
        
        # --- 2. 更新关节 (Joints) ---
        # 我们只在 J1, J2, J3 的位置画大圆点 (Indices 1, 2, 3)
        # Index 0 是基座底，Index 4 是固定杆末端(法兰)
        joints_xyz = r_pts[1:4] 
        self.viz_joints.set_data(joints_xyz[:, 0], joints_xyz[:, 1])
        self.viz_joints.set_3d_properties(joints_xyz[:, 2])
        
        # --- 3. 更新软体臂 ---
        # 为了视觉连续，起点设为刚性臂末端
        sx = np.concatenate(([r_pts[-1,0]], s_pts[:,0]))
        sy = np.concatenate(([r_pts[-1,1]], s_pts[:,1]))
        sz = np.concatenate(([r_pts[-1,2]], s_pts[:,2]))
        
        self.viz_soft.set_data(sx, sy)
        self.viz_soft.set_3d_properties(sz)
        
        # --- 4. 法兰连接处 ---
        flange = r_pts[-1]
        self.viz_flange.set_data([flange[0]], [flange[1]])
        self.viz_flange.set_3d_properties([flange[2]])
        
        # --- 5. 坐标轴 ---
        if self.show_axis:
            axis_len = 200 
            origin = T_base[:3, 3]
            rotation = T_base[:3, :3]
            basis = np.eye(3) 
            for i, line in enumerate(self.axis_lines):
                end_p = origin + rotation @ basis[i] * axis_len
                line.set_data([origin[0], end_p[0]], [origin[1], end_p[1]])
                line.set_3d_properties([origin[2], end_p[2]])

        if not self.is_auto:
            self.fig.canvas.draw_idle()

if __name__ == "__main__":
    sim = HybridSimulation()