"""
文件功能：主程序入口与可视化界面 (UI优化版 + 基座显示)

更新日志：
1. 恢复了末端坐标轴 (RGB Axes) 的显示。
2. 增大了模式选择按钮的尺寸和字体，便于点击。
3. 包含了 TrajectoryPlanner 轨迹规划功能。
4. [新增] 集成了静态基座 (Base) 的 3D 可视化。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.mplot3d import Axes3D

# 导入你的自定义模块 (假设这些文件在同级目录下)
# 如果没有这些文件，请确保 robot_model 和 ik_solver 类已定义
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

    def generate_base_geometry(self, mount_pos):
        
        x_base, y_base, z_base = mount_pos
        
        height = 600   # 基座柱子的高度
        radius = 250   # 底盘半径
        
        # 1. 中心支柱 (从安装点向下延伸 height 长度)
        pillar_top = np.array([x_base, y_base, z_base])
        pillar_btm = np.array([x_base, y_base, z_base - height])
        
        # 2. 底盘圆圈 (位于柱子底部)
        theta = np.linspace(0, 2*np.pi, 30)
        base_x = x_base + radius * np.cos(theta)
        base_y = y_base + radius * np.sin(theta)
        base_z = np.full_like(theta, z_base - height) # Z轴高度与柱底一致
        
        return pillar_top, pillar_btm, base_x, base_y, base_z

    def setup_visuals(self):
        # 1. 先计算一次正运动学，获取机械臂实际的起始位置 P0
        # 这样无论你的 kinematic 模型里把原点定义在哪里，基座都会自动对齐
        r_pts_init, _, _, _ = self.arm_model.forward_kinematics(self.current_q)
        mount_point = r_pts_init[0] # 获取 P0 点 (Base)

        # 2. 动态设置绘图范围 (根据安装点高度调整)
        limit = 1000
        z_start = mount_point[2]
        self.ax3d.set_xlim(-500, 1500)
        self.ax3d.set_ylim(-1000, 1000)
        # Z轴范围：从基座底部再往下一点，到机械臂上方
        self.ax3d.set_zlim(z_start - 800, z_start + 1200) 
        self.ax3d.set_xlabel('X'); self.ax3d.set_ylabel('Y'); self.ax3d.set_zlabel('Z')
        
        # --- [关键修改] 传入 mount_point 生成基座 ---
        p_top, p_btm, bx, by, bz = self.generate_base_geometry(mount_point)
        
        # 绘制支柱
        self.ax3d.plot([p_top[0], p_btm[0]], 
                       [p_top[1], p_btm[1]], 
                       [p_top[2], p_btm[2]], 
                       '-', lw=10, color='#444444', alpha=0.6, solid_capstyle='round')
        
        # 绘制底盘
        self.ax3d.plot(bx, by, bz, '-', lw=3, color='#444444', alpha=0.8)
        
        # 绘制原点连接处的装饰球
        self.ax3d.scatter([p_top[0]], [p_top[1]], [p_top[2]], s=100, c='#333333', marker='o', zorder=10)
        
        # 地面网格 (放到基座底部平面)
        ground_z = p_btm[2]
        xx, yy = np.meshgrid(range(-1000, 2000, 500), range(-1000, 2000, 500))
        self.ax3d.plot_surface(xx, yy, np.full_like(xx, ground_z), color='gray', alpha=0.05)
        
        # --- 下面保持不变 ---
        self.viz_links, = self.ax3d.plot([], [], [], '-', lw=6, c='#4682B4', alpha=0.9, label='Rigid')
        self.viz_fixed, = self.ax3d.plot([], [], [], '-', lw=6, c='#5F9EA0', alpha=0.9)
        self.viz_joints, = self.ax3d.plot([], [], [], 'o', ms=8, mfc='white', mec='black')
        self.viz_soft,   = self.ax3d.plot([], [], [], '-', lw=8, c='#FF8C00', solid_capstyle='round', label='Soft')
        
        self.viz_target, = self.ax3d.plot([], [], [], 'o', ms=10, c='red', alpha=0.8, label='Target')
        self.viz_path,   = self.ax3d.plot([], [], [], '--', lw=1.5, c='red', alpha=0.5)
        
        self.viz_tip_axes = []
        colors = ['r', 'g', 'b']
        for c in colors:
            line, = self.ax3d.plot([], [], [], '-', lw=3, c=c)
            self.viz_tip_axes.append(line)
        
        self.text_info = self.ax3d.text2D(0.02, 0.95, "Mode: MANUAL", transform=self.ax3d.transAxes, fontsize=14, color='blue')
        self.text_soft_tip = self.ax3d.text2D(0.02, 0.85, "Soft Tip: N/A", transform=self.ax3d.transAxes, fontsize=11, color='purple', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
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
        panel_left = 0.72  
        panel_width = 0.20 
        
        # 1. 模式选择 (Radio Button)
        ax_mode = plt.axes([panel_left, 0.65, panel_width, 0.25], facecolor='#f0f0f0')
        self.radio_mode = RadioButtons(ax_mode, ('MANUAL', 'IK-POINT', 'TRAJ-CIRCLE', 'TRAJ-FIG8', 'TRAJ-SPIRAL'))
        self.radio_mode.on_clicked(self.on_mode_change)
        
        ax_mode.set_title("Operation Mode", fontsize=12, pad=10)
        for label in self.radio_mode.labels:
            label.set_fontsize(11) 
        
        # 2. 滑块
        self.sliders = []
        slider_configs = [
            ('J1 Base', -60, 60, 0),
            ('J2 Shoulder', -28, 90, 60),
            ('J3 Elbow', -152, -42, -90),
            ('Bend', -120, 120, 0),
            ('Phi', -180, 180, 0),
            ('Len', 140, 250, 140)
        ]
        
        start_y = 0.55
        for i, (lbl, vmin, vmax, vinit) in enumerate(slider_configs):
            ax = plt.axes([panel_left, start_y - i*0.06, panel_width, 0.03])
            s = Slider(ax, lbl, vmin, vmax, valinit=vinit, valfmt='%0.0f')
            s.label.set_fontsize(10)
            s.on_changed(self.on_slider_manual)
            self.sliders.append(s)
            
        plt.figtext(panel_left, 0.15, "Controls:\n[IK/Traj Mode]:\nArrows: Move Center XY\n+/-: Move Center Z", fontsize=10, color='#333333')

    # ================= 交互逻辑 =================

    def on_mode_change(self, label):
        self.current_mode = label
        self.time_step = 0.0 
        self.text_info.set_text(f"Mode: {label}")
        
        if label.startswith('TRAJ'):
            self.is_traj_running = True
            traj_type = label.split('-')[1]
            self.planner.type = traj_type
            
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
        
        # 1. 提取当前软体参数
        curr_bend = self.current_q[4]
        curr_phi  = self.current_q[5]
        curr_len  = self.current_q[6]
        
        # 2. 计算相对坐标
        tip_local = self.arm_model.get_soft_tip_in_base_frame(
            curr_bend, curr_phi, curr_len, to_real_z_axis=True
        )
        
        # 3. 格式化显示的文本
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

        # 更新末端坐标轴
        origin = T_tip[:3, 3]
        rot_mat = T_tip[:3, :3]
        axis_len = 80 
        
        for i in range(3):
            axis_vec = rot_mat[:, i] 
            end_p = origin + axis_vec * axis_len
            
            line = self.viz_tip_axes[i]
            line.set_data([origin[0], end_p[0]], [origin[1], end_p[1]])
            line.set_3d_properties([origin[2], end_p[2]])
            
        return self.viz_links, self.viz_soft

if __name__ == "__main__":
    app = UnifiedRobotController()