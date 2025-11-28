import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, CheckButtons
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 1. 核心运动学类 (逻辑层)
# ==========================================
class HydraulicSoftArmKinematics:
    def __init__(self):
        self.base_z_offset = 500.0 # mm
        
        # Modified D-H 参数 (4个刚性关节)
        # alpha, a, d 都是常数，theta 是变量
        self.rigid_dh_params = [
            {'alpha': 0, 'a': 190, 'd': 0},             # Link 1
            {'alpha': np.radians(90), 'a': 90, 'd': 0}, # Link 2
            {'alpha': 0, 'a': 605, 'd': 0},             # Link 3
            {'alpha': 0, 'a': 290, 'd': 90}              # Link 4 (末端刚性杆)
        ]
        
        self.soft_segments = 20
        
        # 关节限位
        self.limits = {
            'q1': [-60, 60],
            'q2': [-28, 90],
            'q3': [-152, -42],
            'q4': [-90, 90],
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
        """
        q_degrees: [q1, q2, q3, q4, bend, phi, len]
        """
        # 1. 单位转换
        q_rad = [np.deg2rad(v) for v in q_degrees[:6]]
        length_mm = q_degrees[6]
        
        # 2. 计算刚性部分 (Link 1 - Link 4)
        # T_cum 代表当前关节相对于全局基座的变换矩阵
        T_cum = np.eye(4)
        T_cum[2, 3] = self.base_z_offset
        rigid_points = [T_cum[:3, 3]]
        
        for i in range(4):
            T_i = self.mdh_matrix(self.rigid_dh_params[i]['alpha'], 
                                  self.rigid_dh_params[i]['a'], 
                                  q_rad[i], 
                                  self.rigid_dh_params[i]['d'])
            T_cum = T_cum @ T_i
            rigid_points.append(T_cum[:3, 3])
            
        # 此时 T_cum 就是 Link 4 末端（即软体臂基座）的位姿矩阵
        # 这里的 X 轴方向就是 Link 4 的延伸方向
        T_base_soft = T_cum
        
        # 3. 计算软体部分 (PCC模型) - 在局部坐标系计算
        theta_bend = q_rad[4]
        phi_dir = q_rad[5]
        
        soft_points_local = []
        
        # 避免弯曲角度为0时的除零错误
        if abs(theta_bend) < 1e-4:
            # 直线状态：沿局部 X 轴延伸 (因为 D-H 中 X 是连杆轴线)
            for i in range(self.soft_segments + 1):
                s = (i / self.soft_segments) * length_mm
                soft_points_local.append([s, 0, 0, 1])
            
            # 末端局部变换 (纯平移)
            T_tip_local = np.eye(4)
            T_tip_local[0, 3] = length_mm
        else:
            # 弯曲状态：常曲率圆弧
            R = length_mm / theta_bend
            for i in range(self.soft_segments + 1):
                # 当前弧长对应的圆心角
                sigma = (i / self.soft_segments) * theta_bend
                
                # 在弯曲平面内的坐标 (假设在 X-Y 平面弯曲)
                # 切线起始沿 X 轴，向上弯曲
                x_plane = R * np.sin(sigma)
                y_plane = R * (1 - np.cos(sigma)) # 偏转量
                
                # 引入 Phi (空间旋转角)
                # 绕 X 轴旋转这个平面。
                # 注意：这里 y_plane 是偏转距离。
                # 旋转后，偏转距离分配给 Y 和 Z
                y = y_plane * np.cos(phi_dir)
                z = y_plane * np.sin(phi_dir)
                
                soft_points_local.append([x_plane, y, z, 1])
            
            # 末端局部变换
            # 先计算位置
            tip_x_plane = R * np.sin(theta_bend)
            tip_y_plane = R * (1 - np.cos(theta_bend))
            
            # 构建末端旋转矩阵 (先绕 Z 转 theta_bend，再绕 X 转 phi_dir)
            # 简化处理：构建切线方向作为新的 X 轴
            ct = np.cos(theta_bend)
            st = np.sin(theta_bend)
            cp = np.cos(phi_dir)
            sp = np.sin(phi_dir)
            
            # 这是一个近似的局部变换，用于简单的末端可视化
            T_tip_local = np.eye(4)
            T_tip_local[:3, 3] = [tip_x_plane, tip_y_plane*cp, tip_y_plane*sp]
            
        # 4. 坐标变换：局部 -> 全局
        # 硬连接的核心：所有局部点都左乘 T_base_soft
        # 这保证了软体臂的根部严格跟随 Link 4 的末端
        soft_points_global = (T_base_soft @ np.array(soft_points_local).T).T
        
        # 5. 返回数据
        # rigid_points: 刚性关节坐标
        # soft_points: 软体点云坐标
        # T_base_soft: 软体基座坐标系 (用于画坐标轴)
        # T_tip_global: 软体末端坐标系
        return np.array(rigid_points), soft_points_global[:, :3], T_base_soft, (T_base_soft @ T_tip_local)

# ==========================================
# 2. 混合控制可视化类
# ==========================================
class HybridSimulation:
    def __init__(self):
        self.arm = HydraulicSoftArmKinematics()
        self.is_auto = False
        self.show_axis = True # 是否显示坐标轴
        self.time_step = 0.0
        
        # UI 布局
        self.fig = plt.figure(figsize=(14, 9))
        self.ax3d = plt.axes([0.35, 0.05, 0.60, 0.9], projection='3d')
        
        self.setup_3d_plot()
        self.sliders = []
        self.setup_ui_controls()
        
        # 启动
        self.ani = animation.FuncAnimation(self.fig, self.update_loop, interval=40, blit=False)
        self.update_robot_viz([s.val for s in self.sliders]) # 初始绘制
        
        plt.show()

    def setup_3d_plot(self):
        limit = 1200
        self.ax3d.set_xlim(-limit, limit)
        self.ax3d.set_ylim(-limit, limit)
        self.ax3d.set_zlim(0, 1800)
        self.ax3d.set_xlabel('X')
        self.ax3d.set_ylabel('Y')
        self.ax3d.set_zlabel('Z')
        self.ax3d.set_title("Robot Arm Simulation (Hard Connection Demo)", fontsize=14)
        
        # === 绘图元素 ===
        # 1. 地面基座
        self.ax3d.plot([0,0], [0,0], [0, self.arm.base_z_offset], 'k-', lw=10, alpha=0.3)
        
        # 2. 刚性臂 (蓝色)
        self.line_rigid, = self.ax3d.plot([], [], [], 'o-', lw=6, c='#1f77b4', ms=6, label='Rigid Links')
        
        # 3. 软体臂 (橙色)
        self.line_soft, = self.ax3d.plot([], [], [], '.-', lw=5, c='#ff7f0e', ms=2, alpha=0.9, label='Soft Arm')
        
        # 4. 连接处法兰盘 (黑色点)
        self.flange_point, = self.ax3d.plot([], [], [], 'ko', ms=8, label='Flange (Fixed)')
        
        # 5. 坐标轴可视化 (红X 绿Y 蓝Z)
        self.axis_lines = []
        for c in ['r', 'g', 'b']:
            line, = self.ax3d.plot([], [], [], '-', lw=2, color=c)
            self.axis_lines.append(line)

        self.ax3d.legend(loc='upper right')

    def setup_ui_controls(self):
        axcolor = 'lightgoldenrodyellow'
        slider_configs = [
            ('J1 Base', -60, 60, 0),
            ('J2 Shoulder', -28, 90, 0),
            ('J3 Elbow', -152, -42, -90),
            ('J4 Wrist', -90, 90, 0),     # 旋转这个，观察坐标轴是否旋转
            ('Soft Bend', 0, 120, 0),     # 初始直立
            ('Soft Phi', -180, 180, 0),
            ('Soft Len', 140, 250, 200)
        ]
        
        start_y = 0.85
        for i, (lbl, vmin, vmax, vinit) in enumerate(slider_configs):
            ax = plt.axes([0.05, start_y - i*0.05, 0.20, 0.03], facecolor=axcolor)
            s = Slider(ax, lbl, vmin, vmax, valinit=vinit)
            s.on_changed(self.on_slider_change)
            self.sliders.append(s)

        # 模式切换按钮
        btn_ax = plt.axes([0.05, 0.15, 0.20, 0.06])
        self.btn_mode = Button(btn_ax, 'Switch AUTO/MANUAL', color='lightgray')
        self.btn_mode.on_clicked(self.toggle_mode)
        
        # 显示坐标轴勾选框
        chk_ax = plt.axes([0.05, 0.10, 0.20, 0.04])
        self.chk_axis = CheckButtons(chk_ax, ['Show Flange Axis'], [True])
        self.chk_axis.on_clicked(self.toggle_axis)

    def toggle_mode(self, event):
        self.is_auto = not self.is_auto
        
    def toggle_axis(self, label):
        self.show_axis = not self.show_axis
        # 如果关闭，清空坐标轴数据
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
        
        # 自动生成值
        vals = [
            self.get_auto_wave(t, l['q1'], 0.5),
            self.get_auto_wave(t, l['q2'], 0.5, 1),
            self.get_auto_wave(t, l['q3'], 0.6, 2),
            self.get_auto_wave(t, l['q4'], 0.8, 3), # 注意观察J4
            (np.sin(t)+1)/2 * 100, # Bend
            t * 50 % 360 - 180,    # Phi
            200 + 50*np.sin(t)     # Len
        ]
        
        # 更新滑块 (视觉)
        for i, s in enumerate(self.sliders):
            s.eventson = False # 暂时关闭回调，防止双重计算
            s.set_val(vals[i])
            s.eventson = True
            
        # 驱动绘图
        self.update_robot_viz(vals)

    def on_slider_change(self, val):
        current_q = [s.val for s in self.sliders]
        self.update_robot_viz(current_q)

    def update_robot_viz(self, q):
        # 计算
        # r_pts: 刚性点
        # s_pts: 软体点
        # T_base: 软体基座(连接处)的变换矩阵
        r_pts, s_pts, T_base, T_tip = self.arm.forward_kinematics(q)
        
        # 1. 刚性臂
        self.line_rigid.set_data(r_pts[:,0], r_pts[:,1])
        self.line_rigid.set_3d_properties(r_pts[:,2])
        
        # 2. 软体臂 (确保从刚性末端开始)
        # 将刚性最后一个点作为软体第一个点，确保视觉无缝
        sx = np.concatenate(([r_pts[-1,0]], s_pts[:,0]))
        sy = np.concatenate(([r_pts[-1,1]], s_pts[:,1]))
        sz = np.concatenate(([r_pts[-1,2]], s_pts[:,2]))
        
        self.line_soft.set_data(sx, sy)
        self.line_soft.set_3d_properties(sz)
        
        # 3. 法兰盘位置
        flange_pos = r_pts[-1]
        self.flange_point.set_data([flange_pos[0]], [flange_pos[1]])
        self.flange_point.set_3d_properties([flange_pos[2]])
        
        # 4. 绘制坐标轴 (显示硬连接的随动效果)
        if self.show_axis:
            axis_len = 200 # 坐标轴长度 200mm
            origin = T_base[:3, 3]
            rotation = T_base[:3, :3]
            
            # X轴(红), Y轴(绿), Z轴(蓝)
            # 在 D-H 中，X轴通常是连杆延伸方向
            basis = np.eye(3) 
            for i, line in enumerate(self.axis_lines):
                # 计算轴的终点: 原点 + 旋转矩阵 * 轴向量 * 长度
                end_p = origin + rotation @ basis[i] * axis_len
                line.set_data([origin[0], end_p[0]], [origin[1], end_p[1]])
                line.set_3d_properties([origin[2], end_p[2]])

        if not self.is_auto:
            self.fig.canvas.draw_idle()

if __name__ == "__main__":
    sim = HybridSimulation()