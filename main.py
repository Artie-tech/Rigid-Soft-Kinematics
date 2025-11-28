import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D

# 导入你的自定义模块
from robot_model import HydraulicSoftArmKinematics
from ik_solver import IKSolver

class UnifiedRobotController:
    def __init__(self):
        # 1. 初始化核心模块
        self.arm_model = HydraulicSoftArmKinematics()
        self.ik_solver = IKSolver()
        
        # 2. 状态定义
        self.MODES = ['MANUAL', 'AUTO', 'GRASP']
        self.current_mode_idx = 0 # 默认 MANUAL
        self.time_step = 0.0
        
        # 3. 数据存储
        # [q1, q2, q3, q4(固定), bend, phi, len]
        self.current_q = [0, 0, -90, 0, 0, 0, 180] 
        self.target_pos = np.array([600.0, 0.0, 400.0]) # 目标小球位置
        
        # 4. 初始化绘图
        self.fig = plt.figure(figsize=(15, 9))
        self.ax3d = plt.axes([0.35, 0.05, 0.60, 0.90], projection='3d')
        
        self.setup_visuals()
        self.setup_ui()
        
        # 5. 绑定事件
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # 6. 启动循环
        self.ani = animation.FuncAnimation(self.fig, self.update_loop, interval=30, blit=False)
        print("=== 系统启动 ===")
        print("当前模式: MANUAL")
        print("键盘控制 (仅在 GRASP 模式有效):")
        print("  W/S: 前后移动 (X轴)")
        print("  A/D: 左右移动 (Y轴)")
        print("  Q/E: 上下移动 (Z轴)")
        plt.show()

    def setup_visuals(self):
        limit = 1200
        self.ax3d.set_xlim(-limit, limit)
        self.ax3d.set_ylim(-limit, limit)
        self.ax3d.set_zlim(0, 1800)
        self.ax3d.set_xlabel('X')
        self.ax3d.set_ylabel('Y')
        self.ax3d.set_zlabel('Z')
        
        # 静态环境
        self.ax3d.plot([0,0], [0,0], [0, self.arm_model.base_z_offset], 'k-', lw=5, alpha=0.3)
        self.ax3d.scatter([0], [0], [0], s=300, c='black', marker='s')
        
        # 动态组件 (保留引用以便更新)
        self.viz_links, = self.ax3d.plot([], [], [], '-', lw=8, c='#4682B4', alpha=0.9, label='Rigid Arm')
        self.viz_fixed, = self.ax3d.plot([], [], [], '-', lw=8, c='#505050', alpha=0.9, label='Fixed Ext')
        self.viz_joints, = self.ax3d.plot([], [], [], 'o', ms=12, mfc='white', mec='black')
        self.viz_soft,   = self.ax3d.plot([], [], [], '-', lw=10, c='#FF8C00', alpha=0.7, solid_capstyle='round', label='Soft Arm')
        
        # 目标小球 (仅在 GRASP 模式高亮)
        self.viz_target, = self.ax3d.plot([], [], [], 'o', ms=15, c='gray', alpha=0.3, label='Target')
        
        # 地面
        xx, yy = np.meshgrid(range(-limit, limit+1, 600), range(-limit, limit+1, 600))
        zz = np.zeros_like(xx)
        self.ax3d.plot_surface(xx, yy, zz, color='gray', alpha=0.1)
        self.ax3d.legend(loc='upper right')
        
        # 标题文本
        self.title_text = self.ax3d.text2D(0.05, 0.95, "Mode: MANUAL", transform=self.ax3d.transAxes, fontsize=16, weight='bold', color='blue')

    def setup_ui(self):
        # 侧边栏背景
        axcolor = 'lightgoldenrodyellow'
        
        # 模式切换按钮
        ax_btn = plt.axes([0.05, 0.90, 0.20, 0.05])
        self.btn_mode = Button(ax_btn, 'Switch Mode (Click)', color='lightblue', hovercolor='0.975')
        self.btn_mode.on_clicked(self.toggle_mode)
        
        # 滑块配置
        self.sliders = []
        slider_params = [
            ('J1 Base', -60, 60, 0),
            ('J2 Shoulder', -28, 90, 0),
            ('J3 Elbow', -152, -42, -90),
            # J4 Fixed
            ('Soft Bend', 0, 120, 0),
            ('Soft Phi', -180, 180, 0),
            ('Soft Len', 140, 250, 180)
        ]
        
        start_y = 0.80
        for i, (label, vmin, vmax, vinit) in enumerate(slider_params):
            ax = plt.axes([0.05, start_y - i*0.06, 0.20, 0.03], facecolor=axcolor)
            s = Slider(ax, label, vmin, vmax, valinit=vinit)
            s.on_changed(self.on_slider_manual) # 绑定手动事件
            self.sliders.append(s)
            
        # 操作说明文本
        plt.figtext(0.05, 0.35, "Controls:\n[MANUAL]: Use Sliders\n[AUTO]: Watch demo\n[GRASP]: W/A/S/D/Q/E to move ball", fontsize=10)

    # ================= 逻辑控制 =================
    
    def toggle_mode(self, event):
        # 切换模式循环: MANUAL -> AUTO -> GRASP -> MANUAL
        self.current_mode_idx = (self.current_mode_idx + 1) % 3
        mode = self.MODES[self.current_mode_idx]
        
        # 更新UI外观
        self.btn_mode.label.set_text(f"Mode: {mode}")
        self.title_text.set_text(f"Mode: {mode}")
        
        if mode == 'MANUAL':
            self.title_text.set_color('blue')
            self.viz_target.set_color('gray')
            self.viz_target.set_alpha(0.3)
        elif mode == 'AUTO':
            self.title_text.set_color('purple')
            self.viz_target.set_color('gray')
            self.viz_target.set_alpha(0.3)
        elif mode == 'GRASP':
            self.title_text.set_color('green')
            self.viz_target.set_color('#32CD32') # 亮绿色
            self.viz_target.set_alpha(1.0)

    def on_key_press(self, event):
        # 仅在抓取模式下响应键盘
        if self.MODES[self.current_mode_idx] != 'GRASP':
            return
            
        step = 20.0 # 每次按键移动 20mm
        key = event.key.lower()
        
        if key == 'w': self.target_pos[0] += step # 前
        elif key == 's': self.target_pos[0] -= step # 后
        elif key == 'a': self.target_pos[1] -= step # 左
        elif key == 'd': self.target_pos[1] += step # 右
        elif key == 'q': self.target_pos[2] += step # 上
        elif key == 'e': self.target_pos[2] -= step # 下
        
        # 限制球不钻入地下
        self.target_pos[2] = max(0, self.target_pos[2])

    def on_slider_manual(self, val):
        # 只有在手动模式下，滑块的值才直接驱动机械臂
        if self.MODES[self.current_mode_idx] == 'MANUAL':
            # 读取滑块值构建向量
            vals = [s.val for s in self.sliders]
            # 补全向量: [q1, q2, q3, 0, bend, phi, len]
            self.current_q = vals[:3] + [0] + vals[3:]

    def update_sliders_visual(self, q_vector):
        """
        在自动或抓取模式下，反向更新滑块的显示位置，
        这样切换回手动模式时，滑块不会跳变。
        """
        # q_vector: [q1, q2, q3, 0, bend, phi, len]
        # Sliders:  [q1, q2, q3,    bend, phi, len]
        indices = [0, 1, 2, 4, 5, 6] # 对应滑块的索引
        
        for i, slider_idx in enumerate(indices):
            self.sliders[i].eventson = False # 暂时关闭回调，防止死循环
            self.sliders[i].set_val(q_vector[slider_idx])
            self.sliders[i].eventson = True

    # ================= 主循环 =================
    
    def update_loop(self, frame):
        mode = self.MODES[self.current_mode_idx]
        self.time_step += 0.05
        
        # --- 策略 1: 自动波形 ---
        if mode == 'AUTO':
            t = self.time_step
            # 生成正弦波动作
            q1 = np.sin(t*0.5) * 45
            q2 = np.sin(t*0.5 + 1) * 30 + 10
            q3 = np.sin(t*0.6 + 2) * 40 - 90
            bend = (np.sin(t) + 1) * 50
            phi = (t * 50) % 360 - 180
            length = 180 + np.sin(t*2) * 40
            
            self.current_q = [q1, q2, q3, 0, bend, phi, length]
            self.update_sliders_visual(self.current_q)
            
        # --- 策略 2: 抓取模式 (键盘控制球) ---
        elif mode == 'GRASP':
            # 调用 IK 求解器去追球
            self.current_q = self.ik_solver.solve(self.target_pos, self.current_q)
            self.update_sliders_visual(self.current_q)
            
        # --- 策略 3: 手动模式 ---
        # 手动模式下，数据已经在 on_slider_manual 中更新了，这里只需重绘
        
        # --- 统一绘图 ---
        r_pts, s_pts, _, _ = self.arm_model.forward_kinematics(self.current_q)
        
        # 1. 更新刚性臂
        self.viz_links.set_data(r_pts[0:4, 0], r_pts[0:4, 1])
        self.viz_links.set_3d_properties(r_pts[0:4, 2])
        
        self.viz_fixed.set_data(r_pts[3:5, 0], r_pts[3:5, 1])
        self.viz_fixed.set_3d_properties(r_pts[3:5, 2])
        
        self.viz_joints.set_data(r_pts[1:4, 0], r_pts[1:4, 1])
        self.viz_joints.set_3d_properties(r_pts[1:4, 2])
        
        # 2. 更新软体臂
        sx = np.concatenate(([r_pts[-1,0]], s_pts[:,0]))
        sy = np.concatenate(([r_pts[-1,1]], s_pts[:,1]))
        sz = np.concatenate(([r_pts[-1,2]], s_pts[:,2]))
        self.viz_soft.set_data(sx, sy)
        self.viz_soft.set_3d_properties(sz)
        
        # 3. 更新目标小球 (如果在抓取模式)
        if mode == 'GRASP':
            self.viz_target.set_data([self.target_pos[0]], [self.target_pos[1]])
            self.viz_target.set_3d_properties([self.target_pos[2]])
        
        return self.viz_links, self.viz_soft

if __name__ == "__main__":
    app = UnifiedRobotController()