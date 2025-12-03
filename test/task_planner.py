"""
文件功能：任务空间轨迹规划器

本文件定义了 `TaskPlanner` 类，用于在任务空间（即三维笛卡尔空间）中生成末端执行器的运动轨迹。

主要功能：
- `generate_trajectory`: 核心方法，接收起始点、终止点和中间点。
- `quintic_polynomial_traj`: 使用五次多项式插值，生成平滑的轨迹段。这可以确保位置、速度和加速度在路径点上都是连续的，从而使机器人运动更稳定。
- `generate_circle_traj`: 生成一个圆形轨迹，常用于演示和测试。

与关节空间轨迹规划不同，任务空间规划器关注的是末端执行器在空间中的路径，而具体的关节如何运动则由逆运动学求解器 (`ik_solver`) 负责。
"""

import numpy as np

class TargetObject:
    def __init__(self, position):
        self.initial_pos = np.array(position) # 初始位置
        self.current_pos = np.array(position) # 当前位置
        self.is_grasped = False
        self.color = '#32CD32' # LimeGreen (未抓取颜色)

class GraspTaskPlanner:
    def __init__(self):
        # 定义任务的状态枚举
        self.states = ['IDLE', 'APPROACH', 'GRASPING', 'LIFT', 'RESET']
        self.current_state_idx = 0
        self.progress = 0.0 # 当前状态的进度 0.0 ~ 1.0
        
        # 定义关键帧 (Joint Space Keyframes)
        # 格式: [q1, q2, q3, bend, phi, len]
        # 假设物体在前方 X=800, Z=200 左右的位置
        self.pose_home = np.array([0, 0, -90, 0, 0, 140])
        self.pose_ready = np.array([0, 30, -50, 0, 0, 180]) # 准备姿态
        self.pose_grasp = np.array([0, 45, -65, 120, 0, 220]) # 抓取姿态 (弯曲+伸长)
        self.pose_lift  = np.array([0, 10, -60, 120, 0, 220]) # 举起姿态 (保持弯曲)

        # 初始化物体
        # 根据 pose_grasp 手算的大概位置放置物体，确保能抓到
        self.target = TargetObject([1200, 0, 100]) 

    def update(self, dt):
        """
        根据时间推进状态机，返回当前的关节角度和物体位置
        """
        speed = 0.5 # 动作速度
        self.progress += dt * speed
        
        if self.progress >= 1.0:
            self.progress = 0.0
            self.current_state_idx = (self.current_state_idx + 1) % len(self.states)
            
        state = self.states[self.current_state_idx]
        t = self.progress
        
        # 使用简单的线性插值 (Lerp) 生成平滑轨迹
        # q_current = (1-t)*start + t*end
        
        current_q = self.pose_home.copy()
        
        if state == 'IDLE':
            # 保持 Home
            current_q = self.pose_home
            self.target.is_grasped = False
            self.target.current_pos = self.target.initial_pos.copy()
            self.target.color = '#32CD32'
            
        elif state == 'APPROACH':
            # Home -> Ready -> Grasp
            # 简化为直接从 Home 到 Grasp 的前半段 (Home -> Reach)
            # 这里我们做一个两段插值
            current_q = (1-t) * self.pose_home + t * self.pose_grasp
            # 在接近过程中，弯曲度强制为0，防止还没到就卷起来
            current_q[3] = 0 
            
        elif state == 'GRASPING':
            # 保持位置，执行弯曲动作
            # 实际上 Approach 已经到位了，这里单独演示“卷起”的过程
            base_pose = self.pose_grasp.copy()
            base_pose[3] = 0 # 开始是直的
            target_pose = self.pose_grasp.copy() # 结束是弯的
            
            current_q = (1-t) * base_pose + t * target_pose
            
            # 当弯曲超过一定程度，判定抓住了
            if t > 0.5:
                self.target.is_grasped = True
                self.target.color = '#FF4500' # OrangeRed (抓住变色)
                
        elif state == 'LIFT':
            # Grasp -> Lift
            current_q = (1-t) * self.pose_grasp + t * self.pose_lift
            
        elif state == 'RESET':
            # Lift -> Home
            current_q = (1-t) * self.pose_lift + t * self.pose_home
            self.target.is_grasped = False # 松开
            
            # 模拟物体掉落回原位(简化处理)
            if t > 0.2:
                self.target.current_pos = self.target.initial_pos
                self.target.color = '#32CD32'

        return current_q, self.target, state