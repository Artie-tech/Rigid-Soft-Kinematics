% main_demo.m
% 机器人逆运动学求解演示
clc; clear; close all;

% 1. 初始化求解器
ik_solver = RoboticIKSolver();

% 2. 定义输入：末端目标坐标 (x, y, z) mm
target_pos = [1000; 100; 300]; 

% 3. 定义当前状态 (用于作为优化的初始种子，可以设为全0或上一次的状态)
% 格式: [q1, q2, q3, fixed, bend, phi, len]
current_q = [0, 60, -90, 0, 0, 0, 180]; 

fprintf('=== 开始求解 ===\n');
fprintf('目标坐标: [%.1f, %.1f, %.1f]\n', target_pos(1), target_pos(2), target_pos(3));

% 4. 调用求解
% 返回: 
%   q_result: 完整的关节向量
%   soft_tip_local: 软体末端相对于基座的坐标 [x_side, y_up, z_fwd]
[q_result, soft_tip_local] = ik_solver.solve(target_pos, current_q);

% 5. 提取并显示结果 (符合你的要求)
q_rigid = q_result(1:3);   % 刚性关节角度
soft_params = q_result(5:7); % [Bend, Phi, Len]

fprintf('\n=== 求解结果 ===\n');
fprintf('1. 刚性机械臂关节角度 (q1, q2, q3):\n');
fprintf('   J1 (Base):     %.2f 度\n', q_rigid(1));
fprintf('   J2 (Shoulder): %.2f 度\n', q_rigid(2));
fprintf('   J3 (Elbow):    %.2f 度\n', q_rigid(3));

fprintf('\n2. 软体臂控制参数:\n');
fprintf('   Bend: %.2f 度\n', soft_params(1));
fprintf('   Phi:  %.2f 度\n', soft_params(2));
fprintf('   Len:  %.2f mm\n', soft_params(3));

fprintf('\n3. 软体末端相对于软体基座的坐标 (实机坐标系):\n');
fprintf('   (注意：Z轴为伸长方向)\n');
fprintf('   X (Side): %.2f mm\n', soft_tip_local(1));
fprintf('   Y (Up):   %.2f mm\n', soft_tip_local(2));
fprintf('   Z (Fwd):  %.2f mm\n', soft_tip_local(3));

% 6. 验证 (可选)：计算正向运动学看看是否对得上
[T_final, ~] = ik_solver.Model.forward_kinematics(q_result);
pos_final = T_final(1:3, 4);
err = norm(pos_final - target_pos);
fprintf('\n------------------\n');
fprintf('FK 验证误差: %.4f mm\n', err);