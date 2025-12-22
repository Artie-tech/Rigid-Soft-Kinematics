% test_kinematics.m
% 运动学求解正确性测试脚本
%
% 功能：
% 1. 定义多组测试关节角（Ground Truth）。
% 2. 使用正向运动学 (FK) 计算对应的末端位置。
% 3. 将该位置作为目标，输入逆运动学 (IK) 求解器。
% 4. 验证 IK 求解得到的关节角是否能让末端回到目标位置（计算位置误差）。
% 5. 统计并输出测试结果。

clc; clear; close all;

fprintf('========================================\n');
fprintf('      液压柔性机械臂运动学测试脚本      \n');
fprintf('========================================\n');

% 1. 初始化
solver = RoboticIKSolver();
model = solver.Model;

% 检查 BaseZOffset 是否已修正
if model.BaseZOffset ~= 0
    warning('检测到 BaseZOffset 不为 0 (当前为 %.2f)。请确认是否符合预期。', model.BaseZOffset);
else
    fprintf('BaseZOffset 已确认为 0.0 mm (基座为原点)。\n');
end

% 2. 定义测试用例
% 格式: [q1, q2, q3, fixed(0), bend, phi, len]
% 注意范围:
% Rigid: [-60,60], [-28,90], [-152,-42]
% Soft: Bend [0,100], Len [140,200]
test_cases = [
    % Case 1: 初始位姿
    0, 0, -90, 0, 0, 0, 140;
    
    % Case 2: 仅刚性臂运动
    30, 45, -60, 0, 0, 0, 140;
    
    % Case 3: 刚性臂 + 软体臂伸长
    -30, 60, -100, 0, 0, 0, 180;
    
    % Case 4: 刚性臂 + 软体臂弯曲 (平面内)
    0, 30, -90, 0, 60, 0, 160;
    
    % Case 5: 刚性臂 + 软体臂弯曲 + 旋转 (空间弯曲)
    45, 10, -50, 0, 80, 90, 190;
    
    % Case 6: 复杂组合
    -45, 80, -140, 0, 50, -45, 170;
];

num_cases = size(test_cases, 1);
total_error = 0;
max_error = 0;

fprintf('\n开始测试 %d 个用例...\n', num_cases);

for i = 1:num_cases
    q_gt = test_cases(i, :);
    
    % 2.1 正向运动学 (Ground Truth)
    [T_gt, ~] = model.forward_kinematics(q_gt);
    pos_gt = T_gt(1:3, 4);
    % 计算 GT 的软体相对坐标
    soft_tip_gt = model.get_soft_tip_in_base_frame(q_gt(5), q_gt(6), q_gt(7));
    
    fprintf('\n----------------------------------------\n');
    fprintf('测试用例 %d:\n', i);
    fprintf('  设定关节角: [%.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f]\n', q_gt);
    fprintf('  目标位置:   [%.2f, %.2f, %.2f]\n', pos_gt);
    fprintf('  软体相对位置(GT): [%.2f, %.2f, %.2f]\n', soft_tip_gt);
    
    % 2.2 逆运动学求解
    % 使用一个通用的初始猜测，或者上一次的结果（这里为了独立测试，使用固定初始值）
    q_init = [0, 60, -90, 0, 0, 0, 180]; 
    
    t_start = tic;
    [q_sol, soft_tip_sol] = solver.solve(pos_gt, q_init);
    t_cost = toc(t_start);
    
    % 2.3 验证求解结果 (FK)
    [T_sol, ~] = model.forward_kinematics(q_sol);
    pos_sol = T_sol(1:3, 4);
    
    % 2.4 计算误差
    err = norm(pos_sol - pos_gt);
    total_error = total_error + err;
    if err > max_error
        max_error = err;
    end
    
    fprintf('  求解耗时:   %.4f 秒\n', t_cost);
    fprintf('  求解关节角: [%.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f]\n', q_sol);
    fprintf('  -> 软体状态: Bend=%.1f°, Phi=%.1f°, Len=%.1f mm\n', q_sol(5), q_sol(6), q_sol(7));
    fprintf('  求解位置:   [%.2f, %.2f, %.2f]\n', pos_sol);
    fprintf('  软体相对位置(Sol): [%.2f, %.2f, %.2f]\n', soft_tip_sol);
    fprintf('  位置误差:   %.4f mm\n', err);
    
    if err < 5.0
        fprintf('  结果: PASS\n');
    else
        fprintf('  结果: WARNING (误差 > 5mm)\n');
        
        % 检查关节限位
        % 定义限位 [min, max]
        % q1, q2, q3, q4, bend, phi, len
        limits = [
            -60, 60;    % q1
            -28, 90;    % q2
            -152, -42;  % q3
            -inf, inf;  % q4 (fixed)
            0, 100;     % bend (SoftBendMax)
            -inf, inf;  % phi
            140, 200    % len (SoftLenRange)
        ];
        
        joint_names = {'q1', 'q2', 'q3', 'q4', 'bend', 'phi', 'len'};
        
        fprintf('  [限位检查]:\n');
        is_limit_reached = false;
        for j = 1:7
            val = q_sol(j);
            min_val = limits(j, 1);
            max_val = limits(j, 2);
            
            % 检查是否接近下限 (容差 1.0)
            if val <= min_val + 1.0
                fprintf('    - %s 达到下限 (Val: %.1f, Limit: %.1f)\n', joint_names{j}, val, min_val);
                is_limit_reached = true;
            end
            
            % 检查是否接近上限 (容差 1.0)
            if val >= max_val - 1.0
                fprintf('    - %s 达到上限 (Val: %.1f, Limit: %.1f)\n', joint_names{j}, val, max_val);
                is_limit_reached = true;
            end
        end
        
        if ~is_limit_reached
            fprintf('    未检测到关节达到限位。\n');
        end
    end
end

fprintf('\n========================================\n');
fprintf('测试总结:\n');
fprintf('  平均误差: %.4f mm\n', total_error / num_cases);
fprintf('  最大误差: %.4f mm\n', max_error);
fprintf('========================================\n');
