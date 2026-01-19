% test_runner.m
% 文件功能：IK 算法验证与限位测试运行器 (对齐 test_kinematics.m)
%
% 改进说明：
% 1. 采用与 test_kinematics.m 相同的“真值生成 -> 逆解验证”流程。
% 2. 确保 BaseZOffset 为 0。
% 3. 验证 SolveHydraulicArmIK 的独立运行能力。

clc; clear; format compact;

fprintf('=== 液压软体臂 IK 综合测试 (LabView 接口版) ===\n');
fprintf('当前时间: %s\n\n', datetime("now", "Format", "yyyy-MM-dd HH:mm:ss"));

% --- 1. 定义测试用例 (Ground Truth) ---
% 格式: [q1, q2, q3, fixed(0), bend, phi, len]
% 这些用例与 test_kinematics.m 保持一致
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

% 初始猜测 (通用)
q_init = [0, 60, -90, 0, 0, 0, 180]; 

for i = 1:num_cases
    q_gt = test_cases(i, :);
    
    % 1.1 计算真值位置 (使用本地 FK 函数)
    pos_gt = local_forward_kinematics(q_gt);
    
    fprintf('------------------------------------------------\n');
    fprintf('【Case %d】\n', i);
    fprintf('  设定关节: [%.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f]\n', q_gt);
    fprintf('  目标位置: [%.2f, %.2f, %.2f]\n', pos_gt);
    
    try
        tic;
        % 1.2 调用待测函数
        [q_sol, soft_tip_rel] = SolveHydraulicArmIK(pos_gt, q_init);
        time_cost = toc;
        
        % 1.3 验算结果
        pos_sol = local_forward_kinematics(q_sol);
        err_dist = norm(pos_sol - pos_gt); % 修正：pos_gt 已经是列向量，无需转置
        
        total_error = total_error + err_dist;
        if err_dist > max_error
            max_error = err_dist;
        end
        
        % 1.4 输出信息
        fprintf('  > 计算耗时 : %.4f s\n', time_cost);
        fprintf('  > 求解关节 : [%.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f]\n', q_sol);
        fprintf('  -> 软体状态: Bend=%.1f°, Phi=%.1f°, Len=%.1f mm\n', q_sol(5), q_sol(6), q_sol(7));
        fprintf('  > 软体相对 : [%.2f, %.2f, %.2f] (Z伸长)\n', soft_tip_rel);
        fprintf('  > 最终误差 : %.4f mm ', err_dist);
        
        if err_dist < 5.0
            fprintf('(PASS)\n');
        else
            fprintf('(WARNING)\n');
        end
        
    catch ME
        fprintf(2, '  ERROR: 求解过程发生错误 -> %s\n', ME.message);
        fprintf(2, '  %s\n', ME.stack(1).name);
        fprintf(2, '  Line: %d\n', ME.stack(1).line);
    end
end

fprintf('================================================\n');
fprintf('测试总结:\n');
fprintf('  平均误差: %.4f mm\n', total_error / num_cases);
fprintf('  最大误差: %.4f mm\n', max_error);
fprintf('================================================\n');


%% --- 本地辅助函数：正运动学 (FK) ---
% 必须与 SolveHydraulicArmIK 内部逻辑完全一致
function pos_global = local_forward_kinematics(q_degrees)
    % DH 参数
    DH = [0,          190, 0;
          deg2rad(90), 90, 0;
          0,          605, 0;
          0,          290, 0];
    
    % 【关键】BaseZOffset 已修正为 0
    Base_Z = 0.0;
    
    q_rad = deg2rad(q_degrees(1:3));
    T = eye(4); T(3,4) = Base_Z;
    
    % Rigid Loop
    for i = 1:4
        if i <= 3
            th = q_rad(i);
        else
            th = 0;
        end
        alp = DH(i,1); a = DH(i,2); d = DH(i,3);
        
        % MDH 变换矩阵 (Modified DH)
        M = [cos(th), -sin(th), 0, a;
             sin(th)*cos(alp), cos(th)*cos(alp), -sin(alp), -d*sin(alp);
             sin(th)*sin(alp), cos(th)*sin(alp), cos(alp), d*cos(alp);
             0, 0, 0, 1];
        T = T * M;
    end
    
    % Soft Part
    bend = deg2rad(q_degrees(5)); 
    phi = deg2rad(q_degrees(6)); 
    len = q_degrees(7);
    
    if abs(bend) < 1e-4
        p_loc = [len; 0; 0];
    else
        R = len/bend;
        y_r = R*(1-cos(bend));
        p_loc = [R*sin(bend); y_r*cos(phi); y_r*sin(phi)];
    end
    
    pos_global = T(1:3, 1:3)*p_loc + T(1:3, 4);
end