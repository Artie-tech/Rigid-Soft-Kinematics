% 文件功能：针对液压臂的逆运动学求解函数（作为labview的接口）
%
% 本函数是一个高层封装，专门用于求解 HydraulicSoftArm 的逆运动学问题。
%
% 主要职责：
% 1. 接收一个目标末端位姿 (T_target)。
% 2. 创建 HydraulicSoftArm 对象和 RoboticIKSolver 对象。
% 3. 定义用于求解的匿名函数，如误差函数和雅可比计算函数，并将其传递给
%    通用求解器。
% 4. 调用求解器的 solve 方法，并处理返回结果。
% 5. 返回计算出的关节角度和最终误差。

%用来放入labview进行调用的函数
function [q_new, soft_tip_rel] = SolveHydraulicArmIK(target_pos, current_q)
    % SolveHydraulicArmIK - 软硬结合臂逆运动学核心算法
    %
    % 用于封装 DLL 的独立函数。
    %
    % 输入:
    %   target_pos: [3x1] 或 [1x3] double, 目标末端坐标 (x, y, z) mm
    %   current_q:  [7x1] 或 [1x7] double, 当前关节角度
    %               格式: [J1, J2, J3, Fixed(0), Bend, Phi, Len]
    %
    % 输出:
    %   q_new:        [1x7] double, 计算后的目标关节角度
    %   soft_tip_rel: [1x3] double, 软体末端相对于软体基座的坐标
    %                 (符合实机坐标系: Z轴为伸长方向)
    
    % --- 0. 数据预处理 ---
    % 强制转换为列向量进行内部计算
    target = double(target_pos(:));
    q_start = double(current_q(:));
    
    % --- 1. 参数定义 (原类属性) ---
    % 刚性关节限位 (Rad)
    limits_rad = [deg2rad(-60),  deg2rad(60);
                  deg2rad(-28),  deg2rad(90);
                  deg2rad(-152), deg2rad(-42)];
              
    % 软体参数
    soft_len_range = [140.0, 200.0];
    soft_bend_max = 100.0;
    tolerance = 2.0; % mm
    
    % --- 2. 求解流程 ---
    
    % [Step A] 刚性臂粗定位
    % 仅使用前3个关节作为种子
    q_rigid_seeds = q_start(1:3);
    q_rigid_opt = optimize_rigid_base(target, q_rigid_seeds, limits_rad, soft_len_range);
    
    % [Step B] 软体臂几何解算
    old_phi = q_start(6);
    [q_geo, geo_err] = solve_soft_geometric(target, q_rigid_opt, old_phi, soft_bend_max, soft_len_range);
    
    final_q = q_geo;
    final_err = geo_err;
    
    % [Step C] 全局微调 (Global Refine)
    % 如果几何解误差在 "可挽救范围" (2mm ~ 50mm)，尝试数值优化
    if (geo_err > 2.0) && (geo_err < 50.0)
        [q_ref, ref_err] = global_refine(target, q_geo, tolerance, limits_rad, soft_bend_max, soft_len_range);
        if ref_err < final_err
            final_q = q_ref;
        end
    end
    
    % --- 3. 输出格式化 ---
    % 计算软体末端相对坐标 (实机坐标系)
    % final_q 索引: 5=Bend, 6=Phi, 7=Len
    bend_res = final_q(5);
    phi_res  = final_q(6);
    len_res  = final_q(7);
    
    soft_tip_col = get_soft_tip_real(bend_res, phi_res, len_res);
    
    % 转换为行向量输出 (通用接口通常偏好行向量)
    q_new = final_q(:)';
    soft_tip_rel = soft_tip_col(:)';
    
end

% =========================================================
%  局部辅助函数 (Local Functions)
% =========================================================

function q_rigid_res = optimize_rigid_base(target, start_q_rad, limits, len_range)
    % 刚性基座粗定位优化
    q = deg2rad(start_q_rad(:));
    ideal_reach = mean(len_range);
    % ideal_reach = len_range(1) + 5.0;
    
    for iter = 1:15
        % 构造全关节向量 (软体长度设为 140 防止报错)
        q_full = [rad2deg(q); 0; 0; 0; 140];
        
        [~, T_base] = forward_kinematics(q_full);
        base_pos = T_base(1:3, 4);
        
        vec = target - base_pos;
        dist = norm(vec);
        
        if abs(dist - ideal_reach) < 5.0
            break;
        end
        
        % 期望位置
        direction = vec / (dist + 1e-6);
        desired_pos = target - direction * ideal_reach;
        err_vec = desired_pos - base_pos;
        
        % 数值雅可比
        J = zeros(3, 3);
        delta = 0.001;
        for i = 1:3
            q_tmp = q;
            q_tmp(i) = q_tmp(i) + delta;
            
            q_full_tmp = [rad2deg(q_tmp); 0; 0; 0; 140];
            [~, T_b_tmp] = forward_kinematics(q_full_tmp);
            J(:, i) = (T_b_tmp(1:3, 4) - base_pos) / delta;
        end
        
        % DLS 更新
        dq = (J' * J + 0.1 * eye(3)) \ (J' * err_vec); % 使用 \ 替代 inv 提高速度和精度
        q = q + dq;
        
        % 限位
        for k = 1:3
            q(k) = max(min(q(k), limits(k,2)), limits(k,1));
        end
    end
    q_rigid_res = rad2deg(q);
end

function [q_res, error] = solve_soft_geometric(target, rigid_q, old_phi, bend_max, len_range)
    % 软体几何解算
    q_dummy = [rigid_q(:); 0; 0; 0; 0];
    [~, T_base] = forward_kinematics(q_dummy);
    
    base_pos = T_base(1:3, 4);
    R_base = T_base(1:3, 1:3);
    
    vec_global = target - base_pos;
    vec_local = R_base' * vec_global;
    
    x = vec_local(1); 
    y = vec_local(2); 
    z = vec_local(3);
    
    h = sqrt(y^2 + z^2);
    
    % Phi
    if h < 1e-3
        phi_deg = old_phi;
    else
        phi_deg = rad2deg(atan2(z, y));
    end
    
    % Bend & Len
    if h < 1e-3
        theta_rad = 0;
        arc_len = x;
    else
        theta_rad = 2 * atan2(h, x);
        if abs(theta_rad) < 1e-4
            arc_len = x;
        else
            R = (x^2 + h^2)/(2*h);
            arc_len = R * theta_rad;
        end
    end
    
    % 约束
    final_bend = max(min(rad2deg(theta_rad), bend_max), -bend_max);
    final_len = max(min(arc_len, len_range(2)), len_range(1));
    
    q_res = [rigid_q(:); 0; final_bend; phi_deg; final_len];
    
    [T_tip, ~] = forward_kinematics(q_res);
    error = norm(target - T_tip(1:3, 4));
end

function [q_new, err_new] = global_refine(target, q_start, tol, limits, bend_max, len_range)
    % 全局数值微调 (Levenberg-Marquardt)
    q = q_start;
    active_idx = [1, 2, 3, 5, 6, 7];
    
    [T_tip, ~] = forward_kinematics(q);
    curr_pos = T_tip(1:3, 4);
    curr_err = norm(target - curr_pos);
    
    lambda = 0.01; % 初始阻尼因子
    
    for iter = 1:20
        if curr_err < tol
            break;
        end
        
        J = zeros(3, 6);
        delta_arr = [0.1; 0.1; 0.1; 0; 0.1; 0.1; 1.0];
        
        for k = 1:6
            idx = active_idx(k);
            q_tmp = q;
            delta = delta_arr(idx);
            q_tmp(idx) = q_tmp(idx) + delta;
            
            [T_tmp, ~] = forward_kinematics(q_tmp);
            J(:, k) = (T_tmp(1:3, 4) - curr_pos) / delta;
        end
        
        % LM 更新: (J'J + lambda*I) * dq = J' * err
        err_vec = target - curr_pos;
        H = J' * J;
        g = J' * err_vec;
        
        dq_active = (H + lambda * eye(6)) \ g;
        
        % 试探更新
        q_trial = q;
        for k = 1:6
            idx = active_idx(k);
            q_trial(idx) = q_trial(idx) + dq_active(k);
        end
        
        % 限位保护
        for i = 1:3
            q_trial(i) = max(min(q_trial(i), rad2deg(limits(i,2))), rad2deg(limits(i,1)));
        end
        q_trial(5) = max(min(q_trial(5), bend_max), -bend_max);
        q_trial(7) = max(min(q_trial(7), len_range(2)), len_range(1));
        
        % 检查误差是否减小
        [T_trial, ~] = forward_kinematics(q_trial);
        pos_trial = T_trial(1:3, 4);
        err_trial = norm(target - pos_trial);
        
        if err_trial < curr_err
            % 接受更新
            q = q_trial;
            curr_err = err_trial;
            curr_pos = pos_trial;
            lambda = lambda / 5.0; % 减小阻尼
        else
            % 拒绝更新
            lambda = lambda * 5.0; % 增大阻尼
            if lambda > 1e5
                break; % 阻尼过大，无法收敛
            end
        end
    end
    
    q_new = q;
    err_new = curr_err;
end

function [T_tip_global, T_base_soft] = forward_kinematics(q_degrees)
    % 正向运动学 (包含 DH 参数定义)
    
    % D-H 参数表 [alpha, a, d] (不含 theta)
    % Link 1: alpha=0, a=190, d=0
    % Link 2: alpha=90, a=90, d=0
    % Link 3: alpha=0, a=605, d=0
    % Link 4: alpha=0, a=290, d=0 (Fixed)
    DH_Table = [0,          190, 0;
                deg2rad(90), 90, 0;
                0,          605, 0;
                0,          290, 0];
    
    base_z_offset = 0.0;
    
    q_rad = deg2rad(q_degrees(1:3));
    
    T = eye(4);
    T(3, 4) = base_z_offset;
    
    for i = 1:4
        if i <= 3
            theta = q_rad(i);
        else
            theta = 0;
        end
        
        alpha = DH_Table(i, 1);
        a = DH_Table(i, 2);
        d = DH_Table(i, 3);
        
        ct = cos(theta); st = sin(theta);
        ca = cos(alpha); sa = sin(alpha);
        
        % MDH 变换矩阵 (Modified DH)
        Ti = [ct,    -st,    0,      a;
              st*ca, ct*ca, -sa,     -d*sa;
              st*sa, ct*sa,  ca,     d*ca;
              0,     0,      0,      1];
        T = T * Ti;
    end
    
    T_base_soft = T;
    
    % Soft Part
    bend = deg2rad(q_degrees(5));
    phi  = deg2rad(q_degrees(6));
    len  = q_degrees(7);
    
    T_local = eye(4);
    
    if abs(bend) < 1e-4
        T_local(1, 4) = len;
    else
        R = len / bend;
        x = R * sin(bend);
        y_raw = R * (1 - cos(bend));
        
        y = y_raw * cos(phi);
        z = y_raw * sin(phi);
        
        T_local(1:3, 4) = [x; y; z];
    end
    
    T_tip_global = T_base_soft * T_local;
end

function pos_real = get_soft_tip_real(bend_deg, phi_deg, len_mm)
    % 计算软体末端相对于软体基座的实机坐标
    theta = deg2rad(bend_deg);
    phi = deg2rad(phi_deg);
    
    % 1. 仿真坐标系 (X轴为伸长方向)
    if abs(theta) < 1e-4
        p_sim = [len_mm; 0; 0];
    else
        R = len_mm / theta;
        x_sim = R * sin(theta);
        y_plane = R * (1 - cos(theta));
        
        y_sim = y_plane * cos(phi);
        z_sim = y_plane * sin(phi);
        
        p_sim = [x_sim; y_sim; z_sim];
    end
    
    % 2. 转换到实机坐标系 (Z轴为伸长方向)
    % Sim X (伸长) -> Real Z
    % Sim Y (侧向) -> Real X
    % Sim Z (向上) -> Real Y
    % pos_real = [p_sim(2); p_sim(3); p_sim(1)];
    
    % 修正：根据用户要求，伸长方向为局部 Z 轴
    % 假设 p_sim(1) 是伸长量 (X)，现在要映射到 Z
    % 假设 p_sim(2) 是侧向 (Y)，现在映射到 X
    % 假设 p_sim(3) 是向上 (Z)，现在映射到 Y
    % 这样符合右手系旋转
    pos_real = [p_sim(2); p_sim(3); p_sim(1)];
end

