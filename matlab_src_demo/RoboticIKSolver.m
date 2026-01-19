% 文件功能：通用机器人逆运动学求解器类
%
% 本文件定义了 RoboticIKSolver 类，实现了一个通用的、基于雅可比矩阵的
% 迭代逆运动学 (IK) 算法。
%
% 主要特点：
% - 适用于任何提供了正向运动学和雅可比计算的机器人模型。
% - 采用阻尼最小二乘法 (DLS) 或雅可比伪逆法来计算关节增量，以提高在
%   奇异点附近的稳定性。
% - 包含迭代次数、容忍误差、步长等可配置参数。
% - 核心方法是 solve，接收目标位姿和初始关节角，返回求解结果。

classdef RoboticIKSolver
    % RoboticIKSolver 逆运动学求解器
    
    properties
        Model
        LimitsRad % 刚性关节限位 [min, max] x 3
        SoftLenRange = [140.0, 200.0];
        SoftBendMax = 100.0;
        Tolerance = 2.0; % mm
    end
    
    methods
        function obj = RoboticIKSolver()
            obj.Model = HydraulicSoftArm();
            % 关节限位 (转为弧度)
            obj.LimitsRad = [deg2rad(-60), deg2rad(60);
                             deg2rad(-28), deg2rad(90);
                             deg2rad(-152), deg2rad(-42)];
        end
        
        function [final_q, soft_tip_rel] = solve(obj, target_pos, current_q)
            % 输入: target_pos [x,y,z], current_q [1x7]
            % 输出: final_q [1x7], soft_tip_rel [3x1]
            
            target = target_pos(:); % 确保列向量
            
            % 种子策略
            seeds = current_q(1:3); % 仅取刚性部分作为种子
            
            best_q = [];
            min_error = inf;
            
            % 尝试求解
            % 1. 刚性臂粗定位 (Optimize Rigid Base)
            q_rigid_opt = obj.optimize_rigid_base(target, seeds);
            
            % 2. 软体臂几何解算 (Geometric Solve)
            % 注意：传入 old_phi 以处理奇异点
            old_phi = current_q(6);
            [q_geo, geo_err] = obj.solve_soft_geometric(target, q_rigid_opt, old_phi);
            
            final_cand = q_geo;
            final_err = geo_err;
            
            % 3. 全局微调 (Global Refine) - 如果误差稍大
            if geo_err > 2.0 && geo_err < 50.0
                [q_ref, ref_err] = obj.global_refine(target, q_geo);
                if ref_err < final_err
                    final_cand = q_ref;
                    final_err = ref_err;
                end
            end
            
            % 最终输出处理
            final_q = final_cand;
            
            % 计算此时软体末端相对于基座的坐标 (用于输出)
            soft_tip_rel = obj.Model.get_soft_tip_in_base_frame(...
                final_q(5), final_q(6), final_q(7));
            
            % 简单打印调试
            % fprintf('IK Solver Error: %.2f mm\n', final_err);
        end
        
        function q_rigid_res = optimize_rigid_base(obj, target, start_q_rad)
            % 简单的梯度下降法，让软体基座靠近目标点的"最佳射程"位置
            q = deg2rad(start_q_rad);
            if size(q,1) < size(q,2), q = q'; end % 确保列向量 (3x1)
            
            ideal_reach = mean(obj.SoftLenRange);
                        
            for iter = 1:15
                % FK
                q_full = [rad2deg(q); 0; 0; 0; 140]; 
                
                [~, T_base] = obj.Model.forward_kinematics(q_full);
                base_pos = T_base(1:3, 4);
                
                vec = target - base_pos;
                dist = norm(vec);
                
                % 目标：让 dist 接近 ideal_reach
                if abs(dist - ideal_reach) < 5.0
                    break;
                end
                
                % 计算所需的基座位置
                direction = vec / (dist + 1e-6);
                desired_pos = target - direction * ideal_reach;
                
                err_vec = desired_pos - base_pos;
                
                % 数值雅可比 (3x3)
                J = zeros(3, 3);
                delta = 0.001;
                for i = 1:3
                    q_tmp = q;
                    q_tmp(i) = q_tmp(i) + delta;
                    
                    q_full_tmp = [rad2deg(q_tmp); 0; 0; 0; 140];
                    
                    [~, T_b_tmp] = obj.Model.forward_kinematics(q_full_tmp);
                    J(:, i) = (T_b_tmp(1:3, 4) - base_pos) / delta;
                end
                
                % DLS 更新
                dq = pinv(J' * J + 0.1 * eye(3)) * J' * err_vec;
                q = q + dq;
                
                % 限位
                for k = 1:3
                    q(k) = max(min(q(k), obj.LimitsRad(k,2)), obj.LimitsRad(k,1));
                end
            end
            
            % 输出时转回行向量，方便后续水平拼接
            q_rigid_res = rad2deg(q');
        end
        
        function [q_res, error] = solve_soft_geometric(obj, target, rigid_q, old_phi)
            % 1. 获取软体基座坐标系
            % rigid_q 是行向量，这里使用 horzcat (,)
            q_dummy = [rigid_q, 0, 0, 0, 0];
            [~, T_base] = obj.Model.forward_kinematics(q_dummy);
            
            base_pos = T_base(1:3, 4);
            R_base = T_base(1:3, 1:3);
            
            % 2. 转换目标点到局部
            vec_global = target - base_pos;
            vec_local = R_base' * vec_global; % 旋转矩阵转置即逆
            
            x = vec_local(1); 
            y = vec_local(2); 
            z = vec_local(3);
            
            % 3. 几何解算
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
            
            % 4. 约束
            final_bend = max(min(rad2deg(theta_rad), obj.SoftBendMax), -obj.SoftBendMax);
            final_len = max(min(arc_len, obj.SoftLenRange(2)), obj.SoftLenRange(1));
            
            % 5. 结果
            q_res = [rigid_q, 0, final_bend, phi_deg, final_len];
            
            % 验算误差
            [T_tip, ~] = obj.Model.forward_kinematics(q_res);
            error = norm(target - T_tip(1:3, 4));
        end
        
        function [q_new, err_new] = global_refine(obj, target, q_start)
            % 全局雅可比迭代微调 (Levenberg-Marquardt)
            q = q_start; 
            % 优化索引: 1,2,3 (Rigid), 5(Bend), 6(Phi), 7(Len)
            active_idx = [1, 2, 3, 5, 6, 7];
            
            [T_tip, ~] = obj.Model.forward_kinematics(q);
            curr_pos = T_tip(1:3, 4);
            curr_err = norm(target - curr_pos);
            
            lambda = 0.01; % 初始阻尼因子
            
            for iter = 1:20
                if curr_err < obj.Tolerance
                    break;
                end
                
                % 雅可比 3x6
                J = zeros(3, 6);
                delta_arr = [0.1, 0.1, 0.1, 0, 0.1, 0.1, 1.0]; % 步长
                
                for k = 1:6
                    idx = active_idx(k);
                    q_tmp = q;
                    delta = delta_arr(idx);
                    q_tmp(idx) = q_tmp(idx) + delta;
                    
                    [T_tmp, ~] = obj.Model.forward_kinematics(q_tmp);
                    J(:, k) = (T_tmp(1:3, 4) - curr_pos) / delta;
                end
                
                % LM 更新: (J'J + lambda*I) * dq = J' * err
                err_vec = target - curr_pos;
                H = J' * J;
                g = J' * err_vec;
                
                % 针对不同关节量级，可以加权，这里简化处理
                dq_active = (H + lambda * eye(6)) \ g;
                
                % 试探更新
                q_trial = q;
                for k = 1:6
                    idx = active_idx(k);
                    q_trial(idx) = q_trial(idx) + dq_active(k);
                end
                
                % 限位保护
                for i = 1:3
                    q_trial(i) = max(min(q_trial(i), rad2deg(obj.LimitsRad(i,2))), rad2deg(obj.LimitsRad(i,1)));
                end
                q_trial(5) = max(min(q_trial(5), obj.SoftBendMax), -obj.SoftBendMax);
                q_trial(7) = max(min(q_trial(7), obj.SoftLenRange(2)), obj.SoftLenRange(1));
                
                % 检查误差是否减小
                [T_trial, ~] = obj.Model.forward_kinematics(q_trial);
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
    end
end