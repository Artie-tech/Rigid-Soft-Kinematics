% 文件功能：液压刚柔耦合机械臂运动学模型
%
% 本文件定义了 HydraulicSoftArm 类，用于描述一个包含刚性连杆和末端软体执行器的
% 机械臂。
%
% 主要功能：
% 1. 定义机械臂的结构参数，包括刚性部分的 D-H (Denavit-Hartenberg) 参数和
%    软体部分的几何参数。
% 2. 实现正向运动学 (forwardKinematics)，根据输入的关节角度（刚性+软体），
%    计算出机械臂各关节点的位置以及末端执行器的位姿。
% 3. 采用分段常曲率 (PCC) 模型来描述软体部分的形状。

classdef HydraulicSoftArm
    % HydraulicSoftArm 机器人运动学模型
    % 包含刚性臂 D-H 参数和软体臂 PCC 模型
    
    properties
        BaseZOffset = 0.0; % mm
        
        % D-H 参数结构体数组 (Link 1-4)
        % theta 在运行时输入，这里存储 alpha, a, d
        RigidDH
        
        % 软体段数 (用于绘图精度，不影响几何解算)
        SoftSegments = 20;
    end
    
    methods
        function obj = HydraulicSoftArm()
            % 初始化 D-H 参数 (alpha, a, d)
            % 注意：MATLAB 中角度使用弧度
            obj.RigidDH = struct('alpha', {}, 'a', {}, 'd', {});
            
            % Link 1
            obj.RigidDH(1).alpha = 0; 
            obj.RigidDH(1).a = 190; 
            obj.RigidDH(1).d = 0;
            
            % Link 2
            obj.RigidDH(2).alpha = deg2rad(90); 
            obj.RigidDH(2).a = 90; 
            obj.RigidDH(2).d = 0;
            
            % Link 3
            obj.RigidDH(3).alpha = 0; 
            obj.RigidDH(3).a = 605; 
            obj.RigidDH(3).d = 0;
            
            % Link 4 (固定延长段)
            obj.RigidDH(4).alpha = 0; 
            obj.RigidDH(4).a = 290; 
            obj.RigidDH(4).d = 0;
        end
        
        function T = mdh_matrix(~, alpha, a, theta, d)
            % 标准 MDH 变换矩阵
            ct = cos(theta);
            st = sin(theta);
            ca = cos(alpha);
            sa = sin(alpha);
            
            T = [ct,    -st,    0,      a;
                 st*ca, ct*ca, -sa,     -d*sa;
                 st*sa, ct*sa,  ca,     d*ca;
                 0,     0,      0,      1];
        end
        
        function [T_tip_global, T_base_soft] = forward_kinematics(obj, q_degrees)
            % 输入: 1x7 向量 [q1, q2, q3, q_fixed(0), bend, phi, len] (角度/mm)
            % 输出: T_tip_global (4x4 末端位姿), T_base_soft (4x4 软体基座位姿)
            
            % 1. 提取参数
            q_rigid_rad = deg2rad(q_degrees(1:3));
            theta_bend = deg2rad(q_degrees(5)); % Index 5
            phi_dir = deg2rad(q_degrees(6));    % Index 6
            length_mm = q_degrees(7);           % Index 7
            
            % 2. 刚性部分计算
            T_cum = eye(4);
            T_cum(3, 4) = obj.BaseZOffset;
            
            for i = 1:4
                if i <= 3
                    theta = q_rigid_rad(i);
                else
                    theta = 0; % 第4关节固定
                end
                
                T_i = obj.mdh_matrix(obj.RigidDH(i).alpha, ...
                                     obj.RigidDH(i).a, ...
                                     theta, ...
                                     obj.RigidDH(i).d);
                T_cum = T_cum * T_i;
            end
            
            T_base_soft = T_cum;
            
            % 3. 软体部分计算 (PCC)
            % 获取软体末端相对于软体基座的局部变换 (Sim Frame)
            T_tip_local = eye(4);
            
            if abs(theta_bend) < 1e-4
                % 直线
                T_tip_local(1, 4) = length_mm; % X轴伸长
            else
                % 弯曲
                R = length_mm / theta_bend;
                
                % 几何坐标 (Sim Frame: X伸长, YZ弯曲)
                x = R * sin(theta_bend);
                y_raw = R * (1 - cos(theta_bend));
                
                y = y_raw * cos(phi_dir);
                z = y_raw * sin(phi_dir);
                
                % 这里简化处理，FK只计算位置，若需要旋转矩阵需完整推导Frenet标架
                % 既然主要是要位置，这里只填平移部分
                T_tip_local(1:3, 4) = [x; y; z];
                
                % 若需要末端姿态(旋转)，此处需补充旋转计算，
                % 但目前IK主要基于位置，且篇幅有限，暂略去软体末端旋转矩阵构建
            end
            
            T_tip_global = T_base_soft * T_tip_local;
        end
        
        function pos_real = get_soft_tip_in_base_frame(~, bend_deg, phi_deg, len_mm)
            % 计算软体末端相对于软体基座的坐标
            % 输出: [x; y; z] (实机坐标系：Z轴为伸长方向)
            
            theta = deg2rad(bend_deg);
            phi = deg2rad(phi_deg);
            
            % --- 1. 仿真坐标系 (X轴为伸长方向) ---
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
            
            % --- 2. 转换到实机坐标系 (Z轴为伸长方向) ---
            % Sim X (伸长) -> Real Z
            % Sim Y (侧向) -> Real X
            % Sim Z (向上) -> Real Y
            
            pos_real = [p_sim(2); p_sim(3); p_sim(1)];
        end
    end
end