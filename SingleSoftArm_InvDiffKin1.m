function [l_dot, phi_dot, kappa_dot] = SingleSoftArm_InvDiffKin1(x_dot, y_dot, z_dot, x, y, z, eps_r)
% 单节恒定曲率软体臂的速度级逆运动学映射（含奇异性处理）
%
% 输入：
%   x, y, z         : 末端当前位置
%   x_dot, y_dot, z_dot : 末端速度
%   eps_r           :容差阈值（根据单位调整，例如 mm 或 m）
% 输出：
%   l_dot, phi_dot, kappa_dot : 形变参数变化率
%
% 奇异性处理：
%   当 r = sqrt(x^2 + y^2) < eps_r 时，视为沿 Z 轴运动，
%   此时 phi 无定义，设 phi_dot = 0，并对雅可比进行平滑处理。
%*************************************************************************

  

    % 计算投影半径
    r = sqrt(x^2 + y^2);

    % 初始化输出
    l_dot = 0;
    phi_dot = 0;
    kappa_dot = 0;

    % === 奇异性判断 ===
    if r < eps_r
        % 处于 Z 轴上（或极近），软体臂竖直
        % 此时：phi 无意义，phi_dot = 0
        % 曲率 kappa ≈ 0（直线状态），kappa_dot ≈ 0
        % 弧长 ℓ ≈ |z|，故 l_dot ≈ sign(z) * z_dot（若 z > 0）

        if z <= 0
            % 避免负高度或原点（物理上不合理）
            l_dot = 0;
        else
            l_dot = z_dot;  % 近似：ℓ ≈ z，所以 dℓ/dt ≈ dz/dt
        end

        phi_dot = 0;
        kappa_dot = 0;  % 直线状态，曲率为0，变化率也为0

    else
        % === 正常情况：计算雅可比 ===
        theta = atan(r / z);  % 半弯曲角

        % 预计算常用项
        r2 = r^2;
        r3 = r2 * r;
        z2 = z^2;
        denom_kappa = (r2 + z2)^2;  % (r² + z²)²

        % 初始化雅可比 J = d[l, phi, kappa]^T / d[x, y, z]
        J = zeros(3, 3);

        % 第一行：∂ℓ/∂x, ∂ℓ/∂y, ∂ℓ/∂z
        common_l = theta * (r2 + z2) / r3;
        J(1,1) = x*z / r2 + x * common_l;
        J(1,2) = y*z / r2 + y * common_l;
        J(1,3) = -1 + 2*z*theta / r;

        % 第二行：∂φ/∂x, ∂φ/∂y, ∂φ/∂z
        inv_r2 = 1 / r2;
        J(2,1) = -y * inv_r2;
        J(2,2) =  x * inv_r2;
        J(2,3) =  0;

        % 第三行：∂κ/∂x, ∂κ/∂y, ∂κ/∂z
        factor_k = 2 * (z2 - r2) / (r * denom_kappa);
        J(3,1) = x * factor_k;
        J(3,2) = y * factor_k;
        J(3,3) = -4 * r * z / denom_kappa;

        % 映射速度
        vel_cart = [x_dot; y_dot; z_dot];
        rates = J * vel_cart;

        l_dot      = rates(1);
        phi_dot    = rates(2);
        kappa_dot  = rates(3);
        
        l_dot = real(l_dot);
        phi_dot = real(phi_dot);
        kappa_dot = real(kappa_dot);
    end