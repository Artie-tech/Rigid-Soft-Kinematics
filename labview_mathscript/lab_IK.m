function [q_new, soft_tip_rel] = lab_IK(target_pos, current_q)
    % =========================================================
    % LabVIEW Optimized Hydraulic Arm IK Solver (All-in-One)
    % Modules:
    % 1. Rigid Arm Coarse Positioning
    % 2. Soft Arm Geometric Solve + Anti-Flip Logic
    % 3. Global Numerical Refinement
    % =========================================================
    
    % --- 0. Input Pre-processing ---
    target = double(target_pos(:)); % Target [x;y;z]
    q_last = double(current_q(:));  % Previous joint states [7x1]
    
    % --- 1. Parameter Definitions ---
    % Rigid joint limits (Rad)
    lim_rigid_min = [deg2rad(-60); deg2rad(-28); deg2rad(-152)];
    lim_rigid_max = [deg2rad(60);  deg2rad(90);  deg2rad(-42)];
    
    % Soft arm parameters
    soft_len_min  = 140.0;
    soft_len_max  = 180.0;
    soft_bend_max = 100.0; % Degrees
    tolerance     = 1.0;   % mm (Convergence threshold)
    
    % --- 2. Phase 1: Rigid Arm Coarse Positioning ---
    % Strategy: Treat soft arm as a fixed-length rod, adjust rigid joints to aim base at target
    
    q_rigid_rad = deg2rad(q_last(1:3));
    ideal_reach = soft_len_min + 5.0; % Set an ideal extension length
    
    for k = 1:15
        % Calculate current rigid end (with virtual rod)
        % Manual FK call for performance
        [base_pos, ~] = internal_fk(q_rigid_rad, 0, 0, ideal_reach);
        
        vec = target - base_pos;
        dist = norm(vec);
        if dist < 5.0, break; end % Stop if close enough
        
        % Numerical Jacobian (3x3) - for the first three joints only
        J = zeros(3, 3);
        delta = 0.001;
        for i = 1:3
            q_tmp = q_rigid_rad;
            q_tmp(i) = q_tmp(i) + delta;
            [pos_tmp, ~] = internal_fk(q_tmp, 0, 0, ideal_reach);
            J(:, i) = (pos_tmp - base_pos) / delta;
        end
        
        % DLS Iteration
        dq = (J' * J + 0.1 * eye(3)) \ (J' * vec);
        q_rigid_rad = q_rigid_rad + dq;
        
        % Enforce rigid limits
        for i = 1:3
            q_rigid_rad(i) = max(min(q_rigid_rad(i), lim_rigid_max(i)), lim_rigid_min(i));
        end
    end
    q_rigid_opt = rad2deg(q_rigid_rad);
    
    % --- 3. Phase 2: Soft Arm Geometric Solve + Anti-Flip Logic ---
    
    % Get accurate position and rotation matrix of the rigid base
    [base_pos_abs, T_base_mat] = internal_fk(q_rigid_rad, 0, 0, 0); 
    R_base = T_base_mat(1:3, 1:3);
    
    % Transform target point to soft base local coordinate system
    vec_global = target - base_pos_abs;
    vec_local = R_base' * vec_global;
    
    loc_x = vec_local(1); 
    loc_y = vec_local(2); 
    loc_z = vec_local(3);
    h = sqrt(loc_y^2 + loc_z^2);
    
    % 3.1 Basic geometric calculation
    % Default Phi calculation
    phi_raw = rad2deg(atan2(loc_z, loc_y)); 
    theta_raw = 0;
    arc_len = loc_x;
    
    if h > 1e-3
        theta_rad_calc = 2 * atan2(h, loc_x);
        % Prevent division by zero
        if abs(theta_rad_calc) > 1e-4
            R_curve = (loc_x^2 + h^2)/(2*h);
            arc_len = R_curve * theta_rad_calc;
            theta_raw = rad2deg(theta_rad_calc);
        end
    end
    
    % 3.2 Key logic: Dual-solution optimization (Anti-Flip)
    % Soft arm bending is symmetrical: (30 deg, Phi 0) and (-30 deg, Phi 180) are physically identical.
    % We choose the solution closer to the previous state to prevent jumps.
    
    % Candidate 1: Positive bend
    cand1_bend = theta_raw;
    cand1_phi  = phi_raw;
    
    % Candidate 2: Negative bend (Phi flipped by 180 degrees)
    cand2_bend = -theta_raw;
    if phi_raw > 0
        cand2_phi = phi_raw - 180;
    else
        cand2_phi = phi_raw + 180;
    end
    
    % Get previous state
    last_bend = q_last(5);
    last_phi  = q_last(6);
    
    % Calculate circular distance for Phi (handle -179 to +179 wrap-around)
    d_phi_1 = abs(cand1_phi - last_phi);
    if d_phi_1 > 180, d_phi_1 = 360 - d_phi_1; end
    
    d_phi_2 = abs(cand2_phi - last_phi);
    if d_phi_2 > 180, d_phi_2 = 360 - d_phi_2; end
    
    % Combined distance (Bend difference + Phi difference)
    dist1 = abs(cand1_bend - last_bend) + d_phi_1;
    dist2 = abs(cand2_bend - last_bend) + d_phi_2;
    
    if dist2 < dist1
        final_bend = cand2_bend;
        final_phi  = cand2_phi;
    else
        final_bend = cand1_bend;
        final_phi  = cand1_phi;
    end
    
    % Enforce soft limits
    final_bend = max(min(final_bend, soft_bend_max), -soft_bend_max);
    final_len  = max(min(arc_len, soft_len_max), soft_len_min);
    
    % Combine current solution
    q_current = [q_rigid_opt; 0; final_bend; final_phi; final_len];
    
    % Verify error
    [check_pos, ~] = internal_fk(deg2rad(q_rigid_opt), final_bend, final_phi, final_len);
    geo_err = norm(target - check_pos);
    
    
    % --- 4. Phase 3: Global Refinement ---
    % If geometric error is moderate, use Jacobian iteration for refinement
    
    if (geo_err > 0.5) && (geo_err < 50.0)
        
        active_idx = [1; 2; 3; 5; 6; 7]; % Indices for optimization
        % Damping coefficients: higher damping for soft parts to prevent oscillation
        W_damping = diag([0.005, 0.005, 0.005, 20.0, 20.0, 20.0]);
        
        for iter = 1:10
            % Get current position
            q_rig_rad = deg2rad(q_current(1:3));
            [curr_pos, ~] = internal_fk(q_rig_rad, q_current(5), q_current(6), q_current(7));
            
            err_vec = target - curr_pos;
            if norm(err_vec) < tolerance, break; end
            
            % Calculate 6DOF Jacobian
            J = zeros(3, 6);
            % Step sizes for different joints
            deltas = [0.1; 0.1; 0.1; 0; 0.1; 0.1; 1.0]; 
            
            for k = 1:6
                idx = active_idx(k);
                q_tmp = q_current;
                d_val = deltas(idx);
                q_tmp(idx) = q_tmp(idx) + d_val;
                
                [p_tmp, ~] = internal_fk(deg2rad(q_tmp(1:3)), q_tmp(5), q_tmp(6), q_tmp(7));
                J(:, k) = (p_tmp - curr_pos) / d_val;
            end
            
            % Iterative update
            dq = (J' * J + W_damping) \ (J' * err_vec);
            
            for k = 1:6
                idx = active_idx(k);
                q_current(idx) = q_current(idx) + dq(k);
            end
            
            % Global limit enforcement
            for i = 1:3
                q_current(i) = max(min(q_current(i), rad2deg(lim_rigid_max(i))), rad2deg(lim_rigid_min(i)));
            end
            q_current(5) = max(min(q_current(5), soft_bend_max), -soft_bend_max);
            q_current(7) = max(min(q_current(7), soft_len_max), soft_len_min);
        end
    end
    
    % --- 5. Output Formatting ---
    final_q = q_current(:)'; % Convert to row vector
    q_new = final_q;
    
    % Calculate relative coordinates of soft tip vs soft base for display/control
    % Using corrected geometric mapping
    th_r = deg2rad(final_q(5));
    ph_r = deg2rad(final_q(6));
    len_v = final_q(7);
    
    if abs(th_r) < 1e-4
        x_real = 0; y_real = 0; z_real = len_v;
    else
        R_v = len_v / th_r;
        fwd = R_v * sin(th_r);
        side = R_v * (1 - cos(th_r));
        
        x_real = side * cos(ph_r);
        y_real = side * sin(ph_r);
        z_real = fwd;
    end
    soft_tip_rel = [x_real, y_real, z_real];
end