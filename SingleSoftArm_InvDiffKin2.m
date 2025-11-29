function [l1_dot, l2_dot, l3_dot] = SingleSoftArm_InvDiffKin2(l_dot, phi_dot, kappa_dot, l, phi, kappa,d)

J2= NaN(3, 3);
phi1 = pi/2 - phi;
phi2 = 7*pi/6 - phi;
phi3 = 11*pi/6 - phi;

J2(1,1) = 1 - kappa*d*cos(phi1); %l1-l
J2(1,2) = -l*kappa*d*sin(phi1); %l1-phi
J2(1,3) = -l*d*cos(phi1); %l1-kappa

J2(2,1) = 1 - kappa*d*cos(phi2);
J2(2,2) = -l*kappa*d*sin(phi2);
J2(2,3) = -l*d*cos(phi2);

J2(3,1) = 1 - kappa*d*cos(phi3);
J2(3,2) = -l*kappa*d*sin(phi3);
J2(3,3) = -l*d*cos(phi3);

inputVec = [l_dot; phi_dot; kappa_dot];
resultVec = J2 * inputVec;
l1_dot = resultVec(1);
l2_dot = resultVec(2);
l3_dot = resultVec(3);

l1_dot = real(l1_dot);
l2_dot = real(l2_dot);
l3_dot = real(l3_dot);
