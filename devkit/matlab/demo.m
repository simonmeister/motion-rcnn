disp('======= KITTI 2015 Benchmark Demo =======');
clear all; close all; dbstop error;

% error threshold
tau = [3 0.05];

% stereo demo
disp('Load and show disparity map ... ');
D_est = disp_read('data/disp_est.png');
D_gt  = disp_read('data/disp_gt.png');
d_err = disp_error(D_gt,D_est,tau);
D_err = disp_error_image(D_gt,D_est,tau);
figure,imshow([disp_to_color([D_est;D_gt]);D_err]);
title(sprintf('Disparity Error: %.2f %%',d_err*100));

% flow demo
disp('Load and show optical flow field ... ');
F_est = flow_read('data/flow_est.png');
F_gt  = flow_read('data/flow_gt.png');
f_err = flow_error(F_gt,F_est,tau);
F_err = flow_error_image(F_gt,F_est,tau);
figure,imshow([flow_to_color([F_est;F_gt]);F_err]);
title(sprintf('Flow Error: %.2f %%',f_err*100));
