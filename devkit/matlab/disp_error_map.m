function [E,D_gt_val] = disp_error_map (D_gt,D_est)

D_gt_val = D_gt>=0;
E = abs(D_gt-D_est);
E(D_gt_val==0) = 0;
