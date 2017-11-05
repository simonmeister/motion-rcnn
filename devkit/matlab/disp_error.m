function d_err = disp_error (D_gt,D_est,tau)

E = abs(D_gt-D_est);
n_err   = length(find(D_gt>0 & E>tau(1) & E./abs(D_gt)>tau(2)));
n_total = length(find(D_gt>0));
d_err = n_err/n_total;
