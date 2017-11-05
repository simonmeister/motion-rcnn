function f_err = flow_error (F_gt,F_est,tau)

[E,F_val] = flow_error_map (F_gt,F_est);
F_mag = sqrt(F_gt(:,:,1).*F_gt(:,:,1)+F_gt(:,:,2).*F_gt(:,:,2));
n_err   = length(find(F_val & E>tau(1) & E./F_mag>tau(2)));
n_total = length(find(F_val));
f_err = n_err/n_total;
