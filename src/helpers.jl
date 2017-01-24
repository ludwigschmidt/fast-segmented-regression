
function write_experiment_data(experiment_name, algo_name, n_vals, mses_mean, mses_std, times_mean, times_std)
  mm = mses_mean[algo_name]
  ms = mses_std[algo_name]
  tm = times_mean[algo_name]
  ts = times_std[algo_name]
  f = open(@sprintf("%s_%s.txt", experiment_name, algo_name), "w")
  @printf(f, "n mse_mean mse_std time_mean time_std\n")
  for (ii, n) in enumerate(n_vals)
    @printf(f, "%d %e %e %e %e\n", n, mm[ii], ms[ii], tm[ii], ts[ii])
  end
  close(f)
end

function write_experiment_data_with_ratios(experiment_name, algo_name, ref_algo_name, n_vals, mses_mean, mses_std, times_mean, times_std)
  mm = mses_mean[algo_name]
  ms = mses_std[algo_name]
  tm = times_mean[algo_name]
  ts = times_std[algo_name]
  mr = mses_mean[ref_algo_name]
  tr = times_mean[ref_algo_name]
  f = open(@sprintf("%s_%s.txt", experiment_name, algo_name), "w")
  @printf(f, "n mse_mean mse_std mse_ratio time_mean time_std time_ratio\n")
  for (ii, n) in enumerate(n_vals)
    @printf(f, "%d %e %e %e %e %e %e\n", n, mm[ii], ms[ii], mm[ii] / mr[ii], tm[ii], ts[ii], tr[ii] / tm[ii])
  end
  close(f)
end
