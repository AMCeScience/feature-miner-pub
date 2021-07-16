import Plots.similarity_plot as sim_plot
import Plots.leaveoneout_plot as leave_plot
import Plots.onevsone_plot as one_plot
import Plots.nvsone_plot as n_plot
import Plots.time_plot as time_plot

if __name__ == "__main__":
  sim_plot.plot()
  leave_plot.plot()
  one_plot.plot()
  n_plot.Nvsone_plot()
  time_plot.plot()