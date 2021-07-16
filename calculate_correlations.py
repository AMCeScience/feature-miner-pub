import Correlations.leaveoneout_correlation as leaveoneout
import Correlations.nvsone_correlation as nvsone
import Correlations.similarity_correlation as sim

if __name__ == "__main__":
  leaveoneout.calculate()
  nvsone.calculate()
  sim.calculate()