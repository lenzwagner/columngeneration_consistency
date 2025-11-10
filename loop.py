from Utils.setup import Min_WD_i, Max_WD_i
from cg_behavior import *
from subproblem import *
from Utils.Plots.plots import *
from Utils.aggundercover import *
from datetime import datetime
import os

comb_text = str(0.06) + '_' + str(3)
file = 'perf_Plot_' + comb_text
file3 = 'comb__' + comb_text


path = f'./images/schedules/worker_schedules' + comb_text + '.eps'

combined_list = [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 0, 1, 0, 2, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 0, 1, 0, 2, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 0, 1, 0, 2, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 0, 1, 0, 2, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 0, 1, 0, 2, 2, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 2, 1, 0, 0, 0, 0, 2, 2, 0, 2, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 2, 1, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 3, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 2, 0, 0, 2, 0, 2, 0, 0, 0, 2, 0, 0, 1, 2, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 2, 0, 0, 2, 0, 2, 0, 0, 0, 2, 0, 0, 1, 2, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 2, 0, 0, 2, 0, 2, 0, 0, 0, 2, 1, 0, 0, 2, 0, 1, 2, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 2, 1, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 1, 0, 3, 0, 0, 1, 0, 2, 0, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 3, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 2, 2, 2, 0, 0, 2, 0, 0, 2, 0, 0, 3, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 3, 3, 2, 0, 0, 2, 0, 0, 3, 0, 0, 3, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 3, 3, 2, 0, 0, 2, 0, 0, 3, 0, 0, 3, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 3, 3, 2, 0, 0, 2, 0, 0, 3, 0, 0, 3, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 3, 3, 2, 0, 0, 2, 0, 0, 3, 0, 0, 3, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 3, 3, 2, 0, 0, 2, 0, 0, 3, 0, 0, 3, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 3, 3, 2, 0, 0, 2, 0, 0, 3, 0, 0, 3, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 3, 3, 2, 0, 0, 2, 0, 0, 3, 0, 0, 3, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 3, 3, 2, 0, 0, 2, 0, 0, 3, 0, 0, 2, 0, 0, 2, 0, 3, 0, 0, 1, 0, 2, 0, 0, 0, 2, 0, 0, 2, 2, 3, 1, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 3, 0, 0, 1, 0, 2, 0, 0, 0, 2, 0, 0, 2, 2, 3, 1, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 1, 0, 0, 2, 0, 0, 2, 2, 2, 0, 0, 3, 0, 0, 3, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 2, 2, 2, 0, 1, 2, 0, 0, 3, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 2, 2, 2, 0, 1, 2, 0, 0, 3, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 2, 2, 2, 0, 1, 2, 0, 0, 3, 0, 0, 2, 1, 0, 2, 0, 2, 0, 0, 0, 0, 3, 0, 1, 0, 2, 0, 0, 2, 2, 2, 0, 0, 2, 1, 0, 3, 0, 0, 2, 1, 0, 2, 0, 2, 0, 0, 0, 0, 3, 0, 1, 0, 2, 0, 0, 2, 2, 2, 0, 0, 2, 1, 0, 3, 0, 0, 2, 0, 0, 2, 0, 3, 0, 0, 1, 0, 2, 0, 0, 0, 2, 0, 0, 2, 2, 3, 0, 0, 2, 0, 0, 3, 0, 0, 2, 0, 0, 2, 0, 3, 0, 0, 1, 0, 2, 0, 0, 0, 2, 0, 0, 2, 2, 3, 0, 0, 2, 0, 0, 3, 0, 2, 0, 0, 0, 0, 2, 1, 2, 2, 1, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 2, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 2, 0, 2, 1, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 2, 0, 2, 1, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 2, 0, 2, 1, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 2, 0, 2, 1, 0, 0, 0, 2, 0, 1, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 2, 0, 2, 1, 0, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 3, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 1, 0, 2, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 3, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 1, 0, 2, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 0, 2, 0, 1, 0, 0, 0, 0, 2, 2, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 0, 2, 0, 1, 0, 0, 0, 0, 2, 2, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 2, 2, 0, 0, 2, 0, 0, 2, 0, 1, 0, 1, 1, 0, 2, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 1, 0, 2, 2, 0, 0, 2, 0, 0, 0, 1, 0, 1, 3, 0, 0, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 1, 0, 2, 2, 0, 0, 2, 0, 0, 0, 1, 0, 1, 3, 0, 0, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 1, 0, 2, 2, 0, 0, 2, 0, 0, 0, 1, 0, 1, 3, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 3, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 3, 1, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 3, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 3, 1, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 3, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 3, 1, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 3, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 3, 1, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 3, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 3, 1, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 3, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 3, 1, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 2, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0, 0, 0, 1, 3, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 2, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0, 0, 0, 1, 3, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 2, 2, 0, 1, 2, 0, 0, 0, 1, 0, 1, 3, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 2, 1, 0, 3, 3, 0, 0, 2, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 1, 1, 0, 0, 2, 0, 0, 0, 2, 0, 0, 2, 2, 0, 1, 2, 2, 0, 1, 0, 0, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 1, 3, 3, 0, 0, 2, 2, 0, 0, 0, 0, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 1, 3, 3, 0, 0, 2, 2, 0, 0, 0, 0, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 1, 3, 3, 0, 0, 2, 2, 0, 0, 0, 0, 0, 2, 2, 1, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 2, 0, 0, 2, 2, 0, 1, 2, 2, 0, 1, 0, 0, 0, 2, 2, 1, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 2, 0, 0, 2, 2, 0, 1, 2, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1, 1, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1, 1, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 2, 0, 2, 0, 0, 0, 1, 2, 0, 2, 0, 1, 0, 2, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 2, 0, 2, 0, 0, 0, 1, 2, 0, 2, 0, 1, 0, 2, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 1, 2, 1, 0, 0, 2, 2, 0, 0, 1, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 1, 2, 1, 0, 0, 2, 2, 0, 0, 1, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 0, 3, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 2, 0, 0, 0, 1, 2, 0, 2, 0, 0, 0, 3, 2, 0, 0, 1, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 0, 3, 0, 0, 0, 3, 2, 0, 0, 1, 0, 0, 1, 2, 2, 0, 0, 3, 0, 0, 2, 2, 0, 2, 0, 0, 0, 2, 0, 2, 0, 0, 1, 2, 0, 2, 0, 3, 0, 0, 0, 3, 2, 0, 0, 2, 0, 0, 2, 2, 1, 2, 0, 0, 0, 2, 0, 2, 0, 0, 0, 3, 0, 2, 0, 3, 0, 0, 0, 3, 2, 0, 2, 0, 0, 0, 2, 0, 3, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 1, 2, 0, 0, 3, 0, 0, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 3, 2, 0, 0, 3, 0, 0, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 3, 2, 0, 0, 3, 0, 0, 0, 2, 2, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 3, 0, 0, 0, 2, 2, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 3, 0, 0, 0, 2, 2, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 3, 0, 0, 0, 2, 2, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 3, 0, 0, 0, 2, 2, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 3, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 2, 3, 0, 0, 2, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 2, 3, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 2, 2, 2, 0, 0, 2, 0, 0, 0, 2, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 2, 2, 0, 0, 2, 0, 0, 0, 2, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 2, 3, 2, 0, 0, 2, 0, 0, 0, 2, 0, 1, 2, 1, 1, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 2, 3, 2, 0, 0, 2, 0, 0, 0, 2, 0, 1, 2, 1, 1, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 2, 3, 2, 0, 0, 2, 0, 0, 0, 2, 0, 1, 2, 1, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 2, 3, 0, 0, 0, 2, 2, 0, 0, 0, 0, 1, 2, 3, 3, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 2, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 2, 2, 2, 0, 1, 0, 1, 2, 2, 0, 0, 0, 0, 2, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 1, 0, 0, 0, 2, 2, 2, 0, 0, 0, 1, 2, 2, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 2, 0, 0, 0, 0, 3, 0, 2, 2, 0, 0, 0, 0, 2, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 2, 0, 0, 1, 0, 0, 0, 0, 3, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 1, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 2, 1, 0, 0, 3]


import pandas as pd

def generate_dict_from_excel(file_path, value_I, pattern):

    data = pd.read_excel(file_path)

    # Filter auf die angegebenen Bedingungen
    filtered_row = data[(data['I'] == value_I) & (data['Pattern'] == pattern)]

    # Dictionary erstellen, falls gefilterte Daten existieren
    if not filtered_row.empty:
        result_dict = {
            tuple(map(int, col.split(','))): filtered_row[col].values[0]
            for col in data.columns if ',' in col
        }
        return result_dict
    else:
        print("Keine passenden Daten gefunden.")
        return {}

# Beispielhafte Nutzung
file_path = 'data/data_demand.xlsx'  # Pfad zur Datei
value_I = 100
pattern = 'Medium'

demand_dict = generate_dict_from_excel(file_path, value_I, pattern)

new_max_value = False

# **** Prerequisites ****
# Create Dataframes
eps_ls = [0.06]
chi_ls = [3]
T = list(range(1, 29))
I = list(range(1, 101))
K = [1, 2, 3]

if new_max_value == True:
    adj = 1
else:
    adj =0

# DataFrame
results = pd.DataFrame(columns=['I', 'pattern', 'epsilon', 'chi', 'objval', 'lbound', 'iteration', 'undercover', 'undercover_norm', 'cons', 'cons_norm', 'perf', 'perf_norm', 'max_auto', 'min_auto', 'mean_auto', 'lagrange''undercover', 'undercover_norm_n', 'cons_n', 'cons_norm_n', 'perf_n', 'perf_norm_n', 'max_auto_n', 'min_auto_n', 'mean_auto_n', 'lagrange_n'])
results2 = pd.DataFrame(columns=['I', 'epsilon', 'chi', 'undercover_norm', 'cons_norm', 'understaffing_norm', 'perf_norm', 'undercover_norm_n', 'cons_norm_n', 'understaffing_norm_n', 'perf_norm_n'])

# Times and Parameter
time_Limit = 7200
time_cg = 7200
time_cg_init = 10
prob = 1.0

seed1 = 123 - math.floor(len(I)*len(T)) - adj
print(seed1)
random.seed(seed1)
demand_dict2 = demand_dict_fifty_min(len(T), prob, len(I), 2, 0.25)
print('Demand dict', demand_dict)
max_itr = 200
output_len = 98
mue = 1e-4
threshold = 6e-5

data = pd.DataFrame({
    'I': I + [np.nan] * (max(len(I), len(T), len(K)) - len(I)),
    'T': T + [np.nan] * (max(len(I), len(T), len(K)) - len(T)),
    'K': K + [np.nan] * (max(len(I), len(T), len(K)) - len(K))
})

#fig = visualize_schedule_dual(combined_list, len(T), 100, 100)
#pio.write_image(fig, path, height=230, width=700, engine='kaleido')

# Datanames
current_time = datetime.now().strftime('%Y-%m-%d_%H')
file = f'comb_0.1-1'
#file2 = f'comb_condens_0.06-1'
file_name_csv2 = f'.{os.sep}results{os.sep}study{os.sep}comb{os.sep}low{os.sep}{file}.csv'
file_name_xlsx2 = f'.{os.sep}results{os.sep}study{os.sep}comb{os.sep}low{os.sep}{file}.xlsx'
#file_name_csv2 = f'.{os.sep}results{os.sep}study{os.sep}comb{os.sep}new{os.sep}{file2}.csv'
#file_name_xlsx2 = f'.{os.sep}results{os.sep}study{os.sep}comb{os.sep}new{os.sep}{file2}.xlsx'




## Naive
modelImprovable = True

# Get Starting Solutions
problem_start = Problem(data, demand_dict, 0, Min_WD_i, Max_WD_i, 0)
problem_start.buildLinModel()
problem_start.model.Params.MIPFocus = 1
problem_start.model.Params.Heuristics = 1
problem_start.model.Params.RINS = 10
problem_start.model.Params.TimeLimit = time_cg_init
problem_start.model.update()
problem_start.model.optimize()

# Schedules
# Create
start_values_perf = {(t, s): problem_start.perf[1, t, s].x for t in T for s in K}
start_values_p = {(t): problem_start.p[1, t].x for t in T}
start_values_x = {(t, s): problem_start.x[1, t, s].x for t in T for s in K}
start_values_c = {(t): problem_start.sc[1, t].x for t in T}

# Initialize iterations
itr = 0
last_itr = 0

# Create empty results lists
histories = ["objValHistSP", "timeHist", "objValHistRMP", "avg_rc_hist", "lagrange_hist", "sum_rc_hist", "avg_sp_time",
             "rmp_time_hist", "sp_time_hist"]
histories_dict = {}
for history in histories:
    histories_dict[history] = []
objValHistSP, timeHist, objValHistRMP, avg_rc_hist, lagrange_hist, sum_rc_hist, avg_sp_time, rmp_time_hist, sp_time_hist = histories_dict.values()

X_schedules = {}
for index in I:
    X_schedules[f"Physician_{index}"] = []

Perf_schedules = create_schedule_dict(start_values_perf, 1, T, K)
Cons_schedules = create_schedule_dict(start_values_c, 1, T)
P_schedules = create_schedule_dict(start_values_p, 1, T)
X1_schedules = create_schedule_dict(start_values_x, 1, T, K)

master = MasterProblem(data, demand_dict, max_itr, itr, last_itr, output_len, start_values_perf)
master.buildModel()

# Initialize and solve relaxed model
master.setStartSolution()
master.updateModel()
master.solveRelaxModel()

# Retrieve dual values
duals_i0 = master.getDuals_i()
duals_ts0 = master.getDuals_ts()
print(f"{duals_i0, duals_ts0}")

# Start time count
t0 = time.time()
previous_reduced_cost = float('inf')

while modelImprovable and itr < max_itr:
    print("*{:^{output_len}}*".format(f"Begin Column Generation Iteration {itr}", output_len=output_len))

    # Start
    itr += 1

    # Solve RMP
    rmp_start_time = time.time()
    master.current_iteration = itr + 1
    master.solveRelaxModel()
    rmp_end_time = time.time()
    rmp_time_hist.append(rmp_end_time - rmp_start_time)

    objValHistRMP.append(master.model.objval)
    current_obj = master.model.objval

    # Get and Print Duals
    duals_i = master.getDuals_i()
    duals_ts = master.getDuals_ts()

    # Solve SPs
    modelImprovable = False

    # Build SP
    subproblem = Subproblem(duals_i, duals_ts, data, 1, itr, 0, Min_WD_i, Max_WD_i, 0)
    subproblem.buildModel()

    # Save time to solve SP
    sub_start_time = time.time()
    if previous_reduced_cost < -0.001:
        print("*{:^{output_len}}*".format(f"Use MIP-Gap > 0 in Iteration {itr}", output_len=output_len))
        subproblem.solveModelNOpt(time_cg)
    else:
        print("*{:^{output_len}}*".format(f"Use MIP-Gap = 0 in Iteration {itr}", output_len=output_len))
        subproblem.solveModelOpt(time_cg)
    sub_end_time = time.time()
    sp_time_hist.append(sub_end_time - sub_start_time)

    sub_totaltime = sub_end_time - sub_start_time
    timeHist.append(sub_totaltime)
    index = 1

    keys = ["X", "Perf", "P", "C", "X1"]
    methods = ["getOptX", "getOptPerf", "getOptP", "getOptC", "getOptX"]
    schedules = [X_schedules, Perf_schedules, P_schedules, Cons_schedules, X1_schedules]

    for key, method, schedule in zip(keys, methods, schedules):
        value = getattr(subproblem, method)()
        schedule[f"Physician_{index}"].append(value)

    # Check if SP is solvable
    status = subproblem.getStatus()
    if status != 2:
        raise Exception("*{:^{output_len}}*".format("Pricing-Problem can not reach optimality!", output_len=output_len))

    # Save ObjVal History
    reducedCost = subproblem.model.objval
    objValHistSP.append(reducedCost)

    # Update previous_reduced_cost for the next iteration
    previous_reduced_cost = reducedCost
    print("*{:^{output_len}}*".format(f"Reduced Costs in Iteration {itr}: {reducedCost}", output_len=output_len))

    # Increase latest used iteration
    last_itr = itr + 1

    # Generate and add columns with reduced cost
    if reducedCost < -threshold:
        Schedules = subproblem.getNewSchedule()
        master.addColumn(itr, Schedules)
        master.addLambda(itr)
        master.updateModel()
        modelImprovable = True

    # Update Model
    master.updateModel()

    # Calculate Metrics
    avg_rc = sum(objValHistSP) / len(objValHistSP)
    lagrange = avg_rc + current_obj
    sum_rc = sum(objValHistSP)
    avg_rc_hist.append(avg_rc)
    sum_rc_hist.append(sum_rc)
    lagrange_hist.append(lagrange)
    objValHistSP.clear()
    avg_time = sum(timeHist) / len(timeHist)
    avg_sp_time.append(avg_time)
    timeHist.clear()

    if not modelImprovable:
        print("*" * (output_len + 2))
        break

if modelImprovable and itr == max_itr:
    max_itr *= 2

# Solve Master Problem with integrality restored
master.model.setParam('PoolSearchMode', 2)
master.model.setParam('PoolSolutions', 100)
master.model.setParam('PoolGap', 0.05)
master.finalSolve(time_cg)

status = master.model.Status
if status in (gu.GRB.INF_OR_UNBD, gu.GRB.INFEASIBLE, gu.GRB.UNBOUNDED):
    print("The model cannot be solved because it is infeasible or unbounded")
    gu.sys.exit(1)

if status != gu.GRB.OPTIMAL:
    print(f"Optimization was stopped with status {status}")
    gu.sys.exit(1)


objValHistRMP.append(master.model.objval)

lagranigan_bound = round((objValHistRMP[-2] + sum_rc_hist[-1]), 3)

# Calc Stats
undercoverage_pool = []
understaffing_pool = []
perf_pool = []
cons_pool = []
undercoverage_pool_norm = []
understaffing_pool_norm = []
perf_pool_norm = []
cons_pool_norm = []

sol = master.printLambdas()

ls_sc1 = plotPerformanceList(Cons_schedules, sol)
ls_p1 = plotPerformanceList(Perf_schedules, sol)
ls_x1 = plotPerformanceList(X_schedules, sol)

undercoverage_naive = master.getUndercoverage()

# Loop
for epsilon in eps_ls:
    for chi in chi_ls:

        eps = epsilon
        print(f"")
        print(f"Iteration: {epsilon}-{chi}")
        print(f"")

        ## Column Generation
        # Bevaior
        print('Doing behaviour')
        undercoverage, understaffing, perfloss, consistency, consistency_norm, undercoverage_norm, understaffing_norm, perfloss_norm, results_sc, results_r, autocorell, final_obj, final_lb, itr, lagrangeB, ls_sc_behav, ls_p_behavior, undercoverage_behavior = column_generation_behavior(data, demand_dict, eps, Min_WD_i, Max_WD_i, time_cg_init, max_itr, output_len, chi,
                                    threshold, time_cg, I, T, K, prob)

        # Naive
        print('Doing naive')
        ls_r1 = process_recovery(ls_sc1, chi, len(T))
        undercoverage_ab, understaffing_ab, perfloss_ab, consistency_ab, consistency_norm_ab, undercoverage_norm_ab, understaffing_norm_ab, perfloss_norm_ab, perf_ls_ab, undercover_naive_ab = master.calc_naive(
            ls_p1, ls_sc1, ls_r1, eps, prob)
        cumulative_total = [undercover_naive_ab[j] + undercoverage_naive[j] for j in range(len(undercover_naive_ab))]

        # Plots
        random.seed(0)
        comb_text = str(eps) + '_' + str(chi)
        file = 'perf_Plot_' + comb_text
        file3 = 'comb__' + comb_text

        path = f'./images/schedules/worker_schedules' + comb_text + '.svg'
        #print(f"Ls{combine_lists(ls_sc_behav, ls_sc1, len(T), len(I))}")
        #fig = visualize_schedule_dual(combine_lists(ls_sc_behav, ls_sc1, len(T), len(I)), len(T), len(I), 100)
        #pio.write_image(fig, path, height=230, width=700, engine='kaleido')

       # print(f"Lists: {combine_lists(ls_sc_behav, ls_sc1, len(T), len(I))}")
        #performancePlotAvg(ls_p_behavior, perf_ls_ab, len(T), file, 10, eps, chi)

        a = create_dict_from_list(cumulative_total, len(T), len(K))
        b = create_dict_from_list(undercoverage_behavior, len(T), len(K))

        #plot_relative_undercover_dual(create_dict_from_list(undercoverage_behavior, len(T), len(K)),
                                 #create_dict_from_list(cumulative_total, len(T), len(K)), demand_dict, len(T),
                                      #len(K), 499, file3)


        undercoverage_pool.append(undercoverage_ab)
        understaffing_pool.append(understaffing_ab)
        perf_pool.append(perfloss_ab)
        cons_pool.append(consistency_ab)
        undercoverage_pool_norm.append(undercoverage_norm_ab)
        understaffing_pool_norm.append(understaffing_norm_ab)
        perf_pool_norm.append(perfloss_norm_ab)
        cons_pool_norm.append(consistency_norm_ab)

        #print(f"Solcount: {master.model.SolCount}")
        for k in range(master.model.SolCount):
            master.model.setParam(gu.GRB.Param.SolutionNumber, k)
            vals = master.model.getAttr("Xn", master.lmbda)

            solution = {key: round(value) for key, value in vals.items()}
            sum_lambda = sum(solution.values())
            if abs(sum_lambda - len(I)) > 1e-6:
                #print(f"Skipping infeasible solution {k}: sum of lambda = {sum_lambda}")
                continue

            #print(f"Processing feasible solution {k}")

            ls_sc = plotPerformanceList(Cons_schedules, solution)
            ls_p = plotPerformanceList(Perf_schedules, solution)
            ls_r = process_recovery(ls_sc, chi, len(T))
            ls_x = plotPerformanceList(X_schedules, solution)

            undercoverage_a, understaffing_a, perfloss_a, consistency_a, consistency_norm_a, undercoverage_norm_a, understaffing_norm_a, perfloss_norm_a, perf_ls_a, undercover_naive_a = master.calc_naive(
                ls_p, ls_sc, ls_r, eps, prob)

            undercoverage_pool.append(undercoverage_a)
            understaffing_pool.append(understaffing_a)
            perf_pool.append(perfloss_a)
            cons_pool.append(consistency_a)
            undercoverage_pool_norm.append(undercoverage_norm_a)
            understaffing_pool_norm.append(understaffing_norm_a)
            perf_pool_norm.append(perfloss_norm_a)
            cons_pool_norm.append(consistency_norm_a)

        undercoverage_n = sum(undercoverage_pool) / len(undercoverage_pool)
        understaffing_n = sum(understaffing_pool) / len(understaffing_pool)
        perfloss_n = sum(perf_pool) / len(perf_pool)
        consistency_n = sum(cons_pool) / len(cons_pool)
        undercoverage_norm_n = sum(undercoverage_pool_norm) / len(undercoverage_pool_norm)
        understaffing_norm_n = sum(understaffing_pool_norm) / len(understaffing_pool_norm)
        perfloss_norm_n = sum(perf_pool_norm) / len(perf_pool_norm)
        consistency_norm_n = sum(cons_pool_norm) / len(cons_pool_norm)


        # Data frame
        result = pd.DataFrame([{
            'I': len(I),
            'pattern': "Medium",
            'epsilon': epsilon,
            'chi': chi,
            'objval': final_obj,
            'lbound': final_lb,
            'iteration': itr,
            'undercover': undercoverage,
            'undercover_norm': undercoverage_norm,
            'cons': consistency,
            'cons_norm': consistency_norm,
            'perf': perfloss,
            'perf_norm': perfloss_norm,
            'max_auto': round(max(autocorell), 5),
            'min_auto': round(min(autocorell), 5),
            'mean_auto': round(np.mean(autocorell), 5),
            'lagrange': lagrangeB
        }])

        results = pd.concat([results, result], ignore_index=True)


        result2 = pd.DataFrame([{
            'I': len(I),
            'epsilon': epsilon,
            'chi': chi,
            'undercover_norm': undercoverage_norm,
            'cons_norm': consistency_norm,
            'understaffing_norm': understaffing_norm,
            'perf_norm': perfloss_norm,
            'undercover_norm_n': undercoverage_norm_n,
            'cons_norm_n': consistency_norm_n,
            'understaffing_norm_n': understaffing_norm_n,
            'perf_norm_n': perfloss_norm_n
        }])

        results2 = pd.concat([results2, result2], ignore_index=True)

        print("")
        print("")
        print("")
        print('Master.objval', master.model.objval)
        print("")
        print("")
        print("")

#results.to_csv(file_name_csv, index=False)
#results.to_excel(file_name_xlsx, index=False)
##results2.to_csv(file_name_csv2, index=False)
#results2.to_excel(file_name_xlsx2, index=False)
