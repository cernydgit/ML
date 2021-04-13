#%% INIT

#conda create -n ml python=3.7 pandas jupyter seaborn scikit-learn keras tensorflow
#conda activate ml
#conda info -e
#pip install git+https://github.com/tensorflow/docs
#pip install TA_Lib-0.4.19-cp37-cp37m-win_amd64.whl
#pip install TA_Lib-0.4.19-cp38-cp38-win_amd64.whl

import graph as graphlib
import importlib
import deepblue
import talib as ta
import numpy as np
import tensorflow as tf
import keras.backend as K
import pandas as pd
import matplotlib.pyplot as plt
import operator
from IPython.display import display

importlib.reload(graphlib)
importlib.reload(deepblue)
print("TF version:", tf.__version__)
print("TF CUDA build:", tf.test.is_built_with_cuda()) 
print("TF GPU devices:", tf.config.list_physical_devices('GPU'))

#g = graphlib.Graph()
#g.generate_zigzag(point_count=10000, noise=3, min_trend_legth = 3, max_trend_length = 30)
#g.plot_graph(start=100, length=1000)


def floatrange(start,end,step):
	return list(np.arange(start, end, step))

#%% RUN ----------------------------------------------------------------------------------

train_sample_range = range(3,100,100)
min_profit_range = floatrange(1, 2.5, 0.1)
jma_period_range = range(15,60, 100)
jma_phase_range = range(100,101,100)
target_divergence_range = range(2,6,1)

min_signal = 0.1
silent = True

result = []
graphs = []

for jma_period in jma_period_range:
	for jma_phase in jma_phase_range:
		for train_sample in train_sample_range:
			for min_profit in min_profit_range:
				for target_divergence_period in target_divergence_range:
					g = graphlib.Graph()
					g.generate_zigzag(point_count=10000, min_trend_legth = 3, max_trend_length = 10, min_noise=0.1, max_noise=5)
					g.prepare_training(jma_period=jma_period,jma_phase=jma_phase, target_divergence_period=target_divergence_period)
					graphs.append(g)
					test_profit, total_profit, avg_profit, profit_factor, success_rate, trades = g.analyze(silent=silent, train_sample=4, min_profit=min_profit, train_epochs=100, min_signal = min_signal)
					result.append([target_divergence_period, jma_period, jma_phase, train_sample, min_profit, test_profit + min_profit, test_profit, total_profit, avg_profit, profit_factor, success_rate, trades])

frame = pd.DataFrame(result, columns = ['divergence_period','jma_period','jma_phase','train_sample', 'min_profit', 'test_profit', 'clean_test_profit', 'total_profit', 'avg_profit', 'profit_factor', 'success_rate', 'trades'])

if (len(result) > 1):
	for i in range(len(frame.columns)-7):
		print("CORRELATION: ", frame.columns[i], " --------------------------------------------------")
		frame.plot(kind = 'scatter', x = frame.columns[i], y = 'test_profit')
		plt.show()
		frame.plot(kind = 'scatter', x = frame.columns[i], y = 'clean_test_profit')
		plt.show()
		frame.plot(kind = 'scatter', x = frame.columns[i], y = 'total_profit')
		plt.show()
		frame.plot(kind = 'scatter', x = frame.columns[i], y = 'avg_profit')
		plt.show()
		frame.plot(kind = 'scatter', x = frame.columns[i], y = 'success_rate')
		plt.show()
		frame.plot(kind = 'scatter', x = frame.columns[i], y = 'profit_factor')
		plt.show()
		frame.plot(kind = 'scatter', x = frame.columns[i], y = 'trades')
		plt.show()

	pd.set_option('display.width', 400)
	print("\nRESULTS:")
	print(frame)
	print("\nCORRELATION:")
	print(frame.corr())

#%% SHOW
print("TRADES SEGMENT  --------------------------------------------------")
graphs[0].show_result(100,400, min_signal)
print("TRADES COMPLETE  --------------------------------------------------")
graphs[0].show_result(min_signal=min_signal)
print("Done.")



