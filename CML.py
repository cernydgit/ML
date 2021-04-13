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

def floatrange(start,end,step):
	return list(np.arange(start, end, step))

#%% RUN ----------------------------------------------------------------------------------

train_sample_range = range(10,11,10000)
min_profit_range = floatrange(0, 3, 1000.05)
jma_period = 15
min_signal = 0.1
silent = True
runs = 1

result = []
graphs = []

for train_sample in train_sample_range:
	for min_profit in min_profit_range:
		sum = (0, 0, 0, 0, 0, 0)
		graph = graphlib.create_generated_cycle_graph(silent = False, jma_period = jma_period)
		for i in range(runs): 
			part = graph.analyze(silent=silent, train_sample=4, min_profit=min_profit, train_epochs=100, min_signal = min_signal)
			sum = tuple(map(operator.add, sum, part))
		test_profit, total_profit, avg_profit, profit_factor, success_rate, trades =  tuple(map(lambda x: x/runs, sum))
		result.append([train_sample, min_profit, test_profit + min_profit, test_profit, total_profit, avg_profit, profit_factor, success_rate, trades])
		graphs.append(graph)

frame = pd.DataFrame(result, columns = ['train_sample', 'min_profit', 'test_profit', 'clean_test_profit', 'total_profit', 'avg_profit', 'profit_factor', 'success_rate', 'trades'])

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