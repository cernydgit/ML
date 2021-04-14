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
from softsign_profit import softsign_profit_mean

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

point_count = 10000

train_sample_range = range(5,50,100)
min_profit_range = floatrange(1, 6, 100)
jma_period_range = range(15,60, 100)
jma_phase_range = range(100,101,100)
target_divergence_range = range(2,6,100)

min_signal = 0.05
silent = True

result = []
graphs = []

for jma_period in jma_period_range:
	for jma_phase in jma_phase_range:
		for train_sample in train_sample_range:
			for min_profit in min_profit_range:
				for target_divergence_period in target_divergence_range:
					g = graphlib.Graph()
					graphs.append(g)
					g.generate_zigzag(point_count=point_count, min_trend_legth = 3, max_trend_length = 10, min_noise=0.1, max_noise=5)
					print('TRAINING:')
					g.prepare_training(jma_period=jma_period,jma_phase=jma_phase, target_divergence_period=target_divergence_period)
					#test_profit, total_profit, avg_profit, profit_factor, success_rate, trades = g.analyze(silent=silent, train_sample=train_sample, min_profit=min_profit, train_epochs=100, min_signal = min_signal)
					testing_set_loss, metric, y_test, y_pred = g.train_dnn(sample_size=train_sample, layers = 6, dropout=0.1, epochs=100,  loss=softsign_profit_mean(min_profit), final_activation='softsign') 
					test_profit = -testing_set_loss 
					
					#g.show_result(100,400, min_signal)
					#g.show_result(min_signal=min_signal)

					g.trade(min_signal=min_signal, silent = False)
					g.plot_equity()

					print('REAL:')

					g.generate_zigzag(point_count=point_count, min_trend_legth = 3, max_trend_length = 10, min_noise=0.1, max_noise=5)
					g.prepare_training(jma_period=jma_period,jma_phase=jma_phase, target_divergence_period=target_divergence_period, silent=True)
					g.evaluate(sample_size=train_sample, min_signal=min_signal)
					
					total_profit, avg_profit, profit_factor, success_rate, trades = g.trade(min_signal = min_signal, silent=False)
					g.plot_equity()
					#g.plot_graph(start=1000, length=200, plot_trades = True, filter='input:graph:close')
					#g.plot_indicator(start=1000, length=200, filter='ml:ind:trained')

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
#print("TRADES SEGMENT  --------------------------------------------------")
#graphs[0].show_result(100,400, min_signal)
#print("TRADES COMPLETE  --------------------------------------------------")
#graphs[0].show_result(min_signal=min_signal)

#g.generate_zigzag(point_count=10000, min_trend_legth = 3, max_trend_length = 10, min_noise=0.1, max_noise=5)
#g.prepare_training(jma_period=15,jma_phase=100, target_divergence_period=2)
#g.evaluate(sample_size=4, min_signal=min_signal) #, input_prefix=input_prefix, target=eval_target, zero_origin=zero_origin)

#print("REGEN TRADES SEGMENT  --------------------------------------------------")
#graphs[0].show_result(100,400, min_signal)
#print("REGEN TRADES COMPLETE  --------------------------------------------------")
#graphs[0].show_result(min_signal=min_signal)


print("Done.")



