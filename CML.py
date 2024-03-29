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
from softsign_profit import softsign_profit_sum

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

train_sample_range = range(2,10,300)
min_profit_range = floatrange(0, 2, 500)
jma_period_range = range(5, 20, 100)
jma_phase_range = range(100,101,1000)
target_divergence_range = range(2,100,100)

min_signal = 0
runs = 5

start = 0
max_records = 60000
result = []
graphs = []

for jma_period in jma_period_range:
	for jma_phase in jma_phase_range:
		for train_sample in train_sample_range:
			for min_profit in min_profit_range:
				for target_divergence_period in target_divergence_range:
					for x in range(runs):
						g = graphlib.Graph()
						graphs.append(g)
						g.load("EURUSD15.csv", start = start, max_records = max_records, mult=1000)


						test_start=int(0.8*len(g.close()))
						test_length=int(0.2*len(g.close()))


						g.plot_graph(filter='input:graph:close')

						print('TRAINING SET:')
						g.prepare_training(jma_period=jma_period,jma_phase=jma_phase, target_divergence_period=target_divergence_period, jma_count=3)
						testing_set_loss, metric, y_test, y_pred = g.train_dnn(sample_size=train_sample, layers = 2, layers_reduction=0, dropout=0.1, epochs=400,  loss=softsign_profit_mean(min_profit), final_activation='softsign') 
						test_profit = -testing_set_loss 
						print('test_profit:', test_profit)
						train_length = int(0.8*len(g.close()))
						total_profit, avg_profit, profit_factor, success_rate, trades = g.trade(min_signal=min_signal, silent = False, start = 0, length = train_length)
						g.plot_equity(length = train_length)


						print('VALIDATION SET:')
						test_start=int(0.8*len(g.close()))
						test_length=int(0.2*len(g.close()))
						val_total_profit, val_avg_profit, val_profit_factor, val_success_rate, val_trades = g.trade(min_signal=min_signal, silent = False, start=test_start, length=test_length)

						g.plot_equity(length = test_length)
						#g.plot_graph(start=test_start, length=test_length, plot_trades = True, filter='input:graph:close')
						#g.plot_indicator(start=test_start, length=test_length, filter='ml:ind:trained')
						g.plot_graph(start=test_start, length=400, plot_trades = True, filter='input:graph:close')
						g.plot_indicator(start=test_start, length=400, filter='ml:ind:trained')
						result.append([target_divergence_period, jma_period, jma_phase, train_sample, min_profit, test_profit + min_profit, test_profit, total_profit, avg_profit, profit_factor, success_rate, trades, val_total_profit, val_avg_profit, val_profit_factor, val_success_rate, val_trades])


frame = pd.DataFrame(result, columns = ['divp','jmap','jmaph','sample', 'min_profit', 'test_profit', 'clean_test_profit', 'TOT', 'AVG', 'PF', 'SR', 'T', 'VAL_TOT', 'VAL_AVG', 'VAL_PF', 'VAL_SR', 'VAL_T'])

pd.set_option('display.width', 400)
print("\nRESULTS:")
print(frame)


if (len(result) > 1):
	print('\nSUMMARY:')
	print(frame.describe())
	print("\nCORRELATION:")
	print(frame.corr())



print("Done.")



