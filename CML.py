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
import time
import pandas as pd
import matplotlib.pyplot as plt
import operator
from IPython.display import display


importlib.reload(graphlib)
importlib.reload(deepblue)

print("TF version:", tf.__version__)
print("TF CUDA build:", tf.test.is_built_with_cuda()) 
print("TF GPU devices:", tf.config.list_physical_devices('GPU'))

def softsign_profit_sum(y_true, y_pred):
	return -K.sum(y_true * y_pred)

#min_profit = 1.2  #velikost lotu pridat do vypoctu tradu


def softsign_profit_mean(min_profit=0):
	def softsign_profit_mean(y_true, y_pred):
		return -K.mean(y_true * y_pred -  K.abs(y_pred) * min_profit)
	return softsign_profit_mean;


def floatrange(start,end,step):
	return list(np.arange(start, end, step))


def create_graph(silent = False, jma_period=15):
	g = graphlib.Graph()
	g.generate(trend=40, noise=2, loops=50,point_density=20, swing=0.6, long_swing=2)
	#g.load("US500240.csv")
	#g.load("AUDNZD240.csv")
	#g.load("USDJPY240.csv")
	g.compute_jma_complex(jma_period,100)
	g.compute_target_difference(10)
	if silent == False:
		g.plot_graph(start=100, length=400)
		g.plot_indicator(start=100, length=400)
	return g

def analyze(g, silent = False, train_sample = 10, min_profit=0.0, train_epochs = 100, min_signal=0.1):
	
	train_type = 'dnn'
	activation = 'softsign'
	loss = softsign_profit_mean(min_profit)

	#activation = 'linear'
	#loss = tf.keras.losses.MeanAbsoluteError()

	start = time.time()

	metric = 0
	if train_type == 'rnn':
		testing_set_loss, metric, y_test, y_pred = g.train_rnn(rnn_units=10,sample_size=train_sample, target='ml:ind:target', input_prefix='input:', epochs=train_epochs, loss=loss, final_activation=activation) 
	else:
		testing_set_loss, metric, y_test, y_pred = g.train_dnn(sample_size=train_sample, target='ml:ind:target', input_prefix='input:', epochs=train_epochs, dropout=0, loss=loss, final_activation=activation) 

	if not silent:
		print('Profit:', -testing_set_loss, 'Train time:', time.time()-start)
		#g.show_result(100,400, min_signal)
		g.show_result(min_signal=min_signal)

	total_profit, avg_profit, profit_factor, success_rate, trades = g.trade(min_signal = min_signal, silent=True)
	return -testing_set_loss, total_profit, avg_profit, profit_factor, success_rate, trades


#%% RUN ----------------------------------------------------------------------------------

train_sample_range = range(10,11,10000)
min_profit_range = floatrange(1.5, 3, 1000.05)
jma_period = 15
silent = True
runs = 1
result = []

graph = create_graph(silent = False, jma_period = jma_period)

for train_sample in train_sample_range:
	for min_profit in min_profit_range:
		sum = (0, 0, 0, 0, 0, 0)
		for i in range(runs): 
			part = analyze(graph, silent=silent, train_sample=4, min_profit=min_profit, train_epochs=100)
			sum = tuple(map(operator.add, sum, part))
		test_profit, total_profit, avg_profit, profit_factor, success_rate, trades =  tuple(map(lambda x: x/runs, sum))
		result.append([train_sample, min_profit, test_profit + min_profit, test_profit, total_profit, avg_profit, profit_factor, success_rate, trades])

frame = pd.DataFrame(result, columns = ['train_sample', 'min_profit', 'test_profit', 'clean_test_profit', 'total_profit', 'avg_profit', 'profit_factor', 'success_rate', 'trades'])

for i in range(len(frame.columns)-7):
	print("CORRELATION: ", frame.columns[i], " --------------------------------------------------")
	#frame.plot(kind = 'scatter', x = frame.columns[i], y = 'test_profit')
	#plt.show()
	#frame.plot(kind = 'scatter', x = frame.columns[i], y = 'clean_test_profit')
	#plt.show()
	frame.plot(kind = 'scatter', x = frame.columns[i], y = 'total_profit')
	plt.show()
	frame.plot(kind = 'scatter', x = frame.columns[i], y = 'avg_profit')
	plt.show()
	frame.plot(kind = 'scatter', x = frame.columns[i], y = 'success_rate')
	plt.show()
	#frame.plot(kind = 'scatter', x = frame.columns[i], y = 'profit_factor')
	#plt.show()
	frame.plot(kind = 'scatter', x = frame.columns[i], y = 'trades')
	plt.show()

pd.set_option('display.width', 400)
print("\nRESULTS:")
print(frame)
print("\nCORRELATION:")
print(frame.corr())




#%% SHOW

def analyze_row(graph, row_index):
	row = frame.loc[row_index]
	analyze(graph, silent=False, train_sample=int(row['train_sample']), min_profit=row['min_profit'], train_epochs=100)

#analyze_row(graph,len(frame)-1)
analyze_row(graph,0)

