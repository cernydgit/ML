#%% INIT

#conda create -n ml python=3.7 pandas jupyter seaborn scikit-learn keras tensorflow
#conda activate ml
#conda info -e
#pip install git+https://github.com/tensorflow/docs
#pip install TA_Lib-0.4.19-cp37-cp37m-win_amd64.whl
#pip install TA_Lib-0.4.19-cp38-cp38-win_amd64.whl




import graph
import importlib
import deepblue
import talib as ta
import numpy as np
import tensorflow as tf
import keras.backend as K
import time
import pandas as pd
import matplotlib.pyplot as plt


importlib.reload(graph)
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



def create_graph(silent = False):
	g = graph.Graph()
	g.generate(trend=40, noise=2, loops=50,point_density=20, swing=1, long_swing=0)
	#g.load("US500240.csv")
	#g.load("AUDNZD240.csv")
	#g.load("USDJPY240.csv")
	#g.compute_ema(15,16,1000)
	#g.jmacd(15,3,100)
	#g.jma(5,100)
	#g.jma(70,100)
	##g.jmamom(15,100)
	#g.jmamom(5,100)
	#g.jma(30,100)
	#g.jma(50,100)
	#g.jma(15,100)
	#g.jmacd(5,1,100)
	#g.jmacd(15,1,100)
	g.jmacd(15,5,100)
	#g.jmacd(30,1,100)
	#g.jmacd(50,1,100)
	#g.compute_target_momentum(1)
	g.compute_target_difference(10)
	if silent == False:
		g.plot_graph(start=100, length=400)
		g.plot_indicator(start=100, length=400)
	return g

def analyze(g, silent = False, train_sample = 10, min_profit=0.0, 	train_epochs = 100):
	
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
		print('Profit:', -profit, 'Train time:', time.time()-start)
		g.show_result(100,400, min_signal=0.5)
		g.show_result(min_signal=0.5)

	return -testing_set_loss


#%% RUN


silent = True
runs = 3
train_sample_range = range(10,11,1)
min_profit_range = list(np.arange(0.0, 2.0, 0.4))
result = []

graph = create_graph(silent = False)

for train_sample in train_sample_range:
	for min_profit in min_profit_range:
		profit = 0
		for i in range(runs): profit += analyze(graph, silent=silent, train_sample=4, min_profit=min_profit, train_epochs=100)
		result.append([train_sample, min_profit, profit/runs + min_profit])

frame = pd.DataFrame(result, columns = ['train_sample', 'min_profit', 'profit'])
print(frame)
print(frame.corr())

for i in range(len(frame.columns)-1):
	frame.plot(kind = 'scatter', x = frame.columns[i], y = frame.columns[-1])
	plt.show()

#%% SHOW

row_index = len(frame-1)
row = frame.loc[row_index]
analyze(graph, silent=False, train_sample=row['train_sample'], min_profit=row['min_profit'], train_epochs=100)

