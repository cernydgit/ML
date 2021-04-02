#%% GENERATE/LOAD

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


importlib.reload(graph)
importlib.reload(deepblue)

print("TF version:", tf.__version__)
print("TF CUDA build:", tf.test.is_built_with_cuda()) 
print("TF GPU devices:", tf.config.list_physical_devices('GPU'))


g = graph.Graph()
g.generate(trend=60, noise=0, loops=50,point_density=20, swing=1, long_swing=0)
#g.compute_ema(5,20,10)
g.compute_target_difference(7)
#g.compute_target_momentum(1)
g.plot_graph(start=100, length=200)
g.plot_indicator(start=100, length=200)

#%% TRAIN

train_type = 'dnn'
train_sample = 5
train_epochs = 100
start = time.time()

if train_type == 'rnn':
	loss, metric, y_test, y_pred = g.train_rnn(rnn_units=10,sample_size=train_sample, target='ml:ind:target', input_prefix='input:', epochs=train_epochs, loss='mae')
else:
	loss, metric, y_test, y_pred = g.train_dnn(sample_size=train_sample, target='ml:ind:target', input_prefix='input:', epochs=train_epochs, dropout=0, loss='mae') 

print('Train time:', time.time()-start)

#%% SHOW

#g.show_result(100,200,0)
g.show_result()
