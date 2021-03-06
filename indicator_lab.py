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

import datetime

date_time_str = '14:15'
date_time_obj = datetime.datetime.strptime(date_time_str, '%H:%M')

print('MinInDay:', date_time_obj.time().hour*60+date_time_obj.time().minute)
print('Date:', date_time_obj.date())
print('Time:', date_time_obj.time())
print('Date-time:', date_time_obj)



g = graphlib.Graph()
g.load("EURUSD15.csv", start = 40000, max_records = 20000, mult=1000)
#g.compute_min_distance(5)
#g.compute_max_distance(5)
#g.plot_graph(filter='input:graph:close')
g.jma(10,100)
#g.jma(10,100,input='low')
#g.jma(10,100,input='high')
g.jmamom(10,100)
#g.jmamom(10,100,input='high')
#g.jmamom(10,100,input='low')
g.plot_graph(start=100, length=50, filter = 'jma:')
g.plot_indicator(start=100, length=5000)
