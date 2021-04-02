import graph
import importlib
import deepblue
import talib as ta
import numpy as np
import time
import ta2


importlib.reload(ta2)
importlib.reload(graph)


g = graph.Graph()
g.load("EURUSD240.csv")
#g.generate(trend=60, noise=1, loops=100,point_density=20, swing=1, long_swing=0)
#g.compute_ema(10,11,1)


g.jma(20,100)
g.jmamom(20,100)

g.plot_graph(start=100, length=200)
g.plot_indicator(start=100, length=200)
g.plot_graph(start=11100, length=200)
g.plot_indicator(start=11100, length=200)

