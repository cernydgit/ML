import matplotlib.pyplot as plt
import talib as ta
import ta2
import numpy as np
import random as rnd
import pandas as pd
import math
import time
from softsign_profit import softsign_profit_mean
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import Embedding
from keras.layers import Dropout
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import deepblue


def IntPortion(param):
   if (param > 0): return math.floor(param)
   if (param < 0): return math.ceil(param)
   return 0.0


class Graph:
    trade_profit = 0
    trade_spread = 0
    model = None
    file= None
    #graph_path = 'C:/Users/dcerny/AppData/Roaming/MetaQuotes/Terminal/A270C22676FD87E1F4CE8044BDE1756D/MQL4/Files/'
    graph_path = 'C:/!Code/ML/Symbols/'
    model_path = './Models/'


    def __init__(self):
        self.series = {}
        self.trades = []


    def load(self, file, spread = 0, max_records = 0, start = 0, silent=False, mult = 1):
        self.trades.clear()
        self.file = file
        self.trade_spread = spread
        df = pd.read_csv(self.graph_path + file) #, sep='')
        close = np.array(df[df.columns[5]].tolist())

        if silent == False:
            print('Loaded ', len(close), 'records from', file) 
        close = close[start:]   
        if max_records>0:
            close = close[:max_records]

        self.series['input:graph:close'] = self.scale_min_max(close) * mult
        #self.series['input:graph:close'] = close
        
        

        
        #self.series['ind:macd3'] = ta.MACD(close,3,12)[0]
        #self.series['ind:macd12'] = ta.MACD(close,12,26)[0]


    def compute_cci(self, start,stop,step=1):
        for i in range(start,stop,step):
            self.series['input:ind:cci'+str(i)] = self.scale_min_max(ta.CCI(self.close(), self.close(), self.close(),i))

    def scale_min_max(self,x):
            max = np.max(x[~np.isnan(x)])
            min = np.min(x[~np.isnan(x)])
            x = x-min
            x = x/(max-min)
            return x

    def compute_willr(self, start,stop,step=1):
        for i in range(start,stop,step):
            self.series['input:ind:willr'+str(i)] = self.scale_min_max(ta.WILLR(self.close(),self.close(),self.close(),i))


    def compute_cmo(self, start,stop,step=1):
        for i in range(start,stop,step):
            self.series['input:ind:cmo'+str(i)] = self.scale_min_max(ta.CMO(self.close(),i))


    def compute_roc(self, start,stop,step=1):
        for i in range(start,stop,step):
            self.series['input:ind:roc'+str(i)] = self.scale_min_max(ta.ROC(self.close(),i))

    def compute_mom(self, start,stop,step=1):
        for i in range(start,stop,step):
            self.series['input:ind:mom'+str(i)] =  ta.MOM(self.close(),i) #self.scale_min_max(ta.MOM(self.close(),i))

    def compute_mom_ratio(self, start,stop,step=1):
        for p in range(start,stop,step):
            s = []
            for i in range(len(self.close())):
                if (i<p):
                    s.append(np.nan)
                else:   
                    s.append(self.close()[i]/self.close()[i-p]-1)
            self.series['input:ind:momr'+str(p)] =  np.array(s)



    def compute_ema(self, start,stop,step=1):
        for i in range(start,stop,step):
            self.series['input:graph:ema'+str(i)] = ta.EMA(self.close(),i)

    def compute_kama(self, start,stop,step=1):
        for i in range(start,stop,step):
            self.series['input:graph:kama'+str(i)] = ta.KAMA(self.close(),i)

    def compute_rsi(self, start,stop,step=1):
        for i in range(start,stop,step):
            self.series['input:ind:rsi'+str(i)] = self.scale_min_max(ta.RSI(self.close(),i))







    def train_dnn(self, sample_size, target='ml:ind:target', input_prefix='input:', activation='relu', layers=3, normalize=False, epochs=100, loss='mae', layers_reduction=0, dropout=0, zero_origin=True, final_activation='linear'):
        self.trades.clear()
        #self.series.pop(target, None)
        x_data, y_data, _ = self.get_train_series_for_dnn(sample_size, target=target, input_prefix=input_prefix, zero_origin=zero_origin)

        input_size=len(x_data[0])

        self.model = Sequential()
        self.model.add(Dense(input_size, activation=activation, input_dim = input_size))
        for i in range(layers-1):
            input_size = input_size - layers_reduction
            self.model.add(Dense(input_size, activation=activation))
            if dropout>0:
                self.model.add(Dropout(dropout))
        self.model.add(Dense(1, activation=final_activation))
        self.model.summary()
        test_loss,test_metric,y_test,y_pred = deepblue.execute(self.model, x_data, y_data, loss=loss, metric=loss, epochs=epochs, normalize=normalize)            

        #if self.file is not None:
        #    self.model.save(self.model_path + self.file + '.model')
        
        eval_target = 'ml:graph:trained'
        if 'ind' in target: eval_target = 'ml:ind:trained'

        self.evaluate(sample_size=sample_size, input_prefix=input_prefix, target=eval_target, zero_origin=zero_origin)
        
        return test_loss, test_metric, y_test, y_pred


    def train_rnn(self, sample_size, rnn_units=10, target='ml:ind:target', input_prefix='input:', activation='relu', epochs=100, loss='mae', normalize=False, dropout=0, zero_origin=True, final_activation='linear'):
        self.trades.clear()
        x_data, y_data, _ = self.get_train_series_for_rnn(sample_size, target=target, input_prefix=input_prefix, zero_origin=zero_origin)

        self.model = Sequential()
        #self.model.add(SimpleRNN(10))
        #self.model.add(LSTM(sample_size))
        self.model.add(LSTM(units=rnn_units, input_shape=(sample_size, x_data.shape[2])))
        self.model.add(Dense(1, activation=final_activation))
        #self.model.summary() 
        test_loss,test_metric,y_test,y_pred = deepblue.execute(self.model, x_data, y_data, loss=loss, metric=loss, epochs=epochs, normalize=normalize)            

        #if self.file is not None:
        #    self.model.save(self.model_path + self.file + '.model')

        eval_target = 'ml:graph:trained'
        if 'ind' in target: eval_target = 'ml:ind:trained'
        
        self.evaluate_rnn(sample_size=sample_size, input_prefix=input_prefix, target=eval_target, zero_origin=zero_origin)
        
        return test_loss, test_metric, y_test, y_pred



    def evaluate(self, sample_size, target='ml:ind:trained', input_prefix='input:', min_signal=0, zero_origin=True): 
        #if self.file is not None:
        #    self.model = keras.models.load_model(self.model_path + self.file + '.model')

        x_data, _, start = self.get_train_series_for_dnn(sample_size=sample_size, input_prefix=input_prefix, zero_origin=zero_origin)

        y_pred = self.model.predict(x_data)
        y_pred = [y[0] for y in y_pred]
        y_pred = [np.nan] * (start+sample_size-1) + y_pred
        y_pred = y_pred + [np.nan] * (len(self.close()) - len(y_pred)) 
        #print('y start', start+sample_size-1)

        self.series[target] = np.array(y_pred)
        if 'graph' in target:
            self.series['ml:ind:trained'] = np.array(y_pred)
            if zero_origin: 
                y = self.series[target]
                #y = y+self.close()
                for i in range(sample_size-1,len(y)): y[i]=y[i]+self.close()[i-sample_size+1] 
                self.series[target] = y
            #self.series['ml:ind:trained'] = np.array(y_pred-self.close())
        else:
            self.series['ml:graph:trained'] = np.array(y_pred+self.close())
            #self.series['ml:graph:trained'] = np.array(y_pred+self.series['input:graph:ema6'])

        #self.trade(min_signal=min_signal,silent=silent)        


    def evaluate_rnn(self, sample_size, target='ml:graph:trained', input_prefix='', min_signal=0, zero_origin=True): 
        #if self.file is not None:
            #self.model = keras.models.load_model(self.model_path + self.file + '.model')

        x_data, _, start = self.get_train_series_for_rnn(sample_size=sample_size, input_prefix=input_prefix, zero_origin=zero_origin)

        y_pred = self.model.predict(x_data)
        y_pred = [y[0] for y in y_pred]
        y_pred = [np.nan] * (start+sample_size-1) + y_pred
        y_pred = y_pred + [np.nan] * (len(self.close()) - len(y_pred)) 

        self.series[target] = np.array(y_pred)
        if 'graph' in target:
            self.series['ml:ind:trained'] = np.array(y_pred)
            if zero_origin: 
                y = self.series[target]
                for i in range(sample_size-1,len(y)): y[i]=y[i]+self.close()[i-sample_size-1] 
                self.series[target] = y
        else:
            self.series['ml:graph:trained'] = np.array(y_pred+self.close())        
            #self.series['ml:graph:trained'] = np.array(y_pred+self.series['input:graph:ema6'])


        
    def trade(self, min_signal = 0, silent=False, source='ml:ind:trained'):
        max_trades = 100
        self.trade_min_signal = min_signal
        self.trade_profit = 0
        self.trades.clear()
        equity = []
        open_trades = []
        max_dropdown = 0
        max_equity = 0
        success_trades = 0
        loss_trades = 0
        gross_profit = 0
        gross_loss = 0.0000001

        signals = self.series[source]

        for i in range(0, len(signals)):
            signal = signals[i] 
            
            if (abs(signal) > min_signal): 
                # close open trades in oposite direction
                if len(open_trades) > 0 and self.sgn(signal,min_signal) == -self.sgn(open_trades[0][2]): #if trade is None or self.sgn(signal,min_signal) != self.sgn(trade[2]): 
                    # close open trades
                    for trade in open_trades:
                        trade[1] = i
                        self.trades.append(trade)
                        profit = (self.close()[trade[1]] - self.close()[trade[0]]) * trade[2] - self.trade_spread
                        self.trade_profit += profit
                        if profit > 0:
                            success_trades += 1
                            gross_profit += profit
                        if profit < 0:
                            loss_trades += 1
                            gross_loss -= profit 
                    open_trades.clear()
                # open trade
                if (len(open_trades) < max_trades):
                    open_trades.append([i,-1,signal])

            eq = self.trade_profit
            for trade in open_trades:
                eq += (self.close()[i] - self.close()[trade[0]]) * trade[2]
            equity.append(eq)
            if eq > max_equity: max_equity = eq
            dropdown = max_equity - eq 
            if dropdown > max_dropdown: max_dropdown = dropdown

        self.series['equity'] = np.array(equity)

        avg_profit = np.nan
        total_profit = np.nan
        success_rate = np.nan
        profit_factor = np.nan

        if len(self.trades) > 0:
            avg_profit = self.trade_profit / len(self.trades)
            total_profit = self.trade_profit
            success_rate = 100 * success_trades/len(self.trades)
            profit_factor = gross_profit/gross_loss
            msg = 'Sig:' + str(min_signal) + ', P:' + str(total_profit) + ', Avg:' + str(avg_profit) + ', SR:' + str(success_rate) + ', PF:' + str(profit_factor) + ', D:' + str(max_dropdown) + ', T: ' + str(len(self.trades))
        else:
            msg = 'Profit: 0'

        if silent == False:
            print(msg)

        return total_profit, avg_profit, profit_factor, success_rate, len(self.trades)





    def plot_equity(self, start=0, length=0, fig_x=22, fig_y=4):
        self.plot_series('equity', start=start, length=length, fig_x=fig_x, fig_y=fig_y, plot_trades=False)


    def plot_graph(self,  filter=':graph', start=0, length=0, fig_x=22, fig_y=4, plot_trades=True):
        self.plot_series(filter, start=start, length=length, fig_x=fig_x, fig_y=fig_y, plot_trades=plot_trades)

    def plot_indicator(self, filter=':ind', start=0, length=0, fig_x=22, fig_y=2.5):
        self.plot_series(filter, start=start, length=length, fig_x=fig_x, fig_y=fig_y)


    def plot_series(self, filter, start=0, length=0, fig_x=22, fig_y=5, plot_trades=False):
        plt.figure(figsize=(fig_x,fig_y))
        count= len(self.close())

        if start < 0: start = max(0,count+start)
        if length <= 0: length = count
        end = start+length
        if end > count: end = count

        for k,v in self.series.items():
            if filter in k:
                plt.plot(range(start,end), v[start:end], label=k)    

        colors = ['red', 'gray', 'green']       
        if plot_trades:
            for t in self.trades:
                if (t[0] >= start and t[1] < start+length):
                    plt.plot([t[0],t[1]], [self.close()[t[0]], self.close()[t[1]]], color=colors[self.sgn(t[2])+1], linewidth=3)         

        plt.legend()
        plt.show()                



    def get_train_series_for_dnn(self, sample_size, target = 'ml:graph:target', input_prefix = None, zero_origin = True):
        zero = self.close()[0]

        x_data = []
        y_data = []
        start=sample_size-1
        for i in range(start,len(self.close())-sample_size+1):
            segment = []
            zero = self.close()[i]

            for k,v in self.series.items():
                if k.startswith(target): continue
                if input_prefix is not None and not k.startswith(input_prefix): continue 
                s =  v[i:i+sample_size]
                if 'graph' in k and zero_origin: s = s - zero
                segment.append(s)

            segment = np.array(segment).flatten()
            if np.isnan(segment).any():
                start = max(start,i+1)
                continue

            y = self.series[target][i+sample_size-1]
            if (np.isnan(y)): continue

            if 'graph' in target and zero_origin: y = y-zero

            x_data.append(segment)
            y_data.append(y)
        #print('train data ',x_data)
        return np.array(x_data), np.array(y_data), start

    def get_train_series_for_rnn(self, sample_size, target = 'ml:graph:target', input_prefix = None, zero_origin = True):
        zero = self.close()[0]

        x_data = []
        y_data = []
        start=sample_size-1
        for i in range(start,len(self.close())-sample_size+1):
            segment = []
            zero = self.close()[i]

            for k,v in self.series.items():
                if k.startswith(target): continue
                if input_prefix is not None and not k.startswith(input_prefix): continue 

                s =  v[i:i+sample_size]
                if 'graph' in k and zero_origin: s = s - zero
                segment.append(s)

            segment = np.array(segment).flatten(order='F')
            if np.isnan(segment).any():
                start = max(start,i+1)
                continue

            y = self.series[target][i+sample_size-1]
            if (np.isnan(y)): continue

            if 'graph' in target and zero_origin: y = y-zero

            x_data.append(segment)
            y_data.append(y)
        #print('train data start:',start)
        x_data = np.array(x_data)
        #print('train data shape:', x_data.shape)
        #print('train data before reshape', x_data)
        x_data = x_data.reshape(-1,sample_size,int(x_data.shape[1]/sample_size))
        #print('train data after reshape', x_data)
        return x_data, np.array(y_data), start

















































    def close(self):
        return self.series['input:graph:close']

    def compute_target_momentum(self, period):        
        r = [self.close()[i+period] - self.close()[i] for i in range(len(self.close())-period)]
        r = r + [np.NaN] * period
        self.series['ml:ind:target'] = np.array(r)
        self.series['ml:graph:target'] = np.array(r+self.close())

    def compute_target_shifted(self, period, original, target='target'):        
        r = [self.series[original][i+period] - self.series[original][i] for i in range(len(self.series[original])-period)]
        r = r + [np.NaN] * period
        self.series['ml:ind:'+target] = np.array(r)
        self.series['ml:graph:'+target] = np.array(r+self.series[original])



    def compute_target_difference(self, period):
        r = []
        for i in range(len(self.close())-period):
            segment = self.close()[i:i+period+1]
            diff = np.max(segment) - 2 * segment[0] + np.min(segment)
            r.append(diff)
        r = r + [np.NaN] * period
        self.series['ml:ind:target'] = np.array(r)
        self.series['ml:graph:target'] = np.array(r+self.close())

    def compute_jma_complex(self, period, phase, count=1):
        for i in range(count):
            self.jma(period,phase)
            self.jmamom(period,phase)
            self.jmacd(period,1,phase)
            self.jmacd(period,int(period/2),phase)
            #self.jmacd(period,5,phase)
            period *= 3


    def compute_target_low(self, period):
        r = []
        for i in range(len(self.close())-period):
            segment = self.close()[i:i+period+1]
            diff = np.min(segment) - segment[0]
            r.append(diff)
        r = r + [np.NaN] * period
        self.series['ml:ind:target'] = np.array(r)
        self.series['ml:graph:target'] = np.array(r+self.close())

    def compute_target_low_ratio(self, period):
        r = []
        for i in range(len(self.close())-period):
            segment = self.close()[i:i+period+1]
            #diff = np.min(segment) - segment[0]
            r.append(np.min(segment)/segment[0]-1)
        r = r + [np.NaN] * period
        self.series['ml:ind:target'] = np.array(r)
        self.series['ml:graph:target'] = np.array(r+self.close())




    def generate(self, trend = 10, noise = 1.5, loops = 10, point_density = 100, seed = None, trend_length=100000, swing=1, long_swing = 0):
        self.trades.clear()
        point_count = point_density * loops
        rnd.seed(seed)
        randoms = [trend/1000 * i + noise * rnd.random() for i in range(point_count)]
        x = np.linspace(0, loops * 2 * 3.14, point_count) 
        y =  long_swing * np.sin(x/10) + swing * np.sin(x) + randoms
        self.series['input:graph:close'] = y 
        #self.series['input:graph:close'] = self.scale_min_max(y)




 #----------------------------------------------------------------------------------------



    def load_and_evaluate(self, file, spread, min_signal=0, max_records = 1000, visualize=False):
        self.load(file,spread,max_records=max_records,silent=True)
        self.evaluate(min_signal=min_signal,silent= not visualize, visualize=visualize)
        print(file, self.get_last_signal())

    def get_last_signal(self):
        sig = self.sgn(signals[-1],self.trade_min_signal)
        if sig > 0:
            return 'BUY'
        if sig < 0:
            return 'SELL'
        return 'HALT'

        
   

    def sgn(self, val, limit=0):
        if val < -limit:
            return -1
        if val > limit:
            return 1
        return 0

    def plot_trades(self):
        balance = [0]
        profit = 0
        for t in self.trades:
            profit += (self.values[t[1]] - self.values[t[0]]) * t[2]
            balance.append(profit)

        plt.figure(figsize=(25,5))
        plt.plot(balance)

    def show_result(self, show_start=0, show_length=0, min_signal=0 ):
        result = self.trade(min_signal=min_signal)
        if (show_length == 0): show_length = len(self.close() - show_start)
        self.plot_equity(start=show_start, length=show_length)
        self.plot_graph(start=show_start, length=show_length, plot_trades = True, filter='input:graph:close')
        self.plot_indicator(start=show_start, length=show_length, filter='ml:ind:trained')
        return result

    def jma(self, period, phase):
        self.series['input:graph:jma:'+str(period)+':'+str(phase)] = ta2.jma(self.close(), period, phase)

    def jmacd(self, slow_period, fast_period, phase):
        macd = ta2.jmacd(self.close(), slow_period, fast_period, phase)
        macd = self.scale_min_max(macd) - 0.5;
        self.series['input:ind:jmacd:'+str(slow_period)+':'+str(fast_period)+':'+str(phase)] = macd

    def jmamom(self, jma_period, jma_phase, mom_period=1):
        mom = ta2.jmamom(self.close(), jma_period,  jma_phase, mom_period);
        mom = self.scale_min_max(mom) - 0.5;
        self.series['input:ind:jmamom:'+str(jma_period)+':'+str(jma_phase)+':'+str(mom_period)] = mom


    def analyze(self, silent = False, train_sample = 10, min_profit=0.0, train_epochs = 100, min_signal=0.1):
        train_type = 'dnn'
        activation = 'softsign'
        loss = softsign_profit_mean(min_profit)

	    #activation = 'linear'
	    #loss = tf.keras.losses.MeanAbsoluteError()

        start = time.time()
        metric = 0
        if train_type == 'rnn':
            testing_set_loss, metric, y_test, y_pred = self.train_rnn(rnn_units=10,sample_size=train_sample, epochs=train_epochs, loss=loss, final_activation=activation) 
        else:
            testing_set_loss, metric, y_test, y_pred = self.train_dnn(sample_size=train_sample, epochs=train_epochs, dropout=0, loss=loss, final_activation=activation) 

        if not silent:
            print('Profit:', -testing_set_loss, 'Train time:', time.time()-start)
            self.show_result(100,400, min_signal)
            self.show_result(min_signal=min_signal)

        total_profit, avg_profit, profit_factor, success_rate, trades = self.trade(min_signal = min_signal, silent=True)
        return -testing_set_loss, total_profit, avg_profit, profit_factor, success_rate, trades

    def generate_zigzag(self, point_count = 1000, min_trend_legth = 10, max_trend_length = 100, min_noise=0.1, max_noise=5, max_trend_strength=1.5, seed = None):
        self.trades.clear()
        rnd.seed(seed)
        y = []
        while True:
            last_y = y[-1] if len(y) > 0 else 0
            noise = min_noise + (max_noise - min_noise) * rnd.random()
            trend_legth = int(min_trend_legth + (max_trend_length - min_trend_legth) * rnd.random())
            trend_k = rnd.random() * 2 - 1
            trend_dir = self.sgn(trend_k);
            trend_k *= max_trend_strength
            trend_k = pow(trend_k,2) * trend_dir;

            for i in range(trend_legth):
                y.append(last_y + i * trend_k + (noise * (rnd.random() * 2 - 1)))
                if len(y) >= point_count: 
                    #self.series['input:graph:close'] = self.scale_min_max(np.array(y))
                    self.series['input:graph:close'] = np.array(y)
                    return

    def prepare_training(self, silent = False, jma_period=15, jma_phase=100, target_divergence_period=10, jma_count=1):
        self.compute_jma_complex(jma_period,jma_phase,jma_count)
        self.compute_target_difference(target_divergence_period)
        if silent == False:
            self.plot_graph(start=100, length=200)
            self.plot_indicator(start=100, length=200, filter = "input:ind")




def create_generated_cycle_graph(silent = False, jma_period=15):
	g = Graph()
	g.generate(trend=40, noise=2, loops=500,point_density=20, swing=0.6, long_swing=2) 	#g.load("US500240.csv")
	g.compute_jma_complex(jma_period,100)
	g.compute_target_difference(10)
	if silent == False:
		g.plot_graph(start=100, length=400)
		g.plot_indicator(start=100, length=400)
	return g

