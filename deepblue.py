import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def use_gpu(useGPU):
    num_cores = 4

    if useGPU:
        num_GPU = 1
        num_CPU = 2
    else:
        num_CPU = 2
        num_GPU = 0

    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=num_cores,
                            inter_op_parallelism_threads=num_cores, 
                            allow_soft_placement=True,
                            device_count = {'CPU' : num_CPU,
                                            'GPU' : num_GPU}
                        )

    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)    

    print('Tensorflow version ', tf.__version__)
    print('GPU: ', tf.config.list_physical_devices('GPU'))
    print('CPU: ', tf.config.list_physical_devices('CPU'))


def plot_history(history, metric):
    #print(history.history.keys())
    plotter = tfdocs.plots.HistoryPlotter(smoothing_std=0)
    plt.figure(figsize=(25,5))
    plotter.plot({metric: history}, metric = metric)
    plt.ylabel(metric)
    plt.plot()
    plt.show()


def train_model(model, x_train, y_train, loss, metric, epochs):
    #model.compile(optimizer="adam", loss=loss, metrics=[metric])
    #tf.compat.v1.enable_eager_execution()
    optimizer = keras.optimizers.RMSprop(lr=0.001,rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric]) #, run_eagerly=True)
    #model.run_eagerly=True
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(x_train, y_train, epochs=epochs, validation_split = 0.2, shuffle=True, verbose=0,  callbacks=[early_stop, tfdocs.modeling.EpochDots()])
    print(" done.")
    #plot_history(history, metric)
    plot_history(history, 'loss')
    plt.show()

def test_model(model, x_test, y_test, silent = True):
    loss, metric = model.evaluate(x_test, y_test, verbose=2)
    y_pred = model.predict(x_test)
    
    if not silent:
        print("Testing set loss:",loss)
        plt.scatter(y_test, y_pred)
        plt.show()
    return loss, metric, y_test, y_pred


def prepare_train_data(x_data, y_data, normalize = True):
    #x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=0)
    x_train, x_test, y_train, y_test = train_test_split_tail(x_data, y_data, test_size=0.2)
    #x_train, x_test, y_train, y_test = train_test_split_head(x_data, y_data, test_size=0.2)
    if normalize:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
    print('train data size:', len(x_train))
    #for i in range(3): print(x_train[i],' => ', y_train[i])
    return x_train, x_test, y_train, y_test


def execute(model, x_data, y_data, loss, metric, epochs, normalize = True):
    x_train, x_test, y_train, y_test = prepare_train_data(x_data, y_data, normalize=normalize)
    train_model(model, x_train, y_train, loss=loss, metric=metric, epochs=epochs)
    return test_model(model, x_test, y_test)

def train_test_split_tail(x_data, y_data, test_size=0.2):
    train_size = int(len(x_data) * (1-test_size))
    x_train = x_data[0:train_size]
    y_train = y_data[0:train_size]
    x_test = x_data[train_size:]
    y_test = y_data[train_size:]
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

def train_test_split_head(x_data, y_data, test_size=0.2):
    train_size = int(len(x_data) * (1-test_size))
    x_train = x_data[-train_size:]
    y_train = y_data[-train_size:]
    x_test = x_data[:-train_size]
    y_test = y_data[:-train_size]
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)





