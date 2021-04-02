import numpy as np
import tensorflow as tf
import keras.backend as K


def custom_ae(y_true, y_pred):
	x = K.abs(y_true - y_pred)
	print('y_true', y_true)
	print('y_pred', y_pred)
	print('loss', x)
	return x

def custom_mae(y_true, y_pred):
	x = K.mean(K.abs(y_true - y_pred))
	print('y_true', y_true)
	print('y_pred', y_pred)
	print('loss', x)
	return x

def softsign_profit(y_true, y_pred):
	#x = K.mean(y_true * y_pred)
	x = -K.sum(y_true * y_pred)
	print('y_true', y_true)
	print('y_pred', y_pred)
	print('loss', x)
	return x



y_true = K.constant(np.array([[-1],[0],[1]]))
y_pred = K.constant(np.array([[-10],[10],[10]]))

print('custom_mae',custom_mae(y_true,y_pred))
print('mae',tf.keras.losses.MeanAbsoluteError()(y_true,y_pred))
print('softsign_profit',softsign_profit(y_true,y_pred))


#print('mae',tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)(y_true,y_pred))

#y_true = K.constant(np.array([0,1]))
#y_pred = K.constant(np.array([[1,0],[1,0]]))
#print('sce',tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(y_true,y_pred))

#a = tf.constant([-10000.0, -1, 0.0, 1, 10000.0], dtype = tf.float32)
#print('softsign', tf.keras.activations.softsign(a))
#print('sigmoid', tf.keras.activations.sigmoid(a))