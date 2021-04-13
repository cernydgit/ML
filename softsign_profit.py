import keras.backend as K

def softsign_profit_sum(y_true, y_pred):
	return -K.sum(y_true * y_pred)

def softsign_profit_mean(min_profit=0):
	def softsign_profit_mean(y_true, y_pred):
		return -K.mean(y_true * y_pred -  K.abs(y_pred) * min_profit)
	return softsign_profit_mean;

