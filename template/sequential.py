import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from tensorflow import keras
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.losses import MSE
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau

epochs = 300
learning_rate = 0.01
patience = 30


class ForecastLSTM:
	def __init__(self, random_seed: int = 1234):
		self.random_seed = random_seed

	def set_dataset(self, x, y, name):
		self.x = x
		self.y = y
		self.name = name

	def build_model(
			self,
			n_features: int,
			steps: int = 3,
			lstms: tuple = (64, 32),
			seq_len: int = 27,
			dropout: float = 0.0,
			metrics: str = 'mse',
			single_output: bool = False,
			last_lstm_return_sequences: bool = False,
			dense_units: list = None,
			act: str = "relu",
	):
		"""
		Return LSTM

		:param seq_len: Length of sequences. (Look back window size)
		:param n_features: Number of features. It requires for model input shape.
		:param lstms: Number of cells each LSTM layers.
		:param dropout: Dropout rate.
		:param steps: Length to predict.
		:param metrics: Model loss function metric.
		:param single_output: Whether 'yhat' is a multiple value or a single value.
		:param last_lstm_return_sequences: Last LSTM's `return_sequences`. Allow when `single_output=False` only.
		:param dense_units: Number of cells each Dense layers. It adds after LSTM layers.
		:param activation: Activation function of Layers.
		"""

		tf.random.set_seed(self.random_seed)
		model = Sequential()

		# LSTM -> ... -> LSTM -> Dense(steps)

		if len(lstms) == 1:
			model.add(
				LSTM(
					units=lstms[0],
					activation=act,
					return_sequences=False if single_output else last_lstm_return_sequences,
					input_shape=(seq_len, n_features),
				)
			)
		else:
			for i, n_lstm in enumerate(lstms):
				if i == 0:
					node = LSTM(units=lstms[0], activation=act, return_sequences=True, input_shape=(seq_len, n_features))
					model.add(node)
				else:
					return_sequence = False if single_output else last_lstm_return_sequences
					model.add(
						LSTM(
							units=n_lstm,
							activation=act,
							return_sequences=return_sequence if i == len(lstms) - 1 else True,
						)
					)

		if single_output:  # Single Step, Direct Multi Step
			if dense_units:
				for n_units in dense_units:
					model.add(Dense(units=n_units, activation=act))
			if dropout > 0:
				model.add(Dropout(rate=dropout))
			model.add(Dense(1))
		else:  # Multiple Output Step
			if last_lstm_return_sequences:
				model.add(Flatten())
			if dense_units:
				for n_units in dense_units:
					model.add(Dense(units=n_units, activation=act))
			if dropout > 0:
				model.add(Dropout(rate=dropout))
			model.add(Dense(units=steps))

		# Compile the model
		optimizer = Adam(learning_rate=learning_rate)
		model.compile(optimizer=optimizer, loss=MSE, metrics=metrics)
		return model

	def split_sequences(
			self, dataset: np.array, seq_len: int, steps: int, single_output: bool
	) -> tuple:

		# feature와 y 각각 sequential dataset을 반환할 리스트 생성
		X, y = list(), list()
		# sequence length와 step에 따라 sequential dataset 생성
		for i, _ in enumerate(dataset):
			idx_in = i + seq_len
			idx_out = idx_in + steps
			if idx_out > len(dataset):
				break
			seq_x = dataset[i:idx_in, :-1]
			if single_output:
				seq_y = dataset[idx_out - 1: idx_out, -1]
			else:
				seq_y = dataset[idx_in:idx_out, -1]
			X.append(seq_x)
			y.append(seq_y)
		return np.array(X), np.array(y)

	def reshape_dataset(self, df: pd.DataFrame) -> np.array:
		# y 컬럼을 데이터프레임의 맨 마지막 위치로 이동
		if "y" in df.columns:
			df = df.drop(columns=["y"]).assign(y=df["y"])
		else:
			raise KeyError("Not found target column 'y' in dataset.")

		# shape 변경
		dataset = df.values.reshape(df.shape)
		return dataset

	def split_train_valid_dataset(
			self,
			df: pd.DataFrame,
			seq_len: int,
			steps: int,
			single_output: bool,
			validation_split: float = 0.3,
			verbose: bool = True,
	) -> tuple:
		# dataframe을 numpy array로 reshape
		dataset = self.reshape_dataset(df=df)

		# feature와 y를 sequential dataset으로 분리
		X, y = self.split_sequences(
			dataset=dataset,
			seq_len=seq_len,
			steps=steps,
			single_output=single_output,
		)

		# X, y에서 validation dataset 분리
		dataset_size = len(X)
		train_size = int(dataset_size * (1 - validation_split))
		X_train, y_train = X[:train_size, :], y[:train_size, :]
		X_val, y_val = X[train_size:, :], y[train_size:, :]
		if verbose:
			print(f" >>> X_train: {X_train.shape}")
			print(f" >>> y_train: {y_train.shape}")
			print(f" >>> X_val: {X_val.shape}")
			print(f" >>> y_val: {y_val.shape}")
		return X_train, y_train, X_val, y_val

	def fit_lstm(
			self,
			df: pd.DataFrame = None,
			steps: int = 3,
			dropout: float = 0,
			seq_len: int = 27,
			single_output: bool = True,
			patience: int = 10,
			validation_split: float = 0.1,
			last_lstm_return_sequences: bool = False,
			dense_units: list = None,
			metrics: str = "mse",
			check_point_path: str = None,
			verbose: bool = False,
			plot: bool = True,
	):
		"""
		LSTM 기반 모델 훈련을 진행한다.

		:param df: DataFrame for model train.
		:param steps: Length to predict.
		:param lstm_units: LSTM, Dense Layers
		:param activation: Activation function for LSTM, Dense Layers.
		:param dropout: Dropout ratio between Layers.
		:param seq_len: Length of sequences. (Look back window size)
		:param single_output: Select whether 'y' is a continuous value or a single value.
		"""

		np.random.seed(self.random_seed)
		tf.random.set_seed(self.random_seed)

		if df is not None:
			# 훈련, 검증 데이터셋 생성
			(
				self.X_train,
				self.y_train,
				self.X_val,
				self.y_val,
			) = self.split_train_valid_dataset(
				df=df,
				seq_len=seq_len,
				steps=steps,
				validation_split=validation_split,
				single_output=single_output,
				verbose=verbose,
			)
		else:
			self.X_train = self.X_val = self.x
			self.y_train = self.y_val = self.y

		# LSTM 모델 생성
		n_features = self.X_train.shape[1]
		self.model = self.build_model(
			seq_len=seq_len,
			n_features=n_features,
			dropout=dropout,
			steps=steps,
			last_lstm_return_sequences=last_lstm_return_sequences,
			dense_units=dense_units,
			single_output=single_output,
		)

		# 모델 적합 과정에서 best model 저장
		if check_point_path is not None:
			# create checkpoint
			checkpoint_path = f"checkpoint/lstm_{check_point_path}.h5"
			checkpoint = ModelCheckpoint(
				filepath=checkpoint_path,
				save_weights_only=False,
				save_best_only=True,
				monitor="val_loss",
				verbose=verbose,
			)
			rlr = ReduceLROnPlateau(
				monitor="val_loss", factor=0.5, patience=patience, verbose=verbose
			)
			callbacks = [checkpoint, EarlyStopping(patience=patience), rlr]
		else:
			rlr = ReduceLROnPlateau(
				monitor="val_loss", factor=0.5, patience=patience, verbose=verbose
			)
			callbacks = [EarlyStopping(patience=patience), rlr]

		# 모델 훈련
		self.history = self.model.fit(
			self.X_train,
			self.y_train,
			validation_data=(self.X_val, self.y_val),
			epochs=epochs,
			use_multiprocessing=True,
			workers=8,
			verbose=verbose,
			callbacks=callbacks,
			shuffle=False,
		)

		# 훈련 종료 후 best model 로드
		if check_point_path is not None:
			self.model.load_weights(f"checkpoint/lstm_{check_point_path}.h5")

		# 모델링 과정 시각화
		if plot:
			plt.figure(figsize=(12, 6))
			plt.plot(self.history.history[f"{metrics}"])
			plt.plot(self.history.history[f"val_{metrics}"])
			plt.title("Performance Metric")
			plt.xlabel("Epoch")
			plt.ylabel(f"{metrics}")
			if metrics == "mape":
				plt.axhline(y=10, xmin=0, xmax=1, color="grey", ls="--", alpha=0.5)
			plt.legend(["Train", "Validation"], loc="upper right")
			plt.show()


if __name__ == "__main__":
	df = pd.read_csv('./data.csv')
	stocks = df['0'].unique()
	new_dict = {}
	for s in stocks:
		tmp_df = df[df['0'] == s]
		n = tmp_df.shape[0]
		# print(f'Total number of sequence : {n} for Stock code "{s}"')
		full_arr = []
		for r in range(0, n, 3):
			arr = tmp_df.iloc[r, 2:].tolist()
			full_arr += arr
		new_dict[str(s)] = full_arr[:np.argwhere(np.isnan(full_arr))[0][0]]

	# ARIMA model
	# reference: https://dong-guri.tistory.com/9
	# reference2: https://www.kdnuggets.com/2023/08/times-series-analysis-arima-models-python.html
	p = 0
	d = 0       # differential factor: 시계열 data를 station하게 만들기 위한 차분 계수.
	q = 0

	sample_stock_code = '10002'
	sample_stock_arr = np.array(new_dict[sample_stock_code])
	plot_acf(sample_stock_arr)
	plot_pacf(sample_stock_arr)  ## pacf가 시차 1 이후 0으로 수렴하므로 ar1

	length = len(sample_stock_arr)
	x, y = sample_stock_arr[:int(length * 0.9)], sample_stock_arr[int(length * 0.9):]
	model = sm.tsa.ARIMA(x, order=(p, d, q))
	model_fit = model.fit()
	yhat = model_fit.forecast()[0]
	predictions.append(yhat)
	print(model_fit.summary())

	predictions = []
	yhat = model_fit.forecast()[0]
	predictions.append(yhat)
	history.append(y[0])
	print()


	# LSTM
	# reference: https://velog.io/@lazy_learner/LSTM-%EC%8B%9C%EA%B3%84%EC%97%B4-%EC%98%88%EC%B8%A1-%EB%AA%A8%EB%93%88-%EB%A7%8C%EB%93%A4%EA%B8%B0-1
	df.dropna(inplace=True)
	stock, x, y = df.iloc[:, 1], df.iloc[:, 2:-3], df.iloc[:, -3:].sum(axis=1)

	forecast = ForecastLSTM(random_seed=0)
	forecast.set_dataset(x, y, stock)
	forecast.fit_lstm(steps=3)


	# Split
	# n_data, ratio = len(stock), 0.8

	# print(y.shape)
	#
	# model = ARIMA(train['price'], order=(0, 1, 1))
	# model_fit = model.fit(trend='c', full_output=True, disp=True)
	# print(model_fit.summary())

	print()
