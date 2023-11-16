import pickle
from statsmodels.tsa.arima_model import ARIMA
from transformers import AutoTokenizer, GPT2Model, PreTrainedTokenizer
import keras_nlp
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import time
from tensorflow import keras
from keras.layers import Dense, LSTM, Dropout, Input, Activation, Flatten, Concatenate
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.losses import MSE
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

epochs = 500
learning_rate = 1e-3
patience = 5


class ForecastLSTM:
    def __init__(self, random_seed: int = 1234):
        self.random_seed = random_seed
        self.pre_set = False
        self.X_train, self.y_train, self.X_val, self.y_val = None, None, None, None
        self.X_train_aux, self.X_val_aux = None, None
        self.history = None

    def set_dataset(self, train_valid_dict: dict):
        self.X_train = np.concatenate(train_valid_dict["train_x"], axis=0)
        self.y_train = np.concatenate(train_valid_dict["train_y"], axis=0)
        self.X_val = np.concatenate(train_valid_dict["valid_x"], axis=0)
        self.y_val = np.concatenate(train_valid_dict["valid_y"], axis=0)
        self.X_train_aux = np.concatenate(train_valid_dict["train_aux"], axis=0)
        self.X_val_aux = np.concatenate(train_valid_dict["val_aux"], axis=0)
        self.pre_set = True

    def build_model(
        self,
        n_features: int = 1,
        steps: int = 3,
        lstms: tuple = (64, 32),
        seq_len: int = 11,
        dropout: float = 0.0,
        metrics: str = "mse",
        single_output: bool = False,
        last_lstm_return_sequences: bool = False,
        dense_units: tuple = (),
        act: str = "relu",
        aux: bool = False,
    ):
        """
        Return LSTM model
        # https://velog.io/@lighthouse97/Tensorflow%EB%A1%9C-%EB%AA%A8%EB%8D%B8%EC%9D%84-%EB%A7%8C%EB%93%9C%EB%8A%94-3%EA%B0%80%EC%A7%80-%EB%B0%A9%EB%B2%95
        # 위 링크 참조 build 바꾸기


        :param seq_len: Length of sequences. (Look back window size)
        :param n_features: Number of features. It requires for model input shape.
        :param lstms: Number of cells each LSTM layers.
        :param dropout: Dropout rate.
        :param steps: Length to predict.
        :param metrics: Model loss function metric.
        :param single_output: Whether 'yhat' is a multiple value or a single value.
        :param last_lstm_return_sequences: Last LSTM's `return_sequences`. Allow when `single_output=False` only.
        :param dense_units: Number of cells each Dense layers. It adds after LSTM layers.
        :param act: Activation function of Layers.
        """

        tf.random.set_seed(self.random_seed)
        model = Sequential()

        # LSTM -> ... -> LSTM -> Dense(steps)
        # return_seq = True if len(lstms) != 1 else False if single_output else last_lstm_return_sequences
        #
        # # return_seq = False if single_output else last_lstm_return_sequences if len(lstms) == 1 else True
        #
        # model.add(LSTM(units=lstms[0], activation=act, return_sequences=return_seq, input_shape=(seq_len, n_features)))
        #
        # for i, n_lstm in enumerate(lstms, start=1):
        #     return_seq = True if i != (len(lstms) - 1) else False if single_output else last_lstm_return_sequences
        #     model.add(
        #         LSTM(units=n_lstm, activation=act, return_sequences=return_seq, input_shape=(seq_len, n_features))
        #     )

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
                    node = LSTM(
                        units=lstms[0], activation=act, return_sequences=True, input_shape=(seq_len, n_features)
                    )
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

        if not single_output and last_lstm_return_sequences:
            model.add(Flatten())
        for n_units in dense_units:
            model.add(Dense(units=n_units, activation=act))
        model.add(Dropout(rate=dropout))
        model.add(Dense(1 if single_output else steps))

        # Compile the model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=MSE, metrics=metrics)
        return model

    def build_model_func(
        self,
        steps: int = 3,
        lstms: tuple = (64, 32),
        seq_len: int = 11,
        dropout: float = 0.0,
        metrics: str = "mse",
        single_output: bool = False,
        last_lstm_return_sequences: bool = False,
        dense_units: tuple = (),
        act: str = "relu",
        aux: bool = (False,),
    ):
        """
        Return LSTM model
        # https://keras.io/ko/getting-started/functional-api-guide/
        # 위 링크 참조 build 바꾸기


        :param seq_len: Length of sequences. (Look back window size)
        :param lstms: Number of cells each LSTM layers.
        :param dropout: Dropout rate.
        :param steps: Length to predict.
        :param metrics: Model loss function metric.
        :param single_output: Whether 'yhat' is a multiple value or a single value.
        :param last_lstm_return_sequences: Last LSTM's `return_sequences`. Allow when `single_output=False` only.
        :param dense_units: Number of cells each Dense layers. It adds after LSTM layers.
        :param act: Activation function of Layers.
        """

        tf.random.set_seed(self.random_seed)
        main_inputs = Input(shape=(seq_len, 1), name="main_inputs")

        # LSTM -> ... -> LSTM -> Dense(steps)

        # return_seq = False if single_output else last_lstm_return_sequences if len(lstms) == 1 else True
        #
        # x = LSTM(units=lstms[0], activation=act, return_sequences=return_seq, input_shape=(seq_len, 1))(main_inputs)
        #
        # for i, n_lstm in enumerate(lstms, start=1):
        #     return_seq = False if single_output else last_lstm_return_sequences if i == len(lstms) - 1 else True
        #     x = LSTM(units=n_lstm, activation=act, return_sequences=return_seq)(x)

        if len(lstms) == 1:
            x = LSTM(
                units=lstms[0],
                activation=act,
                return_sequences=False if single_output else last_lstm_return_sequences,
                input_shape=(seq_len, 1),
            )(main_inputs)

        else:
            for i, n_lstm in enumerate(lstms):
                if i == 0:
                    x = LSTM(units=lstms[0], activation=act, return_sequences=True, input_shape=(seq_len, 1))(
                        main_inputs
                    )
                else:
                    return_sequence = False if single_output else last_lstm_return_sequences
                    x = LSTM(
                        units=n_lstm,
                        activation=act,
                        return_sequences=return_sequence if i == len(lstms) - 1 else True,
                    )(x)

        if not single_output and last_lstm_return_sequences:
            x = Flatten()(x)
        if aux:
            aux_inputs = Input(shape=(20,), name="aux_inputs")
            x = Concatenate()((x, aux_inputs))
        for n_units in dense_units:
            x = Dense(units=n_units, activation=act)(x)
        x = Dropout(rate=dropout)(x)
        outputs = Dense(1 if single_output else steps)(x)

        if aux:
            model = Model(inputs=[main_inputs, aux_inputs], outputs=[outputs])
        else:
            model = Model(inputs=main_inputs, outputs=[outputs])

        # Compile the model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=MSE, metrics=metrics)
        return model

    def split_sequences(self, dataset: np.array, seq_len: int, steps: int, single_output: bool) -> tuple:
        # feature와 y 각각 sequential dataset을 반환할 리스트 생성
        X, y = list(), list()
        # sequence length와 step에 따라 sequential dataset 생성
        for i, _ in enumerate(dataset):
            idx_in = i + seq_len
            idx_out = idx_in + steps
            if idx_out > len(dataset):
                break
            seq_x = dataset[i:idx_in, :-1]
            # if single_output:
            #     seq_y = dataset[idx_out - 1 : idx_out, -1]
            # else:
            #     seq_y = dataset[idx_in:idx_out, -1]

            seq_y = dataset[idx_in:idx_out, -1]
            if single_output:
                seq_y = dataset[idx_out - 1 : idx_out, -1].sum(keepdims=True)
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
        validation_split: float = 0.1,
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
        return X_train, y_train, X_val, y_val

    def fit_lstm(
        self,
        df: pd.DataFrame = None,
        steps: int = 3,
        seq_len: int = 27,
        single_output: bool = True,
        last_lstm_return_sequences: bool = False,
        dense_units: tuple = (),
        metrics: str = "mse",
        check_point_path: str = None,
        plot: bool = True,
        aux_input: bool = False,
        patience: int = 50,
    ):
        """
        LSTM 기반 모델 훈련을 진행한다.

        :param df: DataFrame for model train.
        :param steps: Length to predict.
        :param lstm_units: LSTM, Dense Layers
        :param activation: Activation function for LSTM, Dense Layers.
        :param seq_len: Length of sequences. (Look back window size)
        :param single_output: Select whether 'y' is a continuous value or a single value.
        """

        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)

        if df is not None:
            # 훈련, 검증 데이터셋 생성
            self.X_train, self.y_train, self.X_val, self.y_val = self.split_train_valid_dataset(
                df=df,
                seq_len=seq_len,
                steps=steps,
                single_output=single_output,
            )
        else:
            if not self.pre_set:
                return

        # LSTM 모델 생성
        # n_features = self.X_train.shape[1]
        model_api = self.build_model_func if aux_input else self.build_model
        self.model = model_api(
            seq_len=seq_len,
            steps=steps,
            last_lstm_return_sequences=last_lstm_return_sequences,
            dense_units=dense_units,
            single_output=single_output,
        )
        self.model.summary()

        callbacks = []
        # 모델 적합 과정에서 best model 저장
        if check_point_path is not None:
            checkpoint = ModelCheckpoint(
                filepath=f"checkpoint/lstm_{check_point_path}.h5",
                save_weights_only=False,
                save_best_only=True,
                monitor="val_loss",
            )
            callbacks.append(checkpoint)
        rlr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=patience)
        callbacks += [EarlyStopping(patience=patience), rlr]

        # 모델 훈련
        self.history = self.model.fit(
            [self.X_train, self.X_train_aux] if aux_input else [self.X_train],
            self.y_train,
            validation_data=([self.X_val, self.X_val_aux] if aux_input else [self.X_val], self.y_val),
            epochs=epochs,
            use_multiprocessing=True,
            workers=8,
            callbacks=callbacks,
            shuffle=True,
            batch_size=64,
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
            plt.savefig(f"int({time.time()}).png")
            # plt.show()


def arima_model(sequence, p=1, d=0, q=1, code: str = None):
    # p = 0
    # d = 0  # differential factor: 시계열 data를 station하게 만들기 위한 차분 계수.
    # q = 0

    kernel_size = 5
    sequence = np.convolve(sequence, np.ones(kernel_size) / kernel_size)
    length = len(sequence)

    # Checkout auto-correlation & Partial auto-correlation
    # fig, (ax1, ax2, ax3) = plt.subplots(3)
    # plot_acf(sequence, ax=ax1)
    # plot_acf(d1 := np.diff(sequence), ax=ax2)
    # plot_acf(np.diff(d1), ax=ax3)
    # plot_pacf(d1)
    # plt.show()

    x, y = sequence[: int(length * 0.9)], sequence[int(length * 0.9) :]
    arr = x.tolist()
    model_fit = None
    for _ in y:
        arima = sm.tsa.ARIMA(arr, order=(p, d, q))
        model_fit = arima.fit()
        yhat = model_fit.forecast()[0]
        arr.append(yhat)
    print(model_fit.summary())

    plt.figure(figsize=(16, 8))
    plt.plot(x.tolist() + y.tolist(), color="blue", label="True")
    plt.plot(x.tolist() + [sum(y) / len(y)] * len(y), color="skyblue", label="True-Mean")
    plt.plot(arr, color="red", label="Prediction")
    mse = np.mean((arr[-len(y) :] - y) ** 2)
    plt.legend()
    plt.grid(True)
    plt.title(f"Stock Code: {code if code is not None else '??'} & MSE: {mse:.02f}")
    plt.tight_layout()
    plt.show()
    return mse


def embedding(tknzr, model, sequence):
    # 토큰화
    inputs = tknzr(sequence, return_tensors="pt")
    # 임베딩
    return model(**inputs).detach().numpy()


def tokenize(tknzr, sequence, max_length=20) -> np.ndarray:
    if isinstance(sequence, int):
        sequence = [str(sequence)]
    return np.pad(t := (tknzr(sequence)["input_ids"]), (0, max_length - len(t)))


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    ktokenizer = keras_nlp.models.BertTokenizer.from_preset("bert_tiny_en_uncased")
    model = GPT2Model.from_pretrained("gpt2")

    df = pd.read_csv("./data.csv")
    raw_df = pd.read_excel("./online_retail_II.xlsx")

    tkn_dict = {
        str(code): tokenize(tokenizer, descript)
        for (code, descript), _ in raw_df.groupby(by=["StockCode", "Description"])
    }
    stocks = df["0"].unique()
    new_dict = {}
    for s in stocks:
        tmp_df = df[df["0"] == s]
        n = tmp_df.shape[0]
        full_arr = []
        for r in range(0, n, 3):
            arr = tmp_df.iloc[r, 2:].tolist()
            full_arr += arr
        new_dict[str(s)] = full_arr[: np.argwhere(np.isnan(full_arr))[0][0]]

    # ######## ARIMA model
    # # reference: https://dong-guri.tistory.com/9
    # # reference2: https://www.kdnuggets.com/2023/08/times-series-analysis-arima-models-python.html
    #
    # errors = []
    # for code, arr in new_dict.items():
    #     errors.append(arima_model(np.array(arr), code=code))
    # print(f"Total MSE: {np.mean(errors)}")

    # LSTM
    # reference: https://velog.io/@lazy_learner/LSTM-%EC%8B%9C%EA%B3%84%EC%97%B4-%EC%98%88%EC%B8%A1-%EB%AA%A8%EB%93%88-%EB%A7%8C%EB%93%A4%EA%B8%B0-1
    # df.dropna(inplace=True)
    # stock, x, y = df.iloc[:, 1], df.iloc[:, 2:-3], df.iloc[:, -3:].sum(axis=1)

    forecast = ForecastLSTM(random_seed=0)
    forecast_step = 6
    len_sequence = 24

    # train specific stock code
    code = "10002"
    sample_df = pd.DataFrame(np.stack(((arr := new_dict[code])[:-1], arr[1:]), axis=1), columns=["x", "y"])
    forecast.fit_lstm(
        df=sample_df, steps=3, single_output=True, last_lstm_return_sequences=False, dense_units=(32, 16), patience=50
    )
    with open("./specific_train_history", "wb") as file_pi:
        pickle.dump(forecast.history, file_pi)

    train_valid = defaultdict(list)
    for k, v in new_dict.items():
        tmp_df = pd.DataFrame(np.stack((v[:-1], v[1:]), axis=1), columns=["x", "y"])
        tv = forecast.split_train_valid_dataset(tmp_df, steps=forecast_step, seq_len=len_sequence, single_output=False)
        train_aux = np.stack([tkn_dict[k]] * len(tv[0]))
        valid_aux = np.stack([tkn_dict[k]] * len(tv[2]))
        train_valid["train_x"].append(tv[0])
        train_valid["train_y"].append(tv[1])
        train_valid["valid_x"].append(tv[2])
        train_valid["valid_y"].append(tv[3])
        train_valid["train_aux"].append(train_aux)
        train_valid["val_aux"].append(valid_aux)

    forecast.set_dataset(train_valid_dict=train_valid)
    # train entire stock code
    forecast.fit_lstm(
        df=None,
        steps=forecast_step,
        seq_len=len_sequence,
        single_output=True,
        last_lstm_return_sequences=False,
        dense_units=(32, 16),
        aux_input=True,
        patience=10,
    )
    with open("./entire_train_history_w_aux", "wb") as file_pi:
        pickle.dump(forecast.history, file_pi)

    forecast.fit_lstm(
        df=None,
        steps=forecast_step,
        seq_len=len_sequence,
        single_output=True,
        last_lstm_return_sequences=False,
        dense_units=(32, 16),
        aux_input=False,
        patience=10,
    )
    with open("./entire_train_history_wo_aux", "wb") as file_pi:
        pickle.dump(forecast.history, file_pi)

    print()
