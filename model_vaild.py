import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
import json


class TimeModel:
    def __init__(self, num_units=64, model_path="GCM_model.h5"):
        self.scaler_ts_X = StandardScaler()
        self.scaler_static_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.num_units  = num_units
        self.model_save_path = model_path
        with open('pre-process/time_serise_attribute.json', 'r') as file:
            time_serise_attribute = json.load(file)
        with open('pre-process/static_attribute.json', 'r') as file:
            static_attribute = json.load(file)
        tmp_folder = 'pre-process/tmp_data'
        tmp_files = os.listdir(tmp_folder)

        all_data = []

        for file in tmp_files:
            if file.endswith('.csv'):
                patient_data = pd.read_csv(os.path.join(tmp_folder, file))
                all_data.append(patient_data)

        data = pd.concat(all_data, ignore_index=True)
        data = data.drop(columns=['Date'])

        target_attribute = [
            '15 min',
            '30 min',
            '45 min',
            '60 min'
        ]


        # 分离特征和目标值
        time_series_features = data[time_serise_attribute].values
        static_features = data[static_attribute].values
        targets = data[target_attribute].values

        def create_sequences(features, targets, static_features, time_steps=10):
            ts_X, static_X, y = [], [], []
            for i in range(len(features) - time_steps):
                ts_X.append(features[i:i+time_steps])
                static_X.append(static_features[i])
                y.append(targets[i+time_steps])
            return np.array(ts_X), np.array(static_X), np.array(y)
        
        ts_X, static_X, y = create_sequences(time_series_features, targets, static_features)

        # 数据标准化
        ts_X = self.scaler_ts_X.fit_transform(ts_X.reshape(-1, ts_X.shape[-1])).reshape(ts_X.shape)

        static_X = self.scaler_static_X.fit_transform(static_X)

        y = self.scaler_y.fit_transform(y)

        self.ts_X_train, self.ts_X_test, self.static_X_train, self.static_X_test, self.y_train, self.y_test = train_test_split(
            ts_X, static_X, y, test_size=0.2, random_state=42
        )

        self.model = None

        # return ts_X_train, ts_X_test, static_X_train, static_X_test, y_train, y_test


        
        


    def build_model(self):
         # 时序数据输入
        ts_input = Input(shape=(self.ts_X_train.shape[1], self.ts_X_train.shape[2]))

        # Define a function for residual LSTM block
        def residual_lstm_block(x, num_units):
            # First LSTM layer
            lstm1 = LSTM(num_units, return_sequences=True)(x)
            # Second LSTM layer
            lstm2 = LSTM(num_units, return_sequences=True)(lstm1)
            # Residual connection
            res = Add()([x, lstm2])
            return res

        # Stack multiple residual LSTM layers
        num_lstm_layers = 20
        lstm_output = ts_input
        for _ in range(num_lstm_layers):
            lstm_output = residual_lstm_block(lstm_output, self.num_units)

        # Flatten LSTM output
        ts_embedding = Flatten()(lstm_output)

        # 静态特征输入
        static_input = Input(shape=(self.static_X_train.shape[1],))

        # 添加八层 Dense 编码器，逐步减少单元数
        y = Dense(64, activation='relu')(static_input)
        y = Dense(56, activation='relu')(y)
        y = Dense(48, activation='relu')(y)
        y = Dense(40, activation='relu')(y)
        y = Dense(36, activation='relu')(y)
        y = Dense(32, activation='relu')(y)
        y = Dense(32, activation='relu')(y)  # 保持 32 维
        y = Dense(32, activation='relu')(y)  # 保持 32 维
        static_embedding = y

        # Cross-Attention层
        def cross_attention(query, key, value):
            attention_scores = tf.matmul(query, key, transpose_b=True)
            attention_weights = tf.nn.softmax(attention_scores, axis=-1)
            attended_vector = tf.matmul(attention_weights, value)
            return attended_vector

        # 使用静态特征作为查询，时序特征作为键和值
        query1 = tf.expand_dims(static_embedding, axis=1)
        key1 = tf.expand_dims(ts_embedding, axis=1)
        value1 = tf.expand_dims(ts_embedding, axis=1)
        cross_attention_output1 = cross_attention(query1, key1, value1)
        cross_attention_output1 = Flatten()(cross_attention_output1)

        # 使用时序特征作为查询，静态特征作为键和值
        query2 = tf.expand_dims(ts_embedding, axis=1)
        key2 = tf.expand_dims(static_embedding, axis=1)
        value2 = tf.expand_dims(static_embedding, axis=1)
        cross_attention_output2 = cross_attention(query2, key2, value2)
        cross_attention_output2 = Flatten()(cross_attention_output2)

        # 合并两个 cross-attention 输出
        merged_attention_output = Concatenate()([cross_attention_output1, cross_attention_output2])

        # 解码层，从合并后的维度逐渐减少到 4
        z = Dense(64, activation='relu')(merged_attention_output)
        z = Dense(56, activation='relu')(z)
        z = Dense(48, activation='relu')(z)
        z = Dense(40, activation='relu')(z)
        z = Dense(36, activation='relu')(z)
        z = Dense(32, activation='relu')(z)
        z = Dense(16, activation='relu')(z)
        output = Dense(4)(z)  # 输出层，预测4个目标值

        # 构建和编译模型
        self.model = Model(inputs=[ts_input, static_input], outputs=output)
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

    
    def train_model(self, epochs=50, batch_size=32, validation_split=0.2):
        history = self.model.fit([self.ts_X_train, self.static_X_train], self.y_train,
                                 epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        self.model.save(self.model_save_path)  # 保存模型
        return history
        
    def evaluate_model(self):
        test_loss, test_mae = self.model.evaluate([self.ts_X_test, self.static_X_test], self.y_test)
        print(f'Test loss: {test_loss}, Test MAE: {test_mae}')
        return f'Test loss: {test_loss}, Test MAE: {test_mae}'


    def predict(self):
        ts_X = self.ts_X_predict # + self.ts_X_test 
        static_X = self.static_X_predict # + self.static_X_test
        # y_pred = self.model.predict([self.ts_X_test, self.static_X_test])
        # y_pred = self.scaler_y.inverse_transform(y_pred)
        # y_test = self.scaler_y.inverse_transform(self.y_test)
        y_pred = self.model.predict([ts_X, static_X])
        y_pred = self.scaler_y.inverse_transform(y_pred)
        return y_pred
    
    def load_model(self, path):
        self.model = load_model(path)
        
    def load_data(self, path):
        
        data = pd.read_csv(path)
        with open('pre-process/time_serise_attribute.json', 'r') as file:
            time_serise_attribute = json.load(file)
        with open('pre-process/static_attribute.json', 'r') as file:
            static_attribute = json.load(file)
        
        data = data.drop(columns=['Date'])
        # 分离特征和目标值
        time_series_features = data[time_serise_attribute].values
        static_features = data[static_attribute].values

        def create_sequences(features, static_features, time_steps=10):
            ts_X, static_X = [], []
            for i in range(len(features) - time_steps):
                ts_X.append(features[i:i+time_steps])
                static_X.append(static_features[i])
            return np.array(ts_X), np.array(static_X)
        
        ts_X, static_X = create_sequences(time_series_features, static_features)

        # 数据标准化
        ts_X = self.scaler_ts_X.fit_transform(ts_X.reshape(-1, ts_X.shape[-1])).reshape(ts_X.shape)

        static_X = self.scaler_static_X.fit_transform(static_X)
        
        self.static_X_predict = static_X,
        self.ts_X_predict = ts_X
    


        # self.ts_X_train, self.ts_X_test, self.static_X_train, self.static_X_test, self.y_train, self.y_test = train_test_split(
        #     ts_X, static_X, y, test_size=0.2, random_state=42
        # )


import json
    
if __name__ == "__main__":
    model = TimeModel()
    model.load_model("GCM_model.h5")
    # model.load_data("data_final.csv")
    # y_pred = model.predict()
    # with open("vaild_y_pred.json", "w") as file:
    #     y_pred_str = json.dumps(y_pred.tolist(), indent=4)
    #     file.write(y_pred_str)
        
    model.load_data("data1.csv")
    y_pred = model.predict()
    with open("vaild1.json", "w") as file:
        y_pred_str = json.dumps(y_pred.tolist(), indent=4)
        file.write(y_pred_str)
        
    model.load_data("data2.csv")
    y_pred = model.predict()
    with open("vaild2.json", "w") as file:
        y_pred_str = json.dumps(y_pred.tolist(), indent=4)
        file.write(y_pred_str)
        
    model.load_data("data3.csv")
    y_pred = model.predict()
    with open("vaild3.json", "w") as file:
        y_pred_str = json.dumps(y_pred.tolist(), indent=4)
        file.write(y_pred_str)
        
    model.load_data("data4.csv")
    y_pred = model.predict()
    with open("vaild4.json", "w") as file:
        y_pred_str = json.dumps(y_pred.tolist(), indent=4)
        file.write(y_pred_str)
        
    model.load_data("data5.csv")
    y_pred = model.predict()
    with open("vaild5.json", "w") as file:
        y_pred_str = json.dumps(y_pred.tolist(), indent=4)
        file.write(y_pred_str)