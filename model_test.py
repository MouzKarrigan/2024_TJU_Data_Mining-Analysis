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
        ts_X_train, ts_X_test, static_X_train, static_X_test, y_train, y_test = \
        self.ts_X_train, self.ts_X_test, self.static_X_train, self.static_X_test, self.y_train, self.y_test

        # 时序数据输入
        ts_input = Input(shape=(ts_X_train.shape[1], ts_X_train.shape[2]))
        # ts_embedding = TCN(64)(ts_input)
        ts_embedding = LSTM(self.num_units)(ts_input)
        ts_embedding = Flatten()(ts_embedding)

        # 静态特征输入
        static_input = Input(shape=(static_X_train.shape[1],))
        static_embedding = Dense(64, activation='relu')(static_input)

        # Cross-Attention层
        def cross_attention(x1, x2):
            attention_scores = tf.matmul(x1, x2, transpose_b=True)
            attention_weights = tf.nn.softmax(attention_scores, axis=-1)
            attended_vector = tf.matmul(attention_weights, x2)
            return attended_vector

        cross_attention_output = cross_attention(tf.expand_dims(ts_embedding, axis=1), tf.expand_dims(static_embedding, axis=1))
        cross_attention_output = Flatten()(cross_attention_output)

        # 解码层
        output = Dense(64, activation='relu')(cross_attention_output)
        output = Dense(4)(output)  # 输出层，预测3个目标值

        # 构建和编译模型
        self.model = Model(inputs=[ts_input, static_input], outputs=output)
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

        # model.summary()

    
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
        y_pred = self.model.predict([self.ts_X_test, self.static_X_test])
        y_pred = self.scaler_y.inverse_transform(y_pred)
        y_test = self.scaler_y.inverse_transform(self.y_test)
        return y_pred, y_test
    
    def load_model(self, path):
        self.model = load_model(path)
    

import json
    
if __name__ == "__main__":
    model = TimeModel()
    model.load_model("GCM_model.h5")
    y_pred, y_test = model.predict()
    with open("y_pred.json", "w") as file:
        y_pred_str = json.dumps(y_pred.tolist(), indent=4)
        file.write(y_pred_str)
    with open("y_test.json", "w") as file:
        y_test_str = json.dumps(y_test.tolist(), indent=4)
        file.write(y_test_str)
        
    mae = np.mean(np.abs(y_pred - y_test))
    print(f'MAE: {mae}')