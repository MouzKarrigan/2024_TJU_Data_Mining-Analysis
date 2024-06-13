import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate
from tensorflow_addons.layers import TCN
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D

# 时序数据输入
ts_input = Input(shape=(ts_X_train.shape[1], ts_X_train.shape[2]))
ts_embedding = TCN(64)(ts_input)
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
output = Dense(3)(output)  # 输出层，预测3个目标值

# 构建和编译模型
model = Model(inputs=[ts_input, static_input], outputs=output)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

model.summary()
