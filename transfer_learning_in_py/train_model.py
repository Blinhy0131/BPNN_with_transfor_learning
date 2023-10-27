from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp


class EarlyStoppingByLoss(keras.callbacks.Callback):
    def __init__(self, monitor, value):
        super().__init__()
        self.monitor = monitor
        self.value = value

    #callback會呼叫此function
    def on_epoch_end(self, epoch, logs={}):
        current_loss = logs.get(self.monitor)
        #判斷是否有達到停止標準 如果有 就停止訓練
        if current_loss is not None and current_loss < self.value:
            print(f"Epoch {epoch + 1}: Stopping training as loss is less than {self.value}")
            self.model.stop_training = True

fs=1/1e2
x_min=0
x_max=10
x=np.arange(x_min , x_max , fs)
y=np.sin(2*x)

# 建立訓練模型 模型名稱叫做model
model = keras.Sequential([
    keras.layers.Input(shape=(1,)),    # input
    keras.layers.Dense(100, activation='tanh'),  # 第一層
    keras.layers.Dense(100, activation='tanh'),   # 第二層
    keras.layers.Dense(100, activation='tanh'),   # 第二層
    keras.layers.Dense(100, activation='tanh'),   # 第二層
    keras.layers.Dense(1)             # output
])

adam = keras.optimizers.Adam(epsilon=1e-7)
# 建立模型 使用adam優化 使用平方差誤差函數
model.compile(optimizer=adam, loss='mean_squared_error')

# 設定停止條件
stop_condition=EarlyStoppingByLoss(monitor='loss',value=1e-4)

# 建立訓練模型 訓練100次 每次疊代完會呼叫stop_condition
history = model.fit(x, y, epochs=100,callbacks=[stop_condition]) 

# 給新的X進行訓練
x_new = np.arange(x_min, x_max,fs)  
y_pred = model.predict(x_new)

# 保存模型
model_json = model.to_json()
with open("bpnn_model.json", "w") as json_file:
    json_file.write(model_json)

# 保存連結權重
model.save_weights("bpnn_model_weights.h5")

# 繪出mse
plt.figure(1)
plt.plot(history.history['loss'])
plt.title('Training Loss (stop at mse<1e-4)')
plt.figure(2)#畫出結果
plt.plot(x, y, label='Original Data')
plt.plot(x_new, y_pred, label='Model Prediction', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()